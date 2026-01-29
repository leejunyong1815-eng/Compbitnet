import os
import sys
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoConfig, AutoTokenizer,
    DataCollatorForLanguageModeling, get_scheduler
)
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# --- Add the model directory to the Python path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'model')))

# --- Import refactored modules directly ---
try:
    # 복소수 모델 컴포넌트
    from model.complex_model import RealImagGPTLMHeadModel, RealImagConfig
    from model.complex_layers import (
        RealImagLinear, RealImagBitNetLinearPhaseQuant, 
        RealImagTernaryLinear, RealImagFourValLinear, RealImagFiveValLinear, RealImagSixValLinear, RealImagSevenValLinear
    
    )
    # 실수(Real) 모델 컴포넌트 (새로 추가됨)
    from model.real_model_components import RealGPTLMHeadModel, RealBitNetLinear
except ImportError as e:
    print(f"Fatal Error: Could not import required modules.")
    print("Please ensure this script is run from the directory and all model files are in './model/'.")
    print(f"Import error: {e}")
    exit()

def get_model_parameters(model):
    """Returns the total and trainable parameters of a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Model & tokenizer loader
def get_model_and_tokenizer(model_name: str, model_type: str, use_flash_attention: bool = False):
    """
    Loads the appropriate model and tokenizer based on the specified model_type.
    """
    # 1. Config 로드
    base_config = AutoConfig.from_pretrained(model_name)
    config_dict = base_config.to_dict()
    
    # Complex/Real 공통 Config 설정 업데이트
    config_dict['mlp_act'] = 'modrelu' # Complex용, Real은 SiLU 등 내부 정의 따름
    config_dict['use_flash_attention'] = use_flash_attention
    # GQA 헤드 설정 (없으면 기본 헤드 수와 동일하게)
    if 'num_key_value_heads' not in config_dict:
        config_dict['num_key_value_heads'] = config_dict['num_attention_heads']
        
    custom_config = RealImagConfig(**config_dict)

    print(f"Loading Model Type: {model_type}")

    # 2. 모델 분기
    if model_type == "standard":
        print("Initializing RealGPTLMHeadModel with Standard Linear Layers...")
        # PyTorch 기본 Linear 사용 (RoPE, SwiGLU 등 구조는 Custom)
        model = RealGPTLMHeadModel(custom_config, linear_layer_class=torch.nn.Linear)
        
    elif model_type == "bitnet":
        print("Initializing RealGPTLMHeadModel with RealFiveValLinear (BitNet logic)...")
        # 새로 만든 RealFiveValLinear 사용 (Complex BitNet과 동일 로직)
        model = RealGPTLMHeadModel(custom_config, linear_layer_class=RealBitNetLinear)
        
    elif model_type in ["complex", "complexbitnet", "complex3bitnet", "complex4bitnet", "complex5bitnet",
                        "complex6bitnet", "complex7bitnet"]:
        print(f"Loading custom Real/Imaginary model (type: {model_type})...")

        quant_layer_map = {
            "complex": RealImagLinear,
            "complexbitnet": RealImagBitNetLinearPhaseQuant,
            "complex3bitnet": RealImagTernaryLinear,
            "complex4bitnet": RealImagFourValLinear,
            "complex5bitnet": RealImagFiveValLinear,
            "complex6bitnet": RealImagSixValLinear,
            "complex7bitnet": RealImagSevenValLinear,
        }
        linear_layer_class = quant_layer_map[model_type]
        print(f"Using linear layer: {linear_layer_class.__name__}")
        
        model = RealImagGPTLMHeadModel(custom_config, linear_layer_class=linear_layer_class)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    total_params_cust, trainable_params_cust = get_model_parameters(model)
    print(f"\nCustom Model Parameters:")
    print(f"  - Total: {total_params_cust:,}")
    print(f"  - Trainable: {trainable_params_cust:,}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer



def parse_args():
    p = argparse.ArgumentParser(description="Train a Causal Language Model")
    p.add_argument("--model_name",    type=str, default="EleutherAI/pythia-70m")
    p.add_argument("--model_type",    choices=["standard", "complex", "bitnet", "complexbitnet", "complex3bitnet", "complex4bitnet", "complex5bitnet", "complex6bitnet", "complex7bitnet"],
                   default="standard", help="Type of model architecture to train.")
    p.add_argument("--use_flash_attention", action="store_true",
                   help="Enable FlashAttention for complex models (requires no padding).")
    p.add_argument("--test_mode", action="store_true",
                   help="Load only a tiny fraction (0.01%%) of the data for a quick test run.")
    p.add_argument("--train_dir",     type=str, required=True, help="Path to the training dataset.")
    p.add_argument("--eval_dir",      type=str, required=True, help="Path to the evaluation dataset.")
    p.add_argument("--output_dir",    type=str, default="checkpoints", help="Directory to save checkpoints.")
    p.add_argument("--batch_size",    type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=0.1)
    p.add_argument("--betas",         type=float, nargs=2, default=[0.9, 0.95])
    p.add_argument("--eps",           type=float, default=1e-8)
    p.add_argument("--epochs",        type=int, default=1)
    p.add_argument("--warmup_steps",  type=int, default=750)
    p.add_argument("--resume_ckpt",   type=str, default=None, help="Path to a checkpoint folder to resume from.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=6,
                   help="Number of steps to accumulate gradients before an optimizer update.")
    p.add_argument("--mlp_ratio",     type=float, default=1.0, help="MLP ratio for complex models (default: 1.0).")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", help="The scheduler type to use (e.g., 'linear', 'cosine', 'constant').")
    return p.parse_args()


def main():
    args = parse_args()
    
    output_dir = os.path.join(os.getcwd(), args.output_dir, args.model_name.split('/')[-1], args.model_type +"_lr"+str(args.learning_rate))
    os.makedirs(output_dir, exist_ok=True)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    model, tokenizer = get_model_and_tokenizer(args.model_name, args.model_type, args.use_flash_attention)

    train_ds = load_from_disk(args.train_dir)
    eval_ds  = load_from_disk(args.eval_dir)

    if args.test_mode:
        print("--- RUNNING IN TEST MODE ---")
        train_subset_size = max(1, int(len(train_ds) * 0.0001))
        eval_subset_size = max(1, int(len(eval_ds) * 0.0001))
        train_ds = train_ds.select(range(train_subset_size))
        eval_ds = eval_ds.select(range(eval_subset_size))
        print(f"Using {train_subset_size} training samples and {eval_subset_size} evaluation samples.")
        print("--------------------------")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, return_tensors="pt"
    )
    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=args.batch_size, persistent_workers=True,
        collate_fn=data_collator, num_workers=8, pin_memory=True
    )
    eval_loader = DataLoader(
        eval_ds, shuffle=False, batch_size=args.batch_size,
        collate_fn=data_collator, num_workers=8, pin_memory=True
    )

    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=args.learning_rate)
    
    total_update_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    
    warmup = min(args.warmup_steps, total_update_steps // 10)
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        pass
        scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_update_steps,
        )
    else:
        scheduler = DummyScheduler(
            optimizer, total_num_steps=total_update_steps, warmup_num_steps=args.warmup_steps
        )
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    start_epoch = 1
    global_step = 0
    log_path = os.path.join(output_dir, "logs")
    if args.resume_ckpt:
        accelerator.load_state(args.resume_ckpt)
        try:
            global_step = int(open(os.path.join(args.resume_ckpt, "last_step.txt")).read().strip())
            start_epoch = int(open(os.path.join(args.resume_ckpt, "last_epoch.txt")).read().strip())
        except (IOError, ValueError):
            print("Could not read step/epoch from checkpoint, starting from scratch.")
            global_step, start_epoch = 0, 1

    tb_writer = SummaryWriter(log_dir=log_path, purge_step=global_step)
    best_eval_loss = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True, unit="batch"
        )
        for step, batch in enumerate(pbar, 1):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
            # if step % args.gradient_accumulation_steps == 0:
            #     global_step += 1

            # if accelerator.distributed_type == DistributedType.DEEPSPEED:
            #     boundary = model.is_gradient_accumulation_boundary()
            #     if boundary:
            #         print("gradient update!")

            # Update tqdm postfix and TensorBoard
            if accelerator.is_local_main_process:
                lr = scheduler.get_last_lr()[0]
                cur_loss = loss.item() # * args.gradient_accumulation_steps
                pbar.set_postfix({"loss": f"{cur_loss:.4f}", "lr": f"{lr:.2e}", "step": global_step})
                tb_writer.add_scalar("train/loss", cur_loss, global_step)
                tb_writer.add_scalar("train/lr", lr, global_step)

            if global_step % 1000 == 0 and global_step != 0:
                ckpt = os.path.join(output_dir, f"step-{global_step}")
                accelerator.save_state(ckpt)
                open(os.path.join(ckpt, "last_step.txt"), "w").write(str(global_step))
                open(os.path.join(ckpt, "last_epoch.txt"), "w").write(str(epoch))

        ckpt = os.path.join(output_dir, f"fin-step-{global_step}")
        accelerator.save_state(ckpt)
        open(os.path.join(ckpt, "last_step.txt"), "w").write(str(global_step))
        open(os.path.join(ckpt, "last_epoch.txt"), "w").write(str(epoch))

        # model.eval()
        # eval_loss = 0.0
        # for batch in tqdm(eval_loader, desc="Eval  ", disable=not accelerator.is_local_main_process, dynamic_ncols=True):
        #     with torch.no_grad():
        #         outputs = model(**batch)
        #         loss = accelerator.gather(outputs.loss)
        #         eval_loss += loss.mean().item()
        
        # eval_loss /= len(eval_loader)

        # if accelerator.is_local_main_process:
        #     tb_writer.add_scalar("eval/loss", eval_loss, global_step)
        #     print(f"Epoch {epoch} — eval loss: {eval_loss:.4f}")

        #     if eval_loss < best_eval_loss:
        #         best_eval_loss = eval_loss
        #         print(f"New best eval loss: {best_eval_loss:.4f}. Saving best model.")
        #         best_ckpt_path = os.path.join(output_dir, "best_model")
        #         accelerator.save_state(best_ckpt_path)

    if accelerator.is_local_main_process:
        tb_writer.close()
        print(f"Training complete. Final checkpoints and logs are in: {output_dir}")

if __name__ == "__main__":
    main()