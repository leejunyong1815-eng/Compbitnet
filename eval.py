import os
import sys
import argparse
import torch
import math
from tqdm.auto import tqdm
from datasets import load_from_disk
from transformers import (
    AutoConfig, AutoTokenizer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import DataLoader
from accelerate import Accelerator


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'model')))

try:

    from model.complex_model import RealImagGPTLMHeadModel, RealImagConfig
    from model.complex_layers import (
        RealImagLinear, RealImagBitNetLinearPhaseQuant, 
        RealImagTernaryLinear, RealImagFourValLinear, RealImagFiveValLinear, RealImagSixValLinear, RealImagSevenValLinear
    )

    from model.real_model_components import RealGPTLMHeadModel, RealBitNetLinear
    
except ImportError as e:
    print(f"Fatal Error: Could not import required modules.")
    print("Please ensure model files are in './model/'.")
    print(f"Import error: {e}")
    sys.exit(1)


def get_model_parameters(model):
    """Returns the total and trainable parameters of a model."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def get_model_and_tokenizer(model_name: str, model_type: str, use_flash_attention: bool = False):
    """
    Loads the appropriate model and tokenizer based on the specified model_type.
    """

    base_config = AutoConfig.from_pretrained(model_name)
    config_dict = base_config.to_dict()
    

    config_dict['mlp_act'] = 'modrelu' 
    config_dict['use_flash_attention'] = use_flash_attention

    if 'num_key_value_heads' not in config_dict:
        config_dict['num_key_value_heads'] = config_dict['num_attention_heads']
        
    custom_config = RealImagConfig(**config_dict)

    print(f"Loading Model Type: {model_type}")


    if model_type == "standard":
        print("Initializing RealGPTLMHeadModel with Standard Linear Layers...")

        model = RealGPTLMHeadModel(custom_config, linear_layer_class=torch.nn.Linear)
        
    elif model_type == "bitnet":
        print("Initializing RealGPTLMHeadModel with RealFiveValLinear (BitNet logic)...")

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

    # total_params_cust, trainable_params_cust = get_model_parameters(model)
    # print(f"\nCustom Model Parameters:")
    # print(f"  - Total: {total_params_cust:,}")
    # print(f"  - Trainable: {trainable_params_cust:,}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer



def load_weights(model, ckpt_path):
    print(f"Loading weights from: {ckpt_path}")
    

    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")

    elif os.path.isdir(ckpt_path):

        bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:

            print("Warning: Could not find 'pytorch_model.bin' inside directory. Trying Accelerator native load...")

            return False
    else:
        raise ValueError(f"Checkpoint path not found: {ckpt_path}")


    if "module" in state_dict:
        state_dict = state_dict["module"]
    

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {len(unexpected)}")
        
    print("Weights loaded successfully!")
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a Causal Language Model")
    p.add_argument("--model_name",    type=str, default="EleutherAI/pythia-70m")
    p.add_argument("--model_type",    choices=["standard", "complex", "bitnet", "complexbitnet", "complex3bitnet", "complex4bitnet", "complex5bitnet","complex6bitnet", "complex7bitnet"],
                   required=True, help="Type of model architecture.")
    p.add_argument("--ckpt_path",     type=str, required=True, help="Path to the checkpoint (file or directory).")
    p.add_argument("--eval_dir",      type=str, required=True, help="Path to the evaluation dataset (arrow format).")
    p.add_argument("--batch_size",    type=int, default=32)

    p.add_argument("--use_flash_attention", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    

    accelerator = Accelerator()
    device = accelerator.device
    print(f"Running on device: {device}")

    model, tokenizer = get_model_and_tokenizer(args.model_name, args.model_type, args.use_flash_attention)
    
    loaded_manually = load_weights(model, args.ckpt_path)
    
    if not loaded_manually:
        print("Using Accelerator to load checkpoint state...")
        accelerator.load_state(args.ckpt_path)
        
    model.eval()

    print(f"Loading evaluation dataset from {args.eval_dir}...")
    eval_ds = load_from_disk(args.eval_dir)
    print(f"Evaluation samples: {len(eval_ds)}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, return_tensors="pt"
    )

    eval_loader = DataLoader(
        eval_ds, shuffle=False, batch_size=args.batch_size,
        collate_fn=data_collator, num_workers=4, pin_memory=True
    )

    model, eval_loader = accelerator.prepare(model, eval_loader)

    total_loss = 0.0
    total_steps = 0
    
    print("Starting evaluation loop...")
    progress_bar = tqdm(range(len(eval_loader)), disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(eval_loader):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            
            gathered_loss = accelerator.gather(loss.repeat(args.batch_size))
            
            total_loss += gathered_loss.mean().item()
            total_steps += 1
            progress_bar.update(1)

    avg_loss = total_loss / total_steps
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")

    if accelerator.is_local_main_process:
        print("\n" + "="*40)
        print(f"Evaluation Result")
        print(f"Model: {args.model_type}")
        print(f"Checkpoint: {args.ckpt_path}")
        print("-" * 20)
        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Perplexity (PPL): {perplexity:.4f}")
        print("="*40)

        result_file = os.path.join(os.path.dirname(args.ckpt_path), "eval_result.txt")
        with open(result_file, "w") as f:
            f.write(f"Loss: {avg_loss}\nPPL: {perplexity}\n")
        print(f"Results saved to {result_file}")

if __name__ == "__main__":
    main()
