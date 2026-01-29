import os
import sys
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoConfig, AutoTokenizer,
    DataCollatorForLanguageModeling, get_scheduler
)
import json
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
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



# def parse_args():
#     p = argparse.ArgumentParser(description="Train a Causal Language Model")
#     p.add_argument("--model_name",    type=str, default="EleutherAI/pythia-70m")
#     p.add_argument("--model_type",    choices=["standard", "complex", "bitnet", "complexbitnet", "complex3bitnet", "complex4bitnet"],
#                    default="standard", help="Type of model architecture to train.")
#     p.add_argument("--use_flash_attention", action="store_true",
#                    help="Enable FlashAttention for complex models (requires no padding).")
#     p.add_argument("--test_mode", action="store_true",
#                    help="Load only a tiny fraction (0.01%%) of the data for a quick test run.")
#     p.add_argument("--train_dir",     type=str, required=True, help="Path to the training dataset.")
#     p.add_argument("--eval_dir",      type=str, required=True, help="Path to the evaluation dataset.")
#     p.add_argument("--output_dir",    type=str, default="checkpoints", help="Directory to save checkpoints.")
#     p.add_argument("--batch_size",    type=int, default=20)
#     p.add_argument("--learning_rate", type=float, default=1e-3)
#     p.add_argument("--weight_decay",  type=float, default=0.1)
#     p.add_argument("--betas",         type=float, nargs=2, default=[0.9, 0.95])
#     p.add_argument("--eps",           type=float, default=1e-8)
#     p.add_argument("--epochs",        type=int, default=3)
#     p.add_argument("--warmup_steps",  type=int, default=750)
#     p.add_argument("--resume_ckpt",   type=str, default=None, help="Path to a checkpoint folder to resume from.")
#     p.add_argument("--gradient_accumulation_steps", type=int, default=6,
#                    help="Number of steps to accumulate gradients before an optimizer update.")
#     return p.parse_args()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",    type=str, default="EleutherAI/pythia-410m")
    p.add_argument("--model_type",    choices=["standard", "complex", "bitnet", "complexbitnet", "complex3bitnet", "complex4bitnet","complex5bitnet","complex6bitnet","complex7bitnet"],
                   default="complex5bitnet", help="Type of model to load")
    p.add_argument("--output_dir",    type=str, default="./evaluations/")
    p.add_argument("--ckpt_path",   type=str, default='/mnt/148TB/LJY/my_bitnet_10BT_ckpt/pythia-410m/complex5bitnet_lr0.0006/fin-step-4674/pytorch_model/mp_rank_00_model_states.pt'),    
    # 
    p.add_argument("--use_flash_attention", action="store_true",
                   help="Enable FlashAttention for complex models (requires no padding).")
    return p.parse_args()



def main():
    args = parse_args()
    
    # output_dir = os.path.join(os.getcwd(), args.output_dir, args.model_name.split('/')[-1], args.model_type)
    # os.makedirs(output_dir, exist_ok=True)

    accelerator = Accelerator()#gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    model, tokenizer = get_model_and_tokenizer(args.model_name, args.model_type, args.use_flash_attention)

    model, tokenizer = accelerator.prepare(model, tokenizer)

    temp = torch.load(args.ckpt_path)

    model.load_state_dict(temp['module'])

    # https://github.com/EvolvingLMMs-Lab/lmms-eval
    results = evaluator.simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=tokenizer, dtype="bf16"),#,torch_dtype="float32"),
        # tasks=['arc_easy'],
        tasks=["lambada_openai","arc_easy","arc_challenge","sciq"],   # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md
        # tasks=['arc_easy,piqa,winogrande,arc_challenge,lambda_openai,wsc273,sciq,logiqa'],   # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md
        verbosity="WARNING",
        device="cuda:0",
        num_fewshot=5,
        batch_size="auto", # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
    )
    model_name = args.model_name.split("/")[-1]
    
    with open(args.output_dir+args.model_type+"_"+model_name+".json","w") as json_file:
        json.dump(results['results'], json_file)

    return results
    
if __name__ == "__main__":
    results = main()

                        #    ╱|、
                        #   (˚ˎ。7  
                        #   |、˜〵          
                        #   じしˍ,)ノ 

    print(results['results'])
    