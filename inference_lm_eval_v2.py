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

    from model.complex_model import RealImagGPTLMHeadModel, RealImagConfig
    from model.complex_layers import (
        RealImagLinear, RealImagBitNetLinearPhaseQuant, 
        RealImagTernaryLinear, RealImagFourValLinear, RealImagFiveValLinear, RealImagSixValLinear, RealImagSevenValLinear
    
    )

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

    total_params_cust, trainable_params_cust = get_model_parameters(model)
    print(f"\nCustom Model Parameters:")
    print(f"  - Total: {total_params_cust:,}")
    print(f"  - Trainable: {trainable_params_cust:,}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",    type=str, default="EleutherAI/pythia-410m")
    p.add_argument("--model_type",    choices=["standard", "complex", "bitnet", "complexbitnet", "complex3bitnet", "complex4bitnet","complex5bitnet","complex6bitnet","complex7bitnet"],
                   default="complex5bitnet", help="Type of model to load")
    p.add_argument("--output_dir",    type=str, default="./evaluations/")
    p.add_argument("--ckpt_path",   type=str, default='./pythia-410m/complex5bitnet_lr0.0006/fin-step-4674/pytorch_model/mp_rank_00_model_states.pt'),    
    # 
    p.add_argument("--use_flash_attention", action="store_true",
                   help="Enable FlashAttention for complex models (requires no padding).")
    return p.parse_args()



def main():
    args = parse_args()
    
    accelerator = Accelerator()
    
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
    