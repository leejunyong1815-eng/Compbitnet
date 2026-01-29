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

# --- 1. 경로 설정 (학습 코드와 동일) ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'model')))

# --- 2. 모듈 임포트 (학습 코드와 동일) ---
try:
    # 복소수 모델 컴포넌트
    from model.complex_model import RealImagGPTLMHeadModel, RealImagConfig
    from model.complex_layers import (
        RealImagLinear, RealImagBitNetLinearPhaseQuant, 
        RealImagTernaryLinear, RealImagFourValLinear, RealImagFiveValLinear, RealImagSixValLinear, RealImagSevenValLinear
    )
    # 실수(Real) 모델 컴포넌트
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

# --- 3. 모델 로더 (학습 코드의 로직 복사) ---
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

    # total_params_cust, trainable_params_cust = get_model_parameters(model)
    # print(f"\nCustom Model Parameters:")
    # print(f"  - Total: {total_params_cust:,}")
    # print(f"  - Trainable: {trainable_params_cust:,}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# --- 4. 체크포인트 로드 유틸리티 ---
def load_weights(model, ckpt_path):
    print(f"Loading weights from: {ckpt_path}")
    
    # 1. 파일인 경우 (.pt, .bin)
    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
    # 2. 폴더인 경우 (Accelerator save_state로 저장된 폴더)
    elif os.path.isdir(ckpt_path):
        # pytorch_model.bin 또는 model.safetensors 찾기
        bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            # mp_rank_00_model_states.pt 같은 파일이 있는지 확인 (DeepSpeed 등)
            # 여기서는 간단히 pytorch_model.bin을 우선적으로 찾음
            print("Warning: Could not find 'pytorch_model.bin' inside directory. Trying Accelerator native load...")
            # Accelerator load는 optimizer까지 로드하므로 메모리를 많이 먹지만, 구조상 어쩔 수 없을 때 사용
            return False # Accelerator로 로드하도록 신호
    else:
        raise ValueError(f"Checkpoint path not found: {ckpt_path}")

    # DataParallel/DeepSpeed 등으로 인해 'module.'이나 'module' 키가 씌워진 경우 처리
    if "module" in state_dict:
        state_dict = state_dict["module"]
    
    # 키 이름 정리 ('module.' 접두사 제거)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # 모델에 로드
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
    
    # Accelerator 초기화 (Multi-GPU 평가 지원)
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Running on device: {device}")

    # 1. 모델 & 토크나이저 초기화 (빈 껍데기)
    model, tokenizer = get_model_and_tokenizer(args.model_name, args.model_type, args.use_flash_attention)
    
    # 2. 가중치 로드
    loaded_manually = load_weights(model, args.ckpt_path)
    
    # 만약 수동 로드(load_state_dict)가 실패했다면(폴더 구조 특수성 등), Accelerator로 로드 시도
    if not loaded_manually:
        print("Using Accelerator to load checkpoint state...")
        # 주의: 이 경우 Optimizer State도 같이 로드될 수 있음
        # 모델을 먼저 prepare 해야 할 수도 있지만, load_state는 보통 모델 구조에 맞게 들어감
        accelerator.load_state(args.ckpt_path)
        
    model.eval() # 평가 모드 전환

    # 3. 데이터셋 로드 (학습 코드와 동일)
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

    # 4. Accelerator Prepare (Device 배치 및 DDP 설정)
    model, eval_loader = accelerator.prepare(model, eval_loader)

    # 5. 평가 루프
    total_loss = 0.0
    total_steps = 0
    
    print("Starting evaluation loop...")
    progress_bar = tqdm(range(len(eval_loader)), disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(eval_loader):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            
            # Multi-GPU 환경에서 Loss 수집
            gathered_loss = accelerator.gather(loss.repeat(args.batch_size))
            
            # 평균 계산 (Batch size 고려 안 하고 단순 평균하면 오차가 있을 수 있으나, 일반적으로는 이렇게 모니터링함)
            total_loss += gathered_loss.mean().item()
            total_steps += 1
            progress_bar.update(1)

    # 최종 결과 계산
    avg_loss = total_loss / total_steps
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")

    # 결과 출력 (메인 프로세스에서만)
    if accelerator.is_local_main_process:
        print("\n" + "="*40)
        print(f"Evaluation Result")
        print(f"Model: {args.model_type}")
        print(f"Checkpoint: {args.ckpt_path}")
        print("-" * 20)
        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Perplexity (PPL): {perplexity:.4f}")
        print("="*40)

        # 결과 파일 저장 (선택 사항)
        result_file = os.path.join(os.path.dirname(args.ckpt_path), "eval_result.txt")
        with open(result_file, "w") as f:
            f.write(f"Loss: {avg_loss}\nPPL: {perplexity}\n")
        print(f"Results saved to {result_file}")

if __name__ == "__main__":
    main()
