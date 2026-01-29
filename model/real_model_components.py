import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type

# --- Import configurations and utils from existing files ---
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .complex_model import RealImagConfig

try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    flash_attn_func = None

# ==============================================================================
# 0. Utils & Quantization Helpers
# ==============================================================================

def activation_quant(x: torch.Tensor):
    """
    BitNet b1.58 uses 8-bit activations.
    Range: [-128, 127] scaled to input range.
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

# ==============================================================================
# 1. Layers: Norm, RoPE, Linear (Modified for {-1, 0, 1})
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)

class RealRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32).to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        if seq_len is None:
            seq_len = x.shape[2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RealBitNetLinear(nn.Module):
    """
    Standard BitNet b1.58 Linear Layer.
    Weights are quantized to {-1, 0, 1}.
    """
    def __init__(self, in_features: int, out_features: int, eps: float = 1e-5, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        # Init is handled by model._init_weights

    def weight_quant(self, w: torch.Tensor):
        """
        Quantize weights to {-1, 0, 1}
        Formula: Round(w / mean(|w|)).clamp(-1, 1) * mean(|w|)
        """
        # Calculate scale (mean of absolute weights)
        scale = w.abs().mean().clamp_(min=self.eps)
        
        # Normalize and round
        w_scaled = w / scale
        w_quant = w_scaled.round().clamp_(-1, 1)
        
        # Dequantize (Restore scale for STE backward pass)
        w_dequant = w_quant * scale
        return w_dequant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quantize Weights (Ternary {-1, 0, 1})
        # STE (Straight-Through Estimator): Forward uses quantized, Backward uses full precision gradient
        w_quant = self.weight_quant(self.weight)
        ste_w = self.weight + (w_quant - self.weight).detach()
        
        # 2. Linear Operation
        # Note: In a real kernel, this would use INT2/INT8 GEMM. Here we simulate it with FP.
        out = F.linear(x, ste_w, self.bias)
        return out

# ==============================================================================
# 2. Attention and Block
# ==============================================================================

class RealAttention(nn.Module):
    def __init__(self, config: RealImagConfig, linear_layer_class: Type[nn.Module]):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads else self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.use_flash_attention = config.use_flash_attention and FLASH_ATTENTION_AVAILABLE
        self.rope_base = config.rope_base

        self.q_proj = linear_layer_class(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = linear_layer_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = linear_layer_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = linear_layer_class(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rope = RealRotaryEmbedding(self.head_dim, base=self.rope_base)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, _ = x.shape
        
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(v, seq_len=S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        if self.use_flash_attention:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = flash_attn_func(q, k, v, dropout_p=self.dropout, causal=True)
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Causal Mask Fix (Correct logic)
            causal_mask = torch.tril(torch.ones((S, S), device=x.device, dtype=torch.bool))
            attn_weights.masked_fill_(~causal_mask.view(1, 1, S, S), torch.finfo(attn_weights.dtype).min)

            if attention_mask is not None:
                 if attention_mask.dim() == 2:
                     expanded_mask = attention_mask[:, None, None, :]
                 else:
                     expanded_mask = attention_mask
                 attn_weights.masked_fill_(expanded_mask == 0, torch.finfo(attn_weights.dtype).min)
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2)

        out = out.contiguous().view(B, S, self.hidden_size)
        return self.o_proj(out)

class RealSwiGLU(nn.Module):
    def __init__(self, config: RealImagConfig, linear_layer_class: Type[nn.Module]):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gate_proj = linear_layer_class(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = linear_layer_class(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = linear_layer_class(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        return self.down_proj(self.act_fn(self.gate_proj(x_norm)) * self.up_proj(x_norm))

class RealBitNetSwiGLU(nn.Module):
    def __init__(self, config: RealImagConfig, linear_layer_class: Type[nn.Module]):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gate_proj = linear_layer_class(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = linear_layer_class(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = linear_layer_class(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Norm
        x_norm = self.norm(x)
        
        # BitNet: Quantize activations before Linear
        quant_x = x_norm + (activation_quant(x_norm) - x_norm).detach()
        
        gate = self.gate_proj(quant_x)
        up = self.up_proj(quant_x)
        
        gated = F.silu(gate) * up
        
        # Quantize activations before Down Proj
        quant_gated = gated + (activation_quant(gated) - gated).detach()
        
        return self.down_proj(quant_gated)

class RealBlock(nn.Module):
    def __init__(self, config: RealImagConfig, linear_layer_class: Type[nn.Module]):
        super().__init__()
        self.ln1 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = RealAttention(config, linear_layer_class)
        # Check if using BitNet layer (Supports both correct BitNet and legacy 5-val if needed)
        self.is_bitnet = "BitNet" in linear_layer_class.__name__ or "FiveVal" in linear_layer_class.__name__

        if self.is_bitnet:
            self.mlp = RealBitNetSwiGLU(config, linear_layer_class)
        else:
            self.mlp = RealSwiGLU(config, linear_layer_class)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x_norm = self.ln1(x)
        if self.is_bitnet:
            x_norm = x_norm + (activation_quant(x_norm) - x_norm).detach()
        
        attn_out = self.attn(x_norm, attention_mask)
        x = residual + attn_out
        
        mlp_out = self.mlp(x)
        x = x + mlp_out
        return x

class RealGPTLMHeadModel(PreTrainedModel):
    config_class = RealImagConfig

    def __init__(self, config: RealImagConfig, linear_layer_class: Type[nn.Module] = nn.Linear):
        super().__init__(config)
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            RealBlock(config, linear_layer_class) for _ in range(config.num_hidden_layers)
        ])
        
        self.final_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.embed.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, (nn.Linear, RealBitNetLinear)):
            module.weight.data.normal_(mean=0.0, std=std)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
            
        for name, p in module.named_parameters():
            if "o_proj.weight" in name or "down_proj.weight" in name:
                p.data.normal_(mean=0.0, std=std / math.sqrt(2 * self.config.num_hidden_layers))

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        
        x = self.embed(input_ids)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x, attention_mask)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(loss=loss, logits=logits)