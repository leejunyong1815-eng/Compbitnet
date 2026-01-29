import torch
import torch.nn as nn
from typing import Tuple, Optional, Type

# --- Import refactored modules ---
from .complex_rope import RealImagRotaryEmbedding
from .complex_attention import (
    scaled_dot_product_attention_real_imag,
    scaled_dot_product_attention_flash_real_imag,
    FLASH_ATTENTION_AVAILABLE
)
from .complex_norm import ComplexRMSNorm #RealImagLayerNorm
from .complex_activations import ACT2CFN
from .complex_layers import activation_quant

class RealImagSelfAttention(nn.Module):
    """
    A self-attention module that operates on separate real and imaginary tensors.
    The specific linear layer used is determined by the `linear_layer_class` argument.
    """
    def __init__(self, hidden_size: int, num_heads: int, linear_layer_class: Type[nn.Module], use_flash_attention: bool = False, rope_base: float = 10000.0, dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.use_flash_attention = use_flash_attention and FLASH_ATTENTION_AVAILABLE
        if use_flash_attention and not FLASH_ATTENTION_AVAILABLE:
            print("Warning: `use_flash_attention=True` but flash-attn is not installed. Falling back to standard attention.")
        
        # Initialize linear projections using the provided layer class
        self.q_proj = linear_layer_class(hidden_size, hidden_size)
        self.k_proj = linear_layer_class(hidden_size, hidden_size)
        self.v_proj = linear_layer_class(hidden_size, hidden_size)
        self.o_proj = linear_layer_class(hidden_size, hidden_size)
        
        self.rope = RealImagRotaryEmbedding(self.head_dim, base=rope_base)
        self.dropout = dropout

    def _split_heads(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x_real.shape
        x_real = x_real.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        x_imag = x_imag.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        return x_real, x_imag

    def _merge_heads(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, S, D = x_real.shape
        x_real = x_real.transpose(1, 2).contiguous().view(B, S, H * D)
        x_imag = x_imag.transpose(1, 2).contiguous().view(B, S, H * D)
        return x_real, x_imag

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        q_r, q_i = self.q_proj(x_real, x_imag)
        k_r, k_i = self.k_proj(x_real, x_imag)
        v_r, v_i = self.v_proj(x_real, x_imag)

        q_r, q_i = self._split_heads(q_r, q_i)
        k_r, k_i = self._split_heads(k_r, k_i)
        v_r, v_i = self._split_heads(v_r, v_i)

        q_r, q_i = self.rope(q_r, q_i)
        k_r, k_i = self.rope(k_r, k_i)

        if self.use_flash_attention:
            out_r, out_i, attn_weights = scaled_dot_product_attention_flash_real_imag(
                q_r, q_i, k_r, k_i, v_r, v_i, self.dropout, is_causal
            )
        else:
            out_r, out_i, attn_weights = scaled_dot_product_attention_real_imag(
                q_r, q_i, k_r, k_i, v_r, v_i, attn_mask, self.dropout, is_causal
            )

        out_r, out_i = self._merge_heads(out_r, out_i)
        out_r, out_i = self.o_proj(out_r, out_i)
        
        return out_r, out_i, attn_weights

class RealImagMLP(nn.Module):
    """A two-layer MLP that operates on separate real and imaginary tensors."""
    def __init__(self, hidden_size: int, intermediate_size: int, linear_layer_class: Type[nn.Module], act: str = "modrelu"):
        super().__init__()
        self.fc_in = linear_layer_class(hidden_size, intermediate_size)
        self.act_fn = ACT2CFN[act](hidden_size=intermediate_size)
        self.fc_out = linear_layer_class(intermediate_size, hidden_size)
        
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_r, h_i = self.fc_in(x_real, x_imag)
        h_r, h_i = self.act_fn(h_r, h_i)
        out_r, out_i = self.fc_out(h_r, h_i)
        return out_r, out_i

class RealImagBlock(nn.Module):
    """A simplified Transformer block where the MLP handles its own normalization."""
    def __init__(self, config: "RealImagConfig", linear_layer_class: Type[nn.Module]):
        super().__init__()
        self.ln1 = ComplexRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.is_bitnet_variant = any(name in linear_layer_class.__name__ for name in ["BitNet", "Ternary", "FourVal", "FiveVal", "SixVal", "SevenVal"])
        
        use_gqa = hasattr(config, "num_key_value_heads") and (config.num_key_value_heads != config.num_attention_heads)
        
        if use_gqa:
            self.attn = RealImagGQA(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                linear_layer_class=linear_layer_class,
                use_flash_attention=getattr(config, 'use_flash_attention', False),
                rope_base=config.rope_theta,
                dropout=config.dropout
            )
        else:
            self.attn = RealImagSelfAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                linear_layer_class=linear_layer_class,
                use_flash_attention=getattr(config, 'use_flash_attention', False),
                rope_base=config.rope_theta,
                dropout=config.dropout
            )

        if self.is_bitnet_variant:
            self.mlp = BitNetSwiGLU(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                linear_layer_class=linear_layer_class,
                eps=config.layer_norm_eps
            )
        else:
            self.mlp = RealImagSwiGLU(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                linear_layer_class=linear_layer_class,
                act=config.mlp_act,
                eps=config.layer_norm_eps
            )

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Attention 부분
        ln1_r, ln1_i = self.ln1(x_real, x_imag)

        if self.is_bitnet_variant:
            ln1_r = ln1_r + (activation_quant(ln1_r) - ln1_r).detach()
            ln1_i = ln1_i + (activation_quant(ln1_i) - ln1_i).detach()
        else:
            ln1_r, ln1_i = ln1_r, ln1_i
        
        attn_out_r, attn_out_i, _ = self.attn(ln1_r, ln1_i, attn_mask=attn_mask)
        res_r_1 = x_real + attn_out_r
        res_i_1 = x_imag + attn_out_i
        
        # 2. MLP 부분
        # VVVVVV --- 수정된 부분 --- VVVVVV
        # ln2 없이 바로 mlp를 호출합니다. mlp가 내부적으로 정규화를 처리합니다.
        mlp_out_r, mlp_out_i = self.mlp(res_r_1, res_i_1)
        # ^^^^^^ --- 수정된 부분 --- ^^^^^^

        res_r_2 = res_r_1 + mlp_out_r
        res_i_2 = res_i_1 + mlp_out_i
        
        return res_r_2, res_i_2

class RealImagGQA(nn.Module):
    """
    A self-attention module implementing Grouped-Query Attention (GQA)
    for separate real and imaginary tensors.
    """
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_key_value_heads: int, # GQA를 위한 인자
                 linear_layer_class: Type[nn.Module],
                 use_flash_attention: bool = False,
                 rope_base: float = 10000.0,
                 dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads # 그룹 수 계산
        self.head_dim = hidden_size // num_heads
        
        self.use_flash_attention = use_flash_attention and FLASH_ATTENTION_AVAILABLE
        if use_flash_attention and not FLASH_ATTENTION_AVAILABLE:
            print("Warning: `use_flash_attention=True` but flash-attn is not installed. Falling back.")

        # --- GQA를 위한 프로젝션 레이어 수정 ---
        # Q는 전체 헤드 수를 사용
        self.q_proj = linear_layer_class(hidden_size, hidden_size)
        # K, V는 key_value 헤드 수만큼의 차원만 가짐
        self.k_proj = linear_layer_class(hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = linear_layer_class(hidden_size, self.num_key_value_heads * self.head_dim)
        
        self.o_proj = linear_layer_class(hidden_size, hidden_size)
        
        self.rope = RealImagRotaryEmbedding(self.head_dim, base=rope_base)
        self.dropout = dropout

    def _repeat_kv(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expands the key and value tensors to match the number of query heads.
        """
        B, H_kv, S, D = x_real.shape
        if self.num_key_value_groups == 1:
            return x_real, x_imag
        
        # [B, H_kv, S, D] -> [B, H_kv, 1, S, D]
        x_real = x_real.unsqueeze(2)
        x_imag = x_imag.unsqueeze(2)
        
        # [B, H_kv, 1, S, D] -> [B, H_kv, n_groups, S, D]
        x_real = x_real.expand(B, H_kv, self.num_key_value_groups, S, D)
        x_imag = x_imag.expand(B, H_kv, self.num_key_value_groups, S, D)
        
        # [B, H_kv, n_groups, S, D] -> [B, H_q, S, D]
        x_real = x_real.reshape(B, self.num_heads, S, D)
        x_imag = x_imag.reshape(B, self.num_heads, S, D)
        
        return x_real, x_imag

    def _split_heads(self, x_real: torch.Tensor, x_imag: torch.Tensor, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D_total = x_real.shape
        head_dim = D_total // num_heads
        x_real = x_real.view(B, S, num_heads, head_dim).transpose(1, 2)
        x_imag = x_imag.view(B, S, num_heads, head_dim).transpose(1, 2)
        return x_real, x_imag

    def _merge_heads(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, S, D = x_real.shape
        x_real = x_real.transpose(1, 2).contiguous().view(B, S, H * D)
        x_imag = x_imag.transpose(1, 2).contiguous().view(B, S, H * D)
        return x_real, x_imag

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        q_r, q_i = self.q_proj(x_real, x_imag)
        k_r, k_i = self.k_proj(x_real, x_imag)
        v_r, v_i = self.v_proj(x_real, x_imag)

        q_r, q_i = self._split_heads(q_r, q_i, self.num_heads)
        k_r, k_i = self._split_heads(k_r, k_i, self.num_key_value_heads)
        v_r, v_i = self._split_heads(v_r, v_i, self.num_key_value_heads)

        q_r, q_i = self.rope(q_r, q_i)
        k_r, k_i = self.rope(k_r, k_i)
        
        # --- KV 헤드 반복 ---
        k_r, k_i = self._repeat_kv(k_r, k_i)
        v_r, v_i = self._repeat_kv(v_r, v_i)

        if self.use_flash_attention:
            out_r, out_i, attn_weights = scaled_dot_product_attention_flash_real_imag(
                q_r, q_i, k_r, k_i, v_r, v_i, self.dropout, is_causal
            )
        else:
            out_r, out_i, attn_weights = scaled_dot_product_attention_real_imag(
                q_r, q_i, k_r, k_i, v_r, v_i, attn_mask, self.dropout, is_causal
            )

        out_r, out_i = self._merge_heads(out_r, out_i)
        out_r, out_i = self.o_proj(out_r, out_i)
        
        return out_r, out_i, attn_weights
    
class BitNetSwiGLU(nn.Module):
    """
    A SwiGLU-like MLP block adapted for BitNet-style quantization.
    It handles normalization and activation quantization internally and only once.
    """
    # __init__ 인자에 linear_layer_class가 있는지 확인
    def __init__(self, hidden_size: int, intermediate_size: int, linear_layer_class: Type[nn.Module], eps: float = 1e-6):
        super().__init__()
        # 1. 정규화 레이어가 LayerNorm이 아닌 ComplexRMSNorm인지 확인
        self.norm = ComplexRMSNorm(hidden_size, eps=eps)
        
        # 2. linear_layer_class를 인자로 받아 사용하는지 확인
        self.gate_proj = linear_layer_class(hidden_size, intermediate_size, eps=eps)
        self.up_proj = linear_layer_class(hidden_size, intermediate_size, eps=eps)
        self.down_proj = linear_layer_class(intermediate_size, hidden_size, eps=eps)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_x_real, norm_x_imag = self.norm(x_real, x_imag)
        
        quant_x_re = norm_x_real + (activation_quant(norm_x_real) - norm_x_real).detach()
        quant_x_im = norm_x_imag + (activation_quant(norm_x_imag) - norm_x_imag).detach()
        
        gate_r, gate_i = self.gate_proj(quant_x_re, quant_x_im)
        up_r, up_i = self.up_proj(quant_x_re, quant_x_im)
        
        gated_r = gate_r * up_r - gate_i * up_i
        gated_i = gate_r * up_i + gate_i * up_r

        quant_gated_r = gated_r + (activation_quant(gated_r) - gated_r).detach()
        quant_gated_i = gated_i + (activation_quant(gated_i) - gated_i).detach()

        return self.down_proj(quant_gated_r, quant_gated_i)
    
class RealImagSwiGLU(nn.Module):
    """
    A SwiGLU-based MLP that now includes its own normalization layer.
    """
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 linear_layer_class: Type[nn.Module],
                 act: str = "complex_silu",
                 eps: float = 1e-6): # eps 인자 추가
        super().__init__()
        
        # VVVVVV --- 수정된 부분 --- VVVVVV
        # 1. 정규화 레이어를 내장합니다.
        self.norm = ComplexRMSNorm(hidden_size, eps=eps)
        # ^^^^^^ --- 수정된 부분 --- ^^^^^^
        
        self.gate_proj = linear_layer_class(hidden_size, intermediate_size)
        self.up_proj = linear_layer_class(hidden_size, intermediate_size)
        self.down_proj = linear_layer_class(intermediate_size, hidden_size)
        self.act_fn = ACT2CFN[act]()

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # VVVVVV --- 수정된 부분 --- VVVVVV
        # 2. forward 시작 시 정규화를 먼저 수행합니다.
        norm_r, norm_i = self.norm(x_real, x_imag)
        # ^^^^^^ --- 수정된 부분 --- ^^^^^^

        gate_r, gate_i = self.gate_proj(norm_r, norm_i)
        up_r, up_i = self.up_proj(norm_r, norm_i)
        
        activated_gate_r, activated_gate_i = self.act_fn(gate_r, gate_i)
        
        gated_r = activated_gate_r * up_r - activated_gate_i * up_i
        gated_i = activated_gate_r * up_i + activated_gate_i * up_r
        
        return self.down_proj(gated_r, gated_i)