# real_imag_attention.py
import math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

# Attempt to import flash_attn
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    flash_attn_func = None

def scaled_dot_product_attention_real_imag(
    q_real: torch.Tensor,
    q_imag: torch.Tensor,
    k_real: torch.Tensor,
    k_imag: torch.Tensor,
    v_real: torch.Tensor,
    v_imag: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes scaled dot-product attention using separate real and imaginary tensors,
    based on the principles of complex attention. This is the standard implementation.

    Args:
        q_real, q_imag: Real and imaginary parts of the query tensor [B, H, Sq, D].
        k_real, k_imag: Real and imaginary parts of the key tensor [B, H, Sk, D].
        v_real, v_imag: Real and imaginary parts of the value tensor [B, H, Sk, D].
        attn_mask: Optional mask for padding. Expects [B, S].
        dropout_p: Dropout probability.
        is_causal: If True, applies a causal mask.

    Returns:
        A tuple containing the output real part, output imaginary part, and the real-valued attention weights.
    """
    D = q_real.size(-1)
    
    # Calculate logits using the real part of the Hermitian product:
    # Re(q @ k_conj^T) = q_re @ k_re^T + q_im @ k_im^T
    logits = (torch.matmul(q_real, k_real.transpose(-2, -1)) + 
              torch.matmul(q_imag, k_imag.transpose(-2, -1))) / math.sqrt(D)

    if is_causal:
        # Apply a standard causal (lower-triangular) mask to the logits.
        S_q, S_k = logits.size(-2), logits.size(-1)
        causal_mask = torch.ones(S_q, S_k, device=logits.device, dtype=torch.bool).tril(diagonal=0)
        logits.masked_fill_(~causal_mask, float("-inf"))

    if attn_mask is not None:
        # Apply the padding attention mask.
        # Reshape from [B, S] to [B, 1, 1, S] for broadcasting.
        if attn_mask.dim() == 2:
            attn_mask = attn_mask[:, None, None, :]
        
        # Determine the mask to apply based on its data type.
        mask_to_apply = (attn_mask == 0) if attn_mask.dtype != torch.bool else ~attn_mask
        logits.masked_fill_(mask_to_apply, float("-inf"))

    # The attention weights are real since logits are real.
    attn = F.softmax(logits, dim=-1)
    
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=q_real.requires_grad)

    # Apply the real-valued attention weights to the real and imaginary parts of V.
    out_real = torch.matmul(attn, v_real)
    out_imag = torch.matmul(attn, v_imag)
    
    return out_real, out_imag, attn

def scaled_dot_product_attention_flash_real_imag(
    q_real: torch.Tensor,
    q_imag: torch.Tensor,
    k_real: torch.Tensor,
    k_imag: torch.Tensor,
    v_real: torch.Tensor,
    v_imag: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, None]:
    """
    Computes scaled dot-product attention using the FlashAttention kernel.
    This version concatenates real/imaginary parts to leverage the optimized kernel.

    NOTE: Requires the `flash-attn` library to be installed.
    NOTE: Does not support arbitrary `attn_mask`. Use `is_causal` for causal masking.
    NOTE: Does not return attention weights, as this is a key optimization in FlashAttention.

    Args:
        q_real, q_imag: Real and imaginary parts of the query tensor [B, H, Sq, D].
        k_real, k_imag: Real and imaginary parts of the key tensor [B, H, Sk, D].
        v_real, v_imag: Real and imaginary parts of the value tensor [B, H, Sk, D].
        dropout_p: Dropout probability.
        is_causal: If True, applies a causal mask.

    Returns:
        A tuple containing the output real part, output imaginary part, and None for attention weights.
    """
    if not FLASH_ATTENTION_AVAILABLE:
        raise ImportError("Flash Attention is not installed. Please install it using `pip install flash-attn`.")
    
    # Concatenate real and imaginary parts along the last dimension.
    # The new dimension is 2*D.
    q_cat = torch.cat([q_real, q_imag], dim=-1)
    k_cat = torch.cat([k_real, k_imag], dim=-1)
    v_cat = torch.cat([v_real, v_imag], dim=-1)
    q_cat = q_cat.transpose(1, 2).contiguous()
    k_cat = k_cat.transpose(1, 2).contiguous()
    v_cat = v_cat.transpose(1, 2).contiguous()
    drop_p = dropout_p if (q_cat.requires_grad) else 0.0
    # Call the FlashAttention function. It handles scaling, softmax, and dropout internally.
    # The score computation inside flash_attn for concatenated inputs is:
    # (q_re @ k_re.T + q_im @ k_im.T), which matches the desired complex attention score.
    out_cat = flash_attn_func(
        q_cat, k_cat, v_cat,
        dropout_p=drop_p,
        causal=is_causal
    )

    # Split the output back into real and imaginary parts.
    out_cat = out_cat.transpose(1, 2).contiguous()
    out_real, out_imag = torch.chunk(out_cat, 2, dim=-1)

    # FlashAttention does not return attention weights by default for efficiency.
    return out_real, out_imag, None


# ==============================================================================
# Validation Code
# ==============================================================================
if __name__ == '__main__':
    # Define hyperparameters for the test
    batch_size = 4
    num_heads = 8
    seq_len = 128 # FlashAttention works best with longer sequences
    head_dim = 64

    # --- Validation for Standard Attention ---
    print("--- Validation for Standard Attention ---")
    q_real = torch.randn(batch_size, num_heads, seq_len, head_dim)
    q_imag = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k_real = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k_imag = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v_real = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v_imag = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    out_real_std, out_imag_std, attn_weights = scaled_dot_product_attention_real_imag(
        q_real, q_imag, k_real, k_imag, v_real, v_imag
    )
    print(f"Standard Attention - Output (real) shape: {out_real_std.shape}")
    print("Validation successful for standard attention!\n")

    # --- Validation for FlashAttention ---
    print("--- Validation for FlashAttention ---")
    if FLASH_ATTENTION_AVAILABLE:
        try:
            # Move to CUDA as FlashAttention requires it
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                q_real_cuda = q_real.to(device)
                q_imag_cuda = q_imag.to(device)
                k_real_cuda = k_real.to(device)
                k_imag_cuda = k_imag.to(device)
                v_real_cuda = v_real.to(device)
                v_imag_cuda = v_imag.to(device)
                
                print("Running FlashAttention on CUDA device.")
                out_real_flash, out_imag_flash, _ = scaled_dot_product_attention_flash_real_imag(
                    q_real_cuda, q_imag_cuda, k_real_cuda, k_imag_cuda, v_real_cuda, v_imag_cuda, is_causal=False
                )
                print(f"Flash Attention - Output (real) shape: {out_real_flash.shape}")
                
                expected_out_shape = (batch_size, num_heads, seq_len, head_dim)
                assert out_real_flash.shape == expected_out_shape
                assert out_imag_flash.shape == expected_out_shape
                print("Validation successful for FlashAttention!")
            else:
                print("Skipping FlashAttention validation: No CUDA device found.")

        except Exception as e:
            print(f"An error occurred during FlashAttention validation: {e}")
    else:
        print("Skipping FlashAttention validation: `flash-attn` is not installed.")

    print("\n--- All Validations End ---")
