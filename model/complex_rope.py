# real_imag_rope.py
import torch
import torch.nn as nn
from typing import Tuple

class RealImagRotaryEmbedding(nn.Module):
    """
    Applies Rotary Positional Embedding (RoPE) to separate real and imaginary tensors.
    This is achieved by an elementwise multiplication that rotates the input in the complex plane.
    The input tensors are expected to have the sequence dimension at -2 and the feature
    dimension at -1.
    """
    def __init__(self, head_dim: int, base: float = 10000.0):
        """
        Initializes the RealImagRotaryEmbedding module.

        Args:
            head_dim: The feature dimension of the input tensors.
            base: The base value for the inverse frequency calculation.
        """
        super().__init__()
        self.head_dim = head_dim
        # Calculate the inverse frequencies for the positional encoding.
        inv = torch.arange(0, head_dim, dtype=torch.float32) / head_dim
        self.register_buffer("inv_freq", 1.0 / (base ** inv), persistent=False)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor, seq_len: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the rotary embedding.

        Args:
            x_real: The tensor representing the real part of the input. Shape [B, H, S, D].
            x_imag: The tensor representing the imaginary part of the input. Shape [B, H, S, D].
            seq_len: The length of the sequence. If None, it's inferred from the input.

        Returns:
            A tuple containing the rotated real and imaginary tensors.
        """
        # Determine the sequence length.
        S = x_real.size(-2) if seq_len is None else seq_len
        
        # Create the time steps tensor.
        t = torch.arange(S, device=x_real.device, dtype=self.inv_freq.dtype)
        
        # Calculate the rotation angles (theta). Shape: [S, D]
        angles = torch.einsum("s,d->sd", t, self.inv_freq)
        
        # The rotation is equivalent to multiplying by e^(i*theta).
        # e^(i*theta) = cos(theta) + i*sin(theta)
        rot_re = torch.cos(angles)
        rot_im = torch.sin(angles)
        
        # Broadcast the rotation tensors to match the input shape [B, H, S, D].
        while rot_re.dim() < x_real.dim():
            rot_re = rot_re.unsqueeze(0)
            rot_im = rot_im.unsqueeze(0)
            
        # Perform the elementwise complex multiplication:
        # y = x * rot = (x_re + i*x_im) * (rot_re + i*rot_im)
        # y_re = x_re * rot_re - x_im * rot_im
        # y_im = x_re * rot_im + x_im * rot_re
        y_real = x_real * rot_re - x_imag * rot_im
        y_imag = x_real * rot_im + x_imag * rot_re
        
        return y_real, y_imag

# ==============================================================================
# Validation Code
# ==============================================================================
if __name__ == '__main__':
    # Define hyperparameters for the test
    batch_size = 4
    num_heads = 8
    seq_len = 16
    head_dim = 32

    print("--- Validation Start ---")
    
    # 1. Create dummy input tensors
    x_real = torch.randn(batch_size, num_heads, seq_len, head_dim)
    x_imag = torch.randn(batch_size, num_heads, seq_len, head_dim)
    print(f"Input real shape: {x_real.shape}")
    print(f"Input imag shape: {x_imag.shape}")

    # 2. Instantiate the rotary embedding module
    rope = RealImagRotaryEmbedding(head_dim=head_dim)
    
    # 3. Get the rotated outputs
    y_real, y_imag = rope(x_real, x_imag)
    print(f"Output real shape: {y_real.shape}")
    print(f"Output imag shape: {y_imag.shape}")

    # 4. Check if the output shapes are as expected
    expected_shape = (batch_size, num_heads, seq_len, head_dim)
    assert y_real.shape == expected_shape, f"Real part shape mismatch! Expected {expected_shape}, but got {y_real.shape}"
    assert y_imag.shape == expected_shape, f"Imaginary part shape mismatch! Expected {expected_shape}, but got {y_imag.shape}"

    # 5. Check if the magnitude of the vectors is preserved after rotation
    input_magnitude_sq = x_real.pow(2) + x_imag.pow(2)
    output_magnitude_sq = y_real.pow(2) + y_imag.pow(2)
    
    is_close = torch.allclose(input_magnitude_sq, output_magnitude_sq, atol=1e-6)
    print(f"\nIs magnitude preserved after rotation? {is_close}")
    assert is_close, "Magnitude was not preserved after rotation."

    print("\nValidation successful! All tensor shapes and properties are correct.")
    print("--- Validation End ---")
