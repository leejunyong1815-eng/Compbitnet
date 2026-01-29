# real_imag_activations.py
import torch
import torch.nn as nn
from typing import Tuple

class CReLU(nn.Module):
    """
    Complex ReLU activation function refactored to use separate real and imaginary tensors.
    It applies the ReLU function independently to the real and imaginary parts.
    """
    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the CReLU activation.

        Args:
            z_real: A tensor representing the real part of the input.
            z_imag: A tensor representing the imaginary part of the input.

        Returns:
            A tuple containing the activated real and imaginary parts.
        """
        return torch.relu(z_real), torch.relu(z_imag)

class ZReLU(nn.Module):
    """
    zReLU activation function refactored to use separate real and imaginary tensors.
    It zeros out elements where either the real or the imaginary part is not positive.
    """
    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the ZReLU activation.

        Args:
            z_real: A tensor representing the real part of the input.
            z_imag: A tensor representing the imaginary part of the input.

        Returns:
            A tuple containing the activated real and imaginary parts.
        """
        # Create a boolean mask where both real and imaginary parts are positive.
        mask = (z_real > 0) & (z_imag > 0)
        # The mask is cast to float (0.0 or 1.0) and applied to both parts.
        return z_real * mask, z_imag * mask

class ModReLU(nn.Module):
    """
    ModReLU activation function refactored to use separate real and imaginary tensors.
    It scales the input by a factor derived from its magnitude and a learnable bias.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Initializes the ModReLU module.

        Args:
            hidden_size: The size of the hidden layer, used for the bias parameter shape.
            eps: A small epsilon value to prevent division by zero.
        """
        super().__init__()
        # The bias is a real-valued parameter, designed to be broadcastable.
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the ModReLU activation.

        Args:
            z_real: A tensor representing the real part of the input.
            z_imag: A tensor representing the imaginary part of the input.

        Returns:
            A tuple containing the activated real and imaginary parts.
        """
        # Calculate the magnitude (modulus) from the real and imaginary parts.
        # Using .pow(2) for squaring the tensors.
        r = torch.sqrt(z_real.pow(2) + z_imag.pow(2) + self.eps)
        
        # Calculate the scale factor based on the magnitude and the learnable bias.
        scale = torch.relu(r + self.b) / (r + self.eps)
        
        # Apply the real-valued scale factor to both the real and imaginary parts.
        return scale * z_real, scale * z_imag

class ComplexSiLU(nn.Module):
    """
    Complex SiLU (Sigmoid Linear Unit) activation function.
    Applies the SiLU function to the real part and passes the imaginary part through.
    This is a common and simple approach to extend SiLU to complex numbers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.silu = nn.SiLU()

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the Complex SiLU activation.
        """
        # 실수부에만 SiLU를 적용합니다.
        return self.silu(z_real),  z_imag

# ACT2CFN 딕셔너리에도 추가합니다.
ACT2CFN = {
    "crelu": lambda hidden_size=None: CReLU(),
    "zrelu": lambda hidden_size=None: ZReLU(),
    "modrelu": lambda hidden_size: ModReLU(hidden_size),
    "complex_silu": lambda hidden_size=None: ComplexSiLU(), # <-- 추가
}

# Note: The factory functions in this dictionary remain the same.
# However, the way you call the forward pass of the created modules has changed.
# For example, instead of `act_fn = ACT2CFN["crelu"](); act_fn(z_complex)`,
# you would now use `act_fn(z_real, z_imag)`.
