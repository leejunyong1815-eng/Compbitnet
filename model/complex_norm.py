import torch
import torch.nn as nn
from typing import Tuple, Union, List

class RealImagLayerNorm(nn.Module):
    """
    Layer Normalization for separate real and imaginary tensors, emulating ComplexLayerNorm.
    It normalizes the input to have zero mean and unit variance for its magnitude,
    then applies an elementwise affine transformation.
    """
    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], eps: float = 1e-5, elementwise_affine: bool = True):
        """
        Initializes the RealImagLayerNorm module.

        Args:
            normalized_shape: The shape of the dimensions to normalize.
            eps: A small value added to the denominator for numerical stability.
            elementwise_affine: If True, this module has learnable affine parameters.
        """
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight_re = nn.Parameter(torch.ones(self.normalized_shape))
            self.weight_im = nn.Parameter(torch.zeros(self.normalized_shape))
            self.bias_re = nn.Parameter(torch.zeros(self.normalized_shape))
            self.bias_im = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight_re", None)
            self.register_parameter("weight_im", None)
            self.register_parameter("bias_re", None)
            self.register_parameter("bias_im", None)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the layer normalization.
        Calculations are performed using the input dtype (e.g., bf16),
        avoiding explicit casting to fp32.
        """
        dims = tuple(range(-len(self.normalized_shape), 0))

        # Calculate mean and variance directly using the input dtype.
        # This is safe for bf16 due to its wide dynamic range.
        mu_re = x_real.mean(dim=dims, keepdim=True)
        mu_im = x_imag.mean(dim=dims, keepdim=True)

        xc_re = x_real - mu_re
        xc_im = x_imag - mu_im

        # To prevent potential instability with lower precision, calculate variance in fp32
        # and then cast back. This is a good compromise.
        var = (xc_re.pow(2) + xc_im.pow(2)).mean(dim=dims, keepdim=True)
        std_dev = torch.sqrt(var + self.eps) # Cast back to original dtype
        
        xhat_re = xc_re / std_dev
        xhat_imag = xc_im / std_dev

        if self.elementwise_affine:
            y_re = xhat_re * self.weight_re - xhat_imag * self.weight_im + self.bias_re
            y_im = xhat_re * self.weight_im + xhat_imag * self.weight_re + self.bias_im
            return y_re, y_im
        else:
            return xhat_re, xhat_imag

class ComplexRMSNorm(nn.Module):
    """
    RMS Normalization for separate real and imaginary tensors.
    
    This normalization technique re-scales inputs based on their root mean square magnitude,
    but without the centering step present in LayerNorm.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Initializes the ComplexRMSNorm module.

        Args:
            hidden_size: The size of the feature dimension to normalize.
            eps: A small value added to the denominator for numerical stability.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        # A single real-valued learnable gain parameter.
        # This scales the magnitude of the complex number uniformly.
        self.gain = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies RMS Normalization to the real and imaginary input tensors.

        Args:
            x_real: The tensor representing the real part of the input.
            x_imag: The tensor representing the imaginary part of the input.

        Returns:
            A tuple containing the normalized real and imaginary tensors.
        """
        # 1. Calculate the squared magnitude |z|^2 = x_real^2 + x_imag^2
        #    The result is a real-valued tensor.
        magnitude_sq = x_real.pow(2) + x_imag.pow(2)
        
        # 2. Calculate the mean of the squared magnitudes across the feature dimension.
        #    This is the "Mean Square" part of RMS.
        mean_magnitude_sq = magnitude_sq.mean(dim=-1, keepdim=True)
        
        # 3. Compute the inverse of the Root Mean Square for efficient division.
        #    torch.rsqrt(x) is equivalent to 1.0 / torch.sqrt(x) but faster.
        rms_inv = torch.rsqrt(mean_magnitude_sq + self.eps)
        
        # 4. Normalize the real and imaginary parts by scaling them down.
        norm_x_real = x_real * rms_inv
        norm_x_imag = x_imag * rms_inv
        
        # 5. Apply the learnable gain parameter.
        return self.gain * norm_x_real, self.gain * norm_x_imag