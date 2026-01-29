# real_imag_layers.py
import math
import torch
import torch.nn as nn
from typing import Tuple

class RealImagLinear(nn.Module):
    """
    A linear layer that operates on separate real and imaginary tensors,
    emulating a complex linear transformation.

    The transformation is equivalent to Wz, where W and z are complex:
    Wz = (W_r + iW_i)(z_r + iz_i) = (W_r z_r - W_i z_i) + i(W_i z_r + W_r z_i)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initializes the RealImagLinear layer.

        Args:
            in_features: The number of input features for each of the real and imaginary parts.
            out_features: The number of output features for each of the real and imaginary parts.
            bias: If set to False, the layer will not learn an additive bias.
        """
        super().__init__()
        # We need two weight matrices to handle the real and imaginary transformations.
        self.weight_re = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_im = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias_re = nn.Parameter(torch.zeros(out_features))
            self.bias_im = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_re', None)
            self.register_parameter('bias_im', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the weight and bias parameters.
        The initialization is similar to Kaiming uniform for complex numbers.
        """
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_re)
        std = math.sqrt(1.0 / fan_in)  # Adjusted std based on complex Kaiming
        
        # Initialize real and imaginary weights from a normal distribution
        nn.init.normal_(self.weight_re, 0, std)
        nn.init.normal_(self.weight_im, 0, std)
        
        if self.bias_re is not None:
            nn.init.zeros_(self.bias_re)
            nn.init.zeros_(self.bias_im)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the linear transformation using real and imaginary components.

        Args:
            x_real: The tensor representing the real part of the input.
            x_imag: The tensor representing the imaginary part of the input.

        Returns:
            A tuple containing the transformed real and imaginary tensors.
        """
        # Calculate the real part of the output
        # y_real = (x_real @ W_r^T) - (x_imag @ W_i^T)
        y_real = x_real @ self.weight_re.t() - x_imag @ self.weight_im.t()
        
        # Calculate the imaginary part of the output
        # y_imag = (x_real @ W_i^T) + (x_imag @ W_r^T)
        y_imag = x_real @ self.weight_im.t() + x_imag @ self.weight_re.t()

        if self.bias_re is not None:
            y_real = y_real + self.bias_re
            y_imag = y_imag + self.bias_im
            
        return y_real, y_imag

class RealImagBitNetLinearPhaseQuant(nn.Module):
    """
    A BitNet-style linear layer for complex numbers using PhaseQuant.
    It quantizes the weights to {+1, -1, +i, -i} and uses full-precision inputs.
    The implementation follows the architecture from the provided diagram.
    """
    def __init__(self, in_features: int, out_features: int, eps: float = 1e-6):
        """
        Initializes the RealImagBitNetLinearPhaseQuant layer.

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            eps: A small value for numerical stability in LayerNorm and scaling.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        # Full precision weights are stored for training.
        self.weight_re = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_im = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

        # LayerNorm is applied to the concatenated real and imaginary parts of the input.
        self.layernorm = nn.LayerNorm(2 * in_features, eps=eps)

    def reset_parameters(self):
        """Initializes the full-precision weights."""
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_re)
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(self.weight_re, 0, std)
        nn.init.normal_(self.weight_im, 0, std)

    def phase_quantize_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs Phase Quantization on the weights.
        A complex weight w = w_re + i*w_im is quantized to the nearest value
        in {+1, -1, +i, -i} based on which quadrant it lies in.
        """
        abs_re = self.weight_re.abs()
        abs_im = self.weight_im.abs()

        # Create masks to determine if the real or imaginary part is dominant.
        mask_re_dominant = abs_re >= abs_im
        mask_im_dominant = ~mask_re_dominant

        quant_w_re = torch.zeros_like(self.weight_re)
        quant_w_im = torch.zeros_like(self.weight_im)

        # If real part is dominant, quantize to +1 or -1.
        quant_w_re[mask_re_dominant] = self.weight_re[mask_re_dominant].sign()
        # If imaginary part is dominant, quantize to +i or -i.
        quant_w_im[mask_im_dominant] = self.weight_im[mask_im_dominant].sign()
        
        return quant_w_re, quant_w_im

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass.
        """
        # 1. Apply LayerNorm to the input.
        x_cat = torch.cat([x_real, x_imag], dim=-1)
        norm_x_cat = self.layernorm(x_cat)
        norm_x_real, norm_x_imag = torch.chunk(norm_x_cat, 2, dim=-1)

        # 2. Quantize weights using PhaseQuant and Straight-Through Estimator (STE).
        # STE allows gradients to flow through the quantization step.
        quant_w_re, quant_w_im = self.phase_quantize_weights()
        ste_w_re = self.weight_re + (quant_w_re - self.weight_re).detach()
        ste_w_im = self.weight_im + (quant_w_im - self.weight_im).detach()

        # 3. Calculate scaling factor for the weights (dequantization scale).
        # This is the average absolute value of the full-precision weights.
        s_w_re = self.weight_re.abs().mean()
        s_w_im = self.weight_im.abs().mean()
        
        # 4. Perform the complex linear transformation using normalized input and STE weights.
        y_real = norm_x_real @ ste_w_re.t() - norm_x_imag @ ste_w_im.t()
        y_imag = norm_x_real @ ste_w_im.t() + norm_x_imag @ ste_w_re.t()

        # 5. Dequantize the output by scaling with the weight scales.
        dequant_y_real = y_real * s_w_re
        dequant_y_imag = y_imag * s_w_im
        
        return dequant_y_real, dequant_y_imag

class RealImagTernaryLinear(nn.Module):
    """
    A BitNet-style linear layer that quantizes the real and imaginary parts of its weights
    to ternary values {-1, 0, 1}.
    """
    def __init__(self, in_features: int, out_features: int, threshold: float = 0.05, eps: float = 1e-6):
        """
        Initializes the RealImagTernaryLinear layer.

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            threshold: The threshold for quantizing weights to 0.
            eps: A small value for numerical stability.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.threshold = threshold

        # Full precision weights are stored for training.
        self.weight_re = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_im = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

        # LayerNorm is applied to the concatenated real and imaginary parts of the input.
        self.layernorm = nn.LayerNorm(2 * in_features, eps=eps)

    def reset_parameters(self):
        """Initializes the full-precision weights."""
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_re)
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(self.weight_re, 0, std)
        nn.init.normal_(self.weight_im, 0, std)

    def ternarize_weights(self, w: torch.Tensor) -> torch.Tensor:
        """Quantizes a weight tensor to {-1, 0, 1} based on a threshold."""
        output = torch.zeros_like(w)
        # Positive values above threshold become 1
        output[w > self.threshold] = 1
        # Negative values below -threshold become -1
        output[w < -self.threshold] = -1
        # Values in [-threshold, threshold] remain 0
        return output

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass."""
        # 1. Apply LayerNorm to the input.
        x_cat = torch.cat([x_real, x_imag], dim=-1)
        norm_x_cat = self.layernorm(x_cat)
        norm_x_real, norm_x_imag = torch.chunk(norm_x_cat, 2, dim=-1)

        # 2. Ternarize weights and use Straight-Through Estimator (STE).
        quant_w_re = self.ternarize_weights(self.weight_re)
        quant_w_im = self.ternarize_weights(self.weight_im)
        ste_w_re = self.weight_re + (quant_w_re - self.weight_re).detach()
        ste_w_im = self.weight_im + (quant_w_im - self.weight_im).detach()

        # 3. Calculate scaling factor for the weights (dequantization scale).
        s_w_re = self.weight_re.abs().mean()
        s_w_im = self.weight_im.abs().mean()
        
        # 4. Perform the complex linear transformation using normalized input and STE weights.
        y_real = norm_x_real @ ste_w_re.t() - norm_x_imag @ ste_w_im.t()
        y_imag = norm_x_real @ ste_w_im.t() + norm_x_imag @ ste_w_re.t()

        # 5. Dequantize the output by scaling with the weight scales.
        dequant_y_real = y_real * s_w_re
        dequant_y_imag = y_imag * s_w_im
        
        return dequant_y_real, dequant_y_imag

class RealImagFourValLinear(nn.Module):
    """
    A BitNet-style linear layer that quantizes the real and imaginary parts of its weights
    to four values {-1, -0.5, 0.5, 1}.
    """
    def __init__(self, in_features: int, out_features: int, high_threshold: float = 0.75, low_threshold: float = 0.25, eps: float = 1e-6):
        """
        Initializes the RealImagFourValLinear layer.

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            high_threshold: The threshold for quantizing weights to +/-1.
            low_threshold: The threshold for quantizing weights to +/-0.5.
            eps: A small value for numerical stability.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

        # Full precision weights are stored for training.
        self.weight_re = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_im = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

        # LayerNorm is applied to the concatenated real and imaginary parts of the input.
        self.layernorm = nn.LayerNorm(2 * in_features, eps=eps)

    def reset_parameters(self):
        """Initializes the full-precision weights."""
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_re)
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(self.weight_re, 0, std)
        nn.init.normal_(self.weight_im, 0, std)

    def quantize_weights_four_val(self, w: torch.Tensor) -> torch.Tensor:
        """Quantizes a weight tensor to {-1, -0.5, 0.5, 1} based on thresholds."""
        output = torch.zeros_like(w)
        # Quantize to 1
        output[w > self.high_threshold] = 1.0
        # Quantize to 0.5
        output[(w > self.low_threshold) & (w <= self.high_threshold)] = 0.5
        # Quantize to -0.5
        output[(w < -self.low_threshold) & (w >= -self.high_threshold)] = -0.5
        # Quantize to -1
        output[w < -self.high_threshold] = -1.0
        return output

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass."""
        # 1. Apply LayerNorm to the input.
        x_cat = torch.cat([x_real, x_imag], dim=-1)
        norm_x_cat = self.layernorm(x_cat)
        norm_x_real, norm_x_imag = torch.chunk(norm_x_cat, 2, dim=-1)

        # 2. Quantize weights and use Straight-Through Estimator (STE).
        quant_w_re = self.quantize_weights_four_val(self.weight_re)
        quant_w_im = self.quantize_weights_four_val(self.weight_im)
        ste_w_re = self.weight_re + (quant_w_re - self.weight_re).detach()
        ste_w_im = self.weight_im + (quant_w_im - self.weight_im).detach()

        # 3. Calculate scaling factor for the weights (dequantization scale).
        s_w_re = self.weight_re.abs().mean()
        s_w_im = self.weight_im.abs().mean()
        
        # 4. Perform the complex linear transformation using normalized input and STE weights.
        y_real = norm_x_real @ ste_w_re.t() - norm_x_imag @ ste_w_im.t()
        y_imag = norm_x_real @ ste_w_im.t() + norm_x_imag @ ste_w_re.t()

        # 5. Dequantize the output by scaling with the weight scales.
        dequant_y_real = y_real * s_w_re
        dequant_y_imag = y_imag * s_w_im
        
        return dequant_y_real, dequant_y_imag

# ==============================================================================
# Validation Code
# ==============================================================================
if __name__ == '__main__':
    # Define hyperparameters for the test
    batch_size = 8
    seq_len = 12
    in_features = 32
    out_features = 64
    expected_shape = (batch_size, seq_len, out_features)

    # --- Validation for RealImagLinear ---
    print("--- Validation for RealImagLinear ---")
    x_real_1 = torch.randn(batch_size, seq_len, in_features)
    x_imag_1 = torch.randn(batch_size, seq_len, in_features)
    linear_layer = RealImagLinear(in_features=in_features, out_features=out_features)
    y_real_1, y_imag_1 = linear_layer(x_real_1, x_imag_1)
    assert y_real_1.shape == expected_shape
    assert y_imag_1.shape == expected_shape
    print("Validation successful for RealImagLinear!\n")

    # --- Validation for RealImagBitNetLinearPhaseQuant ---
    print("--- Validation for RealImagBitNetLinearPhaseQuant ---")
    x_real_2 = torch.randn(batch_size, seq_len, in_features)
    x_imag_2 = torch.randn(batch_size, seq_len, in_features)
    bitnet_layer = RealImagBitNetLinearPhaseQuant(in_features=in_features, out_features=out_features)
    y_real_2, y_imag_2 = bitnet_layer(x_real_2, x_imag_2)
    assert y_real_2.shape == expected_shape
    assert y_imag_2.shape == expected_shape
    print("Validation successful for RealImagBitNetLinearPhaseQuant!\n")

    # --- Validation for RealImagTernaryLinear ---
    print("--- Validation for RealImagTernaryLinear ---")
    x_real_3 = torch.randn(batch_size, seq_len, in_features)
    x_imag_3 = torch.randn(batch_size, seq_len, in_features)
    ternary_layer = RealImagTernaryLinear(in_features=in_features, out_features=out_features)
    y_real_3, y_imag_3 = ternary_layer(x_real_3, x_imag_3)
    assert y_real_3.shape == expected_shape
    assert y_imag_3.shape == expected_shape
    print("Validation successful for RealImagTernaryLinear!\n")

    # --- Validation for RealImagFourValLinear ---
    print("--- Validation for RealImagFourValLinear ---")
    x_real_4 = torch.randn(batch_size, seq_len, in_features)
    x_imag_4 = torch.randn(batch_size, seq_len, in_features)
    four_val_layer = RealImagFourValLinear(in_features=in_features, out_features=out_features)
    y_real_4, y_imag_4 = four_val_layer(x_real_4, x_imag_4)
    assert y_real_4.shape == expected_shape
    assert y_imag_4.shape == expected_shape
    print("Validation successful for RealImagFourValLinear!")
    
    print("\n--- All Validations End ---")

