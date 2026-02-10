# same as complex_layers_new_bup.py (250819)
import math
import torch
import torch.nn as nn
from typing import Tuple


def activation_quant(x):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): _description_

    Returns:
        _type_: _description_
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


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

    def reset_parameters(self):
        """Initializes the full-precision weights."""
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_re)
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(self.weight_re, 0, std)
        nn.init.normal_(self.weight_im, 0, std)

    def weight_quant(self, w: torch.Tensor):
        scale = w.abs().mean()
        e = w.mean()
        u = (w - e).sign() * scale
        return u

def forward(self, quant_x_re: torch.Tensor, quant_x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass on pre-quantized activations.
        """
        # Input normalization and activation quantization are now done externally.
        
        dequant_w_re = self.weight_quant(self.weight_re)
        dequant_w_im = self.weight_quant(self.weight_im)

        ste_w_re = self.weight_re + (dequant_w_re - self.weight_re).detach()
        ste_w_im = self.weight_im + (dequant_w_im - self.weight_im).detach()

        dequant_y_real = quant_x_re @ ste_w_re.t() - quant_x_im @ ste_w_im.t()
        dequant_y_imag = quant_x_re @ ste_w_im.t() + quant_x_im @ ste_w_re.t()

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

    def reset_parameters(self):
        """Initializes the full-precision weights."""
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_re)
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(self.weight_re, 0, std)
        nn.init.normal_(self.weight_im, 0, std)

    def ternarize_weights(self, w: torch.Tensor):
        scale = 1.0/w.abs().mean().clamp_(min=1e-5)
        temp = (w*scale).round().clamp_(-1,1)   # -1, 0, 1 quantized
        u = temp/scale
        return u

    def forward(self, quant_x_re: torch.Tensor, quant_x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass on pre-quantized activations."""
        # Input normalization and activation quantization are now done externally.

        dequant_w_re = self.ternarize_weights(self.weight_re)
        dequant_w_im = self.ternarize_weights(self.weight_im)

        ste_w_re = self.weight_re + (dequant_w_re - self.weight_re).detach()
        ste_w_im = self.weight_im + (dequant_w_im - self.weight_im).detach()

        dequant_y_real = quant_x_re @ ste_w_re.t() - quant_x_im @ ste_w_im.t()
        dequant_y_imag = quant_x_re @ ste_w_im.t() + quant_x_im @ ste_w_re.t()

        return dequant_y_real, dequant_y_imag


class RealImagFourValLinear(nn.Module):
    """
    A BitNet-style linear layer that quantizes the real and imaginary parts of its weights
    to four values {-1, -0.5, 0.5, 1}.
    """
    def __init__(self, in_features: int, out_features: int, high_threshold: float = 1, low_threshold: float = 0.0, eps: float = 1e-6):
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

    def reset_parameters(self):
        """Initializes the full-precision weights."""
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_re)
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(self.weight_re, 0, std)
        nn.init.normal_(self.weight_im, 0, std)


    def quantize_weights_four_val(self, w: torch.Tensor) -> torch.Tensor:
        scale = w.abs().mean().clamp_(min=self.eps)
        w_scaled = w / scale
        
        # 1. Round to nearest 0.5 step
        w_quantized_int = (w_scaled * 2.0).round() / 2.0
        
        mask_zeros = w_quantized_int.abs() < 1e-5
        original_sign = w_scaled.sign()
        original_sign[original_sign == 0] = 1.0  

        w_quantized_int[mask_zeros] = 0.5 * original_sign[mask_zeros]

        w_quantized_int.clamp_(-1.0, 1.0)
        
        return w_quantized_int * scale
    
    def forward(self, quant_x_re: torch.Tensor, quant_x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass (Follows the 'Pattern A')"""
        # Input normalization and activation quantization are now done externally.

        # 2. Quantize weights and use Straight-Through Estimator (STE).
        dequant_w_re = self.quantize_weights_four_val(self.weight_re)
        dequant_w_im = self.quantize_weights_four_val(self.weight_im)
        
        ste_w_re = self.weight_re + (dequant_w_re - self.weight_re).detach()
        ste_w_im = self.weight_im + (dequant_w_im - self.weight_im).detach()

        # 3. Perform the complex linear transformation
        # The faulty scaling (*s_w_re) has been removed.
        dequant_y_real = quant_x_re @ ste_w_re.t() - quant_x_im @ ste_w_im.t()
        dequant_y_imag = quant_x_re @ ste_w_im.t() + quant_x_im @ ste_w_re.t()

        return dequant_y_real, dequant_y_imag
    
class RealImagFiveValLinear(nn.Module):
    """
    A BitNet-style linear layer that quantizes weights to five values {-1, -0.5, 0, 0.5, 1}.
    This refactored version ONLY performs weight quantization and linear transformation.
    """
    def __init__(self, in_features: int, out_features: int, eps: float = 1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        self.weight_re = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_im = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_re)
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(self.weight_re, 0, std)
        nn.init.normal_(self.weight_im, 0, std)

    def quantize_weights(self, w: torch.Tensor) -> torch.Tensor:
        scale = w.abs().mean().clamp_(min=self.eps)
        w_scaled = w / scale
        w_quantized_int = (w_scaled * 2).round() / 2
        w_quantized_int.clamp_(-1.0, 1.0)
        w_dequantized = w_quantized_int * scale
        return w_dequantized

    def forward(self, quant_x_re: torch.Tensor, quant_x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dequant_w_re = self.quantize_weights(self.weight_re)
        dequant_w_im = self.quantize_weights(self.weight_im)

        ste_w_re = self.weight_re + (dequant_w_re - self.weight_re).detach()
        ste_w_im = self.weight_im + (dequant_w_im - self.weight_im).detach()

        output_y_real = quant_x_re @ ste_w_re.t() - quant_x_im @ ste_w_im.t()
        output_y_imag = quant_x_re @ ste_w_im.t() + quant_x_im @ ste_w_re.t()

        return output_y_real, output_y_imag

class RealImagSixValLinear(nn.Module):
    """
    A BitNet-style linear layer that quantizes weights to six values:
    {-2, -1, -0.5, 0.5, 1, 2} (No Zero).
    Based on RealImagFourValLinear logic but extended to include +/- 2.0.
    """
    def __init__(self, in_features: int, out_features: int, eps: float = 1e-6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        self.weight_re = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_im = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_re)
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(self.weight_re, 0, std)
        nn.init.normal_(self.weight_im, 0, std)

    def quantize_weights(self, w: torch.Tensor) -> torch.Tensor:
        """
        Quantizes weights to {-2, -1, -0.5, 0.5, 1, 2}.
        Logic:
          |w| < 0.75       -> 0.5  (forces small values to 0.5)
          0.75 <= |w| < 1.5 -> 1.0
          |w| >= 1.5       -> 2.0
        """
        scale = w.abs().mean().clamp_(min=self.eps)
        w_scaled = w / scale
        
        # Absolute value based quantization
        w_abs = w_scaled.abs()
        w_sign = w_scaled.sign()
        
        w_sign[w_sign == 0] = 1.0 

        # Create output tensor
        w_quant_abs = torch.zeros_like(w_abs)
        
        # Threshold mapping for 0.5, 1.0, 2.0
        # 1.5 is the midpoint between 1 and 2
        # 0.75 is the midpoint between 0.5 and 1
        
        mask_2 = w_abs >= 1.5
        mask_1 = (w_abs >= 0.75) & (w_abs < 1.5)
        mask_05 = w_abs < 0.75
        
        w_quant_abs[mask_2] = 2.0
        w_quant_abs[mask_1] = 1.0
        w_quant_abs[mask_05] = 0.5
        
        # Apply sign and scale back
        w_dequantized = w_sign * w_quant_abs * scale
        return w_dequantized

    def forward(self, quant_x_re: torch.Tensor, quant_x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dequant_w_re = self.quantize_weights(self.weight_re)
        dequant_w_im = self.quantize_weights(self.weight_im)
        
        ste_w_re = self.weight_re + (dequant_w_re - self.weight_re).detach()
        ste_w_im = self.weight_im + (dequant_w_im - self.weight_im).detach()

        dequant_y_real = quant_x_re @ ste_w_re.t() - quant_x_im @ ste_w_im.t()
        dequant_y_imag = quant_x_re @ ste_w_im.t() + quant_x_im @ ste_w_re.t()

        return dequant_y_real, dequant_y_imag


class RealImagSevenValLinear(nn.Module):
    """
    A BitNet-style linear layer that quantizes weights to seven values:
    {-2, -1, -0.5, 0, 0.5, 1, 2} (With Zero).
    Based on RealImagFiveValLinear logic but extended to include +/- 2.0.
    """
    def __init__(self, in_features: int, out_features: int, eps: float = 1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        self.weight_re = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_im = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_re)
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(self.weight_re, 0, std)
        nn.init.normal_(self.weight_im, 0, std)

    def quantize_weights(self, w: torch.Tensor) -> torch.Tensor:
        """
        Quantizes weights to {-2, -1, -0.5, 0, 0.5, 1, 2}.
        Logic:
          |w| < 0.25       -> 0
          0.25 <= |w| < 0.75 -> 0.5
          0.75 <= |w| < 1.5 -> 1.0
          |w| >= 1.5       -> 2.0
        """
        scale = w.abs().mean().clamp_(min=self.eps)
        w_scaled = w / scale
        
        w_abs = w_scaled.abs()
        w_sign = w_scaled.sign()
        
        w_quant_abs = torch.zeros_like(w_abs)
        
        mask_2 = w_abs >= 1.5
        mask_1 = (w_abs >= 0.75) & (w_abs < 1.5)
        mask_05 = (w_abs >= 0.25) & (w_abs < 0.75)
        # mask_0 is implicit (zeros initialized)
        
        w_quant_abs[mask_2] = 2.0
        w_quant_abs[mask_1] = 1.0
        w_quant_abs[mask_05] = 0.5
        
        w_dequantized = w_sign * w_quant_abs * scale
        return w_dequantized

    def forward(self, quant_x_re: torch.Tensor, quant_x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dequant_w_re = self.quantize_weights(self.weight_re)
        dequant_w_im = self.quantize_weights(self.weight_im)

        ste_w_re = self.weight_re + (dequant_w_re - self.weight_re).detach()
        ste_w_im = self.weight_im + (dequant_w_im - self.weight_im).detach()

        output_y_real = quant_x_re @ ste_w_re.t() - quant_x_im @ ste_w_im.t()
        output_y_imag = quant_x_re @ ste_w_im.t() + quant_x_im @ ste_w_re.t()

        return output_y_real, output_y_imag

# class QuantizedComplexLinear(nn.Module):
#     """
#     A 'pure' quantized linear layer that only handles weight quantization and
#     the linear transformation. It expects pre-normalized and pre-quantized activations.
#     """
#     def __init__(self, in_features: int, out_features: int, eps: float = 1e-5):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.eps = eps

#         self.weight_re = nn.Parameter(torch.empty(out_features, in_features))
#         self.weight_im = nn.Parameter(torch.empty(out_features, in_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_re)
#         std = math.sqrt(1.0 / fan_in)
#         nn.init.normal_(self.weight_re, 0, std)
#         nn.init.normal_(self.weight_im, 0, std)

#     def quantize_weights(self, w: torch.Tensor) -> torch.Tensor:
#         # RealImagFiveValLinear에서 가져온 가중치 양자화 로직
#         scale = w.abs().mean().clamp_(min=self.eps)
#         w_scaled = w / scale
#         w_quantized_int = (w_scaled * 2).round() / 2
#         w_quantized_int.clamp_(-1.0, 1.0)
#         w_dequantized = w_quantized_int * scale
#         return w_dequantized

#     def forward(self, quant_x_re: torch.Tensor, quant_x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         # 정규화나 활성화 양자화 없이 바로 선형 변환 수행
#         dequant_w_re = self.quantize_weights(self.weight_re)
#         dequant_w_im = self.quantize_weights(self.weight_im)

#         ste_w_re = self.weight_re + (dequant_w_re - self.weight_re).detach()
#         ste_w_im = self.weight_im + (dequant_w_im - self.weight_im).detach()

#         output_y_real = quant_x_re @ ste_w_re.t() - quant_x_im @ ste_w_im.t()
#         output_y_imag = quant_x_re @ ste_w_im.t() + quant_x_im @ ste_w_re.t()

#         return output_y_real, output_y_imag
    


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

    # --- Reusable setup for quantized layers ---
    def get_quantized_inputs(bs, sl, in_feat):
        x_real = torch.randn(bs, sl, in_feat)
        x_imag = torch.randn(bs, sl, in_feat)
        x_cat = torch.cat([x_real, x_imag], dim=-1)
        
        # External LayerNorm
        layernorm = nn.LayerNorm(2 * in_feat)
        norm_x_cat = layernorm(x_cat)
        norm_x_real, norm_x_imag = torch.chunk(norm_x_cat, 2, dim=-1)
        
        # External Activation Quantization + STE
        quant_x_re = norm_x_real + (activation_quant(norm_x_real) - norm_x_real).detach()
        quant_x_im = norm_x_imag + (activation_quant(norm_x_imag) - norm_x_imag).detach()
        
        return quant_x_re, quant_x_im

    # --- Validation for RealImagLinear ---
    # (This one doesn't need pre-quantized inputs)
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
    quant_x_re_2, quant_x_im_2 = get_quantized_inputs(batch_size, seq_len, in_features)
    bitnet_layer = RealImagBitNetLinearPhaseQuant(in_features=in_features, out_features=out_features)
    y_real_2, y_imag_2 = bitnet_layer(quant_x_re_2, quant_x_im_2)
    assert y_real_2.shape == expected_shape
    assert y_imag_2.shape == expected_shape
    print("Validation successful for RealImagBitNetLinearPhaseQuant!\n")

    # --- Validation for RealImagTernaryLinear ---
    print("--- Validation for RealImagTernaryLinear ---")
    quant_x_re_3, quant_x_im_3 = get_quantized_inputs(batch_size, seq_len, in_features)
    ternary_layer = RealImagTernaryLinear(in_features=in_features, out_features=out_features)
    y_real_3, y_imag_3 = ternary_layer(quant_x_re_3, quant_x_im_3)
    assert y_real_3.shape == expected_shape
    assert y_imag_3.shape == expected_shape
    print("Validation successful for RealImagTernaryLinear!\n")

    # --- Validation for RealImagFourValLinear ---
    print("--- Validation for RealImagFourValLinear ---")
    quant_x_re_4, quant_x_im_4 = get_quantized_inputs(batch_size, seq_len, in_features)
    four_val_layer = RealImagFourValLinear(in_features=in_features, out_features=out_features)
    y_real_4, y_imag_4 = four_val_layer(quant_x_re_4, quant_x_im_4)
    assert y_real_4.shape == expected_shape
    assert y_imag_4.shape == expected_shape
    print("Validation successful for RealImagFourValLinear!")

    # --- Validation for RealImagFiveValLinear ---
    print("\n--- Validation for RealImagFiveValLinear ---")
    quant_x_re_5, quant_x_im_5 = get_quantized_inputs(batch_size, seq_len, in_features)
    five_val_layer = RealImagFiveValLinear(in_features=in_features, out_features=out_features)
    y_real_5, y_imag_5 = five_val_layer(quant_x_re_5, quant_x_im_5)
    assert y_real_5.shape == expected_shape
    assert y_imag_5.shape == expected_shape
    print("Validation successful for RealImagFiveValLinear!\n")
    
    print("\n--- All Validations End ---")