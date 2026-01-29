# real_imag_embedding.py
import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

class RealImagEmbedding(nn.Module):
    """
    Two real embeddings are used to create separate real and imaginary output tensors.
    This approach maintains the use of PyTorch's optimized nn.Embedding layer.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Initializes the embedding layers for real and imaginary parts.

        Args:
            num_embeddings: The size of the dictionary of embeddings.
            embedding_dim: The size of each embedding vector.
        """
        super().__init__()
        self.emb_re = nn.Embedding(num_embeddings, embedding_dim)
        self.emb_im = nn.Embedding(num_embeddings, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the weights of the embedding layers using a normal distribution.
        """
        for emb in (self.emb_re, self.emb_im):
            # Initialize weights with a mean of 0 and a standard deviation
            # calculated based on the embedding dimension.
            nn.init.normal_(emb.weight, mean=0.0, std=1.0 / math.sqrt(emb.weight.size(1)))

    def forward(self, input_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Looks up the embeddings for the given input IDs.

        Args:
            input_ids: A LongTensor containing the indices to be embedded.

        Returns:
            A tuple of two tensors: the real part and the imaginary part of the embeddings.
        """
        # Return the real and imaginary parts as a tuple of tensors.
        return self.emb_re(input_ids), self.emb_im(input_ids)


class RealFromRealImagLinear(nn.Module):
    """
    A real-valued linear head that operates on separate real and imaginary hidden states.
    It can either have its own weights or use tied weights from an embedding layer.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, tie_weights: Optional[Tuple[nn.Parameter, nn.Parameter]] = None):
        """
        Initializes the linear layer.

        Args:
            in_features: The number of input features for each of the real and imaginary parts.
            out_features: The number of output features (vocab size).
            bias: If set to False, the layer will not learn an additive bias.
            tie_weights: A tuple of (weight_re, weight_im) from an embedding layer.
                         If provided, the layer will not create its own weights.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if tie_weights is not None:
            # --- Tied Embeddings 사용 ---
            self.weight_re, self.weight_im = tie_weights
            self.register_parameter("weight", None) # 자체 weight 파라미터는 없음을 명시
        else:
            # --- 독립적인 가중치 사용 ---
            self.weight = nn.Parameter(torch.empty(out_features, 2 * in_features))
            nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(in_features))
            self.weight_re, self.weight_im = None, None
            
        # Tied embeddings에서는 bias를 사용하지 않는 것이 일반적입니다.
        if bias and tie_weights is None:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear transformation.
        """
        feat = torch.cat([x_real, x_imag], dim=-1)
        
        if self.weight is not None:
            # 독립적인 가중치를 사용할 경우
            y = feat @ self.weight.t()
        else:
            # Tied Embeddings를 사용할 경우
            # 두 개의 임베딩 가중치를 합쳐서 사용
            tied_weight = torch.cat([self.weight_re, self.weight_im], dim=1)
            y = feat @ tied_weight.t()
            
        return y + self.bias if self.bias is not None else y

# ==============================================================================
# Validation Code
# ==============================================================================
if __name__ == '__main__':
    # Define hyperparameters for the test
    batch_size = 4
    seq_len = 10
    num_embeddings = 100  # Vocabulary size
    embedding_dim = 32    # Hidden dimension
    output_features = 5   # For the final linear layer

    print("--- Validation Start ---")
    
    # 1. Create dummy input data
    input_ids = torch.randint(0, num_embeddings, (batch_size, seq_len))
    print(f"Input IDs shape: {input_ids.shape}")

    # 2. Instantiate the embedding layer
    embedding_layer = RealImagEmbedding(num_embeddings, embedding_dim)
    
    # 3. Get real and imaginary embeddings
    z_real, z_imag = embedding_layer(input_ids)
    print(f"Real part (z_real) shape: {z_real.shape}")
    print(f"Imaginary part (z_imag) shape: {z_imag.shape}")

    # 4. Instantiate the linear head
    linear_head = RealFromRealImagLinear(in_features=embedding_dim, out_features=output_features)

    # 5. Get final logits from the linear head
    logits = linear_head(z_real, z_imag)
    print(f"Output logits shape: {logits.shape}")

    # 6. Check if the output shape is as expected
    expected_shape = (batch_size, seq_len, output_features)
    assert logits.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, but got {logits.shape}"
    
    print("\nValidation successful! All tensor shapes are correct.")
    print("--- Validation End ---")
