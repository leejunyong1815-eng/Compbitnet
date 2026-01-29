import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Type

# --- Import Hugging Face classes for compatibility ---
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# --- Import previously refactored modules ---
from .complex_attention_block import RealImagBlock
from .complex_embedding import RealImagEmbedding, RealFromRealImagLinear
from .complex_norm import ComplexRMSNorm #RealImagLayerNorm
from .complex_layers import RealImagLinear # Import the default layer

class RealImagConfig(PretrainedConfig):
    """
    Configuration class for the Real/Imaginary Transformer model, inheriting
    from Hugging Face's PretrainedConfig for compatibility.
    """
    model_type = "real_imag_gpt"

    def __init__(
        self,
        vocab_size=50257,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=None,
        intermediate_size=8192,
        mlp_ratio=1,
        rope_base=10000.0,
        dropout=0.0,
        layer_norm_eps=1e-5,
        mlp_act="complex_silu",
        rope_theta=1000000.0,
        max_position_embeddings=2048,
        pad_token_id=None,
        tie_word_embeddings=True,
        use_flash_attention=False, # Added for convenience
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = int(hidden_size*mlp_ratio)
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = int(intermediate_size*mlp_ratio)
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.rope_base = rope_base
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.mlp_act = mlp_act
        self.max_position_embeddings = max_position_embeddings
        self.use_flash_attention = use_flash_attention
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class RealImagGPTLMHeadModel(PreTrainedModel):
    """
    A complete Transformer Language Model that operates on separate real and imaginary
    tensors. It is initialized with a specific linear layer class for quantization.
    """
    config_class = RealImagConfig

    def __init__(self, config: RealImagConfig, linear_layer_class: Type[nn.Module] = RealImagLinear):
        super().__init__(config)
        
        self.embed = RealImagEmbedding(config.vocab_size, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            RealImagBlock(config=config, linear_layer_class=linear_layer_class)
            for _ in range(config.num_hidden_layers)
        ])
        
        self.final_norm = ComplexRMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        # --- Tied Embeddings 적용 로직 ---
        if getattr(config, "tie_word_embeddings", True):
            print("INFO: Tying word embeddings for lm_head.")
            self.lm_head = RealFromRealImagLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                tie_weights=(self.embed.emb_re.weight, self.embed.emb_im.weight)
            )
        else:
            print("INFO: Using a separate lm_head.")
            self.lm_head = RealFromRealImagLinear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        
        x_real, x_imag = self.embed(input_ids)
        x_real = self.drop(x_real)
        x_imag = self.drop(x_imag)
        
        # --- FIX: Enforce correct dtype for FlashAttention ---
        # When using FlashAttention, intermediate operations (like LayerNorm) might
        # upcast tensors to float32. This block ensures that the tensors are
        # cast back to the model's expected dtype (fp16/bf16) before being
        # passed into the next block, preventing a dtype mismatch error.
        # expected_dtype = next(self.parameters()).dtype
        # if self.config.use_flash_attention:
        #     if x_real.dtype != expected_dtype:
        #         x_real = x_real#.to(expected_dtype)
        #         x_imag = x_imag#.to(expected_dtype)
        
        for blk in self.blocks:
            x_real, x_imag = blk(x_real, x_imag, attn_mask=attention_mask)
            # Re-cast after each block if using FlashAttention
            # if self.config.use_flash_attention and x_real.dtype != expected_dtype:
            #     x_real = x_real#.to(expected_dtype)
            #     x_imag = x_imag#.to(expected_dtype)

        x_real, x_imag = self.final_norm(x_real, x_imag)
        # Re-cast after the final norm as well
        # if self.config.use_flash_attention and x_real.dtype != expected_dtype:
        #     x_real = x_real#.to(expected_dtype)
        #     x_imag = x_imag#.to(expected_dtype)

        logits = self.lm_head(x_real, x_imag)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
