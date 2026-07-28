"""
dflash_draft.py — DFlash draft model definition (from-scratch training).

Defines ``MYDraft`` and its supporting ``DFlashDraftModel``. ``__init__``
copies weights from the last K layers of the target base model, so the draft
starts warm without needing an external checkpoint. Use this for training a
draft from scratch on large SFT data.

To continue-finetune from an existing DFlash checkpoint, see the sibling
module ``dflash_draft_resume.py`` (``MYDraftFromDFlash``).
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import Unpack

# FlexAttention imports
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None

# Import HunyuanVL components
import sys

# Add project root to Python path
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



from transformers import AutoConfig, DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)

from train.dflash_utils import extract_context_feature, sample


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = None
        
        # config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
            # Ensure query, key, value have the same dtype (RoPE may upcast q/k to float32 while v stays in bfloat16)
        target_dtype = v.dtype
        if k.dtype != target_dtype:
            k = k.to(target_dtype)
        if q.dtype != target_dtype:
            q = q.to(target_dtype)

        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = self.config.dflash_config.get("target_layer_ids", None)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
        target_hidden: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def generate_with_mtp_speculative(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: list[int] | None = None,
        target: nn.Module = None,
        max_new_tokens: int = 1024,
        eos_token_id: int | None = None,
        temperature: float = 0.0,
        **kwargs,
    ):
        self.eval() 
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # Prefill stage
        output = target(
            input_ids=input_ids, # 1 * n
            position_ids=position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            logits_to_keep=1, 
            output_hidden_states=True,
            **kwargs,
        )
        
        output_ids[:, :num_input_tokens] = input_ids
        next_token = sample(output.logits, temperature)
        output_ids[:, num_input_tokens:num_input_tokens+1] = next_token
        target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)

        position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
        # position_ids = torch.concat([position_ids, torch.arange(num_input_tokens, num_input_tokens+block_size, device=target.device).unsqueeze(0).unsqueeze(0).expand(1, 4, -1)], dim=2)
        
        # torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)

        # Decode stage
        acceptance_lengths = []
        num_decoding_steps = 0
        start = input_ids.shape[1]
        while start < max_length:
            
            block_output_ids = output_ids[:, start : start + block_size].clone()
            # block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size + 1:, :])
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits) # 表示 next_i + 1

            output = target(
                input_ids=block_output_ids, # block_size 个
                position_ids=None,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
                **kwargs,
            )

            posterior = sample(output.logits, temperature) # block_size + 1 个
            acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            # next_token = posterior[:, acceptance_length: acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)[:, :acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length)
            num_decoding_steps += 1
            
            # Check for EOS token
            if eos_token_id is not None:
                # Normalize eos_token_id to list
                eos_list = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
                # Check if any EOS token appears in generated tokens
                generated_tokens = output_ids[:, num_input_tokens:start]
                if any((generated_tokens == eos_id).any() for eos_id in eos_list):
                    break
        
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        
        # Truncate at first EOS token
        if eos_token_id is not None:
            eos_list = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
            eos_tensor = torch.tensor(eos_list, device=output_ids.device)
            # Find first EOS token in generated sequence
            generated_tokens = output_ids[0, num_input_tokens:]
            stop_token_indices = torch.isin(generated_tokens, eos_tensor).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[:, :num_input_tokens + stop_token_indices[0] + 1]
        
        # Calculate and print statistics
        total_generated = output_ids.shape[1] - num_input_tokens
        step_decoding = num_decoding_steps
        num_accepted_total = sum(acceptance_lengths)
        num_drafted_total = len(acceptance_lengths) * (block_size - 1)  # 每步 draft block_size-1 个 token
        acceptance_rate = num_accepted_total / num_drafted_total if num_drafted_total > 0 else 0.0
        avg_acceptance_length = sum(acceptance_lengths) / len(acceptance_lengths) if len(acceptance_lengths) > 0 else 0.0
        
        full_accept_count = sum(1 for l in acceptance_lengths if l == (block_size - 1))
        full_accept_rate = full_accept_count / len(acceptance_lengths) if len(acceptance_lengths) > 0 else 0.0
        
        print(f"[MTP Speculative] Generated {total_generated}/{step_decoding} tokens, "
              f"acceptance rate: {acceptance_rate:.2%} "
              f"({num_accepted_total}/{num_drafted_total}), "
              f"avg acceptance length: {avg_acceptance_length:.2f}, "
              f"full accept rate: {full_accept_rate:.2%} ({full_accept_count}/{len(acceptance_lengths)})")
        return output_ids, avg_acceptance_length, full_accept_rate


class MYDraft(Qwen3PreTrainedModel):
    """
    MYDraft — speculative decoding draft model compatible with Transformers Trainer.

    Components
    ----------
    target_model : HunYuanVLForConditionalGeneration  (frozen)
        Provides last-layer hidden_states for the draft model's context.
    draft_model : DFlashDraftModel
        Draft model (K layers)

    Forward
    -------
    Input  : same kwargs as target_model.forward()  +  optional `labels`
    Output : CausalLMOutputWithPast
        .logits  — [batch, T*N, vocab_size]  draft logits (T positions × N mask tokens)
        .loss    — None  (loss computation is left to the trainer / caller)
    """

    base_model_prefix = "target_model"
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(
        self,
        config: Qwen3Config,
        target_model: nn.Module,
        num_draft_layers: int = 5,
        only_draft: bool = False,
        use_distill: bool = False,
        loop_num: int = 1,
        sample_block_num: int = 16,
    ):
        """
        Parameters
        ----------
        config : Qwen3Config
            Config of the target model (passed to PreTrainedModel.__init__).
        target_model : nn.Module
            A HunYuanVLForConditionalGeneration instance.
            Its parameters are frozen inside this constructor.
        num_draft_layers : int
            K — number of decoder layers to copy from the tail of target_model.
        num_mask_tokens : int
            N — number of mask tokens (= number of tokens predicted per step).
        """
        super().__init__(config)
        self.config = config
        self.loss_decay_gamma = 7.0
        self.use_distill = use_distill
        self.loop_num = loop_num
        self.sample_block_num = sample_block_num
        num_draft_layers = self.config.num_draft_layers

        # ── Target model (frozen) ──────────────────────────────────────────
        self.target_model = target_model
        for p in self.target_model.parameters():
            p.requires_grad = False
        self.target_model.eval()

        hidden_size  = config.hidden_size
        self.hidden_size = hidden_size
        # config = copy.deepcopy(config)
        # Load the DFlash draft-model config template (config.json + dflash.py).
        # Default: train/configs/ next to this file. Override with the env var
        # HYOCR_DFLASH_CONFIG_DIR to point at any directory that contains a
        # compatible config.json + dflash.py (e.g. the dflash/ subfolder that
        # ships with the HuggingFace HunyuanOCR release).
        import os
        _dflash_cfg_dir = os.environ.get(
            "HYOCR_DFLASH_CONFIG_DIR",
            str(Path(__file__).parent / "configs"),
        )
        config_dflash = AutoConfig.from_pretrained(_dflash_cfg_dir, trust_remote_code=True)
        config_dflash._attn_implementation = "flex_attention"
        config_dflash.num_hidden_layers = num_draft_layers
        config_dflash.block_size = config.block_size
        self.draft_model = DFlashDraftModel(config_dflash)
        
        # Initialize draft_model layers from the last num_draft_layers of target_model
        self._initialize_draft_from_target(num_draft_layers)
        
        # MTP configuration
        self.block_size = config.block_size
        self.mask_token_id = getattr(config, "mask_token_id", 120817)  # <｜hy_place▁holder▁no▁799｜>
        
        self.only_draft = only_draft

    def _initialize_draft_from_target(self, num_draft_layers: int):
        """
        Initialize draft_model layers from the last num_draft_layers of target_model.
        
        We copy weights from target_model.model.layers[-num_draft_layers:] to draft_model.layers.
        
        Note: The attention modules have different structures:
        - Target: HunYuanVLAttention (standard self-attention)
        - Draft: HunYuanVLDFlashAttention (cross-attention with target hidden states)
        
        We copy:
        - MLP (fully compatible)
        - LayerNorm (input_layernorm, post_attention_layernorm)
        - Attention projections (q_proj, k_proj, v_proj, o_proj) where structure matches
        - Attention LayerNorms (query_layernorm, key_layernorm)
        """
        target_layers = self.target_model.model.layers
        total_layers = len(target_layers)
        
        if num_draft_layers > total_layers:
            raise ValueError(
                f"num_draft_layers ({num_draft_layers}) cannot exceed total target layers ({total_layers})"
            )
        
        # Get the last num_draft_layers from target model
        start_layer_idx = total_layers - num_draft_layers
        
        print(f"Initializing {num_draft_layers} draft layers from target layers {start_layer_idx} to {total_layers-1}")
        
        for draft_idx, target_idx in enumerate(range(start_layer_idx, total_layers)):
            target_layer = target_layers[target_idx]
            draft_layer = self.draft_model.layers[draft_idx]
            
            # Copy MLP weights (fully compatible)
            draft_layer.mlp.load_state_dict(target_layer.mlp.state_dict())
            
            # Copy LayerNorm weights
            draft_layer.input_layernorm.load_state_dict(target_layer.input_layernorm.state_dict())
            draft_layer.post_attention_layernorm.load_state_dict(target_layer.post_attention_layernorm.state_dict())
            
            # Copy attention projection weights
            draft_layer.self_attn.q_proj.load_state_dict(target_layer.self_attn.q_proj.state_dict())
            draft_layer.self_attn.k_proj.load_state_dict(target_layer.self_attn.k_proj.state_dict())
            draft_layer.self_attn.v_proj.load_state_dict(target_layer.self_attn.v_proj.state_dict())
            draft_layer.self_attn.o_proj.load_state_dict(target_layer.self_attn.o_proj.state_dict())
            
            # Copy attention LayerNorms
            draft_layer.self_attn.q_norm.load_state_dict(target_layer.self_attn.query_layernorm.state_dict())
            draft_layer.self_attn.k_norm.load_state_dict(target_layer.self_attn.key_layernorm.state_dict())
            
            print(f"  Initialized draft layer {draft_idx} from target layer {target_idx}")
        
        # Copy final norm from target model
        self.draft_model.norm.load_state_dict(self.target_model.model.norm.state_dict())
        print("Initialized draft_model.norm from target_model.model.norm")


    # ------------------------------------------------------------------
    # MTP input construction
    # ------------------------------------------------------------------

    def _build_mtp_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        labels: torch.LongTensor,
        cu_seqlens: torch.LongTensor,
    ):
        """Build MTP inputs for next-block prediction."""
        device = input_ids.device
        block_size = self.block_size
        mask_token_id = self.mask_token_id

        input_ids = input_ids.squeeze(0)
        labels = labels.squeeze(0) if labels is not None else None

        num_samples = len(cu_seqlens) - 1
        total_len = len(input_ids)

        all_mtp_input_ids = []
        all_mtp_position_ids = []
        all_mask_labels = []
        all_target_indices = []
        all_sample_ids = []
        all_context_position_ids = []
        all_weight_mask = []

        for sample_idx in range(num_samples):
            sample_start = cu_seqlens[sample_idx].item()
            sample_end = cu_seqlens[sample_idx + 1].item()
            sample_len = sample_end - sample_start

            sample_input_ids = input_ids[sample_start:sample_end]
            sample_labels = labels[sample_start:sample_end] if labels is not None else None
            sample_position_ids = torch.arange(sample_len, device=device).unsqueeze(0)
            all_context_position_ids.append(sample_position_ids)

            if sample_labels is not None:
                pred_mask = (sample_labels != -100)
            else:
                pred_mask = torch.ones(sample_len, dtype=torch.bool, device=device)

            pred_positions = torch.where(pred_mask)[0]
            if len(pred_positions) == 0:
                continue

            sample_block_num = self.sample_block_num
            if len(pred_positions) > sample_block_num:
                rand_indices = torch.randperm(
                    len(pred_positions), device=device
                )[:sample_block_num]
                pred_positions = pred_positions[rand_indices].sort()[0]

            num_blocks = len(pred_positions)
            block_total_len = num_blocks * block_size

            block_offsets = torch.arange(block_size, device=device).unsqueeze(0)
            target_indices = pred_positions.unsqueeze(1) + block_offsets
            target_indices_flat = target_indices.flatten()

            valid_mask = target_indices_flat < sample_len

            sample_mtp_input_ids = torch.full(
                (block_total_len,), mask_token_id, dtype=torch.long, device=device
            )
            first_token_mask = torch.zeros(
                block_total_len, dtype=torch.bool, device=device
            )
            first_token_mask[::block_size] = True

            valid_first_mask = first_token_mask & valid_mask
            safe_indices = target_indices_flat[valid_first_mask].clamp(max=sample_len - 1)
            sample_mtp_input_ids[valid_first_mask] = sample_input_ids[safe_indices]

            sample_mtp_position_ids = torch.zeros(
                (1, block_total_len), dtype=torch.long, device=device
            )

            for i in range(block_total_len):
                target_idx = target_indices_flat[i].item()
                if valid_mask[i]:
                    sample_mtp_position_ids[:, i] = sample_position_ids[:, target_idx]
                else:
                    last_valid_idx = sample_len - 1
                    offset = target_idx - last_valid_idx
                    sample_mtp_position_ids[:, i] = (
                        sample_position_ids[:, last_valid_idx] + offset
                    )

            label_indices = target_indices_flat
            label_valid_mask = label_indices < sample_len

            # Same-position prediction (matches SpecForge): position k inside
            # a block is responsible for predicting the token at
            # ``anchor_pos + k`` of the original sample. For k == 0 (the
            # anchor / block-start slot) we still record the label so distill
            # / debug paths have it, but we mask it out from the loss via
            # ``sample_weight_mask`` below (pos_in_block > 0).
            sample_mask_labels = torch.full(
                (block_total_len,), -100, dtype=torch.long, device=device
            )
            safe_label_indices = label_indices[label_valid_mask].clamp(max=sample_len - 1)
            sample_mask_labels[label_valid_mask] = sample_input_ids[safe_label_indices]

            # ----------------------------------------------------------------
            # Per-slot loss weight (float in {0, 1}). Mirrors the official
            # SpecForge ``OnlineDFlashModel.forward`` weight_mask:
            #   weight = block_keep * in_bounds * (pos_in_block > 0)
            #          * loss_mask_at_label_position
            # In our packed setup every block in pred_positions is kept by
            # construction (block_keep == True), so we only need:
            #   in_bounds (label_valid_mask) * (pos_in_block > 0)
            #   * pred_mask[label_pos]   (== loss_mask in SpecForge)
            # ----------------------------------------------------------------
            pos_in_block = torch.arange(block_size, device=device).repeat(num_blocks)
            sample_weight_mask = label_valid_mask.float() * (pos_in_block > 0).float()
            # Gate by the label-position pred_mask (i.e. only count slots whose
            # to-be-predicted token is itself a real label, not -100 padding).
            label_pred_mask = torch.zeros(
                block_total_len, dtype=torch.bool, device=device
            )
            label_pred_mask[label_valid_mask] = pred_mask[safe_label_indices]
            sample_weight_mask = sample_weight_mask * label_pred_mask.float()

            all_mtp_input_ids.append(sample_mtp_input_ids)
            all_mtp_position_ids.append(sample_mtp_position_ids)
            all_mask_labels.append(sample_mask_labels)
            all_target_indices.append(target_indices_flat + sample_start)
            all_sample_ids.append(
                torch.full((block_total_len,), sample_idx, dtype=torch.long, device=device)
            )
            all_weight_mask.append(sample_weight_mask)

        if len(all_mtp_input_ids) == 0:
            empty = torch.zeros((1, 0), dtype=torch.long, device=device)
            empty_pos = torch.zeros((1, 4, 0), dtype=torch.long, device=device)
            empty_idx = torch.zeros((1, 0), dtype=torch.long, device=device)
            empty_w = torch.zeros((1, 0), dtype=torch.float32, device=device)
            return (
                empty,
                empty_pos,
                empty_pos,
                torch.zeros((1, 0, total_len), dtype=torch.bool, device=device),
                empty.fill_(-100),
                None,
                empty_idx,
                empty_w,
            )

        mtp_input_ids = torch.cat(all_mtp_input_ids, dim=0)
        mtp_position_ids = torch.cat(all_mtp_position_ids, dim=1)
        context_position_ids = torch.cat(all_context_position_ids, dim=1)
        mask_labels = torch.cat(all_mask_labels, dim=0)
        target_indices_flat = torch.cat(all_target_indices, dim=0)
        sample_ids = torch.cat(all_sample_ids, dim=0)
        weight_mask = torch.cat(all_weight_mask, dim=0)

        # NOTE: the previous implementation prepended a -100 pad and shifted
        # every label by one slot (``[pad_label, mask_labels[:-1]]``). That
        # is *wrong* for MTP same-position prediction: it cross-contaminates
        # block boundaries (block i's last slot would inherit block i+1's
        # first label) and is not what SpecForge does. We removed it and
        # rely on ``weight_mask`` (with pos_in_block > 0) to mask the anchor
        # slot inside each block.

        total_mtp_len = len(mtp_input_ids)

        mtp_attention_mask = torch.zeros(
            total_mtp_len, total_len + total_mtp_len, dtype=torch.bool, device=device
        )

        mtp_offset = 0
        for sample_blocks_data in all_target_indices:
            num_blocks = len(sample_blocks_data) // block_size
            for block_idx in range(num_blocks):
                block_start = mtp_offset + block_idx * block_size
                block_end = block_start + block_size

                first_token_idx = block_start
                first_target_idx = target_indices_flat[first_token_idx].item()

                sample_idx = sample_ids[first_token_idx].item()
                sample_start = cu_seqlens[sample_idx].item()
                sample_end = cu_seqlens[sample_idx + 1].item()

                if first_target_idx > sample_start:
                    mtp_attention_mask[
                        block_start:block_end, sample_start:first_target_idx
                    ] = True
                mtp_attention_mask[
                    block_start:block_end,
                    total_len + block_start:total_len + block_end,
                ] = True
            mtp_offset += num_blocks * block_size

        mtp_input_ids = mtp_input_ids.unsqueeze(0)
        mtp_attention_mask = mtp_attention_mask.unsqueeze(0)
        mask_labels = mask_labels.unsqueeze(0)

        # Build label_global_indices for distillation: which position in the
        # original input_ids does each MTP slot's label correspond to?
        # target_indices_flat already has + sample_start applied above.
        # Out-of-range slots (label_valid_mask was False) get -1 sentinel.
        # Then apply the same [pad_label, ...[:-1]] left-shift as mask_labels
        # so the indices stay aligned with the (already left-shifted) labels.
        label_global_indices = target_indices_flat.clone()
        # Mark invalid positions (those that fall past the sample end) as -1.
        # Each MTP slot belongs to exactly one sample (tracked in sample_ids);
        # a slot is valid iff its absolute target index falls before that
        # sample's end boundary in cu_seqlens.
        sample_end_lookup = cu_seqlens.to(device)  # [num_samples + 1]
        sample_id_per_slot = sample_ids  # [total_mtp_len]
        per_slot_sample_end = sample_end_lookup[sample_id_per_slot + 1]
        valid_label_mask = label_global_indices < per_slot_sample_end
        label_global_indices = torch.where(
            valid_label_mask,
            label_global_indices,
            torch.full_like(label_global_indices, -1),
        )
        label_global_indices = label_global_indices.unsqueeze(0)
        # No left-shift here either (matches the same-position alignment of
        # ``mask_labels`` above).

        block_mask = None
        if FLEX_ATTENTION_AVAILABLE:
            num_blocks = (total_mtp_len + block_size - 1) // block_size
            block_first_target_indices = torch.zeros(
                1, num_blocks, dtype=torch.long, device=device
            )
            block_sample_ids = torch.zeros(
                1, num_blocks, dtype=torch.long, device=device
            )
            block_valid_mask = torch.zeros(
                1, num_blocks, dtype=torch.bool, device=device
            )

            block_first_indices = torch.arange(
                0, total_mtp_len, block_size, device=device
            )
            if len(block_first_indices) > 0:
                block_first_target_indices[0, :len(block_first_indices)] = (
                    target_indices_flat[block_first_indices]
                )
                block_sample_ids[0, :len(block_first_indices)] = sample_ids[
                    block_first_indices
                ]
                block_valid_mask[0, :len(block_first_indices)] = True

            block_mask = self._create_flex_attention_mask(
                block_first_target_indices,
                block_sample_ids,
                block_valid_mask,
                cu_seqlens,
                total_len,
                total_mtp_len,
                block_size,
                device,
            )

        weight_mask = weight_mask.unsqueeze(0)

        return (
            mtp_input_ids,
            context_position_ids,
            mtp_position_ids,
            mtp_attention_mask,
            mask_labels,
            block_mask,
            label_global_indices,
            weight_mask,
        )

    def _create_flex_attention_mask(
        self,
        block_first_target_indices: torch.Tensor,
        block_sample_ids: torch.Tensor,
        block_valid_mask: torch.Tensor,
        cu_seqlens: torch.Tensor,
        total_len: int,
        total_mtp_len: int,
        block_size: int,
        device: torch.device,
    ):
        total_seq_len = total_len + total_mtp_len

        def mask_fn(b, h, q_idx, kv_idx):
            q_block_id = q_idx // block_size
            first_target_idx = block_first_target_indices[b, q_block_id]
            sample_idx = block_sample_ids[b, q_block_id]
            is_valid_block = block_valid_mask[b, q_block_id]
            sample_start = cu_seqlens[sample_idx]
            sample_end = cu_seqlens[sample_idx + 1]

            is_context = kv_idx < total_len
            mask_context = (
                is_context
                & (sample_start <= kv_idx)
                & (kv_idx < first_target_idx)
                & (kv_idx < sample_end)
            )
            is_draft = kv_idx >= total_len
            kv_block_id = (kv_idx - total_len) // block_size
            mask_draft = is_draft & (q_block_id == kv_block_id)
            return (mask_context | mask_draft) & is_valid_block

        try:
            return create_block_mask(
                mask_fn, B=1, H=None, Q_LEN=total_mtp_len,
                KV_LEN=total_seq_len, device=device,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"Warning: FlexAttention block mask creation failed (OOM): {e}")
            return None
        except Exception as e:
            print(f"Warning: Failed to create FlexAttention block mask: {e}")
            return None

    # ------------------------------------------------------------------
    # forward — Transformers Trainer compatible
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: list[int] | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # target_model must run in eval mode regardless of trainer state.
        self.target_model.eval()
        with torch.no_grad():
            target_out = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=None,
                labels=None,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                use_cache=False,
                **kwargs,
            )
        target_hidden = extract_context_feature(
            target_out.hidden_states, self.draft_model.target_layer_ids
        )

        # NOTE: distillation has been removed (loss is now exactly aligned
        # with SpecForge's OnlineDFlashModel). We therefore no longer need
        # to materialize the target's full-sequence logits.

        (
            mtp_input_ids,
            context_position_ids,
            mtp_position_ids,
            _mtp_attention_mask,
            mask_labels,
            block_mask,
            _label_global_indices,
            weight_mask,
        ) = self._build_mtp_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            cu_seqlens=attention_mask,
        )

        # Embed MTP query tokens via the target's embed_tokens (training-only;
        # at inference vLLM uses the draft body's own embed_tokens).
        noise_embedding = self.target_model.model.embed_tokens(mtp_input_ids)

        # Full position_ids for RoPE: [target_position_ids | mtp_position_ids]
        # because key_states = concat([k_ctx, k_noise], dim=1).
        full_position_ids = torch.cat(
            [context_position_ids, mtp_position_ids], dim=-1
        )

        draft_output = self.draft_model(
            position_ids=full_position_ids,
            attention_mask=block_mask,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden.detach(),
            past_key_values=None,
            use_cache=False,
        )
        hidden_states = draft_output

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        # Use the target's lm_head at training time (tie_word_embeddings=True
        # in HunyuanOCR — vLLM will load the draft's own lm_head from this
        # checkpoint via process_eagle_weight if requested).
        logits = self.target_model.lm_head(hidden_states[:, slice_indices, :])

        # ------------------------------------------------------------------
        # Loss / accuracy — fully aligned with SpecForge's
        # ``OnlineDFlashModel.forward`` (block-wise CE with explicit
        # weight_mask, optional in-block exponential loss decay, and a
        # weight-free binary accuracy on valid slots).
        # ------------------------------------------------------------------
        if mask_labels is not None and weight_mask.numel() > 0:
            device = logits.device
            sliced_labels = (
                mask_labels[:, slice_indices]
                if not isinstance(logits_to_keep, int) or logits_to_keep > 0
                else mask_labels
            )
            sliced_weight_mask = (
                weight_mask[:, slice_indices]
                if not isinstance(logits_to_keep, int) or logits_to_keep > 0
                else weight_mask
            )

            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_labels = sliced_labels.reshape(-1)
            flat_weights = sliced_weight_mask.reshape(-1).to(torch.float32)

            # Replace -100 sentinels with 0 so cross_entropy doesn't index
            # out of vocab; the corresponding flat_weights entries are 0
            # already so they contribute nothing to the loss.
            safe_labels = flat_labels.clone()
            safe_labels[safe_labels < 0] = 0

            # Binary mask for accuracy = weight_mask without loss decay.
            binary_eval_mask = flat_weights.clone()

            # Apply in-block loss decay: pos k gets weight exp(-(k-1)/gamma);
            # pos 0 (anchor) is already zeroed via weight_mask but we still
            # clamp k-1 at >=0 to be safe.
            if (
                self.loss_decay_gamma is not None
                and self.loss_decay_gamma > 0
                and flat_weights.numel() > 0
            ):
                bs = self.block_size
                total_mtp_len = flat_weights.numel()
                pos_in_block = torch.arange(total_mtp_len, device=device) % bs
                decay = torch.exp(
                    -(pos_in_block.float() - 1.0).clamp(min=0.0)
                    / float(self.loss_decay_gamma)
                )
                flat_weights = flat_weights * decay

            loss_per_token = F.cross_entropy(
                flat_logits.float(), safe_labels, reduction="none"
            )
            valid_token_count = flat_weights.sum() + 1e-6
            ce_loss = (loss_per_token * flat_weights).sum() / valid_token_count
            loss = ce_loss

            with torch.no_grad():
                pred_ids = torch.argmax(flat_logits, dim=-1)
                correct = (pred_ids == safe_labels) & (binary_eval_mask > 0.5)
                actual_token_count = binary_eval_mask.sum() + 1e-6
                accuracy = correct.sum().float() / actual_token_count
        else:
            ce_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            loss = None
            accuracy = torch.tensor(0.0, device=logits.device)

        # Distillation has been removed in this version (per request to
        # match SpecForge's loss exactly). We still emit a zero scalar so
        # the Trainer's logging code (which reads outputs["loss"]["distill_loss"])
        # keeps working without changes.
        distill_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # Pack the main loss and per-component scalars into a single dict
        # so the custom Trainer can read them via ``outputs["loss"]``.
        # ``loss["loss"]`` is the scalar used for backprop; the rest are
        # logging-only tensors.
        loss_dict = {
            "loss": loss if loss is not None else ce_loss,
            "ce_loss": ce_loss,
            "distill_loss": distill_loss,
            "accuracy": accuracy,
        }

        out = CausalLMOutputWithPast(
            loss=loss_dict,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
        return out

    # ------------------------------------------------------------------
    # Gradient checkpointing override
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Override to prevent Transformers from applying gradient checkpointing
        to draft_layers.  The draft layers use monkey-patched forwards that are
        incompatible with torch.utils.checkpoint.checkpoint().
        """
        # Do NOT call super() — that would wrap all sub-modules including draft_layers.
        # Just mark use_cache=False so Trainer is happy.
        self.config.use_cache = False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_trainable_parameters(self):
        """Return only the trainable parameters (requires_grad=True)."""
        return [p for p in self.parameters() if p.requires_grad]

    def print_parameter_info(self):
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_total  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_total     = total_params - trainable_total

        # breakdown by named sub-module
        target_params    = sum(p.numel() for p in self.target_model.parameters())
        draft_trainable  = sum(p.numel() for p in self.draft_model.parameters() if p.requires_grad)

        print(f"Target model parameters  (frozen)    : {target_params:,}")
        print(f"Draft model parameters   (trainable) : {draft_trainable:,}")
        print(f"Total parameters                     : {total_params:,}")
        print(f"Total frozen                         : {frozen_total:,}")
        print(f"Total trainable                      : {trainable_total:,}")
        ratio = trainable_total / total_params * 100 if total_params > 0 else 0.0
        print(f"Trainable ratio                      : {ratio:.2f}%")
