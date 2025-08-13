import torch
import torch.nn.functional as F
import diffusers
import diffusers.models.attention_processor
import typing
import spattn.sparseattn_functionals
import spattn.mask_utils


mask_utils = spattn.mask_utils
sparseattn_functionals = spattn.sparseattn_functionals
FusedCogVideoXAttnProcessor2_0 = diffusers.models.attention_processor.FusedCogVideoXAttnProcessor2_0
CogVideoXAttnProcessor2_0 = diffusers.models.attention_processor.CogVideoXAttnProcessor2_0


class CustomProcessor(CogVideoXAttnProcessor2_0):
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.processor_kwargs = kwargs
        self.num_layers = kwargs["num_layers"]
        self.ditrun = 0
        self.current_layer = 0
        if kwargs["processor"] == "baseline":
            self.attn_fn = sparseattn_functionals.baseline_attn
        elif kwargs["processor"] == "cached":
            self.attn_fn = sparseattn_functionals.cached_attn_cuda
            if kwargs["compress"]:
                self.processor_kwargs["mask_cache"] = mask_utils.CompressMaskCache(kwargs["compute_cache_at"])
            elif kwargs["offload"]:
                self.processor_kwargs["mask_cache"] = mask_utils.OffloadMaskCache(kwargs["compute_cache_at"])
            else:
                self.processor_kwargs["mask_cache"] = mask_utils.NaiveMaskCache(kwargs["compute_cache_at"])
        elif kwargs["processor"] == "bitmask":
            self.attn_fn = sparseattn_functionals.bitmaskcached_attn_cuda
            if kwargs["compress"]:
                self.processor_kwargs["mask_cache"] = mask_utils.CompressedBitMaskCache(kwargs["compute_cache_at"])
            else:
                self.processor_kwargs["mask_cache"] = mask_utils.BitMaskCache(kwargs["compute_cache_at"])

        elif kwargs["processor"] == "topk":
            self.attn_fn = sparseattn_functionals.attn_computed_with_sparse_mask_cuda
        elif kwargs["processor"] == "lsh":
            raise NotImplementedError("LSH not implemented yet")
        else:
            raise ValueError("Error: Unrecognized type of attention processor/not implemented")


    def __call__(
        self,
        attn: diffusers.models.attention_processor.Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: typing.Optional[torch.Tensor] = None,
        image_rotary_emb: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # hidden_states = self.attn_fn(query, key, value, self.current_layer, self.ditrun, **self.processor_kwargs)


        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class CustomProcessorFused(FusedCogVideoXAttnProcessor2_0):
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.processor_kwargs = kwargs
        self.num_layers = kwargs["num_layers"]
        self.ditrun = 0
        self.current_layer = 0
        if kwargs["processor"] == "baseline":
            self.attn_fn = sparseattn_functionals.baseline_attn
        elif kwargs["processor"] == "cached":
            self.attn_fn = sparseattn_functionals.cached_attn_cuda
            if kwargs["compress"]:
                self.processor_kwargs["mask_cache"] = mask_utils.CompressMaskCache(kwargs["compute_cache_at"])
            elif kwargs["offload"]:
                self.processor_kwargs["mask_cache"] = mask_utils.OffloadMaskCache(kwargs["compute_cache_at"])
            else:
                self.processor_kwargs["mask_cache"] = mask_utils.NaiveMaskCache(kwargs["compute_cache_at"])
        elif kwargs["processor"] == "bitmask":
            self.attn_fn = sparseattn_functionals.bitmaskcached_attn_cuda
            if kwargs["compress"]:
                self.processor_kwargs["mask_cache"] = mask_utils.CompressedBitMaskCache(kwargs["compute_cache_at"])
            else:
                self.processor_kwargs["mask_cache"] = mask_utils.BitMaskCache(kwargs["compute_cache_at"])

        elif kwargs["processor"] == "topk":
            self.attn_fn = sparseattn_functionals.attn_computed_with_sparse_mask_cuda
        elif kwargs["processor"] == "lsh":
            raise NotImplementedError("LSH not implemented yet")
        else:
            raise ValueError("Error: Unrecognized type of attention processor/not implemented")


    def __call__(
        self,
        attn: diffusers.models.attention_processor.Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: typing.Optional[torch.Tensor] = None,
        image_rotary_emb: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        qkv = attn.to_qkv(hidden_states)
        split_size = qkv.shape[-1] // 3
        query, key, value = torch.split(qkv, split_size, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # hidden_states = self.attn_fn(query, key, value, self.current_layer, self.ditrun, **self.processor_kwargs)


        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states
