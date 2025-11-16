import torch
import torch.nn.functional as F
import diffusers
import diffusers.models.attention_processor
import diffusers.models.transformers
import typing
import spattn.sparseattn_functionals
import spattn.mask_utils
import sdpa_naive_cache
import sdpa_naive_cache_compressed
import sdpa_topcdf_mask
import os
import time

mask_utils = spattn.mask_utils
sparseattn_functionals = spattn.sparseattn_functionals
HunyuanVideoAttnProcessor2_0 = diffusers.models.transformers.transformer_hunyuan_video.HunyuanVideoAttnProcessor2_0

def benchmark_and_run_attn(attn_fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.time()
    N = 20
    for i in range(N):
        hidden_states = attn_fn(*args, **kwargs)
    torch.cuda.synchronize()
    t1 = time.time()
    print('time = ', (t1-t0)/N, 's')


def sdpa_naive_cache_attn(query, key, value, current_layer, ditrun, **kwargs):
    """Scaled-Dot-Product Attention with naive cache mask (threshold-based, column-wise max).
    
    Pure PyTorch implementation, no CUDA dependencies.
    """
    # Inject model name for SDPA output filenames
    os.environ.setdefault("MODEL_NAME", "hunyuan")
    blocksz = kwargs.get("blocksz", 16)
    thresh = kwargs.get("thresh", 0.001)
    mask_cache = kwargs.get("mask_cache", None)
    log_flops = os.environ.get("SDPA_LOG_FLOPS", "1").lower() in {"1", "true", "yes"}
    
    return sdpa_naive_cache.sdpa_with_naive_cache_mask(
        query, key, value, 
        blocksz=blocksz, 
        thresh=thresh,
        layer_idx=current_layer, 
        iter_idx=ditrun, 
        log_flops=log_flops,
        mask_cache=mask_cache
    )

def sdpa_naive_cache_compressed_attn(query, key, value, current_layer, ditrun, **kwargs):
    """Scaled-Dot-Product Attention with naive cache mask + bit-packing compression (pure PyTorch)."""
    # Inject model name for SDPA output filenames
    os.environ.setdefault("MODEL_NAME", "hunyuan")
    blocksz = kwargs.get("blocksz", 16)
    thresh = kwargs.get("thresh", 0.001)
    mask_cache = kwargs.get("mask_cache", None)
    log_flops = os.environ.get("SDPA_LOG_FLOPS", "1").lower() in {"1", "true", "yes"}
    
    return sdpa_naive_cache_compressed.sdpa_with_naive_cache_mask_compressed(
        query, key, value, 
        blocksz=blocksz, 
        thresh=thresh,
        layer_idx=current_layer, 
        iter_idx=ditrun, 
        log_flops=log_flops,
        mask_cache=mask_cache
    )

def sdpa_topcdf_attn(query, key, value, current_layer, ditrun, **kwargs):
    """Scaled-Dot-Product Attention with Top-CDF block mask injected via attn_mask."""
    # Inject model name for SDPA output filenames
    os.environ.setdefault("MODEL_NAME", "hunyuan")
    blocksz = kwargs.get("blocksz", 16)
    tau = kwargs.get("tau", 0.95)
    gamma_q = kwargs.get("gamma_q", 0.5)
    gamma_k = kwargs.get("gamma_k", 0.5)
    log_flops = os.environ.get("SDPA_LOG_FLOPS", "1").lower() in {"1", "true", "yes"}
    return sdpa_topcdf_mask.sdpa_with_topcdf_mask(
        query, key, value,
        blocksz=blocksz,
        tau=tau, gamma_q=gamma_q, gamma_k=gamma_k,
        layer_idx=current_layer, iter_idx=ditrun,
        log_flops=log_flops
    )

class CustomProcessor(HunyuanVideoAttnProcessor2_0):
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
        elif kwargs["processor"] == "sdpa_cached":
            # Dense SDPA with naive cache mask (threshold-based, pure PyTorch)
            if "blocksz" not in self.processor_kwargs:
                self.processor_kwargs["blocksz"] = 16
            # Initialize cache
            compute_cache_at = kwargs.get("compute_cache_at", [0])
            self.processor_kwargs["mask_cache"] = sdpa_naive_cache.NaiveMaskCache(compute_cache_at)
            self.attn_fn = sdpa_naive_cache_attn
        elif kwargs["processor"] == "sdpa_cached_compressed":
            # Dense SDPA with naive cache mask + bit-packing compression (pure PyTorch)
            if "blocksz" not in self.processor_kwargs:
                self.processor_kwargs["blocksz"] = 16
            compute_cache_at = kwargs.get("compute_cache_at", [0])
            self.processor_kwargs["mask_cache"] = sdpa_naive_cache_compressed.CompressedBitMaskCache(compute_cache_at)
            self.attn_fn = sdpa_naive_cache_compressed_attn
        elif kwargs["processor"] == "sdpa_topcdf16":
            # Dense SDPA with Top-CDF block mask (pure PyTorch)
            if "blocksz" not in self.processor_kwargs:
                self.processor_kwargs["blocksz"] = 16
            self.attn_fn = sdpa_topcdf_attn
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
        encoder_hidden_states: typing.Optional[torch.Tensor] = None,
        attention_mask: typing.Optional[torch.Tensor] = None,
        image_rotary_emb: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        # 5. Attention
        hidden_states = self.attn_fn(query, key, value, self.current_layer, self.ditrun, **self.processor_kwargs)

        if self.current_layer == self.num_layers-1:
            self.ditrun += 1
        self.current_layer = (self.current_layer + 1) % self.num_layers

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


