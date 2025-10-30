import torch
import torch.nn.functional as F
import diffusers
import diffusers.models.transformers
import typing
import spattn.sparseattn_functionals
import spattn.mask_utils
import time
import sdpa_topcdf_mask

mask_utils = spattn.mask_utils
sparseattn_functionals = spattn.sparseattn_functionals
WanAttnProcessor = diffusers.models.transformers.transformer_wan.WanAttnProcessor2_0

def benchmark_and_run_attn(attn_fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.time()
    N = 20
    for i in range(N):
        hidden_states = attn_fn(*args, **kwargs)
    torch.cuda.synchronize()
    t1 = time.time()
    print('time = ', (t1-t0)/N, 's')


def sdpa_topcdf_attn(query, key, value, current_layer, ditrun, **kwargs):
    """Scaled-Dot-Product Attention with a 16-token Top-CDF mask injected via attn_mask.

    Accepts per-layer/per-head threshold tables in kwargs (same keys as sparse version):
      - is_sparse (ignored here)
      - cdfthreshd (tau), simthreshd1 (gamma_q), simthreshd2 (gamma_k)
    """
    heads = query.shape[1]

    def pick_table(name: str, default: float, reshape: str):
        table = kwargs.get(name, None)
        if table is None:
            t = torch.full((1, heads), float(default), device=query.device, dtype=torch.float32)
        else:
            t = torch.as_tensor(table, device=query.device, dtype=torch.float32)
            if t.ndim == 2:  # [L,H]
                t = t[current_layer]
            # now [H]
            if t.ndim == 1:
                t = t.unsqueeze(0)  # [1,H]
        if reshape == "4d":
            return t.view(1, heads, 1, 1)  # broadcast over [B,H,n_q,1]
        if reshape == "3d":
            return t.view(1, heads, 1)     # broadcast over [B,H,n_*]
        return t

    tau = pick_table("cdfthreshd", kwargs.get("tau", 0.90), reshape="4d")
    gamma_q = pick_table("simthreshd1", kwargs.get("gamma_q", 0.60), reshape="3d")
    gamma_k = pick_table("simthreshd2", kwargs.get("gamma_k", 0.60), reshape="3d")

    blocksz = kwargs.get("blocksz", 16)
    return sdpa_topcdf_mask.sdpa_with_topcdf_mask(query, key, value, blocksz=blocksz, tau=tau, gamma_q=gamma_q, gamma_k=gamma_k)


class MyCustomProcessor(WanAttnProcessor):
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
        elif kwargs["processor"] == "topcdf":
            self.attn_fn = sparseattn_functionals.attn_topcdf_cuda
        elif kwargs["processor"] == "sdpa_topcdf16":
            # Dense SDPA with 16-token Top-CDF mask injected via attn_mask (validation mode)
            if "blocksz" not in self.processor_kwargs:
                self.processor_kwargs["blocksz"] = 16
            self.attn_fn = sdpa_topcdf_attn
        elif kwargs["processor"] == "lsh":
            raise NotImplementedError("LSH not implemented yet")
        elif kwargs["processor"] == "2x":
            self.attn_fn = sparseattn_functionals.attn_2xmask_cuda

        else:
            raise ValueError("Error: Unrecognized type of attention processor/not implemented")



    # overloading function right here
    def __call__(
        self,
        attn: diffusers.models.attention_processor.Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: typing.Optional[torch.Tensor] = None,
        attention_mask: typing.Optional[torch.Tensor] = None,
        rotary_emb: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = self.attn_fn(query, key, value, self.current_layer, self.ditrun, **self.processor_kwargs)
        # if self.ditrun %13 == 1:
        #     benchmark_and_run_attn(self.attn_fn, query, key, value, self.current_layer, self.ditrun, **self.processor_kwargs)
        #     torch.cuda.synchronize()
        #     t0 = time.time()
        #     N = 20
        #     for i in range(N):
        #         hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        #     torch.cuda.synchronize()
        #     t1 = time.time()
        #     print('time = ', (t1-t0)/N, 's')



        if self.current_layer == self.num_layers-1:
            self.ditrun += 1
        self.current_layer = (self.current_layer + 1) % self.num_layers

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


