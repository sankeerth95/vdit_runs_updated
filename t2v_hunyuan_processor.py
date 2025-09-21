import torch
import torch.nn.functional as F
import diffusers
import diffusers.models.attention_processor
import diffusers.models.transformers
import typing
import spattn.sparseattn_functionals
import spattn.mask_utils
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
        # print("current layer = ", self.current_layer)
        # print("dtype = ", query.dtype, key.dtype, value.dtype)
        # print("shape = ", query.shape, key.shape, value.shape)
        # assert(query.dtype == torch.bfloat16)
        # assert(key.dtype == torch.bfloat16)
        # assert(value.dtype == torch.bfloat16)
        # assert query.shape[0] == key.shape[0] == value.shape[0] == 1
        # assert query.shape[1] == key.shape[1] == value.shape[1] == 24
        # assert query.shape[2] == key.shape[2] == value.shape[2] == 75856
        # assert query.shape[3] == key.shape[3] == value.shape[3] == 128

        # torch.cuda.synchronize()
        # print("start")
        # attention_mask[:,:,:,-300:] = True
        # hidden_states2 = F.scaled_dot_product_attention(
            # query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # ) # attention mask is nonzero here
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



        # torch.cuda.synchronize()
        # print("done, maxdiff = ", (hidden_states-hidden_states2).abs().max())
        # print((attention_mask[0,0,0,-300:]==False).sum()) # 245
        # print((attention_mask==False).sum()) # 245

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


