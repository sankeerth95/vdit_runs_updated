import torch
import torch.nn.functional as F
import diffusers
import diffusers.models.transformers
import typing
import sdpa_topcdf_mask
import sdpa_naive_cache
import time
import os
from pathlib import Path

# Conditional imports for CUDA-based sparse attention (not needed for sdpa_topcdf16)
try:
    import spattn.sparseattn_functionals
    import spattn.mask_utils
    mask_utils = spattn.mask_utils
    sparseattn_functionals = spattn.sparseattn_functionals
    SPATTN_AVAILABLE = True
except ImportError:
    mask_utils = None
    sparseattn_functionals = None
    SPATTN_AVAILABLE = False

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


class MyCustomProcessor(WanAttnProcessor):
    def __new__(cls, *args, **kwargs):
        """Override __new__ to bypass base class factory pattern.
        
        WanAttnProcessor.__new__ returns WanAttnProcessor instead of subclass.
        We need to bypass it to create an actual MyCustomProcessor instance.
        """
        # Create instance of THIS class, not base class
        return object.__new__(cls)
    
    def __init__(self):
        """Initialize processor with default values.
        
        Call configure(**kwargs) after instantiation to set up sparse attention.
        This pattern is required for diffusers 0.35.1+ compatibility.
        """
        # Call base class __init__ directly (skip __new__)
        WanAttnProcessor.__init__(self)
        
        # Set default values - will be configured via configure() method
        self.processor_kwargs = {}
        self.num_layers = 0
        self.ditrun = 0
        self.current_layer = 0
        self.attn_fn = None
        # Debug logging
        self.debug_file = Path("wan_debug.md")
        self.max_debug_calls = 20
        self.debug_calls = 0
        self._wrote_header = False

    def _log_md(self, lines):
        try:
            if not self._wrote_header:
                with self.debug_file.open('w', encoding='utf-8') as f:
                    f.write("# WAN Debug Log\n\n")
                    f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                self._wrote_header = True
            with self.debug_file.open('a', encoding='utf-8') as f:
                for line in lines:
                    f.write(line)
                f.write("\n")
        except Exception:
            # Never crash model execution due to logging
            pass

    def configure(self, **kwargs):
        """Configure sparse attention parameters after instantiation.
        
        Args:
            processor (str): Type of attention processor to use.
                Options: "baseline", "cached", "bitmask", "topk", "2x"
            num_layers (int): Total number of transformer layers
            compute_cache_at (int): Denoising step to compute cache (for cached/bitmask)
            compress (bool): Whether to compress cached masks
            offload (bool): Whether to offload cache to CPU
            **kwargs: Additional parameters passed to sparse attention functions
        """
        self.processor_kwargs = kwargs
        self.num_layers = kwargs["num_layers"]

        # Optional debug controls
        # Inject model name default for SDPA output filenames
        os.environ.setdefault("MODEL_NAME", "wandistilled")
        if "debug_file" in kwargs and kwargs["debug_file"]:
            self.debug_file = Path(kwargs["debug_file"]).expanduser()
        if "max_debug_calls" in kwargs and kwargs["max_debug_calls"] is not None:
            self.max_debug_calls = int(kwargs["max_debug_calls"])

        if kwargs["processor"] == "baseline":
            if not SPATTN_AVAILABLE:
                raise ImportError("spattn module required for 'baseline' processor but not found")
            self.attn_fn = sparseattn_functionals.baseline_attn
        elif kwargs["processor"] == "cached":
            if not SPATTN_AVAILABLE:
                raise ImportError("spattn module required for 'cached' processor but not found")
            self.attn_fn = sparseattn_functionals.cached_attn_cuda
            if kwargs.get("compress", False):
                self.processor_kwargs["mask_cache"] = mask_utils.CompressMaskCache(kwargs["compute_cache_at"])
            elif kwargs.get("offload", False):
                self.processor_kwargs["mask_cache"] = mask_utils.OffloadMaskCache(kwargs["compute_cache_at"])
            else:
                self.processor_kwargs["mask_cache"] = mask_utils.NaiveMaskCache(kwargs["compute_cache_at"])
        elif kwargs["processor"] == "bitmask":
            if not SPATTN_AVAILABLE:
                raise ImportError("spattn module required for 'bitmask' processor but not found")
            self.attn_fn = sparseattn_functionals.bitmaskcached_attn_cuda
            if kwargs.get("compress", False):
                self.processor_kwargs["mask_cache"] = mask_utils.CompressedBitMaskCache(kwargs["compute_cache_at"])
            else:
                self.processor_kwargs["mask_cache"] = mask_utils.BitMaskCache(kwargs["compute_cache_at"])

        elif kwargs["processor"] == "sdpa_topcdf":
            # Dense SDPA with Top‑CDF block mask (pure PyTorch)
            if "blocksz" not in self.processor_kwargs:
                self.processor_kwargs["blocksz"] = 16
            # Defaults if not provided
            self.processor_kwargs.setdefault("tau", 0.95)
            self.processor_kwargs.setdefault("gamma_q", 0.5)
            self.processor_kwargs.setdefault("gamma_k", 0.5)
            self.attn_fn = lambda q, k, v, current_layer, ditrun, **kw: sdpa_topcdf_mask.sdpa_with_topcdf_mask(
                q, k, v,
                blocksz=kw.get("blocksz", 16),
                tau=kw.get("tau", 0.95),
                gamma_q=kw.get("gamma_q", 0.5),
                gamma_k=kw.get("gamma_k", 0.5),
                layer_idx=current_layer,
                iter_idx=ditrun,
                log_flops=os.environ.get("SDPA_LOG_FLOPS", "1").lower() in {"1", "true", "yes"},
            )

        elif kwargs["processor"] == "sdpa_cached":
            # Dense SDPA with naive cache mask (threshold-based, pure PyTorch)
            if "blocksz" not in self.processor_kwargs:
                self.processor_kwargs["blocksz"] = 16
            # Initialize cache
            compute_cache_at = kwargs.get("compute_cache_at", [0])
            self.processor_kwargs["mask_cache"] = sdpa_naive_cache.NaiveMaskCache(compute_cache_at)
            self.attn_fn = lambda q, k, v, current_layer, ditrun, **kw: sdpa_naive_cache.sdpa_with_naive_cache_mask(
                q, k, v,
                blocksz=kw.get("blocksz", 16),
                thresh=kw.get("thresh", 0.001),
                layer_idx=current_layer,
                iter_idx=ditrun,
                log_flops=os.environ.get("SDPA_LOG_FLOPS", "1").lower() in {"1", "true", "yes"},
                mask_cache=kw.get("mask_cache", None),
            )

        elif kwargs["processor"] == "topk":
            if not SPATTN_AVAILABLE:
                raise ImportError("spattn module required for 'topk' processor but not found")
            self.attn_fn = sparseattn_functionals.attn_computed_with_sparse_mask_cuda
        elif kwargs["processor"] == "lsh":
            raise NotImplementedError("LSH not implemented yet")
        elif kwargs["processor"] == "2x":
            if not SPATTN_AVAILABLE:
                raise ImportError("spattn module required for '2x' processor but not found")
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
        # Lightweight debug: capture shapes at start of some calls
        do_debug = self.debug_calls < self.max_debug_calls or self.current_layer in (0, self.num_layers - 1)
        if do_debug:
            try:
                self._log_md([
                    "## Call\n",
                    f"- ditrun: {self.ditrun}\n",
                    f"- current_layer: {self.current_layer} / {self.num_layers}\n",
                    f"- hidden_states: {tuple(hidden_states.shape)}\n",
                    f"- encoder_hidden_states: {None if encoder_hidden_states is None else tuple(encoder_hidden_states.shape)}\n",
                ])
                self.debug_calls += 1
            except Exception:
                pass
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
        if do_debug:
            try:
                self._log_md([
                    "### QKV Shapes After Unflatten\n",
                    f"- heads: {attn.heads}\n",
                    f"- query: {tuple(query.shape)} (B, H, L, D)\n",
                    f"- key: {tuple(key.shape)}\n",
                    f"- value: {tuple(value.shape)}\n",
                ])
            except Exception:
                pass

        if rotary_emb is not None:
            # Debug RoPE format once per run
            if do_debug and self.ditrun == 0 and self.current_layer == 0:
                try:
                    if isinstance(rotary_emb, (tuple, list)):
                        self._log_md([
                            "### RoPE Format Detected\n",
                            f"- Type: (cos, sin) tuple (WAN 2.2 format)\n",
                            f"- cos shape: {tuple(rotary_emb[0].shape)}\n",
                            f"- sin shape: {tuple(rotary_emb[1].shape)}\n",
                        ])
                    else:
                        self._log_md([
                            "### RoPE Format Detected\n",
                            f"- Type: Complex tensor (WAN 2.1 format)\n",
                            f"- freqs shape: {tuple(rotary_emb.shape)}\n",
                        ])
                except Exception:
                    pass

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs):
                """Apply rotary embeddings - supports both formats:
                1. WAN 2.1: Complex tensor freqs
                2. WAN 2.2: (cos, sin) tuple
                """
                # Check if freqs is a tuple/list (cos, sin format - WAN 2.2)
                if isinstance(freqs, (tuple, list)) and len(freqs) == 2:
                    cos, sin = freqs
                    # hidden_states shape: (B, H, L, D)
                    original_shape = hidden_states.shape
                    
                    # Reshape into pairs: (B, H, L, D/2, 2)
                    x_pairs = hidden_states.reshape(*hidden_states.shape[:-1], -1, 2)
                    x1 = x_pairs[..., 0]  # (B, H, L, D/2)
                    x2 = x_pairs[..., 1]  # (B, H, L, D/2)
                    
                    cos_orig_shape = cos.shape
                    sin_orig_shape = sin.shape
                    
                    # cos/sin are often (B, L, 1, D) but query is (B, H, L, D)
                    # Transpose to match: (B, L, 1, D) -> (B, 1, L, D)
                    if len(cos.shape) == 4 and cos.shape[1] != 1:
                        cos = cos.transpose(1, 2)  # (B, L, 1, D) -> (B, 1, L, D)
                        sin = sin.transpose(1, 2)
                    
                    # cos/sin might be full dim (D) or half dim (D/2)
                    # If full dim, reshape and take first of each pair
                    if cos.shape[-1] == hidden_states.shape[-1]:
                        cos = cos.reshape(*cos.shape[:-1], -1, 2)[..., 0]
                        sin = sin.reshape(*sin.shape[:-1], -1, 2)[..., 0]
                    
                    # Debug shapes if this is an early call
                    if do_debug and self.ditrun == 0 and self.current_layer == 0:
                        try:
                            self._log_md([
                                "#### RoPE Application Details\n",
                                f"- Input hidden_states: {tuple(original_shape)}\n",
                                f"- x_pairs (reshaped): {tuple(x_pairs.shape)}\n",
                                f"- x1 (even): {tuple(x1.shape)}\n",
                                f"- x2 (odd): {tuple(x2.shape)}\n",
                                f"- cos (original): {tuple(cos_orig_shape)}\n",
                                f"- sin (original): {tuple(sin_orig_shape)}\n",
                                f"- cos (after reshape): {tuple(cos.shape)}\n",
                                f"- sin (after reshape): {tuple(sin.shape)}\n",
                            ])
                        except Exception:
                            pass
                    
                    # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
                    rotated_x1 = x1 * cos - x2 * sin
                    rotated_x2 = x1 * sin + x2 * cos
                    
                    # Interleave back to (B, H, L, D)
                    rotated = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
                    
                    if do_debug and self.ditrun == 0 and self.current_layer == 0:
                        try:
                            self._log_md([
                                f"- rotated_x1: {tuple(rotated_x1.shape)}\n",
                                f"- rotated_x2: {tuple(rotated_x2.shape)}\n",
                                f"- rotated (final): {tuple(rotated.shape)}\n",
                            ])
                        except Exception:
                            pass
                    
                    return rotated.type_as(hidden_states)
                else:
                    # Complex tensor format (WAN 2.1)
                    if do_debug and self.ditrun == 0 and self.current_layer == 0:
                        try:
                            self._log_md([
                                "#### RoPE Application Details (Complex)\n",
                                f"- Input hidden_states: {tuple(hidden_states.shape)}\n",
                                f"- freqs (complex tensor): {tuple(freqs.shape)}\n",
                            ])
                        except Exception:
                            pass
                    
                    dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                    x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                    
                    if do_debug and self.ditrun == 0 and self.current_layer == 0:
                        try:
                            self._log_md([
                                f"- x_rotated (complex): {tuple(x_rotated.shape)}\n",
                                f"- x_out (final): {tuple(x_out.shape)}\n",
                            ])
                        except Exception:
                            pass
                    
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
            if do_debug:
                try:
                    self._log_md([
                        "### Image Cross Attention\n",
                        f"- key_img: {tuple(key_img.shape)}\n",
                        f"- value_img: {tuple(value_img.shape)}\n",
                        f"- hidden_states_img: {tuple(hidden_states_img.shape)} (B, L, C)\n",
                    ])
                except Exception:
                    pass

        hidden_states = self.attn_fn(query, key, value, self.current_layer, self.ditrun, **self.processor_kwargs)
        if do_debug:
            try:
                self._log_md([
                    "### Output After Sparse Attention\n",
                    f"- hidden_states (pre-proj): {tuple(hidden_states.shape)} (B, L, C)\n",
                ])
            except Exception:
                pass
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
        if do_debug:
            try:
                self._log_md([
                    "### After Output Projection\n",
                    f"- hidden_states: {tuple(hidden_states.shape)} (B, L, C)\n",
                    "\n---\n\n",
                ])
            except Exception:
                pass
        return hidden_states


