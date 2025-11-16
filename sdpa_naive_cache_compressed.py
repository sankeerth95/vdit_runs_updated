"""Pure PyTorch implementation of naive cache mask with bit-packing compression.

This module combines:
- Naive cache mask selection (threshold-based, column-wise max)
- Bit-packing compression (8 bools → 1 byte) for 8x memory reduction

Key differences from sdpa_naive_cache.py:
- Uses CompressedBitMaskCache instead of NaiveMaskCache
- Stores block-level masks with bit-packing for memory efficiency
- Same mask selection logic (column-wise max + threshold)
"""

import math
import os
from typing import Union

import torch
import torch.nn.functional as F


class CompressedBitMaskCache:
    """Compressed cache for storing block-level attention masks using bit packing.
    
    Stores block-level masks [B, H, n_q, n_k] with 8x compression via bit-packing.
    Each byte stores 8 boolean values.
    
    Memory comparison (for B=1, H=24, n_q=512, n_k=512):
    - Uncompressed bool: 1 * 24 * 512 * 512 = 6.3 MB
    - Bit-packed uint8: 6.3 MB / 8 = 0.79 MB (8x reduction)
    """
    
    def __init__(self, compute_cache_at):
        """
        Args:
            compute_cache_at: List of iteration indices where mask should be computed.
                             E.g., [0] means compute only at first iteration.
        """
        self.cache = {}  # layer_idx -> (packed_mask_uint8, shape_tuple)
        self.compute_cache_at = compute_cache_at
    
    def should_compute(self, iter_idx):
        """Check if we should compute mask at this iteration."""
        return iter_idx in self.compute_cache_at
    
    def get_mask(self, layer_idx):
        """Retrieve and unpack cached block-level mask for a layer.
        
        Returns:
            torch.Tensor: Boolean mask [B, H, n_q, n_k] or None if not cached.
        """
        if layer_idx not in self.cache:
            return None
        
        packed_mask, original_shape = self.cache[layer_idx]
        device = packed_mask.device
        
        # Unpack: each byte contains 8 bools
        # packed_mask: [..., n_bytes] -> [..., n_bytes, 8] -> [..., n_padded]
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=device)
        unpacked = (packed_mask.unsqueeze(-1) & powers) != 0  # [..., n_bytes, 8]
        
        # Flatten last two dims and trim to original size
        n_padded = unpacked.shape[-2] * 8
        mask_flat = unpacked.reshape(*unpacked.shape[:-2], n_padded)
        
        # Trim to original n_k dimension
        B, H, n_q, n_k = original_shape
        mask = mask_flat[..., :n_k]  # [B, H, n_q, n_k]
        
        return mask
    
    def update_cache(self, layer_idx, block_mask):
        """Pack and store block-level mask for a layer.
        
        Args:
            layer_idx: Layer index
            block_mask: Boolean tensor [B, H, n_q, n_k]
        """
        assert block_mask.dim() == 4, "block_mask must be [B, H, n_q, n_k]"
        device = block_mask.device
        original_shape = block_mask.shape
        B, H, n_q, n_k = original_shape
        
        # Pad last dim (n_k) to multiple of 8
        n_k_padded = ((n_k + 7) // 8) * 8
        if n_k_padded > n_k:
            pad_size = n_k_padded - n_k
            pad = torch.zeros((B, H, n_q, pad_size), dtype=block_mask.dtype, device=device)
            block_mask = torch.cat([block_mask, pad], dim=-1)
        
        # Reshape to group by 8: [..., n_k_padded] -> [..., n_bytes, 8]
        grouped = block_mask.reshape(B, H, n_q, n_k_padded // 8, 8)
        
        # Pack 8 bools into 1 byte using bit positions
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=device)
        packed = (grouped.to(torch.uint8) * powers).sum(dim=-1).to(torch.uint8)  # [..., n_bytes]
        
        self.cache[layer_idx] = (packed.contiguous(), original_shape)
    
    def get_memory_usage(self):
        """Get total memory usage of cached masks in bytes."""
        total_bytes = 0
        for packed_mask, _ in self.cache.values():
            total_bytes += packed_mask.numel() * packed_mask.element_size()
        return total_bytes


def build_naive_cache_block_mask_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    blocksz: int = 16,
    thresh: float = 0.001,
    layer_idx: int | None = None,
    iter_idx: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute block mask using chunked streaming to avoid OOM for large sequences.
    
    Instead of materializing full S×S attention matrix, processes query blocks one at a time.
    For each query block, computes attention to all keys, then reduces to block-level decision.
    
    Memory: O(blocksz × S) instead of O(S × S)
    
    Args:
        q, k: [B, H, S, D]
        blocksz: block size (e.g., 16)
        thresh: threshold for keeping blocks
        layer_idx, iter_idx: for logging
    
    Returns:
        (keep_blocks, S): block-level mask [B, H, n_q, n_k] and sequence length S
    """
    assert q.shape == k.shape, "q and k must be same shape for self-attention"
    B, H, S, D = q.shape
    device = q.device
    
    n_q = (S + blocksz - 1) // blocksz
    n_k = n_q
    
    # Allocate output block mask
    keep_blocks = torch.zeros(B, H, n_q, n_k, dtype=torch.bool, device=device)
    
    # Process one query block at a time
    for qi in range(n_q):
        q_start = qi * blocksz
        q_end = min(q_start + blocksz, S)
        q_chunk = q[:, :, q_start:q_end, :]  # [B, H, blocksz_actual, D]
        blocksz_q = q_end - q_start
        
        # Compute attention from this query block to all keys
        logits_chunk = torch.matmul(q_chunk, k.transpose(-1, -2)) / math.sqrt(D)  # [B, H, blocksz_q, S]
        P_chunk = torch.softmax(logits_chunk.float(), dim=-1).to(logits_chunk.dtype)  # [B, H, blocksz_q, S]
        
        # Pad key length to multiple of blocksz, then reshape to blocks
        S_padded = n_k * blocksz
        if S_padded != S:
            pad_len = S_padded - S
            pad = torch.zeros((B, H, blocksz_q, pad_len), dtype=P_chunk.dtype, device=P_chunk.device)
            P_chunk = torch.cat([P_chunk, pad], dim=-1)  # [B, H, blocksz_q, S_padded]
        
        # [B, H, blocksz_q, S_padded] -> [B, H, blocksz_q, n_k, blocksz]
        P_blocks = P_chunk.view(B, H, blocksz_q, n_k, blocksz)
        # Column-wise max over query tokens within the query block (dim=2)
        col_max_in_patch = P_blocks.max(dim=2).values  # [B, H, n_k, blocksz]
        # Keep if any key token in the key block exceeds threshold
        keep_row = (col_max_in_patch > thresh).any(dim=-1)  # [B, H, n_k]
        keep_blocks[:, :, qi, :] = keep_row
    
    return keep_blocks, S


def build_naive_cache_block_mask_vectorized(
    q: torch.Tensor,
    k: torch.Tensor,
    blocksz: int = 16,
    thresh: float = 0.001,
    return_blocks_only: bool = False,
    layer_idx: int | None = None,
    iter_idx: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, int]:
    """Compute threshold-based block mask using column-wise max (naive cache approach).
    
    WARNING: Materializes full S×S attention matrix - will OOM for large S (e.g., Hunyuan).
    Use build_naive_cache_block_mask_chunked for large sequences.
    
    Algorithm:
        1. Compute token-level attention: softmax(Q @ K^T / sqrt(D)) → P [B,H,S,S]
        2. For each (query block i, key block j) patch P[i_block, j_block]:
           - Take column-wise max over query tokens in the patch
           - If ANY key token in the patch has max > threshold → keep (i,j)

    
    Args:
        q, k: [B, H, S, D]
        blocksz: block size (e.g., 16)
        thresh: threshold for keeping blocks (e.g., 0.001)
        return_blocks_only: if True, return block-level mask only
        layer_idx: layer index (for logging)
        iter_idx: iteration index (for logging)
    
    Returns:
        If return_blocks_only is False (default):
            keep_tokens: boolean tensor [B, H, S, S]; True = keep (allowed attention)
        If return_blocks_only is True:
            (keep_blocks, S): block-level mask [B, H, n_q, n_k] and sequence length S
    """
    assert q.shape == k.shape, "q and k must be same shape for self-attention"
    B, H, S, D = q.shape
    device = q.device
    
    n_q = (S + blocksz - 1) // blocksz
    n_k = n_q
    
    # Compute token-level attention probabilities P: [B, H, S, S]
    # logits = q @ k^T / sqrt(D)
    logits_full = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)  # [B, H, S, S]
    P = torch.softmax(logits_full, dim=-1)  # [B, H, S, S]
    del logits_full
    
    # Build block-level keep mask by scanning patches
    # Vectorized: pad to full blocks, reshape, then reduce over block dims
    S_padded = n_q * blocksz
    if S_padded != S:
        P_padded = F.pad(P, (0, S_padded - S, 0, S_padded - S), value=0.0)
    else:
        P_padded = P
    # [B,H,S_pad,S_pad] -> [B,H,n_q,blocksz,n_k,blocksz]
    P_blocks = P_padded.view(B, H, n_q, blocksz, n_k, blocksz)
    # Column-wise max over query tokens within each (qi,kj) patch
    # max over the query-token axis inside the query block (dim=3)
    col_max_in_patch = P_blocks.max(dim=3).values  # [B,H,n_q,n_k,blocksz]
    # If ANY key token in the key block exceeds threshold → keep (qi,kj)
    keep_blocks = (col_max_in_patch > thresh).any(dim=-1)  # [B,H,n_q,n_k]
    
    if return_blocks_only:
        return keep_blocks, S
    
    # Expand to token level: [B, H, S, S]
    keep_q = keep_blocks.repeat_interleave(blocksz, dim=2)[:, :, :S, :]  # [B, H, S, n_k]
    keep_tokens = keep_q.repeat_interleave(blocksz, dim=3)[:, :, :, :S]  # [B, H, S, S]
    return keep_tokens


def build_naive_cache_block_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    blocksz: int = 16,
    thresh: float = 0.001,
    return_blocks_only: bool = False,
    layer_idx: int | None = None,
    iter_idx: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, int]:
    """Automatically choose between chunked and vectorized mask computation.
    
    Uses chunked (memory-efficient) for large sequences, vectorized (faster) for small ones.
    """
    B, H, S, D = q.shape
    
    # Estimate memory for full S×S matrix in GB
    estimated_memory_gb = (B * H * S * S * 2) / (1024**3)  # 2 bytes for bfloat16
    
    # Use chunked if estimated memory > 10 GB
    if estimated_memory_gb > 10.0:
        # print(f"[MaskBuilder] Using chunked (S={S}, est. {estimated_memory_gb:.1f} GB)")
        keep_blocks, _S = build_naive_cache_block_mask_chunked(q, k, blocksz, thresh, layer_idx, iter_idx)
        if return_blocks_only:
            return keep_blocks, S
        # Expand to token level when requested
        keep_q = keep_blocks.repeat_interleave(blocksz, dim=2)[:, :, :S, :]
        keep_tokens = keep_q.repeat_interleave(blocksz, dim=3)[:, :, :, :S]
        return keep_tokens
    else:
        # print(f"[MaskBuilder] Using vectorized (S={S}, est. {estimated_memory_gb:.1f} GB)")
        return build_naive_cache_block_mask_vectorized(q, k, blocksz, thresh, return_blocks_only, layer_idx, iter_idx)


def sdpa_with_naive_cache_mask_compressed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    blocksz: int = 16,
    thresh: float = 0.001,
    layer_idx: int | None = None,
    iter_idx: int | None = None,
    log_flops: bool = False,
    mask_cache: CompressedBitMaskCache | None = None,
) -> torch.Tensor:
    """Run SDPA with naive cache mask (threshold-based, column-wise max) and bit-packing compression.
    
    Uses CompressedBitMaskCache for 8x memory reduction compared to uncompressed storage.
    
    Shapes: q, k, v are [B, H, S, D]. Returns o: [B, H, S, D].
    """
    B, H, S, D = q.shape
    
    # Check if we should use cache
    use_cache = mask_cache is not None
    compute_mask = not use_cache or mask_cache.should_compute(iter_idx)
    
    if compute_mask:
        # Build block-level mask
        # print(f"[CompressedCache] Computing mask for layer={layer_idx}, iter={iter_idx}")
        keep_blocks, S = build_naive_cache_block_mask(
            q, k, blocksz, thresh, return_blocks_only=True,
            layer_idx=layer_idx, iter_idx=iter_idx
        )
        
        # Cache it if caching is enabled
        if use_cache:
            mask_cache.update_cache(layer_idx, keep_blocks)
            # print(f"[CompressedCache] Cached mask for layer={layer_idx}, memory={mask_cache.get_memory_usage()/1024**2:.2f} MB")
    else:
        # Use cached mask
        # print(f"[CompressedCache] Using cached mask for layer={layer_idx}, iter={iter_idx}")
        keep_blocks = mask_cache.get_mask(layer_idx)
        if keep_blocks is None:
            raise RuntimeError(f"No cached mask found for layer {layer_idx}")
    
    # N = B*H, L = S
    q3 = q.reshape(B * H, S, D)  # [N, L, E]
    k3 = k.reshape(B * H, S, D)
    v3 = v.reshape(B * H, S, D)
    
    # Chunked queries to reduce peak memory (don't build full S×S mask!)
    chunk = int(os.environ.get("SDPA_CHUNK", "1024"))
    outputs = []
    
    # For FLOP estimate per head
    kept_per_head_total = torch.zeros(H, dtype=torch.float64, device=q3.device) if log_flops else None
    
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        for start in range(0, S, chunk):
            end = min(start + chunk, S)
            W = end - start
            q_chunk = q3[:, start:end, :].contiguous()  # [N, W, E]
            
            # Build [N, W, S] mask lazily for this chunk from block-level mask
            token_idx = torch.arange(start, end, device=q3.device)
            q_block_idx = (token_idx // blocksz).to(torch.long)
            
            # Vectorized build of [B, H, W, S] mask for this chunk from block-level mask
            # Select per-row query-block decisions and expand keys back to tokens
            # keep_blocks: [B, H, n_q, n_k], q_block_idx: [W]
            allowed_blocks_rows = torch.index_select(keep_blocks, dim=2, index=q_block_idx)  # [B, H, W, n_k]
            allowed_tokens = allowed_blocks_rows.repeat_interleave(blocksz, dim=-1)[..., :S]  # [B, H, W, S]
            mask_chunk_bhs = ~allowed_tokens  # [B, H, W, S] (True = mask out)
            
            mask_chunk = mask_chunk_bhs.reshape(B * H, W, S)  # [N, W, S]
            
            # Convert boolean mask -> additive bias (0 for keep, -inf for mask)
            attn_bias = torch.zeros_like(mask_chunk, dtype=q_chunk.dtype)
            attn_bias = attn_bias.masked_fill(mask_chunk, float("-inf"))
            
            o_chunk = F.scaled_dot_product_attention(
                q_chunk, k3, v3, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
            )  # [N, W, E]
            outputs.append(o_chunk)
            
            # Accumulate kept counts per head for FLOPs
            if log_flops:
                kept_per_row = (~mask_chunk).sum(dim=-1).to(torch.float64)  # [N, W]
                kept_per_head_step = kept_per_row.sum(dim=1)  # [N]
                kept_per_head_step = kept_per_head_step.view(B, H).sum(dim=0)  # [H]
                kept_per_head_total += kept_per_head_step
    
    o3 = torch.cat(outputs, dim=1)  # [N, L, E]
    
    # Write FLOPs estimate if requested
    if log_flops:
        try:
            flops_per_head = (4.0 * float(D)) * kept_per_head_total.detach().cpu().numpy()
            flops_total = float(flops_per_head.sum())
            keep_ratio_per_head = (kept_per_head_total / (float(B) * float(S) * float(S))).detach().cpu().numpy()
            
            # Block-level keep ratio (before expansion to tokens)
            block_keep_ratio_head = keep_blocks.float().mean(dim=(0, 2, 3)).detach().cpu().numpy()
            
            # Default filename: <model>_<blocksz>_<method>.md
            model = os.environ.get("MODEL_NAME", "model")
            default_path = f"{model}_{blocksz}_cache.md"
            path = os.environ.get("SDPA_FLOPS_PATH", default_path)
            
            # Write header if file is new/empty
            write_header = False
            try:
                write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
            except Exception:
                write_header = True
            
            with open(path, "a") as f:
                if write_header:
                    f.write(f"blocksz={blocksz}\n")
                f.write(f"layer={layer_idx if layer_idx is not None else -1} H={H} D={D} L={S} ")
                f.write(f"iter={iter_idx if iter_idx is not None else -1} ")
                f.write("keep_ratio_head=" + ",".join(f"{x:.6f}" for x in keep_ratio_per_head) + " ")
                f.write("block_keep_ratio_head=" + ",".join(f"{x:.6f}" for x in block_keep_ratio_head) + " ")
                f.write("flops_head=" + ",".join(f"{int(x)}" for x in flops_per_head) + " ")
                f.write(f"flops_total={int(flops_total)}\n")
        except Exception:
            pass
    
    return o3.reshape(B, H, S, D)

