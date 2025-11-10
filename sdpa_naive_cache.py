"""Pure PyTorch implementation of naive cache mask for SDPA.

This module implements threshold-based sparse attention mask generation
without relying on CUDA kernels, enabling flexible block sizes (e.g., 16).

Key differences from Top-CDF:
- Uses column-wise max + threshold (conservative OR logic)
- Simpler criterion: max(attention_probs) > thresh
- No self-similarity checks
"""

import math
import os
from typing import Union

import torch
import torch.nn.functional as F


class NaiveMaskCache:
    """Simple cache for storing attention masks on GPU."""
    
    def __init__(self, compute_cache_at):
        """
        Args:
            compute_cache_at: List of iteration indices where mask should be computed.
                             E.g., [0] means compute only at first iteration.
        """
        self.cache = {}  # layer_idx -> mask
        self.compute_cache_at = compute_cache_at
    
    def should_compute(self, iter_idx):
        """Check if we should compute mask at this iteration."""
        return iter_idx in self.compute_cache_at
    
    def get_mask(self, layer_idx):
        """Retrieve cached mask for a layer."""
        return self.cache.get(layer_idx, None)
    
    def update_cache(self, layer_idx, mask):
        """Store mask for a layer."""
        self.cache[layer_idx] = mask


def build_naive_cache_block_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    blocksz: int = 16,
    thresh: float = 0.001,
    return_blocks_only: bool = False,
    layer_idx: int | None = None,
    iter_idx: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, int]:
    """Compute threshold-based block mask using column-wise max (naive cache approach).
    
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


def sdpa_with_naive_cache_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    blocksz: int = 16,
    thresh: float = 0.001,
    layer_idx: int | None = None,
    iter_idx: int | None = None,
    log_flops: bool = False,
    mask_cache: NaiveMaskCache | None = None,
) -> torch.Tensor:
    """Run SDPA with naive cache mask (threshold-based, column-wise max).
    
    Shapes: q, k, v are [B, H, S, D]. Returns o: [B, H, S, D].
    """
    B, H, S, D = q.shape
    
    # Check if we should use cache
    use_cache = mask_cache is not None
    compute_mask = not use_cache or mask_cache.should_compute(iter_idx)
    
    if compute_mask:
        # Build block-level mask
        # print(f"[NaiveCache] Computing mask for layer={layer_idx}, iter={iter_idx}")
        keep_blocks, S = build_naive_cache_block_mask(
            q, k, blocksz, thresh, return_blocks_only=True,
            layer_idx=layer_idx, iter_idx=iter_idx
        )
        
        # Cache it if caching is enabled
        if use_cache:
            mask_cache.update_cache(layer_idx, keep_blocks)
            # print(f"[NaiveCache] Cached mask for layer={layer_idx}")
    else:
        # Use cached mask
        # print(f"[NaiveCache] Using cached mask for layer={layer_idx}, iter={iter_idx}")
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
            
            # Allocate per-chunk boolean mask [B, H, W, S]
            mask_chunk_bhs = torch.empty(B, H, W, S, dtype=torch.bool, device=q3.device)
            
            i = 0
            while i < W:
                qb = int(q_block_idx[i].item())
                j = i + 1
                while j < W and int(q_block_idx[j].item()) == qb:
                    j += 1
                rows = j - i
                
                allowed_blocks = keep_blocks[:, :, qb, :]  # [B, H, n_k]
                allowed_tokens = allowed_blocks.repeat_interleave(blocksz, dim=-1)[..., :S]  # [B, H, S]
                mask_rows = ~allowed_tokens  # [B, H, S] (True = mask out)
                mask_chunk_bhs[:, :, i:j, :] = mask_rows.unsqueeze(2)  # [B, H, rows, S]
                i = j
            
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
            
            # Default filename depends on block size
            default_path = f"sdpa_naive_cache_flops{blocksz}block.md"
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

