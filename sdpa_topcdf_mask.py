import math
import os
from typing import Tuple, Union

import torch
import torch.nn.functional as F


def _blockmean(x: torch.Tensor, blocksz: int) -> torch.Tensor:
    """Mean-pooled block representatives over the sequence dimension.

    Args:
        x: [B, H, S, D]
        blocksz: block size
    Returns:
        [B, H, n_blocks, D]
    """
    B, H, S, D = x.shape
    device, dtype = x.device, x.dtype

    full = S // blocksz
    if full > 0:
        x_prefix = x[:, :, : full * blocksz, :].contiguous()
        mean_full = x_prefix.view(B, H, full, blocksz, D).mean(dim=3)
    else:
        mean_full = torch.empty(B, H, 0, D, device=device, dtype=dtype)

    rem = S % blocksz
    if rem > 0:
        mean_tail = x[:, :, full * blocksz :, :].mean(dim=2, keepdim=True)
    else:
        mean_tail = torch.empty(B, H, 0, D, device=device, dtype=dtype)

    return torch.cat((mean_full, mean_tail), dim=2)


def _self_similarity(x: torch.Tensor, x_bar: torch.Tensor, blocksz: int) -> torch.Tensor:
    """Mean cosine similarity between tokens and their block mean.

    Returns shape [B, H, n_blocks].
    """
    B, H, S, D = x.shape
    device, dtype = x.device, x.dtype

    full = S // blocksz
    x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
    x_bar_norm = torch.nn.functional.normalize(x_bar, p=2, dim=-1)

    parts = []
    if full > 0:
        xb = x_norm[:, :, : full * blocksz, :].reshape(B, H, full, blocksz, D)
        xbb = x_bar_norm[:, :, :full, :].unsqueeze(3)
        cos = (xb * xbb).sum(dim=-1).mean(dim=-1)  # [B,H,full]
        parts.append(cos)

    rem = S % blocksz
    if rem > 0:
        xr = x_norm[:, :, full * blocksz :, :]
        xbr = x_bar_norm[:, :, full : full + 1, :]
        cos_r = (xr * xbr).sum(dim=-1).mean(dim=-1, keepdim=True)  # [B,H,1]
        parts.append(cos_r)

    if parts:
        return torch.cat(parts, dim=2)
    return torch.empty(B, H, 0, device=device, dtype=dtype)


def build_topcdf_block_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    blocksz: int = 16,
    tau: Union[float, torch.Tensor] = 0.9,
    gamma_q: Union[float, torch.Tensor] = 0.6,
    gamma_k: Union[float, torch.Tensor] = 0.6,
) -> torch.Tensor:
    """Compute a 16×16 block Top‑CDF keep mask and expand to token level.

    Args:
        q, k: [B, H, S, D]
        blocksz: block size for both query and key blocks (default 16)
        tau: Top‑CDF threshold (scalar or broadcastable to [1,H,1,1])
        gamma_q, gamma_k: self‑similarity thresholds (scalar or broadcastable to [1,H,1])

    Returns:
        keep_tokens: boolean tensor [B, H, S, S]; True = keep (allowed attention)
    """
    assert q.shape == k.shape, "q and k must be same shape for self-attention"
    B, H, S, D = q.shape
    device = q.device

    n_q = (S + blocksz - 1) // blocksz
    n_k = n_q

    q_bar = _blockmean(q, blocksz)
    k_bar = _blockmean(k, blocksz)

    q_sim = _self_similarity(q, q_bar, blocksz)
    k_sim = _self_similarity(k, k_bar, blocksz)

    if not torch.is_tensor(gamma_q):
        gamma_q = torch.tensor(float(gamma_q), device=device, dtype=q.dtype)
    if not torch.is_tensor(gamma_k):
        gamma_k = torch.tensor(float(gamma_k), device=device, dtype=k.dtype)
    gamma_q = gamma_q.to(device=device, dtype=q.dtype)
    gamma_k = gamma_k.to(device=device, dtype=k.dtype)

    q_similar = (q_sim >= gamma_q).to(torch.bool)  # [B,H,n_q]
    k_similar = (k_sim >= gamma_k).to(torch.bool)  # [B,H,n_k]

    logits = (q_bar @ k_bar.transpose(-1, -2)) / math.sqrt(D)  # [B,H,n_q,n_k]

    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~q_similar.unsqueeze(-1), neg_inf)
    logits = logits.masked_fill(~k_similar.unsqueeze(-2), neg_inf)

    P = torch.softmax(logits, dim=-1)
    P = torch.nan_to_num(P, nan=0.0)

    P_sorted, sorted_idx = torch.sort(P, dim=-1, descending=True)
    cumsum = torch.cumsum(P_sorted, dim=-1)

    if not torch.is_tensor(tau):
        tau = torch.tensor(float(tau), device=device, dtype=P.dtype)
    tau = tau.to(device=device, dtype=P.dtype)
    thresh = tau * P.sum(dim=-1, keepdim=True)

    keep_sorted = (cumsum < thresh)
    cross = (cumsum >= thresh)
    first_cross = cross.float().argmax(dim=-1, keepdim=True)
    keep_sorted.scatter_(-1, first_cross, True)
    keep_sorted[..., 0] |= (P_sorted[..., 0] > 0)

    keep_blocks = torch.zeros_like(P, dtype=torch.bool)
    keep_blocks.scatter_(-1, sorted_idx, keep_sorted)

    # Force keep whole rows/cols for non‑self‑similar blocks
    keep_blocks |= ~q_similar.unsqueeze(-1)
    keep_blocks |= ~k_similar.unsqueeze(-2)

    # Expand to token level: [B,H,S,S]
    keep_q = keep_blocks.repeat_interleave(blocksz, dim=2)[:, :, :S, :]  # [B,H,S,n_k]
    keep_tokens = keep_q.repeat_interleave(blocksz, dim=3)[:, :, :, :S]  # [B,H,S,S]
    return keep_tokens


def sdpa_with_topcdf_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    blocksz: int = 16,
    tau: Union[float, torch.Tensor] = 0.9,
    gamma_q: Union[float, torch.Tensor] = 0.6,
    gamma_k: Union[float, torch.Tensor] = 0.6,
    layer_idx: int | None = None,
    log_flops: bool = False,
) -> torch.Tensor:
    """Run SDPA with a Top‑CDF 16×16 block mask injected via attn_mask.

    Shapes: q,k,v are [B, H, S, D]. Returns o: [B, H, S, D].
    """
    B, H, S, D = q.shape
    keep_tokens = build_topcdf_block_mask(q, k, blocksz, tau, gamma_q, gamma_k)
    attn_mask = ~keep_tokens  # True = mask out, shape [B,H,S,S]

    # Use 3D chunked math-kernel path for mask support

    # N = B*H, L = S
    q3 = q.reshape(B * H, S, D)  # [N, L, E]
    k3 = k.reshape(B * H, S, D)
    v3 = v.reshape(B * H, S, D)
    mask3d = attn_mask.reshape(B * H, S, S)  # [N, L, S]

    # Chunked queries to reduce peak memory while preserving full K/V context
    chunk = int(os.environ.get("SDPA_CHUNK", "1024"))
    outputs = []

    # Always use the masked math-kernel path to keep numerics consistent with mask injection,
    # even when the mask is all-False (i.e., has no effect).
    assert mask3d.dtype == torch.bool, f"attn_mask must be bool; got {mask3d.dtype}"
    assert mask3d.device == q3.device, "attn_mask must be on same device as q"

    # For FLOP estimate per head
    kept_per_head_total = torch.zeros(H, dtype=torch.float64, device=q3.device) if log_flops else None

    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        for start in range(0, S, chunk):
            end = min(start + chunk, S)
            q_chunk = q3[:, start:end, :].contiguous()      # [N, W, E]
            mask_chunk = mask3d[:, start:end, :].contiguous()  # [N, W, S]

            # Convert boolean mask -> additive bias (0 for keep, -inf for mask)
            attn_bias = torch.zeros_like(mask_chunk, dtype=q_chunk.dtype)
            attn_bias = attn_bias.masked_fill(mask_chunk, float("-inf"))

            o_chunk = F.scaled_dot_product_attention(q_chunk, k3, v3, attn_mask=attn_bias, dropout_p=0.0, is_causal=False)  # [N, W, E]
            outputs.append(o_chunk)

            # Accumulate kept counts per head for FLOPs (kept = ~mask)
            if log_flops:
                kept_per_row = (~mask_chunk).sum(dim=-1).to(torch.float64)   # [N, W]
                kept_per_head_step = kept_per_row.sum(dim=1)                 # [N]
                kept_per_head_step = kept_per_head_step.view(B, H).sum(dim=0)  # [H]
                kept_per_head_total += kept_per_head_step

    o3 = torch.cat(outputs, dim=1)  # [N, L, E]

    # Write FLOPs estimate if requested: per head and total
    if log_flops:
        try:
            flops_per_head = (4.0 * float(D)) * kept_per_head_total.detach().cpu().numpy()
            flops_total = float(flops_per_head.sum())
            sparsity_per_head = (kept_per_head_total / (float(B) * float(S) * float(S))).detach().cpu().numpy()
            path = os.environ.get("SDPA_FLOPS_PATH", "sdpa_flops.md")
            with open(path, "a") as f:
                f.write(f"layer={layer_idx if layer_idx is not None else -1} H={H} D={D} L={S} ")
                f.write("sparsity_head=" + ",".join(f"{x:.6f}" for x in sparsity_per_head) + " ")
                f.write("flops_head=" + ",".join(f"{int(x)}" for x in flops_per_head) + " ")
                f.write(f"flops_total={int(flops_total)}\n")
        except Exception:
            pass

    return o3.reshape(B, H, S, D)






