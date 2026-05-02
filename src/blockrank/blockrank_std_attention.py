"""
BlockRank Attention Implementation for Transformers

This module implements the BlockRank attention mechanism from the BlockRank paper,
enables efficient attention over block-structured inputs for in-context ranking.

The attention pattern:
- Block 0 (instruction): Causal self-attention only
- Blocks 1..M-2 (documents): Attend to block 0 + causal self-attention
- Block M-1 (query): Attend to all previous blocks + causal self-attention
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable
from typing_extensions import Unpack
from torch import nn

from transformers import AttentionInterface, AttentionMaskInterface
from transformers.models.llama.modeling_llama import TransformersKwargs, repeat_kv
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Set to True only for debugging - adds validation overhead
_DEBUG = False

_BLOCK_ORDER_DEFAULT = "instruction_first"
_DOC_CROSS_ATTN_DEFAULT = False

def set_blockrank_defaults(block_order: Optional[str] = None, doc_cross_attn: Optional[bool] = None):
    global _BLOCK_ORDER_DEFAULT, _DOC_CROSS_ATTN_DEFAULT
    if block_order is not None:
        _BLOCK_ORDER_DEFAULT = block_order
    if doc_cross_attn is not None:
        _DOC_CROSS_ATTN_DEFAULT = bool(doc_cross_attn)

def _resolve_blockrank_flags(module: nn.Module, kwargs):
    block_order = None
    doc_cross_attn = None
    cfg = getattr(module, "config", None)
    if cfg is not None:
        block_order = getattr(cfg, "blockrank_block_order", None)
        doc_cross_attn = getattr(cfg, "blockrank_doc_cross_attn", None)
    block_order = getattr(module, "blockrank_block_order", block_order)
    doc_cross_attn = getattr(module, "blockrank_doc_cross_attn", doc_cross_attn)
    block_order = kwargs.get("block_order", block_order)
    doc_cross_attn = kwargs.get("doc_cross_attn", doc_cross_attn)
    if block_order is None:
        block_order = _BLOCK_ORDER_DEFAULT
    if doc_cross_attn is None:
        doc_cross_attn = _DOC_CROSS_ATTN_DEFAULT
    return block_order, bool(doc_cross_attn)


def _normalize_return_last_block_attn_scores(module: nn.Module, kwargs):
    """
    Resolve layer-specific attention-score requests outside compiled regions.

    `layers_to_return_scores` depends on `module.layer_idx`, which would force
    `torch.compile` to specialize per decoder layer. That interacts poorly with
    gradient checkpointing because the recomputation path may hit a different
    compiled/eager variant than the original forward. Normalize it once here and
    pass only the resulting boolean into the compiled kernel.
    """
    normalized = dict(kwargs)
    layers_to_return_scores = normalized.pop("layers_to_return_scores", None)
    if layers_to_return_scores is not None:
        layer_idx = getattr(module, "layer_idx", None)
        normalized["return_last_block_attn_scores"] = bool(
            layer_idx is not None and layer_idx in layers_to_return_scores
        )
    else:
        normalized["return_last_block_attn_scores"] = bool(
            normalized.get("return_last_block_attn_scores", False)
        )
    return normalized

def _resolve_blockrank_mask_flags(kwargs):
    block_order = None
    doc_cross_attn = None
    cfg = kwargs.get("model_config") or kwargs.get("config")
    if cfg is not None:
        block_order = getattr(cfg, "blockrank_block_order", None)
        doc_cross_attn = getattr(cfg, "blockrank_doc_cross_attn", None)
    block_order = kwargs.get("block_order", block_order)
    doc_cross_attn = kwargs.get("doc_cross_attn", doc_cross_attn)
    if block_order is None:
        block_order = _BLOCK_ORDER_DEFAULT
    if doc_cross_attn is None:
        doc_cross_attn = _DOC_CROSS_ATTN_DEFAULT
    return block_order, bool(doc_cross_attn)

def _build_cross_block_mask(attention_mask: torch.Tensor, query_block: int, key_blocks: list[int], min_dtype: float):
    """
    Build an additive mask for a query block attending to selected key blocks.
    Uses per-block padding/causal mask for self, and key-valid broadcast for non-self.
    """
    B, _, _, H, _ = attention_mask.shape
    min_dtype_t = torch.tensor(min_dtype, device=attention_mask.device, dtype=attention_mask.dtype)
    query_valid = attention_mask[:, :, query_block].max(dim=-1).values > (min_dtype_t / 2)
    mask_list = []
    for block_idx in key_blocks:
        if block_idx == query_block:
            mask_block = attention_mask[:, :, block_idx]
        else:
            mask_key = attention_mask[:, :, block_idx, -1, :]  # (B, 1, H)
            mask_block = mask_key[:, :, None, :].expand(B, 1, H, H)
            mask_block = torch.where(query_valid[:, :, :, None], mask_block, min_dtype_t)
        mask_list.append(mask_block)
    return torch.cat(mask_list, dim=-1)

def check_left_padded_mask(attention_mask: torch.Tensor, verbose: bool = False):
    """
    Check if a (B, 1, M, H, H) attention mask is properly left-padded for each block.

    For each block on M axis:
    - All padding should be on the left
    - After the first valid token, there should be no padding tokens

    Args:
        attention_mask: torch.Tensor of shape (B, 1, M, H, H)
                       where -inf indicates masked (padding) and 0.0 indicates valid
        verbose: If True, return detailed violation information

    Returns:
        is_valid: bool or dict with details about violations
    """
    B, _, M, H, _ = attention_mask.shape

    # For causal masks, check the last row of each block (it sees all tokens in that block)
    # Shape: (B, 1, M, H) - the last row of each H×H block
    last_rows = attention_mask[:, :, :, -1, :]  # (B, 1, M, H)
    last_rows = last_rows.squeeze(1)  # (B, M, H)

    # Determine which positions are valid (0.0) vs masked (-inf)
    # valid_mask: True where token is valid, False where padded
    is_valid_token = (last_rows == 0.0)  # (B, M, H)
    is_padding = ~is_valid_token  # (B, M, H)

    # For proper left-padding:
    # After the first valid token (True), all subsequent tokens should be valid (True)
    # Equivalently: once we see False (padding) after True (valid), it's a violation

    # Compute cumulative OR from left to right
    # If a position has seen any valid token before (including itself), cumsum > 0
    cumsum_valid = torch.cumsum(is_valid_token.float(), dim=-1)  # (B, M, H)

    # A padding token is invalid if it appears after a valid token
    # i.e., is_padding=True AND cumsum_valid > 0 (but we need cumsum_valid from previous positions)
    cumsum_valid_shifted = torch.cat([
        torch.zeros(B, M, 1, device=attention_mask.device),
        cumsum_valid[:, :, :-1]
    ], dim=-1)  # (B, M, H)

    # Violation: padding appears after we've seen a valid token
    violations = is_padding & (cumsum_valid_shifted > 0)  # (B, M, H)

    # Check if each block has any violations
    has_violation = violations.any(dim=-1)  # (B, M)
    is_properly_left_padded = ~has_violation  # (B, M)

    if not verbose:
        return torch.all(~has_violation).item()

    # Find first violation position in each block (for debugging)
    violation_positions = torch.where(violations,
                                     torch.arange(H, device=attention_mask.device).view(1, 1, H),
                                     torch.tensor(H, device=attention_mask.device))  # (B, M, H)
    first_violation_pos = violation_positions.min(dim=-1)  # (B, M)

    # Count valid tokens in each block (from last row)
    num_valid_tokens = is_valid_token.sum(dim=-1)  # (B, M)

    return {
        'is_properly_left_padded': is_properly_left_padded,  # (B, M)
        'has_violation': has_violation,  # (B, M)
        'first_violation_pos': first_violation_pos.values,  # (B, M)
        'num_valid_tokens': num_valid_tokens,  # (B, M)
        'violations_per_block': violations.sum(dim=-1),  # (B, M) - count of violations
    }


def _additive_mask_valid_tokens(mask_row: torch.Tensor) -> torch.Tensor:
    """Return validity mask from additive attention mask rows (0 valid, very negative masked)."""
    threshold = torch.finfo(mask_row.dtype).min / 2
    return mask_row > threshold


def _select_doc_anchor_tokens_from_additive_mask(
    doc_mask_row: torch.Tensor,
    token_compression_mode: str,
    token_compression_last_k: int,
    token_compression_segment_k: int = 10,
    token_compression_segment_anchor: str = "end",
) -> torch.Tensor:
    """Select anchor tokens (last-k / mid+last / segment-wise) from per-doc additive mask rows."""
    mode = token_compression_mode.lower()
    valid = _additive_mask_valid_tokens(doc_mask_row)

    if mode == "last":
        if token_compression_last_k <= 0:
            raise ValueError(
                f"token_compression_last_k must be > 0 when token_compression_mode='last', got {token_compression_last_k}"
            )
        H = valid.shape[-1]
        token_idx = torch.arange(H, device=valid.device).view(*([1] * (valid.dim() - 1)), H)
        last_idx = torch.where(valid, token_idx, torch.full_like(token_idx, -1)).amax(dim=-1)
        first_idx = (last_idx - int(token_compression_last_k) + 1).clamp(min=0)
        return (
            valid
            & (token_idx <= last_idx.unsqueeze(-1))
            & (token_idx >= first_idx.unsqueeze(-1))
        )

    if mode == "mid_last":
        rank = valid.cumsum(dim=-1)
        valid_count = valid.sum(dim=-1, keepdim=True)
        mid_rank = (valid_count + 1) // 2
        last_rank = valid_count
        return valid & ((rank == mid_rank) | (rank == last_rank))

    if mode == "segment":
        if token_compression_segment_k <= 0:
            raise ValueError(
                "token_compression_segment_k must be > 0 when token_compression_mode='segment', "
                f"got {token_compression_segment_k}"
            )
        anchor = token_compression_segment_anchor.lower()
        if anchor not in ("start", "end"):
            raise ValueError(
                "token_compression_segment_anchor must be one of ['start', 'end'] when "
                f"token_compression_mode='segment', got {token_compression_segment_anchor}"
            )
        rank = valid.cumsum(dim=-1) - 1
        if anchor == "start":
            keep = (rank.remainder(int(token_compression_segment_k)) == 0)
        else:
            valid_count = valid.sum(dim=-1, keepdim=True)
            dist_from_end = valid_count - 1 - rank
            keep = (dist_from_end.remainder(int(token_compression_segment_k)) == 0)
        return valid & keep

    return valid


def _normalize_attention_weighted_top_k(attention_weighted_top_k: Optional[int | str]) -> Optional[int]:
    """Normalize optional top-k config; accepts None/null/none."""
    if attention_weighted_top_k is None:
        return None
    if isinstance(attention_weighted_top_k, str):
        raw = attention_weighted_top_k.strip().lower()
        if raw in ("", "none", "null"):
            return None
    top_k = int(attention_weighted_top_k)
    if top_k <= 0:
        raise ValueError(f"attention_weighted_top_k must be > 0 when provided, got {attention_weighted_top_k}")
    return top_k


def _doc_block_range(block_order: str, num_blocks: int) -> range:
    """Return doc block index range excluding the final query block."""
    if block_order == "instruction_first":
        return range(1, max(num_blocks - 1, 1))
    if block_order == "doc_first":
        return range(0, max(num_blocks - 1, 0))
    raise ValueError(f"Unknown block_order: {block_order}")


def _compute_doc_attention_weighted_topk_mask_from_qk(
    query_blocks: torch.Tensor,
    key_blocks: torch.Tensor,
    full_attention_mask: torch.Tensor,
    block_order: str,
    top_k: int,
) -> torch.Tensor:
    """
    Build per-document top-k token mask using each doc's last token attending to that doc.

    query_blocks/key_blocks: (B, N, M, H, D)
    full_attention_mask: (B, 1, M, H, H), additive mask
    returns: (B, num_docs, H) bool mask
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    B, N, M, H, D = query_blocks.shape
    doc_indices = list(_doc_block_range(block_order, M))
    num_docs = len(doc_indices)
    if num_docs == 0:
        return torch.zeros(B, 0, H, dtype=torch.bool, device=query_blocks.device)

    token_idx = torch.arange(H, device=query_blocks.device)
    k_eff = min(int(top_k), H)
    doc_topk_mask = torch.zeros(B, num_docs, H, dtype=torch.bool, device=query_blocks.device)

    for local_doc_idx, block_idx in enumerate(doc_indices):
        q_doc = query_blocks[:, :, block_idx, :, :]  # (B, N, H, D)
        k_doc = key_blocks[:, :, block_idx, :, :]    # (B, N, H, D)
        self_mask = full_attention_mask[:, 0, block_idx, :, :]  # (B, H, H)

        valid = _additive_mask_valid_tokens(self_mask[:, -1, :])  # (B, H)
        last_idx = torch.where(
            valid,
            token_idx.view(1, H),
            torch.full((1, H), -1, device=valid.device, dtype=token_idx.dtype),
        ).amax(dim=-1)  # (B,)
        has_valid = last_idx >= 0
        safe_last_idx = last_idx.clamp(min=0).to(torch.long)

        q_last = torch.gather(
            q_doc,
            dim=2,
            index=safe_last_idx.view(B, 1, 1, 1).expand(B, N, 1, D),
        ).squeeze(2)  # (B, N, D)
        mask_row = torch.gather(
            self_mask,
            dim=1,
            index=safe_last_idx.view(B, 1, 1).expand(B, 1, H),
        ).squeeze(1)  # (B, H)

        logits = torch.einsum("bnd,bnhd->bnh", q_last, k_doc)
        logits = logits + mask_row[:, None, :].to(logits.dtype)
        logits = logits.masked_fill(~has_valid[:, None, None], float("-inf"))

        head_reduced_logits = logits.mean(dim=1)  # (B, H)
        topk_idx = head_reduced_logits.topk(k=k_eff, dim=-1).indices

        keep = torch.zeros(B, H, dtype=torch.bool, device=query_blocks.device)
        keep.scatter_(1, topk_idx, True)
        doc_topk_mask[:, local_doc_idx, :] = keep & valid & has_valid[:, None]

    return doc_topk_mask


def _compress_query_doc_visibility_mask(
    mask_others_blocks: torch.Tensor,
    block_order: str,
    token_compression_mode: str,
    token_compression_last_k: int,
    token_compression_segment_k: int = 10,
    token_compression_segment_anchor: str = "end",
    attention_weighted_top_k: Optional[int | str] = 1,
    query_blocks: Optional[torch.Tensor] = None,
    key_blocks: Optional[torch.Tensor] = None,
    full_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compress doc-token visibility only for the last query block attending to docs."""
    mode = (token_compression_mode or "none").lower()
    if mode not in ("none", "topk", "last", "mid_last", "segment"):
        raise ValueError(f"Unknown token_compression_mode: {token_compression_mode}")
    aw_top_k = _normalize_attention_weighted_top_k(attention_weighted_top_k)
    if mode == "segment":
        # Strict segment-wise selection: do not union with attention-weighted top-k.
        aw_top_k = None

    _, _, prev_blocks, _ = mask_others_blocks.shape
    if block_order == "instruction_first":
        doc_slice = slice(1, prev_blocks)
    elif block_order == "doc_first":
        doc_slice = slice(0, prev_blocks)
    else:
        raise ValueError(f"Unknown block_order: {block_order}")

    out = mask_others_blocks.clone()
    doc_rows = out[:, :, doc_slice, :]
    if doc_rows.numel() == 0:
        return out

    if mode == "segment":
        doc_keep = _select_doc_anchor_tokens_from_additive_mask(
            doc_mask_row=doc_rows,
            token_compression_mode="segment",
            token_compression_last_k=int(token_compression_last_k),
            token_compression_segment_k=int(token_compression_segment_k),
            token_compression_segment_anchor=token_compression_segment_anchor,
        )
    else:
        # Default keep-set is last-k (or mid+last for legacy mode).
        default_mode = "mid_last" if mode == "mid_last" else "last"
        doc_keep = _select_doc_anchor_tokens_from_additive_mask(
            doc_mask_row=doc_rows,
            token_compression_mode=default_mode,
            token_compression_last_k=int(token_compression_last_k),
            token_compression_segment_k=int(token_compression_segment_k),
            token_compression_segment_anchor=token_compression_segment_anchor,
        )

    if aw_top_k is not None:
        if query_blocks is None or key_blocks is None or full_attention_mask is None:
            raise ValueError(
                "query_blocks, key_blocks and full_attention_mask are required when attention_weighted_top_k is used."
            )
        aw_keep = _compute_doc_attention_weighted_topk_mask_from_qk(
            query_blocks=query_blocks,
            key_blocks=key_blocks,
            full_attention_mask=full_attention_mask,
            block_order=block_order,
            top_k=aw_top_k,
        )  # (B, num_docs, H)
        doc_keep = doc_keep | aw_keep.unsqueeze(1)

    masked_value = 0.7 * torch.finfo(out.dtype).min
    out[:, :, doc_slice, :] = torch.where(doc_keep, doc_rows, torch.full_like(doc_rows, masked_value))
    return out

def _eager_blockrank_attention_forward_impl(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    '''
    Eager BlockRank attention implementation.

    Implements the BlockRank attention pattern where:
    - Block 0 attends causally to itself
    - Blocks 1..M-2 attend causally to self and fully to block 0
    - Block M-1 attends fully to all blocks and causally to itself

    Args:
        module: The attention module
        query: (B, N, M*H, D) Query tensor
        key: (B, Nk, M*H, D) Key tensor
        value: (B, Nk, M*H, D) Value tensor
        attention_mask: (B, 1, M, H, H) Additive mask (0 for allowed, -inf for masked)
        scaling: Attention scaling factor
        dropout: Dropout probability

    Returns:
        attn_output: (B, M*H, N, D) Attention output
        attn_weights: Attention weights (for compatibility, returns None)
    '''
    B, N, MH, D = query.shape
    _, Nk, _, _ = key.shape
    assert attention_mask is not None, "BlockRank attention requires an attention mask"
    assert len(attention_mask.shape) == 5, "Attention mask must be 5D for BlockRank attention"
    _, _, M, H, _ = attention_mask.shape
    assert H == MH // M, f"Block size H={H} does not match MH // M = {MH // M}"

    # Repeat K/V heads for GQA/MQA so that key/value heads match query heads
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    # Reshape to block structure
    query = query.view(B, N, M, H, D)
    key = key.view(B, N, M, H, D)
    value = value.view(B, N, M, H, D)
    attention_mask = attention_mask.view(B, 1, M, H, H)  # redundant but explicit
    device, dtype = query.device, query.dtype
    block_order, doc_cross_attn = _resolve_blockrank_flags(module, kwargs)
    if block_order not in ("instruction_first", "doc_first"):
        raise ValueError(f"Unknown block_order: {block_order}")

    # Validate mask is properly left-padded (only in debug mode)
    if _DEBUG:
        assert check_left_padded_mask(attention_mask, verbose=False), \
            "Attention mask is not properly left-padded per block"

    # Output tensor
    out = torch.empty((B, N, M, H, D), device=device, dtype=dtype)

    # Convenience views
    Q = query * scaling  # (B, N, M, H, D)
    K = key             # (B, N, M, H, D)
    V = value           # (B, N, M, H, D)

    # -----------------------------
    # Block 0: causal self-attention (instruction block only)
    # -----------------------------
    if block_order == "instruction_first":
        Q0 = Q[:, :, 0]                 # (B, N, H, D)
        K0 = K[:, :, 0]                 # (B, N, H, D)
        V0 = V[:, :, 0]                 # (B, N, H, D)
        m0 = attention_mask[:, :, 0]    # (B, 1, H, H)

        s0 = torch.matmul(Q0, K0.transpose(-2, -1)) + m0
        p0 = F.softmax(s0, dim=-1, dtype=torch.float32).to(dtype)
        if dropout:
            p0 = F.dropout(p0, p=dropout, training=module.training)
        out[:, :, 0] = torch.matmul(p0, V0)

        # Early return if only one block
        if M == 1:
            out = out.view(B, N, MH, D).transpose(1, 2).contiguous()
            return out, None

    # ------------------------------------------------------------
    # Document blocks: doc-first or doc-cross configurations
    # ------------------------------------------------------------
    if block_order == "instruction_first" and not doc_cross_attn:
        # Optimized default path: middle blocks attend to instruction + self
        if M > 2:
            Q_mid = Q[:, :, 1:M-1]                # (B, N, M-2, H, D)
            K_self = K[:, :, 1:M-1]               # (B, N, M-2, H, D)
            V_self = V[:, :, 1:M-1]               # (B, N, M-2, H, D)

            # Repeat block 0 K/V for each middle block
            K0_rep = K[:, :, 0].unsqueeze(2).expand(B, N, M-2, H, D)  # (B, N, M-2, H, D)
            V0_rep = V[:, :, 0].unsqueeze(2).expand(B, N, M-2, H, D)  # (B, N, M-2, H, D)

            # Concatenate: [K0 | Kself]
            K_mid = torch.cat([K0_rep, K_self], dim=-2)       # (B, N, M-2, 2H, D)
            V_mid = torch.cat([V0_rep, V_self], dim=-2)       # (B, N, M-2, 2H, D)

            # Compute attention scores
            s_mid = torch.matmul(Q_mid, K_mid.transpose(-2, -1))  # (B, N, M-2, H, 2H)

            # Build concatenated mask:
            # - first H columns: broadcast "last valid" row from block 0
            # - next H columns: per-block causal self mask
            mask_first_cols = attention_mask[:, :, 0, -1, :]                     # (B, 1, H)
            mask_first = mask_first_cols.unsqueeze(2).unsqueeze(2)               # (B, 1, 1, 1, H)
            mask_first = mask_first.expand(B, 1, M-2, H, H)                      # (B, 1, M-2, H, H)
            mask_self = attention_mask[:, :, 1:M-1]                              # (B, 1, M-2, H, H)

            # Combine: take minimum (more restrictive) of block 0 mask and self mask for first H cols
            mask_first = torch.minimum(mask_first, mask_self[:, :, :, -1, :, None])  # (B, 1, M-2, H, H)
            m_mid = torch.cat([mask_first, mask_self], dim=-1)                   # (B, 1, M-2, H, 2H)

            s_mid = s_mid + m_mid
            p_mid = F.softmax(s_mid, dim=-1, dtype=torch.float32).to(dtype)
            if dropout:
                p_mid = F.dropout(p_mid, p=dropout, training=module.training)
            out[:, :, 1:M-1] = torch.matmul(p_mid, V_mid)
    else:
        doc_start = 1 if block_order == "instruction_first" else 0
        doc_end = M - 1
        if doc_end > doc_start:
            min_dtype = 0.7 * torch.finfo(dtype).min
            for i in range(doc_start, doc_end):
                if block_order == "instruction_first":
                    key_blocks = [0] + list(range(1, i + 1)) if doc_cross_attn else [0, i]
                else:
                    key_blocks = list(range(0, i + 1)) if doc_cross_attn else [i]
                K_cat = torch.cat([K[:, :, b] for b in key_blocks], dim=-2)
                V_cat = torch.cat([V[:, :, b] for b in key_blocks], dim=-2)
                m_blk = _build_cross_block_mask(attention_mask, i, key_blocks, min_dtype)
                s_blk = torch.matmul(Q[:, :, i], K_cat.transpose(-2, -1)) + m_blk
                p_blk = F.softmax(s_blk, dim=-1, dtype=torch.float32).to(dtype)
                if dropout:
                    p_blk = F.dropout(p_blk, p=dropout, training=module.training)
                out[:, :, i] = torch.matmul(p_blk, V_cat)

    # ------------------------------------------------------------
    # Last block (M-1): full to all blocks, causal to self
    # Concatenate K/V across all blocks
    # ------------------------------------------------------------
    Q_last = Q[:, :, M-1]                                        # (B, N, H, D)
    K_all = K.reshape(B, N, M * H, D)                            # (B, N, M*H, D)
    V_all = V.reshape(B, N, M * H, D)                            # (B, N, M*H, D)

    # Mask for other blocks (0..M-2): take last row and broadcast over query rows.
    # Optional token compression is applied ONLY when the last query block attends to doc blocks.
    token_compression_mode = kwargs.get("token_compression_mode", "none")
    token_compression_last_k = kwargs.get("token_compression_last_k", 1)
    if token_compression_last_k is None:
        token_compression_last_k = 1
    token_compression_last_k = int(token_compression_last_k)
    token_compression_segment_k = kwargs.get("token_compression_segment_k", 10)
    if token_compression_segment_k is None:
        token_compression_segment_k = 10
    token_compression_segment_k = int(token_compression_segment_k)
    token_compression_segment_anchor = kwargs.get("token_compression_segment_anchor", "end")
    attention_weighted_top_k = kwargs.get("attention_weighted_top_k", 1)
    mask_others = attention_mask[:, :, :M-1, -1, :]              # (B, 1, M-1, H)
    mask_others = _compress_query_doc_visibility_mask(
        mask_others_blocks=mask_others,
        block_order=block_order,
        token_compression_mode=token_compression_mode,
        token_compression_last_k=token_compression_last_k,
        token_compression_segment_k=token_compression_segment_k,
        token_compression_segment_anchor=token_compression_segment_anchor,
        attention_weighted_top_k=attention_weighted_top_k,
        query_blocks=Q,
        key_blocks=K,
        full_attention_mask=attention_mask,
    )
    mask_others = mask_others.reshape(B, 1, (M - 1) * H)         # (B, 1, (M-1)*H)
    mask_others = mask_others.unsqueeze(-2).expand(B, 1, H, (M - 1) * H)  # (B, 1, H, (M-1)*H)
    mask_self_last = attention_mask[:, :, M-1]                   # (B, 1, H, H)

    # Combine: minimum of other blocks mask and self mask
    mask_others = torch.minimum(mask_others, mask_self_last[:, :, -1, :, None])
    m_last = torch.cat([mask_others, mask_self_last], dim=-1)    # (B, 1, H, M*H)

    s_last = torch.matmul(Q_last, K_all.transpose(-2, -1)) + m_last  # (B, N, H, M*H)
    p_last = F.softmax(s_last, dim=-1, dtype=torch.float32).to(dtype)
    if dropout:
        p_last = F.dropout(p_last, p=dropout, training=module.training)
    out[:, :, M-1] = torch.matmul(p_last, V_all)

    # Reshape output to expected format
    out = out.view(B, N, MH, D).transpose(1, 2).contiguous()  # (B, M*H, N, D)

    return_last_block_attn_scores = bool(kwargs.get('return_last_block_attn_scores', False))
    num_last_queries = kwargs.get('num_last_queries', 16)

    if return_last_block_attn_scores:
        s_last = s_last[:, :, -num_last_queries:] if s_last.size(-2) >= num_last_queries else s_last  # (B, N, num_last_queries, M*H)
    else:
        s_last = None
    return out, s_last


def eager_blockrank_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    kwargs = _normalize_return_last_block_attn_scores(module, kwargs)
    return _eager_blockrank_attention_forward_impl(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=dropout,
        **kwargs,
    )

def eager_blockrank_attention_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = None,
    attention_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create BlockRank attention mask for eager attention.

    Converts a binary block attention mask (B, M, H) to a 5D causal block mask
    (B, 1, M, H, H) where each block has causal masking and padding is properly handled.

    Args:
        batch_size: Batch size
        q_length: Query length (not used, for interface compatibility)
        kv_length: Key-value length (not used, for interface compatibility)
        q_offset: Query offset (not used, for interface compatibility)
        kv_offset: KV offset (not used, for interface compatibility)
        mask_function: Mask function (not used, for interface compatibility)
        attention_mask: (B, M, H) Binary mask where 1=valid, 0=padding
        dtype: Output dtype
        **kwargs: Additional arguments (may include model config)

    Returns:
        mask: (B, 1, M, H, H) Additive mask (0 for attend, -inf for mask)
    """
    assert attention_mask is not None, "attention_mask must be provided for BlockRank eager attention"
    B, M, H = attention_mask.shape

    # Convert to boolean: True=valid, False=padding
    mask = attention_mask.bool().view(B, 1, M, 1, H)  # (B, 1, M, 1, H)

    # Create causal mask for each block
    causal_mask = torch.tril(torch.ones(H, H, device=mask.device, dtype=torch.bool))  # (H, H)

    # Combine: valid tokens + causal constraint
    mask = mask & (causal_mask[None, None, None, :, :])  # (B, 1, M, H, H)

    # Convert to additive mask: 0 for attend, -inf for mask
    min_dtype = 0.7 * torch.finfo(dtype).min  # Use 0.7 to avoid overflow
    mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)

    return mask

def flex_blockrank_attention_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Mask interface for flex_blockrank attention.
    
    Creates a BlockMask for efficient sparse attention computation.
    This is called once to set up the mask, avoiding overhead in the forward pass.
    
    Args:
        attention_mask: (B, M, H) binary mask where 1=valid, 0=padding
    Returns:
        BlockMask object that will be used by flex_attention
    """
    assert attention_mask is not None, "attention_mask must be provided for BlockRank flex attention"
    
    # Extract dimensions from attention_mask
    # attention_mask shape: (B, M, H) where M=num_blocks, H=block_size
    B, M, H = attention_mask.shape
    MH = M * H  # Total sequence length
    
    # Store as bool tensor for efficient lookup
    block_valid_mask = attention_mask.bool()
    
    block_order, doc_cross_attn = _resolve_blockrank_mask_flags(kwargs)
    if block_order not in ("instruction_first", "doc_first"):
        raise ValueError(f"Unknown block_order: {block_order}")

    # Define BlockRank mask function
    # This captures block_valid_mask, M, H in the closure
    if block_order == "instruction_first":
        def blockrank_mask_fn(b, h, q_idx, kv_idx):
            """
            BlockRank attention mask logic (instruction_first).
            """
            q_block = q_idx // H
            kv_block = kv_idx // H
            q_pos = q_idx % H
            kv_pos = kv_idx % H

            q_valid = block_valid_mask[b, q_block, q_pos]
            kv_valid = block_valid_mask[b, kv_block, kv_pos]
            both_valid = q_valid & kv_valid

            causal = q_pos >= kv_pos
            same_block = q_block == kv_block
            is_block_0 = q_block == 0
            is_last_block = q_block == (M - 1)
            is_doc_block = ~is_block_0 & ~is_last_block
            kv_is_block_0 = kv_block == 0
            kv_is_doc = (kv_block > 0) & (kv_block < (M - 1))

            block_0_pattern = is_block_0 & same_block & causal
            if doc_cross_attn:
                prev_doc = kv_is_doc & (kv_block < q_block)
                doc_pattern = is_doc_block & (kv_is_block_0 | prev_doc | (same_block & causal))
            else:
                doc_pattern = is_doc_block & (kv_is_block_0 | (same_block & causal))

            last_pattern = is_last_block & ((same_block & causal) | (~same_block & (kv_block < q_block)))
            return both_valid & (block_0_pattern | doc_pattern | last_pattern)
    else:
        def blockrank_mask_fn(b, h, q_idx, kv_idx):
            """
            BlockRank attention mask logic (doc_first).
            """
            q_block = q_idx // H
            kv_block = kv_idx // H
            q_pos = q_idx % H
            kv_pos = kv_idx % H

            q_valid = block_valid_mask[b, q_block, q_pos]
            kv_valid = block_valid_mask[b, kv_block, kv_pos]
            both_valid = q_valid & kv_valid

            causal = q_pos >= kv_pos
            same_block = q_block == kv_block
            is_last_block = q_block == (M - 1)
            is_doc_block = q_block < (M - 1)
            kv_is_doc = kv_block < (M - 1)

            if doc_cross_attn:
                prev_doc = kv_is_doc & (kv_block < q_block)
                doc_pattern = is_doc_block & (prev_doc | (same_block & causal))
            else:
                doc_pattern = is_doc_block & (same_block & causal)

            last_pattern = is_last_block & ((same_block & causal) | (~same_block & (kv_block < q_block)))
            return both_valid & (doc_pattern | last_pattern)
    
    # Create BlockMask once here (expensive operation, should be cached)
    # B, H are batch and head dimensions - pattern varies per batch but not per head
    block_mask = create_block_mask(
        blockrank_mask_fn,
        B=B,  # Need batch-specific patterns due to padding
        H=None,  # Pattern is same across all heads (will broadcast)
        Q_LEN=MH,
        KV_LEN=MH,
        device=attention_mask.device,
    )
    
    return block_mask

def flex_blockrank_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],  # This will be the BlockMask now
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    '''
    Flex Attention BlockRank implementation
    
    Functionally identical to eager_blockrank_attention_forward but uses PyTorch's
    flex_attention API with pre-computed BlockMask for efficient sparse computation.

    Shapes:
      query: (B, N, M*H, D) - already in correct format for flex_attention
      key:   (B, Nk, M*H, D)
      value: (B, Nk, M*H, D)
      attention_mask: BlockMask object created by flex_blockrank_attention_mask
    
    Note: flex_attention expects (B, H, S, D) format which matches our (B, N, M*H, D)
    
    Semantics:
      - Block 0 attends causally to itself.
      - Blocks 1..M-2 attend causally to self and fully to block 0.
      - Block M-1 attends fully to all blocks and causally to itself.
    '''
    B, N, MH, D = query.shape
    
    # Repeat K/V heads for GQA/MQA so that key/value heads match query heads
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    # attention_mask is now a BlockMask object created in the mask interface
    block_mask = attention_mask
    
    # query, key, value are already in (B, N, M*H, D) format
    # This is exactly what flex_attention expects: (B, H, S, D)
    # No transpose needed!
    
    # Apply flex attention with pre-computed BlockMask
    attn_output = flex_attention(
        query,
        key, 
        value,
        block_mask=block_mask,
        scale=scaling,
        enable_gqa=False,  # We already handled GQA via repeat_kv
    )
    
    # attn_output shape: (B, N, M*H, D)
    # Transformers expects output as (B, M*H, N, D), so we transpose
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    return attn_output, None  # flex_attention doesn't return attention weights

def _sdpa_blockrank_attention_forward_impl(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    '''
    Eager BlockRank attention implementation.

    Implements the BlockRank attention pattern where:
    - Block 0 attends causally to itself
    - Blocks 1..M-2 attend causally to self and fully to block 0
    - Block M-1 attends fully to all blocks and causally to itself

    Args:
        module: The attention module
        query: (B, N, M*H, D) Query tensor
        key: (B, Nk, M*H, D) Key tensor
        value: (B, Nk, M*H, D) Value tensor
        attention_mask: (B, 1, M, H, H) Additive mask (0 for allowed, -inf for masked)
        scaling: Attention scaling factor
        dropout: Dropout probability

    Returns:
        attn_output: (B, M*H, N, D) Attention output
        attn_weights: Attention weights (for compatibility, returns None)
    '''
    B, N, MH, D = query.shape
    _, Nk, _, _ = key.shape
    assert attention_mask is not None, "BlockRank attention requires an attention mask"
    assert len(attention_mask.shape) == 5, "Attention mask must be 5D for BlockRank attention"
    _, _, M, H, _ = attention_mask.shape
    assert H == MH // M, f"Block size H={H} does not match MH // M = {MH // M}"
    
    # # Convert attention mask to boolean if it has additive format (0.0 for attend, -inf for mask)
    # if attention_mask.dtype != torch.bool and attention_mask.max() < 1e-6:
    #     # Mask is in additive format: 0.0 = attend, -inf = mask
    #     # Convert to boolean: True = attend, False = mask
    #     attention_mask = (attention_mask > -1.0)
    

    # Repeat K/V heads for GQA/MQA so that key/value heads match query heads
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    # Reshape to block structure
    query = query.view(B, N, M, H, D)
    key = key.view(B, N, M, H, D)
    value = value.view(B, N, M, H, D)
    attention_mask = attention_mask.view(B, 1, M, H, H)  # redundant but explicit
    device, dtype = query.device, query.dtype

    # Validate mask is properly left-padded (only in debug mode)
    if _DEBUG:
        assert check_left_padded_mask(attention_mask, verbose=False), \
            "Attention mask is not properly left-padded per block"

    # Output tensor
    out = torch.empty((B, N, M, H, D), device=device, dtype=dtype)

    block_order, doc_cross_attn = _resolve_blockrank_flags(module, kwargs)
    if block_order not in ("instruction_first", "doc_first"):
        raise ValueError(f"Unknown block_order: {block_order}")

    # Convenience views
    Q = query  # (B, N, M, H, D)
    K = key             # (B, N, M, H, D)
    V = value           # (B, N, M, H, D)

    # -----------------------------
    # Block 0: causal self-attention (instruction block only)
    # -----------------------------
    if block_order == "instruction_first":
        Q0 = Q[:, :, 0]                 # (B, N, H, D)
        K0 = K[:, :, 0]                 # (B, N, H, D)
        V0 = V[:, :, 0]                 # (B, N, H, D)
        m0 = attention_mask[:, :, 0]    # (B, 1, H, H)

        out[:, :, 0] = F.scaled_dot_product_attention(
            Q0, K0, V0,
            attn_mask=m0,
            dropout_p=dropout if module.training else 0.0,
            scale=scaling,
        )

        # Early return if only one block
        if M == 1:
            out = out.view(B, N, MH, D).transpose(1, 2).contiguous()
            return out, None

    # ------------------------------------------------------------
    # Document blocks: doc-first or doc-cross configurations
    # ------------------------------------------------------------
    if block_order == "instruction_first" and not doc_cross_attn:
        # Optimized default path: middle blocks attend to instruction + self
        if M > 2:
            Q_mid = Q[:, :, 1:M-1]                # (B, N, M-2, H, D)
            K_self = K[:, :, 1:M-1]               # (B, N, M-2, H, D)
            V_self = V[:, :, 1:M-1]               # (B, N, M-2, H, D)

            # Repeat block 0 K/V for each middle block
            K0_rep = K[:, :, 0].unsqueeze(2).expand(B, N, M-2, H, D)  # (B, N, M-2, H, D)
            V0_rep = V[:, :, 0].unsqueeze(2).expand(B, N, M-2, H, D)  # (B, N, M-2, H, D)

            # Concatenate: [K0 | Kself]
            K_mid = torch.cat([K0_rep, K_self], dim=-2)       # (B, N, M-2, 2H, D)
            V_mid = torch.cat([V0_rep, V_self], dim=-2)       # (B, N, M-2, 2H, D)

            # Build concatenated mask:
            # - first H columns: broadcast "last valid" row from block 0
            # - next H columns: per-block causal self mask
            mask_first_cols = attention_mask[:, :, 0, -1, :]                     # (B, 1, H)
            mask_first = mask_first_cols.unsqueeze(2).unsqueeze(2)               # (B, 1, 1, 1, H)
            mask_first = mask_first.expand(B, 1, M-2, H, H)                      # (B, 1, M-2, H, H)
            mask_self = attention_mask[:, :, 1:M-1]                              # (B, 1, M-2, H, H)

            # Combine: take minimum (more restrictive) of block 0 mask and self mask for first H cols
            mask_first = torch.minimum(mask_first, mask_self[:, :, :, -1, :, None])  # (B, 1, M-2, H, H)
            m_mid = torch.cat([mask_first, mask_self], dim=-1)                   # (B, 1, M-2, H, 2H)

            # Compute attention using SDPA
            out[:, :, 1:M-1] = F.scaled_dot_product_attention(
                Q_mid, K_mid, V_mid,
                attn_mask=m_mid,
                dropout_p=dropout if module.training else 0.0,
                scale=scaling,
            )
    else:
        doc_start = 1 if block_order == "instruction_first" else 0
        doc_end = M - 1
        if doc_end > doc_start:
            min_dtype = 0.7 * torch.finfo(dtype).min
            for i in range(doc_start, doc_end):
                if block_order == "instruction_first":
                    key_blocks = [0] + list(range(1, i + 1)) if doc_cross_attn else [0, i]
                else:
                    key_blocks = list(range(0, i + 1)) if doc_cross_attn else [i]
                K_cat = torch.cat([K[:, :, b] for b in key_blocks], dim=-2)
                V_cat = torch.cat([V[:, :, b] for b in key_blocks], dim=-2)
                m_blk = _build_cross_block_mask(attention_mask, i, key_blocks, min_dtype)
                out[:, :, i] = F.scaled_dot_product_attention(
                    Q[:, :, i], K_cat, V_cat,
                    attn_mask=m_blk,
                    dropout_p=dropout if module.training else 0.0,
                    scale=scaling,
                )
    
    # ------------------------------------------------------------
    # Last block (M-1): full to all blocks, causal to self
    # Concatenate K/V across all blocks
    # ------------------------------------------------------------
    Q_last = Q[:, :, M-1]                                        # (B, N, H, D)
    K_all = K.reshape(B, N, M * H, D)                            # (B, N, M*H, D)
    V_all = V.reshape(B, N, M * H, D)                            # (B, N, M*H, D)

    # Mask for other blocks (0..M-2): take last row and broadcast over query rows.
    # Optional token compression is applied ONLY when the last query block attends to doc blocks.
    token_compression_mode = kwargs.get("token_compression_mode", "none")
    token_compression_last_k = kwargs.get("token_compression_last_k", 1)
    if token_compression_last_k is None:
        token_compression_last_k = 1
    token_compression_last_k = int(token_compression_last_k)
    token_compression_segment_k = kwargs.get("token_compression_segment_k", 10)
    if token_compression_segment_k is None:
        token_compression_segment_k = 10
    token_compression_segment_k = int(token_compression_segment_k)
    token_compression_segment_anchor = kwargs.get("token_compression_segment_anchor", "end")
    attention_weighted_top_k = kwargs.get("attention_weighted_top_k", 1)
    mask_others = attention_mask[:, :, :M-1, -1, :]              # (B, 1, M-1, H)
    mask_others = _compress_query_doc_visibility_mask(
        mask_others_blocks=mask_others,
        block_order=block_order,
        token_compression_mode=token_compression_mode,
        token_compression_last_k=token_compression_last_k,
        token_compression_segment_k=token_compression_segment_k,
        token_compression_segment_anchor=token_compression_segment_anchor,
        attention_weighted_top_k=attention_weighted_top_k,
        query_blocks=Q,
        key_blocks=K,
        full_attention_mask=attention_mask,
    )
    mask_others = mask_others.reshape(B, 1, (M - 1) * H)         # (B, 1, (M-1)*H)
    mask_others = mask_others.unsqueeze(-2).expand(B, 1, H, (M - 1) * H)  # (B, 1, H, (M-1)*H)
    mask_self_last = attention_mask[:, :, M-1]                   # (B, 1, H, H)

    # Combine: minimum of other blocks mask and self mask
    mask_others = torch.minimum(mask_others, mask_self_last[:, :, -1, :, None])
    m_last = torch.cat([mask_others, mask_self_last], dim=-1)    # (B, 1, H, M*H)

    out[:, :, M-1] = F.scaled_dot_product_attention(
        Q_last, K_all, V_all,
        attn_mask=m_last,
        dropout_p=dropout if module.training else 0.0,
        scale=scaling,
    )
    
    return_last_block_attn_scores = bool(kwargs.get('return_last_block_attn_scores', False))
    num_last_queries = kwargs.get('num_last_queries', 16)

    s_last = None
    if return_last_block_attn_scores:
        # Compute attention weights for last num_last_queries tokens only (for compatibility)
        Q_last = Q_last[:, :, -num_last_queries:] if Q_last.size(-2) >= num_last_queries else Q_last  # (B, N, num_last_queries, D) or less
        s_last = torch.matmul(Q_last, K_all.transpose(-2, -1))  # (B, N, num_last_queries, M*H)
        m_last = m_last[:, :, -num_last_queries:] if m_last.size(-2) >= num_last_queries else m_last  # (B, 1, num_last_queries, M*H)
        s_last = s_last + m_last

    # Reshape output to expected format
    out = out.view(B, N, MH, D).transpose(1, 2).contiguous()  # (B, M*H, N, D)

    return out, s_last  # Return last block's attention weights


def sdpa_blockrank_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    kwargs = _normalize_return_last_block_attn_scores(module, kwargs)
    return _sdpa_blockrank_attention_forward_impl(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=dropout,
        **kwargs,
    )


_default_compiled_blockrank_attention_forward = torch.compile(
    _eager_blockrank_attention_forward_impl,
    mode="default",
)
_max_autotune_compiled_blockrank_attention_forward = torch.compile(
    _eager_blockrank_attention_forward_impl,
    mode="max-autotune",
)
_sdpa_compiled_blockrank_attention_forward = torch.compile(
    _sdpa_blockrank_attention_forward_impl,
)


def default_blockrank_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    kwargs = _normalize_return_last_block_attn_scores(module, kwargs)
    return _default_compiled_blockrank_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=dropout,
        **kwargs,
    )


def max_autotune_blockrank_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    kwargs = _normalize_return_last_block_attn_scores(module, kwargs)
    return _max_autotune_compiled_blockrank_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=dropout,
        **kwargs,
    )


def sdpa_compiled_blockrank_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    kwargs = _normalize_return_last_block_attn_scores(module, kwargs)
    return _sdpa_compiled_blockrank_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=dropout,
        **kwargs,
    )

def register_blockrank_attention():
    # Register the BlockRank attention implementation with Transformers
    AttentionInterface.register("eager_blockrank", eager_blockrank_attention_forward)
    AttentionInterface.register("default_blockrank", default_blockrank_attention_forward)
    AttentionInterface.register("max-autotune_blockrank", max_autotune_blockrank_attention_forward)
    for mode in ["default", "max-autotune", "eager"]:
        AttentionMaskInterface.register(f"{mode}_blockrank", eager_blockrank_attention_mask)
    
    AttentionInterface.register(f"flex_blockrank", torch.compile(flex_blockrank_attention_forward))
    AttentionMaskInterface.register(f"flex_blockrank", flex_blockrank_attention_mask)

    AttentionInterface.register(f"sdpa_blockrank", sdpa_blockrank_attention_forward)
    AttentionMaskInterface.register(f"sdpa_blockrank", eager_blockrank_attention_mask)

    AttentionInterface.register(f"sdpa_compiled_blockrank", sdpa_compiled_blockrank_attention_forward)
    AttentionMaskInterface.register(f"sdpa_compiled_blockrank", eager_blockrank_attention_mask)
