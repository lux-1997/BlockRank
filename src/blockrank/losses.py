"""
Auxiliary attention loss for BlockRank training.

This module implements the contrastive loss that optimizes query-document
attention patterns during fine-tuning.
"""

import os
import torch
import torch.nn.functional as F

_AUX_DEBUG_PRINTED = False
_AUX_MODE_WARNED = False

def _should_print_aux_debug() -> bool:
    if os.environ.get("BLOCKRANK_AUX_DEBUG", "0") != "1":
        return False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return False
    global _AUX_DEBUG_PRINTED
    if _AUX_DEBUG_PRINTED:
        return False
    _AUX_DEBUG_PRINTED = True
    return True


def _last_k_valid_token_mask(mask: torch.Tensor, last_k: int = 1) -> torch.Tensor:
    """Keep only the last-k valid tokens on the last dimension."""
    if last_k <= 0:
        raise ValueError(f"last_k must be > 0, got {last_k}")
    valid = mask.bool()
    H = valid.shape[-1]
    token_idx = torch.arange(H, device=valid.device).view(*([1] * (valid.dim() - 1)), H)
    last_idx = torch.where(valid, token_idx, torch.full_like(token_idx, -1)).amax(dim=-1)
    first_idx = (last_idx - int(last_k) + 1).clamp(min=0)
    return (
        valid
        & (token_idx <= last_idx.unsqueeze(-1))
        & (token_idx >= first_idx.unsqueeze(-1))
    )


def _last_valid_token_mask(mask: torch.Tensor) -> torch.Tensor:
    """Backward-compatible alias for last-1 token masking."""
    return _last_k_valid_token_mask(mask, last_k=1)


def _mid_last_valid_token_mask(mask: torch.Tensor) -> torch.Tensor:
    """Keep middle (ceil(L/2)) and last valid tokens on the last dimension."""
    valid = mask.bool()
    rank = valid.cumsum(dim=-1)
    valid_count = valid.sum(dim=-1, keepdim=True)
    mid_rank = (valid_count + 1) // 2
    last_rank = valid_count
    return valid & ((rank == mid_rank) | (rank == last_rank))


def _normalize_optional_top_k(value: int | str | None, field_name: str = "attention_weighted_top_k") -> int | None:
    """Normalize optional top-k config; accepts None/null/none."""
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in ("", "none", "null"):
            return None
    top_k = int(value)
    if top_k <= 0:
        raise ValueError(f"{field_name} must be > 0 when provided, got {value}")
    return top_k


def compress_attention_mask_to_doc_anchor_tokens(
    attention_mask: torch.Tensor,
    block_order: str = "instruction_first",
    token_compression_mode: str = "last",
    last_k: int = 1,
) -> torch.Tensor:
    """
    Keep only selected anchor tokens visible in each document block.
    Non-document blocks are unchanged.
    """
    if attention_mask.dim() != 3:
        return attention_mask

    mode = token_compression_mode.lower()
    if mode not in ("none", "last", "mid_last"):
        raise ValueError(f"Unsupported token_compression_mode for forward compression: {token_compression_mode}")
    if mode == "none":
        return attention_mask

    _, M, _ = attention_mask.shape
    if block_order == "instruction_first":
        doc_slice = slice(1, M - 1)
    elif block_order == "doc_first":
        doc_slice = slice(0, M - 1)
    else:
        raise ValueError(f"Unknown block_order: {block_order}")

    compressed = attention_mask.clone()
    doc_mask = attention_mask[:, doc_slice, :]
    if mode == "last":
        doc_visible_mask = _last_k_valid_token_mask(doc_mask, last_k=last_k)
    else:
        doc_visible_mask = _mid_last_valid_token_mask(doc_mask)
    compressed[:, doc_slice, :] = doc_visible_mask.to(attention_mask.dtype)
    return compressed


def compress_attention_mask_to_last_doc_tokens(
    attention_mask: torch.Tensor,
    block_order: str = "instruction_first",
    last_k: int = 1,
) -> torch.Tensor:
    """Backward-compatible alias for last-k forward compression."""
    return compress_attention_mask_to_doc_anchor_tokens(
        attention_mask=attention_mask,
        block_order=block_order,
        token_compression_mode="last",
        last_k=last_k,
    )


def _select_query_indices_from_labels(
    labels: torch.Tensor,
    h1: int,
    query_token_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select the first supervised token index within the last h1 positions.

    Returns:
        query_indices: (B, 1) indices in [0, h1-1]
        has_supervised: (B,) bool mask indicating at least one supervised token in the window
    """
    last_h1_labels = labels[:, -h1:]  # (B, h1)
    supervised_mask = last_h1_labels > -100
    has_supervised = supervised_mask.any(dim=-1)

    query_indices = supervised_mask.int().argmax(dim=-1)
    query_indices = query_indices + int(query_token_offset)
    query_indices = query_indices.clamp(min=0, max=h1 - 1)
    return query_indices[:, None], has_supervised


def _build_query_selection_mask(
    labels: torch.Tensor,
    h1: int,
    query_token_offset: int = 0,
    query_aggregation_mode: str = "single",
    query_token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build query-token selection mask on the last h1 query positions.

    Returns:
        query_selection_mask: (B, h1) bool
        has_query: (B,) bool, whether the requested query set is non-empty before fallback
        fallback_indices: (B, 1) first supervised fallback index
    """
    fallback_indices, has_supervised = _select_query_indices_from_labels(
        labels=labels,
        h1=h1,
        query_token_offset=query_token_offset,
    )
    B = labels.size(0)
    fallback_mask = torch.zeros(B, h1, dtype=torch.bool, device=labels.device)
    fallback_mask.scatter_(1, fallback_indices, True)

    mode = query_aggregation_mode.lower()
    if mode == "single":
        return fallback_mask, has_supervised, fallback_indices

    if mode not in ("mean_all", "logsumexp_all"):
        raise ValueError(
            f"Unknown query_aggregation_mode: {query_aggregation_mode}. "
            "Expected one of ['single', 'mean_all', 'logsumexp_all']."
        )

    if query_token_mask is None:
        raise ValueError(
            f"query_token_mask is required when query_aggregation_mode='{mode}' "
            "to aggregate over exact query tokens."
        )
    if query_token_mask.dim() != 2:
        raise ValueError(f"query_token_mask must be 2D, got shape={tuple(query_token_mask.shape)}")
    if query_token_mask.size(0) != B:
        raise ValueError(
            f"query_token_mask batch size mismatch: {query_token_mask.size(0)} vs labels batch {B}"
        )
    if query_token_mask.size(1) < h1:
        raise ValueError(
            f"query_token_mask length {query_token_mask.size(1)} is shorter than h1={h1}."
        )

    query_selection_mask = query_token_mask[:, -h1:].bool().to(labels.device)
    has_query = query_selection_mask.any(dim=-1)
    if not bool(has_query.all()):
        query_selection_mask = torch.where(has_query[:, None], query_selection_mask, fallback_mask)
    return query_selection_mask, has_query, fallback_indices


def _aggregate_doc_scores_over_queries(
    doc_scores_by_query: torch.Tensor,
    query_selection_mask: torch.Tensor,
    query_aggregation_mode: str = "single",
) -> torch.Tensor:
    """Aggregate (B, h1, num_docs) doc scores over selected query tokens."""
    mode = query_aggregation_mode.lower()
    if mode in ("single", "mean_all"):
        mask_f = query_selection_mask.to(doc_scores_by_query.dtype)
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (doc_scores_by_query * mask_f.unsqueeze(-1)).sum(dim=1) / denom
    if mode == "logsumexp_all":
        neg_inf = torch.finfo(doc_scores_by_query.dtype).min
        masked = doc_scores_by_query.masked_fill(~query_selection_mask.unsqueeze(-1), neg_inf)
        return torch.logsumexp(masked, dim=1)
    raise ValueError(
        f"Unknown query_aggregation_mode: {query_aggregation_mode}. "
        "Expected one of ['single', 'mean_all', 'logsumexp_all']."
    )


def compute_auxiliary_attention_loss(
    attention_scores: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_ids: torch.Tensor | None = None,
    temperature: float = 0.03,
    return_logits = False,
    query_token_offset: int = 0,
    block_order: str = "instruction_first",
    aux_norm_mode: str = "doc_plus_non_doc",
    token_compression_mode: str = "none",
    token_compression_last_k: int = 1,
    attention_weighted_top_k: int | None = 1,
    query_aggregation_mode: str = "single",
    query_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute InfoNCE contrastive loss on attention scores at first loss position (should correspond to "[" token).

    This loss encourages the model to attend more strongly to relevant documents
    from the "[" token position (first non-masked token in the completion).

    Args:
        attention_scores: (B, N, H1, M*H) - attention logits from specified layer
            Returns last h1 query positions from the last block
        labels: (B, M*H) - label tensor with -100 for masked positions
        attention_mask: (B, M, H) - block-wise attention mask
        answer_ids: (B, max_num_answers) - positive document indices, padded with -1, or None return only logits
        temperature: float - temperature for InfoNCE loss (default: 0.1)
        return_logits: bool - whether to return logits along with loss
        query_token_offset: int - offset to add to bracket token index (if bracket is not at first position in last h1)
        aux_norm_mode: str - normalization mode: "doc_plus_non_doc" (default) or "doc_only"

    Returns:
        loss: scalar tensor - InfoNCE contrastive loss
    """
    B, M, H = attention_mask.shape
    _, N, h1, MH = attention_scores.shape
    assert MH == M * H, "Attention scores last dimension must match M*H"

    # Step 1: Build query-token selection mask on the last h1 positions.
    query_selection_mask, has_query, bracket_indices = _build_query_selection_mask(
        labels=labels,
        h1=h1,
        query_token_offset=query_token_offset,
        query_aggregation_mode=query_aggregation_mode,
        query_token_mask=query_token_mask,
    )
    if not bool(has_query.all()):
        if query_aggregation_mode.lower() == "single":
            print("WARNING: No supervised token found in last h1 positions for some samples.")
        else:
            print("WARNING: No query token found in last h1 positions for some samples; fallback to single-anchor token.")

    # Step 2: Keep all returned last-h1 query positions, then aggregate by query_aggregation_mode.
    bracket_attn_logits = attention_scores  # (B, N, h1, M*H)

    if block_order == "instruction_first":
        doc_start = H
        doc_end = M * H - H
        doc_block_start = 1
        doc_block_end = M - 1
        if aux_norm_mode == "doc_plus_non_doc":
            non_doc_lse = torch.logaddexp(
                torch.logsumexp(bracket_attn_logits[..., :H], dim=-1, keepdim=True),
                torch.logsumexp(bracket_attn_logits[..., -H:], dim=-1, keepdim=True),
            )
    elif block_order == "doc_first":
        doc_start = 0
        doc_end = M * H - H
        doc_block_start = 0
        doc_block_end = M - 1
        if aux_norm_mode == "doc_plus_non_doc":
            non_doc_lse = torch.logsumexp(bracket_attn_logits[..., -H:], dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown block_order: {block_order}")

    if aux_norm_mode not in ("doc_plus_non_doc", "doc_only"):
        raise ValueError(f"Unknown aux_norm_mode: {aux_norm_mode}")
    mode = token_compression_mode.lower()
    if mode not in ("none", "last", "mid_last", "topk"):
        raise ValueError(f"Unknown token_compression_mode: {token_compression_mode}")
    if int(token_compression_last_k) <= 0:
        raise ValueError(f"token_compression_last_k must be > 0, got {token_compression_last_k}")
    aw_top_k = _normalize_optional_top_k(attention_weighted_top_k, field_name="attention_weighted_top_k")

    effective_aux_norm_mode = aux_norm_mode
    if (aw_top_k is not None or mode in ("last", "mid_last")) and aux_norm_mode == "doc_plus_non_doc":
        global _AUX_MODE_WARNED
        if not _AUX_MODE_WARNED:
            _AUX_MODE_WARNED = True
            print(
                "[BlockRankAuxLoss] last-k / attention-weighted compression with aux_norm_mode='doc_plus_non_doc' "
                "is auto-switched to 'doc_only' to avoid non-doc normalization collapse."
            )
        effective_aux_norm_mode = "doc_only"

    # Step 2: Apply softmax over selected document tokens
    num_docs = M - 2 if block_order == "instruction_first" else M - 1
    doc_token_mask = attention_mask[:, doc_block_start:doc_block_end, :].bool()  # (B, num_docs, H)
    doc_logits = bracket_attn_logits[..., doc_start:doc_end]

    last_k_mask = _last_k_valid_token_mask(doc_token_mask, last_k=int(token_compression_last_k))
    if aw_top_k is None:
        # None => use last-k only
        doc_select_mask = last_k_mask
    else:
        # k => use union(last-k, attention-weighted top-k visible from forward mask)
        visible_threshold = torch.finfo(doc_logits.dtype).min / 2
        visible_mask = (doc_logits[:, 0, 0] > visible_threshold).reshape(B, num_docs, H)
        doc_select_mask = last_k_mask | (visible_mask & doc_token_mask)

    doc_logits = doc_logits.masked_fill(
        ~doc_select_mask.reshape(B, 1, 1, num_docs * H),
        float("-inf"),
    )

    doc_attn_lse = torch.logsumexp(doc_logits, dim=-1, keepdim=True) # L, B, N, H1, 1, 1
    if effective_aux_norm_mode == "doc_plus_non_doc":
        with torch.no_grad():
            non_doc_attn_lse = non_doc_lse
        attn_lse = torch.logaddexp(non_doc_attn_lse.detach(), doc_attn_lse) # L, B, N, H1, 1, 1
    else:
        attn_lse = doc_attn_lse
    bracket_attn = (doc_logits - attn_lse).exp()

    # Step 3: Reshape to separate documents
    # Documents are in blocks 1 to M-2 (instruction_first) or 0 to M-2 (doc_first)
    bracket_attn = bracket_attn.reshape(B, N, -1, num_docs, H)  # (B, N, 2, num_docs, H)

    # Step 4: Aggregate to document-level scores
    # Sum over doc tokens, average over heads, then aggregate over selected query tokens.
    doc_scores_by_query = bracket_attn.sum(dim=-1).mean(dim=1)  # (B, h1, num_docs)
    doc_scores = _aggregate_doc_scores_over_queries(
        doc_scores_by_query=doc_scores_by_query,
        query_selection_mask=query_selection_mask,
        query_aggregation_mode=query_aggregation_mode,
    )

    # Step 5: Compute InfoNCE loss with multiple positives
    # answer_ids shape: (B, max_num_answers), values are doc indices or -1 (padding)

    # Apply temperature scaling
    logits = doc_scores / temperature  # (B, num_docs)

    if answer_ids is None:
        assert return_logits, "If answer_ids is None, return_logits must be True."
        return logits

    # Create mask for valid positives (B, num_docs+1) to safely handle -1 padding
    pos_mask = torch.zeros(B, num_docs + 1, dtype=torch.bool, device=logits.device)
    safe_answer_ids = torch.where(answer_ids >= 0, answer_ids, num_docs)  # Map negative -> num_docs
    pos_mask.scatter_(1, safe_answer_ids, True)
    pos_mask = pos_mask[:, :-1]  # Remove last column, back to (B, num_docs)

    # Ignore rows with no valid positives to avoid inf loss from empty positive sets.
    valid_rows = pos_mask.any(dim=1)
    if not bool(valid_rows.any()):
        zero = logits.sum() * 0.0
        return (zero, zero, logits) if return_logits else (zero, zero)

    logits_valid = logits[valid_rows]
    pos_mask_valid = pos_mask[valid_rows]

    # InfoNCE: -log(sum(exp(pos)) / sum(exp(all)))
    pos_logsumexp = torch.logsumexp(logits_valid.masked_fill(~pos_mask_valid, float("-inf")), dim=1)
    all_logsumexp = torch.logsumexp(logits_valid, dim=1)

    loss = -(pos_logsumexp - all_logsumexp).mean()
    accuracy = pos_mask_valid.gather(1, logits_valid.argmax(dim=1, keepdim=True)).float().mean()
    
    return (loss, accuracy, logits) if return_logits else (loss, accuracy)


def compute_auxiliary_attention_loss_copynet(
    attention_scores: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_ids: torch.Tensor | None = None,
    temperature: float = 0.03,
    return_logits = False,
    query_token_offset: int = 0,
    block_order: str = "instruction_first",
    aux_norm_mode: str = "doc_plus_non_doc",
    token_compression_mode: str = "none",
    token_compression_last_k: int = 1,
    attention_weighted_top_k: int | None = 1,
    query_aggregation_mode: str = "single",
    query_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute pointer-style (CopyNet) cross-entropy loss on attention-derived doc scores.

    This mirrors compute_auxiliary_attention_loss up to doc-level logits, but replaces
    InfoNCE with cross-entropy over candidate documents.
    """
    B, M, H = attention_mask.shape
    _, N, h1, MH = attention_scores.shape
    assert MH == M * H, "Attention scores last dimension must match M*H"

    debug = _should_print_aux_debug()
    if debug:
        finite_ratio = torch.isfinite(attention_scores).float().mean().item()
        print(
            "[BlockRankAuxDebug] attention_scores.shape="
            f"{tuple(attention_scores.shape)} labels.shape={tuple(labels.shape)} "
            f"attention_mask.shape={tuple(attention_mask.shape)} block_order={block_order} "
            f"temperature={temperature} query_token_offset={query_token_offset} "
            f"finite_ratio={finite_ratio:.4f} min={attention_scores.min().item():.4f} "
            f"max={attention_scores.max().item():.4f}"
        )

    # Step 1: Build query-token selection mask on the last h1 positions.
    query_selection_mask, has_query, bracket_indices = _build_query_selection_mask(
        labels=labels,
        h1=h1,
        query_token_offset=query_token_offset,
        query_aggregation_mode=query_aggregation_mode,
        query_token_mask=query_token_mask,
    )
    if not bool(has_query.all()):
        if query_aggregation_mode.lower() == "single":
            print("WARNING: No supervised token found in last h1 positions for some samples.")
        else:
            print("WARNING: No query token found in last h1 positions for some samples; fallback to single-anchor token.")

    # Step 2: Keep all returned last-h1 query positions, then aggregate by query_aggregation_mode.
    bracket_attn_logits = attention_scores  # (B, N, h1, M*H)

    if block_order == "instruction_first":
        doc_start = H
        doc_end = M * H - H
        doc_block_start = 1
        doc_block_end = M - 1
        if aux_norm_mode == "doc_plus_non_doc":
            non_doc_lse = torch.logaddexp(
                torch.logsumexp(bracket_attn_logits[..., :H], dim=-1, keepdim=True),
                torch.logsumexp(bracket_attn_logits[..., -H:], dim=-1, keepdim=True),
            )
    elif block_order == "doc_first":
        doc_start = 0
        doc_end = M * H - H
        doc_block_start = 0
        doc_block_end = M - 1
        if aux_norm_mode == "doc_plus_non_doc":
            non_doc_lse = torch.logsumexp(bracket_attn_logits[..., -H:], dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown block_order: {block_order}")

    if aux_norm_mode not in ("doc_plus_non_doc", "doc_only"):
        raise ValueError(f"Unknown aux_norm_mode: {aux_norm_mode}")
    mode = token_compression_mode.lower()
    if mode not in ("none", "last", "mid_last", "topk"):
        raise ValueError(f"Unknown token_compression_mode: {token_compression_mode}")
    if int(token_compression_last_k) <= 0:
        raise ValueError(f"token_compression_last_k must be > 0, got {token_compression_last_k}")
    aw_top_k = _normalize_optional_top_k(attention_weighted_top_k, field_name="attention_weighted_top_k")

    effective_aux_norm_mode = aux_norm_mode
    if (aw_top_k is not None or mode in ("last", "mid_last")) and aux_norm_mode == "doc_plus_non_doc":
        global _AUX_MODE_WARNED
        if not _AUX_MODE_WARNED:
            _AUX_MODE_WARNED = True
            print(
                "[BlockRankAuxLoss] last-k / attention-weighted compression with aux_norm_mode='doc_plus_non_doc' "
                "is auto-switched to 'doc_only' to avoid non-doc normalization collapse."
            )
        effective_aux_norm_mode = "doc_only"

    if debug:
        bracket_valid_ratio = has_query.float().mean().item()
        print(
            "[BlockRankAuxDebug] bracket_valid_ratio="
            f"{bracket_valid_ratio:.4f} bracket_indices={bracket_indices.squeeze(-1).tolist()} "
            f"h1={h1} doc_start={doc_start} doc_end={doc_end} num_docs="
            f"{(M - 2) if block_order == 'instruction_first' else (M - 1)}"
        )

    # Step 2: Apply softmax over selected document tokens
    num_docs = M - 2 if block_order == "instruction_first" else M - 1
    doc_token_mask = attention_mask[:, doc_block_start:doc_block_end, :].bool()
    doc_logits = bracket_attn_logits[..., doc_start:doc_end]

    last_k_mask = _last_k_valid_token_mask(doc_token_mask, last_k=int(token_compression_last_k))
    if aw_top_k is None:
        # None => use last-k only
        doc_select_mask = last_k_mask
    else:
        # k => use union(last-k, attention-weighted top-k visible from forward mask)
        visible_threshold = torch.finfo(doc_logits.dtype).min / 2
        visible_mask = (doc_logits[:, 0, 0] > visible_threshold).reshape(B, num_docs, H)
        doc_select_mask = last_k_mask | (visible_mask & doc_token_mask)

    doc_logits = doc_logits.masked_fill(
        ~doc_select_mask.reshape(B, 1, 1, num_docs * H),
        float("-inf"),
    )

    doc_attn_lse = torch.logsumexp(doc_logits, dim=-1, keepdim=True)
    if effective_aux_norm_mode == "doc_plus_non_doc":
        with torch.no_grad():
            non_doc_attn_lse = non_doc_lse
        attn_lse = torch.logaddexp(non_doc_attn_lse.detach(), doc_attn_lse)
    else:
        attn_lse = doc_attn_lse
    bracket_attn = (doc_logits - attn_lse).exp()

    # Step 3: Reshape to separate documents
    bracket_attn = bracket_attn.reshape(B, N, -1, num_docs, H)  # (B, N, 2, num_docs, H)

    # Step 4: Aggregate to document-level scores
    doc_scores_by_query = bracket_attn.sum(dim=-1).mean(dim=1)  # (B, h1, num_docs)
    doc_scores = _aggregate_doc_scores_over_queries(
        doc_scores_by_query=doc_scores_by_query,
        query_selection_mask=query_selection_mask,
        query_aggregation_mode=query_aggregation_mode,
    )

    logits = doc_scores / temperature  # (B, num_docs)

    if answer_ids is None:
        assert return_logits, "If answer_ids is None, return_logits must be True."
        return logits

    # Use the first valid positive as the target for pointer-style CE.
    valid_mask = answer_ids >= 0
    valid_rows = valid_mask.any(dim=1)
    if not bool(valid_rows.any()):
        zero = logits.sum() * 0.0
        return (zero, zero, logits) if return_logits else (zero, zero)

    first_pos = valid_mask.int().argmax(dim=1, keepdim=True)
    target_doc_idx = answer_ids.gather(1, first_pos).squeeze(1)
    target_doc_idx = torch.where(target_doc_idx >= 0, target_doc_idx, torch.zeros_like(target_doc_idx))

    if debug:
        k = min(5, doc_scores.size(1))
        topk = torch.topk(doc_scores, k=k, dim=1)
        topk_idx = topk.indices[0].detach().cpu().tolist() if doc_scores.size(0) > 0 else []
        topk_val = topk.values[0].detach().cpu().tolist() if doc_scores.size(0) > 0 else []
        ans0 = answer_ids[0].detach().cpu().tolist() if answer_ids.size(0) > 0 else []
        tgt0 = int(target_doc_idx[0].item()) if target_doc_idx.numel() > 0 else -1
        print(
            "[BlockRankAuxDebug] doc_scores_topk_idx="
            f"{topk_idx} doc_scores_topk_val={topk_val} "
            f"answer_ids[0]={ans0} target_doc_idx[0]={tgt0}"
        )

    loss = F.cross_entropy(logits[valid_rows], target_doc_idx[valid_rows], reduction="mean")
    accuracy = (logits[valid_rows].argmax(dim=1) == target_doc_idx[valid_rows]).float().mean()

    return (loss, accuracy, logits) if return_logits else (loss, accuracy)


def compute_doc_last_token_alignment_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    full_attention_mask: torch.Tensor,
    student_attention_mask: torch.Tensor,
    block_order: str = "instruction_first",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Align compressed-token doc representations to full-token doc representations.

    student_hidden: hidden states from compressed-forward (e.g., last-k visible doc tokens)
    teacher_hidden: hidden states from full-forward (all doc tokens visible)
    full_attention_mask: (B, M, H) mask used by teacher forward
    student_attention_mask: (B, M, H) mask used by student forward
    """
    if student_hidden.dim() != 3 or teacher_hidden.dim() != 3:
        raise ValueError("student_hidden and teacher_hidden must be 3D tensors: (B, S, D)")

    B, M, H = full_attention_mask.shape
    S_expected = M * H
    if student_hidden.shape[0] != B or teacher_hidden.shape[0] != B:
        raise ValueError("Batch size mismatch between hidden states and attention masks")
    if student_hidden.shape[1] < S_expected or teacher_hidden.shape[1] < S_expected:
        raise ValueError(
            f"Hidden sequence length is shorter than M*H ({S_expected}). "
            f"student={student_hidden.shape[1]}, teacher={teacher_hidden.shape[1]}"
        )

    # Keep exactly M*H tokens to match block masks.
    student_hidden = student_hidden[:, :S_expected, :]
    teacher_hidden = teacher_hidden[:, :S_expected, :]

    D = student_hidden.shape[-1]
    student_blocks = student_hidden.view(B, M, H, D)
    teacher_blocks = teacher_hidden.view(B, M, H, D)

    if block_order == "instruction_first":
        doc_slice = slice(1, M - 1)
    elif block_order == "doc_first":
        doc_slice = slice(0, M - 1)
    else:
        raise ValueError(f"Unknown block_order: {block_order}")

    teacher_doc = teacher_blocks[:, doc_slice, :, :]
    student_doc = student_blocks[:, doc_slice, :, :]
    teacher_mask = full_attention_mask[:, doc_slice, :].bool()
    student_mask = student_attention_mask[:, doc_slice, :].bool()

    teacher_denom = teacher_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(teacher_doc.dtype)
    student_denom = student_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(student_doc.dtype)
    teacher_repr = (teacher_doc * teacher_mask.unsqueeze(-1).to(teacher_doc.dtype)).sum(dim=-2) / teacher_denom
    student_repr = (student_doc * student_mask.unsqueeze(-1).to(student_doc.dtype)).sum(dim=-2) / student_denom

    # Valid docs are those that exist in both views.
    valid_docs = teacher_mask.any(dim=-1) & student_mask.any(dim=-1)
    if not bool(valid_docs.any()):
        return student_hidden.sum() * 0.0

    cos = F.cosine_similarity(student_repr, teacher_repr.detach(), dim=-1, eps=eps)
    align = 1.0 - cos
    return align.masked_select(valid_docs).mean()
