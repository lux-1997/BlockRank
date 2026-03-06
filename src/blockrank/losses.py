"""
Auxiliary attention loss for BlockRank training.

This module implements the contrastive loss that optimizes query-document
attention patterns during fine-tuning.
"""

import os
import torch
import torch.nn.functional as F

_AUX_DEBUG_PRINTED = False

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

    # Step 1: Find bracket token position (first non-masked token in last h1 positions)
    # Look at last h1 positions in labels to find the "[" bracket token
    last_h1_labels = labels[:, -h1:]  # (B, h1)
    # Find first position where labels > -100 (first non-masked token = bracket)
    bracket_mask = last_h1_labels > -100  # (B, h1)
    bracket_indices = bracket_mask.int().argmax(dim=-1)[:, None] + query_token_offset  # (B,) - index in [0, h1)
    # bracket_indices = torch.hstack([bracket_indices-1, bracket_indices])  # (B, 2) - take bracket and previous token
    if not torch.all(bracket_indices >= 0):
        print("WARNING: Bracket token not found in last h1 positions.")

    # Step 2: Extract attention scores at bracket position for document blocks only
    # attention_scores shape: (B, N, h1, M*H)
    # We need to index each batch item with its specific bracket position
    bracket_attn_logits = attention_scores.take_along_dim(bracket_indices[:, None, :, None], dim=2) # (B, N, 2, M*H)

    if block_order == "instruction_first":
        doc_start = H
        doc_end = M * H - H
        if aux_norm_mode == "doc_plus_non_doc":
            non_doc_lse = torch.logaddexp(
                torch.logsumexp(bracket_attn_logits[..., :H], dim=-1, keepdim=True),
                torch.logsumexp(bracket_attn_logits[..., -H:], dim=-1, keepdim=True),
            )
    elif block_order == "doc_first":
        doc_start = 0
        doc_end = M * H - H
        if aux_norm_mode == "doc_plus_non_doc":
            non_doc_lse = torch.logsumexp(bracket_attn_logits[..., -H:], dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown block_order: {block_order}")

    if aux_norm_mode not in ("doc_plus_non_doc", "doc_only"):
        raise ValueError(f"Unknown aux_norm_mode: {aux_norm_mode}")

    # Step 2: Apply softmax over all document tokens
    # bracket_attn = F.softmax(bracket_attn_logits, dim=-1)  # (B, N, 2, (M-2)*H)
    doc_attn_lse = torch.logsumexp(bracket_attn_logits[..., doc_start:doc_end], dim=-1, keepdim=True) # L, B, N, H1, 1, 1
    if aux_norm_mode == "doc_plus_non_doc":
        with torch.no_grad():
            non_doc_attn_lse = non_doc_lse
        attn_lse = torch.logaddexp(non_doc_attn_lse.detach(), doc_attn_lse) # L, B, N, H1, 1, 1
    else:
        attn_lse = doc_attn_lse
    bracket_attn = (bracket_attn_logits[..., doc_start:doc_end] - attn_lse).exp()

    # Step 3: Reshape to separate documents
    # Documents are in blocks 1 to M-2 (instruction_first) or 0 to M-2 (doc_first)
    num_docs = M - 2 if block_order == "instruction_first" else M - 1
    bracket_attn = bracket_attn.reshape(B, N, -1, num_docs, H)  # (B, N, 2, num_docs, H)

    # Step 4: Aggregate to document-level scores
    # Average over attention heads, signal query tokens, sum over tokens within each document
    doc_scores = bracket_attn.sum(dim=-1).mean(dim=(1,2))  # (B, num_docs)

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
    
    # InfoNCE: -log(sum(exp(pos)) / sum(exp(all)))
    pos_logsumexp = torch.logsumexp(logits.masked_fill(~pos_mask, float('-inf')), dim=1)
    all_logsumexp = torch.logsumexp(logits, dim=1)
    
    loss = -(pos_logsumexp - all_logsumexp).mean()
    accuracy = pos_mask.gather(1, logits.argmax(dim=1, keepdim=True)).float().mean()
    
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

    # Step 1: Find bracket token position (first non-masked token in last h1 positions)
    last_h1_labels = labels[:, -h1:]  # (B, h1)
    bracket_mask = last_h1_labels > -100  # (B, h1)
    bracket_indices = bracket_mask.int().argmax(dim=-1)[:, None] + query_token_offset  # (B,) - index in [0, h1)
    if not torch.all(bracket_indices >= 0):
        print("WARNING: Bracket token not found in last h1 positions.")

    # Step 2: Extract attention scores at bracket position for document blocks only
    bracket_attn_logits = attention_scores.take_along_dim(bracket_indices[:, None, :, None], dim=2) # (B, N, 2, M*H)

    if block_order == "instruction_first":
        doc_start = H
        doc_end = M * H - H
        if aux_norm_mode == "doc_plus_non_doc":
            non_doc_lse = torch.logaddexp(
                torch.logsumexp(bracket_attn_logits[..., :H], dim=-1, keepdim=True),
                torch.logsumexp(bracket_attn_logits[..., -H:], dim=-1, keepdim=True),
            )
    elif block_order == "doc_first":
        doc_start = 0
        doc_end = M * H - H
        if aux_norm_mode == "doc_plus_non_doc":
            non_doc_lse = torch.logsumexp(bracket_attn_logits[..., -H:], dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown block_order: {block_order}")

    if aux_norm_mode not in ("doc_plus_non_doc", "doc_only"):
        raise ValueError(f"Unknown aux_norm_mode: {aux_norm_mode}")

    if debug:
        bracket_valid_ratio = (bracket_mask.sum(dim=1) > 0).float().mean().item()
        print(
            "[BlockRankAuxDebug] bracket_valid_ratio="
            f"{bracket_valid_ratio:.4f} bracket_indices={bracket_indices.squeeze(-1).tolist()} "
            f"h1={h1} doc_start={doc_start} doc_end={doc_end} num_docs="
            f"{(M - 2) if block_order == 'instruction_first' else (M - 1)}"
        )

    # Step 2: Apply softmax over all document tokens
    doc_attn_lse = torch.logsumexp(bracket_attn_logits[..., doc_start:doc_end], dim=-1, keepdim=True)
    if aux_norm_mode == "doc_plus_non_doc":
        with torch.no_grad():
            non_doc_attn_lse = non_doc_lse
        attn_lse = torch.logaddexp(non_doc_attn_lse.detach(), doc_attn_lse)
    else:
        attn_lse = doc_attn_lse
    bracket_attn = (bracket_attn_logits[..., doc_start:doc_end] - attn_lse).exp()

    # Step 3: Reshape to separate documents
    num_docs = M - 2 if block_order == "instruction_first" else M - 1
    bracket_attn = bracket_attn.reshape(B, N, -1, num_docs, H)  # (B, N, 2, num_docs, H)

    # Step 4: Aggregate to document-level scores
    doc_scores = bracket_attn.sum(dim=-1).mean(dim=(1,2))  # (B, num_docs)

    logits = doc_scores / temperature  # (B, num_docs)

    if answer_ids is None:
        assert return_logits, "If answer_ids is None, return_logits must be True."
        return logits

    # Use the first valid positive as the target for pointer-style CE.
    valid_mask = answer_ids >= 0
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

    loss = F.cross_entropy(logits, target_doc_idx, reduction="mean")
    accuracy = (logits.argmax(dim=1) == target_doc_idx).float().mean()

    return (loss, accuracy, logits) if return_logits else (loss, accuracy)
