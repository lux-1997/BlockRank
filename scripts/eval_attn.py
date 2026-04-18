import os
import sys
import argparse
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from functools import partial

# Add scripts directory to path for train module imports
sys.path.insert(0, os.path.dirname(__file__))
from train import setup_model_and_tokenizer, load_config, ModelArgs, DataArgs, TrainArgs, logger

from transformers import HfArgumentParser, set_seed
from blockrank.dataset import load_icr_dataset_hf, icr_collate_fn, block_icr_collate_fn
from blockrank.utils import calculate_accuracy, load_qrels
from accelerate import Accelerator, DataLoaderConfiguration
import wandb

class IncrementalMetrics:
    """Incremental metric tracker with proper per-query aggregation."""
    def __init__(self, eval_data, qrels):
        self.answer_ids, self.query_ids, self.doc_ids = eval_data
        self.qrels = qrels
        self.last_idx = 0
        self.agg = {'correct': 0, 'total': 0, 'invalid': 0}
        self.run_dict = {}  # Store run dictionary incrementally, evaluate all at once

    def update(self, all_preds):
        """Incrementally update all metrics (accuracy + ranking)."""
        new_preds = all_preds[self.last_idx:]
        if not new_preds:
            return self._build_results()

        n = len(all_preds)
        ground_truth_raw = self.answer_ids[self.last_idx:n]

        # Normalize ground truth (match calculate_accuracy logic)
        ground_truth = [int(g) if isinstance(g, (int, str, float)) else [int(x) for x in g] for g in ground_truth_raw]

        # Normalize predictions
        normalized_preds = []
        for p in new_preds:
            if isinstance(p, list):
                normalized_preds.append([int(x) for x in p])
            else:
                normalized_preds.append([int(p)])

        # 1. Fast accuracy update (cheap)
        for pred, gt in zip(normalized_preds, ground_truth):
            try:
                top1_pred = [pred[0]] if pred else []
                gt_list = gt if isinstance(gt, list) else [gt]
                self.agg['correct'] += (set(top1_pred).issubset(set(gt_list)))
                self.agg['total'] += 1
            except:
                self.agg['invalid'] += 1
                self.agg['total'] += 1

        # 2. Incremental ranking metrics (only if qrels available)
        if self.qrels and self.query_ids:
            # Build run dict for NEW queries and add to accumulated run_dict
            for i, pred_ranking in enumerate(normalized_preds):
                global_idx = self.last_idx + i
                query_id = str(self.query_ids[global_idx])
                remapped_doc_ids = self.doc_ids[global_idx]

                if query_id not in self.qrels:
                    continue

                # Create ranking scores for this query
                self.run_dict[query_id] = {}
                for rank, doc_idx in enumerate(pred_ranking):
                    if doc_idx < len(remapped_doc_ids):
                        doc_id = str(remapped_doc_ids[doc_idx])
                        self.run_dict[query_id][doc_id] = float(len(pred_ranking) - rank)

                # Assign 0 to unranked docs
                for doc_idx, doc_id in enumerate(remapped_doc_ids):
                    if doc_idx not in pred_ranking:
                        self.run_dict[query_id][str(doc_id)] = 0.0

        self.last_idx = n
        return self._build_results()

    def _build_results(self):
        """Build current metrics from aggregated values."""
        results = {
            'accuracy': 100 * self.agg['correct'] / self.agg['total'] if self.agg['total'] > 0 else 0.0,
            'exact_match': self.agg['correct'],
            'total': self.agg['total'],
            'invalid_predictions': self.agg['invalid'],
            'invalid_rate': 100 * self.agg['invalid'] / self.agg['total'] if self.agg['total'] > 0 else 0.0,
        }

        # Evaluate all queries at once (avoid pytrec_eval evaluator reuse issues)
        if self.run_dict and self.qrels:
            import pytrec_eval
            measures = {'ndcg_cut_1', 'ndcg_cut_3', 'ndcg_cut_5', 'ndcg_cut_10', 'recip_rank'}
            evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, measures)
            query_results = evaluator.evaluate(self.run_dict)

            # Aggregate across all queries (match calculate_accuracy logic exactly)
            num_queries = len(query_results)
            for k in [1, 3, 5, 10]:
                ndcg_sum = sum(qm.get(f'ndcg_cut_{k}', 0.0) for qm in query_results.values())
                mrr_sum = sum(qm.get('recip_rank', 0.0) for qm in query_results.values())
                results[f'ndcg@{k}'] = 100 * ndcg_sum / num_queries
                results[f'mrr@{k}'] = 100 * mrr_sum / num_queries

        return results

def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Mean over valid positions in `mask`; returns 0 when no valid positions exist."""
    mask = mask.to(values.dtype)
    denom = mask.sum(dim=dim).clamp(min=1.0)
    return (values * mask).sum(dim=dim) / denom


def _last_k_valid_token_mask(mask: torch.Tensor, last_k: int = 1) -> torch.Tensor:
    """Return a mask that keeps only the last-k valid tokens on the last dimension."""
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


def _mid_last_valid_token_mask(mask: torch.Tensor) -> torch.Tensor:
    """Return a mask that keeps middle (ceil(L/2)) and last valid tokens."""
    valid = mask.bool()
    rank = valid.cumsum(dim=-1)
    valid_count = valid.sum(dim=-1, keepdim=True)
    mid_rank = (valid_count + 1) // 2
    last_rank = valid_count
    return valid & ((rank == mid_rank) | (rank == last_rank))


def _segment_valid_token_mask(mask: torch.Tensor, segment_k: int = 10, anchor: str = "end") -> torch.Tensor:
    """Return a mask that keeps one valid token every k positions, anchored at start/end."""
    if segment_k <= 0:
        raise ValueError(f"segment_k must be > 0, got {segment_k}")
    anchor = anchor.lower()
    if anchor not in ("start", "end"):
        raise ValueError(f"anchor must be one of ['start', 'end'], got {anchor}")
    valid = mask.bool()
    rank = valid.cumsum(dim=-1) - 1
    if anchor == "start":
        keep = (rank.remainder(int(segment_k)) == 0)
    else:
        valid_count = valid.sum(dim=-1, keepdim=True)
        dist_from_end = valid_count - 1 - rank
        keep = (dist_from_end.remainder(int(segment_k)) == 0)
    return valid & keep


def _normalize_optional_top_k(value: int | str | None, field_name: str = "attention_weighted_top_k") -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in ("", "none", "null"):
            return None
    top_k = int(value)
    if top_k <= 0:
        raise ValueError(f"{field_name} must be > 0 when provided")
    return top_k


def _build_query_selection_mask(
    labels: torch.Tensor,
    h1: int,
    query_aggregation_mode: str = "single",
    query_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    last_h1_labels = labels[:, -h1:]
    supervised_mask = last_h1_labels > -100
    fallback_indices = supervised_mask.int().argmax(dim=-1, keepdim=True)
    B = labels.size(0)
    fallback_mask = torch.zeros(B, h1, dtype=torch.bool, device=labels.device)
    fallback_mask.scatter_(1, fallback_indices, True)

    mode = query_aggregation_mode.lower()
    if mode == "single":
        return fallback_mask
    if mode not in ("mean_all", "logsumexp_all"):
        raise ValueError(
            f"Unknown query_aggregation_mode: {query_aggregation_mode}. "
            "Expected one of ['single', 'mean_all', 'logsumexp_all']."
        )
    if query_token_mask is None:
        raise ValueError(
            f"query_token_mask is required when query_aggregation_mode='{mode}'"
        )
    if query_token_mask.dim() != 2:
        raise ValueError(f"query_token_mask must be 2D, got shape={tuple(query_token_mask.shape)}")
    if query_token_mask.size(0) != B:
        raise ValueError(
            f"query_token_mask batch size mismatch: {query_token_mask.size(0)} vs labels batch {B}"
        )
    if query_token_mask.size(1) < h1:
        raise ValueError(
            f"query_token_mask length {query_token_mask.size(1)} is shorter than h1={h1}"
        )

    qmask = query_token_mask[:, -h1:].bool().to(labels.device)
    has_query = qmask.any(dim=-1)
    return torch.where(has_query[:, None], qmask, fallback_mask)


def _aggregate_doc_scores_over_queries(
    doc_scores_by_query: torch.Tensor,
    query_selection_mask: torch.Tensor,
    query_aggregation_mode: str = "single",
) -> torch.Tensor:
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


def _compute_doc_scores_with_token_compression(
    attention_scores: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    block_order: str = "instruction_first",
    aux_norm_mode: str = "doc_plus_non_doc",
    token_compression_mode: str = "none",
    token_compression_topk: int = 8,
    token_compression_last_k: int = 1,
    token_compression_segment_k: int = 10,
    token_compression_segment_anchor: str = "end",
    attention_weighted_top_k: int | None = 1,
    query_aggregation_mode: str = "single",
    query_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build doc-level ranking scores from attention scores with token compression modes:
    - none: mean over all valid tokens in each document
    - topk: mean over top-k tokens (importance = attention score)
    - last: score averaged over the last-k valid tokens in each document
    - mid_last: score averaged over middle+last valid tokens in each document
    - segment: keep one token every k positions (anchor by token_compression_segment_anchor)
    """
    B, M, H = attention_mask.shape
    _, N, h1, MH = attention_scores.shape
    assert MH == M * H, "Attention scores last dimension must match M*H"

    query_selection_mask = _build_query_selection_mask(
        labels=labels,
        h1=h1,
        query_aggregation_mode=query_aggregation_mode,
        query_token_mask=query_token_mask,
    )

    bracket_attn_logits = attention_scores

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

    num_docs = M - 2 if block_order == "instruction_first" else M - 1
    doc_token_mask = attention_mask[:, doc_block_start:doc_block_end, :].bool()   # (B, num_docs, H)

    mode = token_compression_mode.lower()
    doc_logits = bracket_attn_logits[..., doc_start:doc_end]

    if token_compression_last_k <= 0:
        raise ValueError("token_compression_last_k must be > 0")
    aw_top_k = _normalize_optional_top_k(attention_weighted_top_k, field_name="attention_weighted_top_k")
    if mode == "segment":
        if int(token_compression_segment_k) <= 0:
            raise ValueError("token_compression_segment_k must be > 0")
        if str(token_compression_segment_anchor).lower() not in {"start", "end"}:
            raise ValueError(
                "token_compression_segment_anchor must be one of ['start', 'end']"
            )
        aw_top_k = None

    if mode == "segment":
        doc_select_mask = _segment_valid_token_mask(
            doc_token_mask,
            segment_k=int(token_compression_segment_k),
            anchor=token_compression_segment_anchor,
        )
    else:
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
    if aux_norm_mode == "doc_plus_non_doc":
        attn_lse = torch.logaddexp(non_doc_lse.detach(), doc_attn_lse)
    else:
        attn_lse = doc_attn_lse
    doc_attn = (doc_logits - attn_lse).exp()

    doc_token_scores = doc_attn.reshape(B, N, h1, num_docs, H)  # (B, N, h1, num_docs, H)
    doc_scores_by_query = _masked_mean(doc_token_scores, doc_select_mask.unsqueeze(1).unsqueeze(1), dim=-1).mean(dim=1)
    return _aggregate_doc_scores_over_queries(doc_scores_by_query, query_selection_mask, query_aggregation_mode)


def main():
    # Reuse train.py argument parsing
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default=None)
    cfg_args, remaining = ap.parse_known_args()
    cfg = load_config(cfg_args.config)

    parser = HfArgumentParser((ModelArgs, DataArgs, TrainArgs))
    merged = {**cfg.get("model", {}), **cfg.get("data", {}), **cfg.get("eval", {})}
    margs, dargs, targs = parser.parse_dict(merged)
    # Override model path if checkpoint provided
    if cfg_args.checkpoint:
        margs.model_name_or_path = cfg_args.checkpoint
    if 'blockrank' in margs.attn_implementation:
        margs.use_blockrank = True
        logger.info("BlockRank attention enabled based on attn_implementation=" + margs.attn_implementation)
    token_compression_mode = getattr(targs, "token_compression_mode", "none").lower()
    token_compression_last_k = int(getattr(targs, "token_compression_last_k", 1))
    token_compression_segment_k = int(getattr(targs, "token_compression_segment_k", 10))
    token_compression_segment_anchor = str(getattr(targs, "token_compression_segment_anchor", "end")).lower()
    attention_weighted_top_k = _normalize_optional_top_k(
        getattr(targs, "attention_weighted_top_k", 1),
        field_name="attention_weighted_top_k",
    )
    query_aggregation_mode = getattr(targs, "query_aggregation_mode", "single").lower()
    aux_num_last_queries = int(getattr(targs, "aux_num_last_queries", 32))
    if token_compression_mode not in {"none", "topk", "last", "mid_last", "segment"}:
        raise ValueError(
            f"token_compression_mode must be one of ['none', 'topk', 'last', 'mid_last', 'segment'], got: {token_compression_mode}"
        )
    if token_compression_mode == "topk" and int(getattr(targs, "token_compression_topk", 0)) <= 0:
        raise ValueError("token_compression_topk must be > 0 when token_compression_mode='topk'")
    if token_compression_last_k <= 0:
        raise ValueError("token_compression_last_k must be > 0")
    if token_compression_mode == "segment":
        if token_compression_segment_k <= 0:
            raise ValueError("token_compression_segment_k must be > 0 when token_compression_mode='segment'")
        if token_compression_segment_anchor not in {"start", "end"}:
            raise ValueError(
                "token_compression_segment_anchor must be one of ['start', 'end'] when token_compression_mode='segment'"
            )
    if query_aggregation_mode not in {"single", "mean_all", "logsumexp_all"}:
        raise ValueError(
            f"query_aggregation_mode must be one of ['single', 'mean_all', 'logsumexp_all'], got: {query_aggregation_mode}"
        )

    dataloader_config = DataLoaderConfiguration(
        split_batches=False,
        even_batches=True,
        use_seedable_sampler=True,
    )
    accelerator = Accelerator(dataloader_config=dataloader_config)
    set_seed(targs.seed)

    # Initialize W&B on main process only
    if accelerator.is_local_main_process:
        wandb.init(
            project=getattr(targs, "wandb_project", "blockrank-attn-eval"),
            name=os.path.basename(targs.output_dir) + f"_{os.path.basename(margs.model_name_or_path)}_attn",
            config={
                "model": margs.__dict__,
                "data": dargs.__dict__,
                "eval": targs.__dict__,
                "checkpoint": cfg_args.checkpoint,
                "attn_layer": targs.aux_layer_idx,
            },
            job_type="attn_eval",
        )

    # Load model and tokenizer (reuse from train.py)
    model, tok = setup_model_and_tokenizer(margs, device_map='cuda:0')
    model.eval()

    # Load eval dataset (reuse from train.py)
    with accelerator.main_process_first():
        ds = load_icr_dataset_hf(
            data_path=dargs.data_path,
            tokenizer=tok,
            num_documents=-1,
            seed=dargs.dataset_seed,
            train_test_split=dargs.train_test_split,
            streaming=dargs.streaming,
            eval_mode=True,
            use_blockrank=margs.use_blockrank,
            block_order=margs.block_order,
            query_in_instruction=dargs.query_in_instruction,
            remove_doc_id=dargs.remove_doc_id,
        )
        eval_ds = ds["test"] if ds.get("test", None) is not None else ds["train"]
        qrels = None
        if hasattr(dargs, 'qrels_path') and dargs.qrels_path:
            if os.path.exists(dargs.qrels_path):
                qrels = load_qrels(dargs.qrels_path)
                logger.info(f"Loaded qrels from {dargs.qrels_path}")
            else:
                logger.warning(
                    f"qrels_path does not exist: {dargs.qrels_path}. "
                    "Ranking metrics (ndcg/mrr) will be skipped."
                )
        else:
            logger.warning("No qrels_path provided. Ranking metrics (ndcg/mrr) will be skipped.")

    accelerator.wait_for_everyone()
    logger.info(f"Loaded {len(eval_ds)} examples")

    # Pre-extract data for fast incremental metrics
    metric_tracker = None
    if accelerator.is_local_main_process:
        eval_data = (
            list(eval_ds['answer_ids']),
            list(eval_ds['query_id']) if qrels else None,
            list(eval_ds['remapped_doc_ids']) if qrels else None
        )
        metric_tracker = IncrementalMetrics(eval_data, qrels)
        logger.info("Initialized incremental metric tracker")

    # Setup data collator (reuse from train.py)
    # Select appropriate collate function based on use_blockrank
    pad_to_multiple_of = dargs.__dict__.get("pad_to_multiple_of", 16)
    if margs.use_blockrank:
        data_collator = partial(
            block_icr_collate_fn,
            tok=tok,
            max_block_length=dargs.max_block_length,
            pad_to_multiple_of=pad_to_multiple_of,
            position_id_mode=dargs.position_id_mode,
            block_order=margs.block_order,
        )
        logger.info(f"Using BlockRank collate function with max_block_length={dargs.max_block_length}, pad_to_multiple_of={pad_to_multiple_of}")
    else:
        data_collator = partial(icr_collate_fn, tok=tok, max_seq_length=dargs.max_seq_length, pad_to_multiple_of=pad_to_multiple_of)
        logger.info(f"Using standard collate function with max_seq_length={dargs.max_seq_length}, pad_to_multiple_of={pad_to_multiple_of}")
    batch_size = getattr(targs, "per_device_eval_batch_size", None) or getattr(targs, "eval_batch_size", None) or 1
    dataloader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

    if os.environ.get("PRINT_EVAL_BATCH", "0") == "1" and accelerator.is_local_main_process:
        batch = next(iter(dataloader))

        def _show_sample(i: int = 0):
            ids = batch["input_ids"][i]
            labels = batch.get("labels")
            attn_mask = batch.get("attention_mask")
            pos_ids = batch.get("position_ids")
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            max_tokens = int(os.environ.get("PRINT_EVAL_MAX_TOKENS", "512"))
            print_full = os.environ.get("PRINT_EVAL_FULL", "0") == "1"
            print("Eval batch shapes:", {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")})
            print(f"block_order={margs.block_order} position_id_mode={dargs.position_id_mode}")
            if pad_id is not None:
                num_pad = int((ids == pad_id).sum().item())
                print(f"Token counts: total={ids.numel()} pad={num_pad} nonpad={ids.numel() - num_pad}")
            if pos_ids is not None:
                pos = pos_ids[i]
                pos_min = int(pos.min().item())
                pos_max = int(pos.max().item())
                print(f"Position IDs: min={pos_min} max={pos_max}")
                print("Position IDs head:", pos[:32].tolist())
                print("Position IDs tail:", pos[-32:].tolist())
                if attn_mask is not None:
                    flat_mask = attn_mask[i].reshape(-1).bool()
                    pos_nonpad = pos[flat_mask]
                    print("Nonpad Position IDs head:", pos_nonpad[:10].tolist())
                    print("Nonpad Position IDs tail:", pos_nonpad[-10:].tolist())
                    # Per-document position IDs (first 10 non-pad tokens per block)
                    M, H = attn_mask.shape[1], attn_mask.shape[2]
                    pos_blocks = pos.view(M, H)
                    doc_start = 1 if margs.block_order == "instruction_first" else 0
                    doc_end = M - 1
                    max_docs = int(os.environ.get("PRINT_EVAL_DOCS", "5"))
                    if max_docs <= 0:
                        max_docs = doc_end - doc_start
                    for b in range(doc_start, min(doc_end, doc_start + max_docs)):
                        mask_block = attn_mask[i, b].bool()
                        pos_block = pos_blocks[b][mask_block]
                        print(f"Doc block {b} pos head:", pos_block[:10].tolist())
            if attn_mask is not None:
                print(f"Attention mask shape: {tuple(attn_mask.shape)}")
                if attn_mask.dim() == 3:
                    block_sums = attn_mask[i].sum(dim=-1).tolist()
                    print("Attention mask per-block token counts (first 8):", block_sums[:8])
            flat_mask = None
            if attn_mask is not None:
                if attn_mask.dim() == 3:
                    flat_mask = attn_mask[i].reshape(-1).bool()
                else:
                    flat_mask = attn_mask[i].bool()
            elif pad_id is not None:
                flat_mask = ids != pad_id

            effective_ids = ids[flat_mask] if flat_mask is not None else ids

            if print_full:
                print("\nDecoded input (full, no pad):")
                print(tok.decode(effective_ids, skip_special_tokens=False))
                return

            if flat_mask is not None and flat_mask.any():
                start_idx = int(flat_mask.nonzero(as_tuple=False)[0].item())
            else:
                start_idx = 0

            trimmed_ids = ids[start_idx:]
            trimmed_ids = trimmed_ids[:max_tokens]

            print(f"\nDecoded input (trimmed from idx={start_idx}, first {max_tokens} tokens):")
            print(tok.decode(trimmed_ids, skip_special_tokens=False))
            print("\nDecoded input (trimmed, skip special tokens):")
            print(tok.decode(trimmed_ids, skip_special_tokens=True))

            # String-level cleanup for pad/eos artifacts
            pad_token_str = tok.pad_token or ""
            eos_token_str = tok.eos_token or ""
            cleaned = tok.decode(trimmed_ids, skip_special_tokens=False)
            for t in [pad_token_str, eos_token_str]:
                if t:
                    cleaned = cleaned.replace(t, "")
            print("\nDecoded input (trimmed, string-filtered):")
            print(cleaned.strip())

            # Debug: show first non-100 label token (should be '[')
            if labels is not None:
                lbl = labels[i]
                first_idx = (lbl != -100).nonzero(as_tuple=True)[0]
                if first_idx.numel() > 0:
                    idx = first_idx[0].item()
                    tok_id = int(ids[idx].item())
                    tok_str = tok.decode([tok_id], skip_special_tokens=False)
                    print(f"\nFirst non-100 label token: idx={idx}, id={tok_id}, token={tok_str!r}")
                else:
                    print("\nFirst non-100 label token: not found")


        _show_sample(0)

    # Prepare model and dataloader with Accelerator
    model, dataloader = accelerator.prepare(model, dataloader)
    logger.info(f"Running attention-based evaluation on {accelerator.num_processes} processes...")
    logger.info(f"Using attention layer {targs.aux_layer_idx} for predictions")
    logger.info(
        "Token compression mode=%s topk=%s last_k=%s segment_k=%s segment_anchor=%s attention_weighted_top_k=%s query_aggregation_mode=%s aux_num_last_queries=%s",
        token_compression_mode,
        getattr(targs, "token_compression_topk", None),
        token_compression_last_k,
        token_compression_segment_k,
        token_compression_segment_anchor,
        attention_weighted_top_k,
        query_aggregation_mode,
        aux_num_last_queries,
    )

    # Optimize by preventing computation in layers after the target layer
    unwrapped_model = accelerator.unwrap_model(model)
    target_layer_idx = targs.aux_layer_idx

    # Find the model's layer list
    if hasattr(unwrapped_model, 'model') and hasattr(unwrapped_model.model, 'layers'):
        layers = unwrapped_model.model.layers
    elif hasattr(unwrapped_model, 'layers'):
        layers = unwrapped_model.layers
    else:
        layers = None
        logger.warning("Could not find model layers, will compute all layers")

    original_forwards = []
    if layers is not None and target_layer_idx + 1 < len(layers):
        def identity_forward(self, hidden_states, *args, **kwargs):
            return hidden_states
        
        for i in range(target_layer_idx + 1, len(layers)):
            original_forwards.append((i, layers[i].forward))
            layers[i].forward = identity_forward.__get__(layers[i], type(layers[i]))
        unwrapped_model.lm_head.forward = identity_forward.__get__(unwrapped_model.lm_head, type(unwrapped_model.lm_head))
        logger.info(f"Monkey-patched {len(original_forwards)} layers & LM head after layer {target_layer_idx} with identity forward")

    all_attn_preds = []
    with torch.no_grad():
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc="Evaluating")
        for batch in pbar:
            # Forward pass with attention output
            labels = batch.pop('labels')
            _ = batch.pop('answer_ids', None)
            query_token_mask = batch.pop('query_token_mask', None)
            out = unwrapped_model(
                **batch,
                output_attentions=True,
                layers_to_return_scores=[target_layer_idx],
                num_last_queries=aux_num_last_queries,
                token_compression_mode=token_compression_mode,
                token_compression_last_k=token_compression_last_k,
                token_compression_segment_k=token_compression_segment_k,
                token_compression_segment_anchor=token_compression_segment_anchor,
                attention_weighted_top_k=attention_weighted_top_k,
            )

            doc_scores = _compute_doc_scores_with_token_compression(
                attention_scores=out.attentions[0],
                labels=labels,
                attention_mask=batch['attention_mask'],
                block_order=getattr(unwrapped_model.config, "blockrank_block_order", "instruction_first"),
                aux_norm_mode=getattr(targs, "aux_norm_mode", "doc_plus_non_doc"),
                token_compression_mode=token_compression_mode,
                token_compression_topk=int(getattr(targs, "token_compression_topk", 8)),
                token_compression_last_k=token_compression_last_k,
                token_compression_segment_k=token_compression_segment_k,
                token_compression_segment_anchor=token_compression_segment_anchor,
                attention_weighted_top_k=attention_weighted_top_k,
                query_aggregation_mode=query_aggregation_mode,
                query_token_mask=query_token_mask,
            )

            # Get top-k predictions (k=10 for ranking metrics)
            k = min(10, doc_scores.shape[-1])
            attn_preds = torch.topk(doc_scores, k=k, dim=-1).indices
            attn_preds = accelerator.gather_for_metrics(attn_preds)

            if accelerator.is_local_main_process:
                all_attn_preds.extend([pred.cpu().tolist() for pred in attn_preds])

                # Update metrics incrementally every iteration
                results = metric_tracker.update(all_attn_preds)
                pbar.set_postfix({"acc": f"{results['accuracy']:.2f}%", "ndcg@10": f"{results.get('ndcg@10', 0):.2f}"})

                # Log to wandb every 20 batches to avoid flooding
                if pbar.n % 20 == 0 or pbar.n == len(dataloader):
                    wandb.log({f"intermediate_eval/{k}": v for k, v in results.items()}, step=len(all_attn_preds))

        accelerator.wait_for_everyone()

    # Only main process computes metrics and saves
    if accelerator.is_local_main_process:
        results = calculate_accuracy(all_attn_preds, eval_ds, qrels=qrels)

        # Log to W&B
        wandb.log(results)

        # Save results
        os.makedirs(targs.output_dir, exist_ok=True)
        metrics_file = os.path.join(targs.output_dir, f"attn_eval_{os.path.basename(margs.model_name_or_path)}_metrics.json")
        results_with_config = {
            **results,
            "attn_layer": targs.aux_layer_idx,
            "token_compression_mode": token_compression_mode,
            "token_compression_topk": int(getattr(targs, "token_compression_topk", 8)),
            "token_compression_last_k": token_compression_last_k,
            "token_compression_segment_k": token_compression_segment_k,
            "token_compression_segment_anchor": token_compression_segment_anchor,
            "attention_weighted_top_k": attention_weighted_top_k,
            "query_aggregation_mode": query_aggregation_mode,
            "aux_num_last_queries": aux_num_last_queries,
        }
        with open(metrics_file, "w") as f:
            json.dump(results_with_config, f, indent=2)

        logger.info(f"\n{'='*50}\nAttention-based Evaluation Results:")
        logger.info(f"  Attention Layer: {targs.aux_layer_idx}")
        logger.info(f"  Token Compression Mode: {token_compression_mode}")
        if token_compression_mode == "topk":
            logger.info(f"  Token Compression Top-k: {int(getattr(targs, 'token_compression_topk', 8))}")
        if token_compression_mode == "last":
            logger.info(f"  Token Compression Last-k: {token_compression_last_k}")
        if token_compression_mode == "mid_last":
            logger.info("  Token Compression Anchors: middle + last")
        if token_compression_mode == "segment":
            logger.info(f"  Token Compression Segment-k: {token_compression_segment_k}")
            logger.info(f"  Token Compression Segment Anchor: {token_compression_segment_anchor}")
        logger.info(f"  Attention Weighted Top-k: {attention_weighted_top_k}")
        logger.info(f"  Query Aggregation Mode: {query_aggregation_mode}")
        logger.info(f"  Aux Num Last Queries: {aux_num_last_queries}")
        for k, v in results.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        logger.info(f"{'='*50}\n")
        logger.info(f"Saved to {metrics_file}")

        # Log some example predictions
        examples_table = wandb.Table(
            columns=["Predicted ID", "Ground Truth"],
            data=[
                [str(all_attn_preds[i]), str(eval_ds['answer_ids'][i])]
                for i in range(min(100, len(all_attn_preds)))
            ]
        )
        wandb.log({"predictions_sample": examples_table})

        # Finish W&B run
        wandb.finish()

if __name__ == "__main__":
    main()
