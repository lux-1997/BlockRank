import os, sys, glob
from typing import Dict, List, Optional, Any
from torch.utils.data import Dataset
from .utils import (
    remap_documents,
    create_prompt_completion_format,
)

import torch
import numpy as np
from datasets import load_dataset, DatasetDict, IterableDatasetDict
from transformers import AutoTokenizer
import datasets

datasets.enable_caching()

def load_icr_dataset_hf(
    data_path: str,
    tokenizer: AutoTokenizer,
    num_documents: int = -1,
    seed: Optional[int] = 42,
    train_test_split: float = 1.0,
    streaming: bool = False,
    eval_mode: bool = False,
    use_blockrank: bool = False,
    prompt_type: str | None = None,
    block_order: str = "instruction_first",
    query_in_instruction: bool = True,
    remove_doc_id: bool = False,
    doc_end_token: str | None = None,
    **kwargs,
) -> DatasetDict:
    """
    Returns a DatasetDict with 'train' and 'test' splits, each item containing:
      - messages: list[{'role','content'}]
      - query: str
      - answer_ids: list[int]
      - num_documents: int
    """
    # handle sharded jsonl files, local paths, or HuggingFace hub datasets
    if data_path.endswith('.jsonl') or '*' in data_path or os.path.exists(data_path):
        # Local file handling
        if "*" in data_path:
            import glob
            data_files = sorted(list(glob.glob(data_path)))
        else:
            data_files = data_path
        cache_dir = os.path.join(os.path.dirname(data_path) if not "*" in data_path else os.path.dirname(data_path.split('*')[0]), "hf_cache")
        raw = load_dataset("json", data_files=data_files, split="train", streaming=streaming, cache_dir=cache_dir)
    else:
        # HuggingFace Hub dataset
        split = kwargs.pop("split", "train")
        subset = kwargs.pop("subset", None)
        raw = load_dataset(data_path, subset, split=split, streaming=streaming)

    if streaming:
        raw = raw.shuffle(seed=seed or 42)
        print('WARNING: Streaming mode enabled; train/test split will not be created.')
        ds_dict = IterableDatasetDict({"train": raw})
    else:
        if train_test_split >= 1.0:
            ds_dict = DatasetDict({"train": raw})
        elif train_test_split <= 0.0:
            ds_dict = DatasetDict({"test": raw})
        else:
            ds_dict = raw.train_test_split(test_size=1 - train_test_split, seed=seed or 42)

    PROMPT_SEGMENT_SEP = "<<end_of_block_prompt_segment>>" if use_blockrank else "\n"
    PROMPT_TYPE = prompt_type or ("mistral" if "mistral" in tokenizer.name_or_path.lower() else "qwen")
    print(f"[load_icr_dataset_hf] Using PROMPT_TYPE={PROMPT_TYPE}")
    resolved_doc_end_token = doc_end_token
    if isinstance(doc_end_token, str):
        marker = doc_end_token.strip()
        if marker == "":
            resolved_doc_end_token = None
        elif marker.lower() == "<eos>":
            if tokenizer.eos_token is None:
                raise ValueError("doc_end_token='<eos>' but tokenizer.eos_token is None")
            resolved_doc_end_token = tokenizer.eos_token
    if resolved_doc_end_token:
        print(f"[load_icr_dataset_hf] Appending doc_end_token={repr(resolved_doc_end_token)} to each document block")
    if remove_doc_id:
        print("[load_icr_dataset_hf] remove_doc_id=True: omitting explicit document IDs in prompt blocks")
    print_prompt_example = os.environ.get("PRINT_PROMPT_EXAMPLE", "0") == "1"
    map_num_proc = 1 if eval_mode else max(1, os.cpu_count() - 2)

    def _sample_and_format(example, idx):
        query = example["query"]
        query_id = example.get("query_id", str(idx))
        documents = example["documents"]
        answer_ids = example["answer_ids"]
        if answer_ids is None:
            answer_ids = []
        elif not isinstance(answer_ids, list):
            answer_ids = [answer_ids]
        if isinstance(documents, list):
            if not documents:
                documents = {}
            elif isinstance(documents[0], dict):
                documents = {doc.get("doc_id", str(i)): f'{doc.get("title", "")} {doc.get("text", "")}'.strip() for i, doc in enumerate(documents)}
            else:
                documents = {str(i): doc for i, doc in enumerate(documents)}

        remapped_docs, remapped_doc_ids, remapped_ans_ids = remap_documents(
            documents=documents,
            answer_ids=answer_ids,
            num_samples=num_documents,
            seed=(seed or 42) + idx,
            sample=not eval_mode,
            add_padding_docs=not eval_mode,
        )
        remapped_ans_ids = [int(x) for x in remapped_ans_ids]

        pc = create_prompt_completion_format(
            query,
            remapped_docs,
            [] if eval_mode else remapped_ans_ids,
            sep=PROMPT_SEGMENT_SEP,
            type=PROMPT_TYPE,
            block_order=block_order,
            query_in_instruction=query_in_instruction,
            remove_doc_id=remove_doc_id,
            doc_end_token=resolved_doc_end_token,
        )
        if print_prompt_example and idx == 0:
            prompt_msg = pc["prompt"][0].get("content", "") if pc.get("prompt") else ""
            completion_msg = ""
            if pc.get("completion"):
                completion_msg = pc["completion"][-1].get("content", "")
            print("[Prompt Example] prompt_type=", PROMPT_TYPE)
            print(prompt_msg)
            print("[Prompt Example] completion")
            print(completion_msg)
        return {
            "query": query,
            "query_id": query_id,
            "answer_ids": remapped_ans_ids,
            "remapped_doc_ids": remapped_doc_ids,
            "num_documents": len(remapped_docs),
            **pc,
        }
    
    def _tokenize_batch(batch):
        merged_msgs = [x + y for x, y in zip(batch['prompt'], batch['completion'])]
        if eval_mode:
            raw_prompt_env = os.environ.get("BLOCKRANK_RAW_PROMPT")
            raw_prompt = False if raw_prompt_env is None else (raw_prompt_env == "1")
            if raw_prompt:
                # Use raw message contents without chat template wrappers.
                prompt_texts = []
                for msgs in batch['prompt']:
                    prompt_texts.append("".join(m.get("content", "") for m in msgs))
            else:
                # Build "open" assistant prefix without continue_final_message to avoid template drops.
                prompt_texts = tokenizer.apply_chat_template(
                    batch['prompt'],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            completion_texts = []
            for m in batch["completion"]:
                content = m[-1].get("content", "") if m else ""
                # Drop any accidental template suffixes.
                if "<|im_end|>" in content:
                    content = content.split("<|im_end|>", 1)[0]
                completion_texts.append(content)
            full_texts = [p + c for p, c in zip(prompt_texts, completion_texts)]
            full_input_ids = tokenizer(
                full_texts,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            prompt_lengths = [
                len(x) for x in tokenizer(
                    prompt_texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
            ]
        else:
            full_input_ids = tokenizer.apply_chat_template(merged_msgs)
            prompt_lengths = [len(x) for x in tokenizer.apply_chat_template(batch['prompt'])]

        return {
            'input_ids': full_input_ids,
            'prompt_lengths': prompt_lengths,
        }

    def _block_tokenize_batch(batch):
        def _find_query_token_positions(block_text: str, query_text: str) -> list[int]:
            if not block_text or not query_text:
                return []
            spans = []
            start = 0
            while True:
                idx = block_text.find(query_text, start)
                if idx < 0:
                    break
                spans.append((idx, idx + len(query_text)))
                start = idx + len(query_text)
            if not spans:
                return []

            encoded = tokenizer(
                block_text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_offsets_mapping=True,
            )
            offsets = encoded.get('offset_mapping', [])
            q_positions = []
            for tidx, (c0, c1) in enumerate(offsets):
                if c1 <= c0:
                    continue
                if any((c0 < e and c1 > s) for s, e in spans):
                    q_positions.append(int(tidx))
            return q_positions

        all_block_input_texts = []
        all_last_prompt_query_positions = []
        raw_prompt_env = os.environ.get("BLOCKRANK_RAW_PROMPT")
        raw_prompt = False if raw_prompt_env is None else (raw_prompt_env == "1")
        for itr in range(len(batch['prompt'])):
            if eval_mode and raw_prompt:
                prompt_text = "".join(m.get("content", "") for m in batch["prompt"][itr])
                completion_msgs = batch["completion"][itr]
                completion_text = completion_msgs[-1].get("content", "") if completion_msgs else ""
                if "<|im_end|>" in completion_text:
                    completion_text = completion_text.split("<|im_end|>", 1)[0]
                input_texts = prompt_text + completion_text
            else:
                input_texts = tokenizer.apply_chat_template(batch['prompt'][itr]+batch['completion'][itr], tokenize=False)
            input_texts = input_texts.strip()
            if eval_mode and input_texts.endswith(tokenizer.eos_token):
                input_texts = input_texts[:-len(tokenizer.eos_token)].rstrip()
            block_input_texts = input_texts.split(PROMPT_SEGMENT_SEP)
            n = len(block_input_texts)
            block_input_texts = [f'\n{x}' if i > 0 and i < n-1 else x for i, x in enumerate(block_input_texts)]
            all_block_input_texts.append(block_input_texts)

            query_text = batch.get('query', [''] * len(batch['prompt']))[itr] or ''
            if len(block_input_texts) >= 2:
                last_prompt_block_text = block_input_texts[-2]
                q_positions = _find_query_token_positions(last_prompt_block_text, query_text)
            else:
                q_positions = []
            all_last_prompt_query_positions.append(q_positions)

        indptr = np.cumsum([0] + [len(x) for x in all_block_input_texts])

        all_block_input_ids = tokenizer(
            [x for y in all_block_input_texts for x in y],
            add_special_tokens=False,
            return_attention_mask=False,
        )['input_ids']
        all_block_input_ids = [all_block_input_ids[indptr[i]:indptr[i+1]] for i in range(len(all_block_input_texts))]
        block_lengths = [[len(x) for x in y] for y in all_block_input_ids]
        all_block_input_ids = [[x for y in ex for x in y] for ex in all_block_input_ids]
        return {
            'input_ids': all_block_input_ids,
            'block_lengths': block_lengths,
            'last_prompt_query_positions': all_last_prompt_query_positions,
        }


    ds_dict = ds_dict.map(
        _sample_and_format,
        with_indices=True,
        remove_columns=['documents', 'answer_ids'],
        num_proc=map_num_proc,
        load_from_cache_file=not eval_mode,
    )
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    ds_dict = ds_dict.map(
        _block_tokenize_batch if use_blockrank else _tokenize_batch,
        batched=True,
        batch_size=64,
        remove_columns=['prompt', 'completion'],
        num_proc=map_num_proc,
    )

    ds_dict = ds_dict.with_format("torch")

    return ds_dict
def icr_collate_fn(batch, tok, pad_to_multiple_of=8, max_seq_length=None, always_max_len=False) -> Dict[str, torch.Tensor]:
    pad_token_id = tok.pad_token_id
    padding_side = tok.padding_side
    if always_max_len:
        max_seq_length = max_seq_length or max([item['input_ids'].size(0) for item in batch])
    else:
        max_seq_length = min(max([item['input_ids'].size(0) for item in batch]), max_seq_length or int(1e9))
    
    if pad_to_multiple_of is not None:
        max_seq_length = ((max_seq_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    padding_input_id = torch.full((max_seq_length,), pad_token_id, dtype=torch.long)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['input_ids'].squeeze(0)[:max_seq_length] for item in batch] + [padding_input_id],
        batch_first=True,
        padding_value=pad_token_id,
        padding_side=padding_side,
    )[:-1] # remove extra padding row
    B, S = input_ids.shape # batch size, seq len
    attention_mask = (input_ids != pad_token_id)
    labels = input_ids.clone()
    labels[input_ids == pad_token_id] = -100 # pad tokens not to be predicted
    
    # Adjust prompt_lengths based on padding side and truncation
    prompt_lengths = torch.tensor([item['prompt_lengths'] for item in batch])
    original_lengths = torch.tensor([item['input_ids'].size(0) for item in batch])
    
    if padding_side == 'left':
        # With left padding, prompt positions shift right by padding amount
        padding_amounts = max_seq_length - torch.min(original_lengths, torch.tensor(max_seq_length))
        adjusted_len_prompt = prompt_lengths + padding_amounts
    else:  # right padding
        # With right padding, if sequence is truncated, adjust prompt length
        adjusted_len_prompt = torch.min(prompt_lengths, torch.tensor(max_seq_length))
    
    labels[torch.arange(S)[None, :] < adjusted_len_prompt[:, None]] = -100 # prompt tokens not to be predicted

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

def block_icr_collate_fn(
    batch,
    tok,
    pad_to_multiple_of=16,
    max_block_length=None,
    always_max_len=False,
    permutation_invariant_pos=True,
    position_id_mode: str | None = None,
    block_order: str = "instruction_first",
    preserve_doc_last_token: bool = False,
) -> Dict[str, torch.Tensor]:
    pad_token_id = tok.pad_token_id
    padding_side = tok.padding_side
    B = len(batch)
    # merge completion block into last prompt block
    for item in batch:
        item['last_prompt_block_lengths'] = item['block_lengths'][-2].item()
        item['completion_lengths'] = item['block_lengths'][-1].item()
        item['block_lengths'] = item['block_lengths'][:-1]
        item['block_lengths'][-1] += item['completion_lengths']
        assert sum(item['block_lengths']) == item['input_ids'].size(0), "Block lengths do not sum to input_ids length"

    M = max(len(item['block_lengths']) for item in batch)  # number of prompt blocks
    # Pad missing document blocks with empty blocks (keep last block semantics intact)
    for item in batch:
        if len(item['block_lengths']) < M:
            pad_count = M - len(item['block_lengths'])
            pad_blocks = torch.zeros(pad_count, dtype=item['block_lengths'].dtype)
            item['block_lengths'] = torch.cat(
                [item['block_lengths'][:-1], pad_blocks, item['block_lengths'][-1:]]
            )

    if always_max_len:
        max_block_length = max_block_length or max([item['block_lengths'].max().item() for item in batch])
    else:
        max_block_length = min(max([item['block_lengths'].max().item() for item in batch]), max_block_length or int(1e9))

    if pad_to_multiple_of is not None:
        max_block_length = ((max_block_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    all_block_input_ids = []
    for item in batch:
        indptr = torch.cumsum(torch.cat([torch.tensor([0]), item['block_lengths']]), dim=0)
        block_input_ids = []
        for i in range(len(item['block_lengths'])):
            block_ids = item['input_ids'][indptr[i]:indptr[i+1]]
            if block_ids.numel() > max_block_length:
                # Keep each document block's final token under truncation (used with doc_end_token=<eos>).
                last_block_idx = len(item['block_lengths']) - 1
                if block_order == "instruction_first":
                    is_doc_block = (i > 0) and (i < last_block_idx)
                elif block_order == "doc_first":
                    is_doc_block = i < last_block_idx
                else:
                    raise ValueError(f"Unknown block_order: {block_order}")
                if preserve_doc_last_token and is_doc_block and max_block_length > 1:
                    block_ids = torch.cat([block_ids[:max_block_length - 1], block_ids[-1:]], dim=0)
                else:
                    block_ids = block_ids[:max_block_length]
            block_input_ids.append(block_ids)
        item['block_input_ids'] = block_input_ids
        all_block_input_ids.extend(item['block_input_ids'])

    padding_input_id = torch.full((max_block_length,), pad_token_id, dtype=torch.long)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        all_block_input_ids + [padding_input_id],
        batch_first=True,
        padding_value=pad_token_id,
        padding_side=padding_side,
    )[:-1] # remove extra padding row
    BM, H = input_ids.shape # batch size, seq len
    input_ids = input_ids.view(B, M, H)
    attention_mask = (input_ids != pad_token_id)
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100 # pad tokens not to be predicted
    labels[:, :-1, :] = -100  # only last block trainable


    # Adjust last_prompt_block_lengths based on padding side and truncation
    last_prompt_block_lengths = torch.tensor([item['last_prompt_block_lengths'] for item in batch])
    original_lengths = torch.tensor([item['block_lengths'][-1] for item in batch])
    if os.environ.get("PRINT_LAST_PROMPT_BLOCK_LENGTHS", "0") == "1":
        print(
            "[BlockRank] last_prompt_block_lengths="
            f"{last_prompt_block_lengths.tolist()} "
            f"original_last_block_lengths={original_lengths.tolist()} "
            f"max_block_length={max_block_length}"
        )

    if padding_side == 'left':
        # With left padding, prompt positions shift right by padding amount
        padding_amounts = max_block_length - torch.min(original_lengths, torch.tensor(max_block_length))
        adjusted_len_prompt = last_prompt_block_lengths + padding_amounts
    else:  # right padding
        # With right padding, if sequence is truncated, adjust prompt length
        adjusted_len_prompt = torch.min(last_prompt_block_lengths, torch.tensor(max_block_length))

    labels[:, -1, :] = torch.where(
        torch.arange(max_block_length)[None, :] < adjusted_len_prompt[:, None], 
        -100, 
        labels[:, -1, :]) # prompt tokens not to be predicted

    # Build precise query-token mask on the last block (shape: B, H).
    query_token_mask = torch.zeros((B, H), dtype=torch.bool, device=input_ids.device)
    for bi, item in enumerate(batch):
        qpos = item.get('last_prompt_query_positions', [])
        if torch.is_tensor(qpos):
            qpos = qpos.tolist()
        qpos = [int(x) for x in qpos]

        orig_last_len = item['block_lengths'][-1]
        if torch.is_tensor(orig_last_len):
            orig_last_len = int(orig_last_len.item())
        else:
            orig_last_len = int(orig_last_len)
        effective_last_len = min(orig_last_len, max_block_length)
        left_pad_shift = (max_block_length - effective_last_len) if padding_side == 'left' else 0

        for pos in qpos:
            if 0 <= pos < effective_last_len:
                query_token_mask[bi, pos + left_pad_shift] = True

    if position_id_mode is None:
        position_id_mode = "perm_invariant" if permutation_invariant_pos else "sequential"

    # Position IDs respecting block order
    if position_id_mode == "perm_invariant":
        position_ids = attention_mask.cumsum(-1)
        if block_order == "instruction_first":
            position_ids[:, 1:-1] += position_ids[:, 0].max(dim=-1).values[:, None, None] # offset by instruction block max
            position_ids[:, -1] += 16384  # a large position offset for last block
        else:
            position_ids[:, -1] += 16384  # a large position offset for instruction/query block
        position_ids = torch.clamp_min(position_ids - 1, 0)
        position_ids[~attention_mask] = 0 # pad positions
    elif position_id_mode == "sequential":
        flat_mask = attention_mask.view(B, -1)
        position_ids = flat_mask.cumsum(-1) * flat_mask
        position_ids = torch.clamp_min(position_ids - 1, 0)
    else:
        raise ValueError(f"Unknown position_id_mode: {position_id_mode}")

    # Extract and pad answer_ids from batch items (for auxiliary loss)
    answer_ids_padded = torch.nn.utils.rnn.pad_sequence(
        [item['answer_ids'] for item in batch],
        batch_first=True,
        padding_value=-1
    )  # Shape: (B, max_num_answers), padded with -1

    return {
        'input_ids': input_ids.view(B, M*H),
        'position_ids': position_ids.view(B, M*H),
        'attention_mask': attention_mask,
        'labels': labels.view(B, M*H),
        'num_blocks': torch.tensor(M, dtype=torch.long, device=input_ids.device),
        'answer_ids': answer_ids_padded,  # Padded 2D tensor (B, max_num_answers)
        'query_token_mask': query_token_mask,
    }
