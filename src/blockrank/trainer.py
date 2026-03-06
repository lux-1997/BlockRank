"""
Custom trainer for BlockRank with auxiliary attention loss.

This module provides a custom SFTTrainer that combines the standard
language modeling loss with an auxiliary contrastive loss on attention patterns.
"""

import os
import torch
from trl import SFTTrainer
from .losses import compute_auxiliary_attention_loss, compute_auxiliary_attention_loss_copynet
from peft import PeftType


class BlockRankAuxLossTrainer(SFTTrainer):
    """
    Custom trainer that adds auxiliary attention loss to the standard LM loss.

    The auxiliary loss optimizes query-document attention patterns during training,
    encouraging the model to attend more strongly to relevant documents from the
    "[" bracket token position.

    Configuration parameters (from TrainArgs):
        - use_aux_loss: Enable auxiliary loss (bool)
        - aux_layer_idx: Which transformer layer to extract attention from (int)
        - aux_loss_weight: Weight for combining losses (float, lambda in paper)
        - aux_temperature: Temperature for InfoNCE loss (float, tau in paper)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the trainer and enforce use_reentrant=False for gradient checkpointing
        when auxiliary loss is enabled.
        """
        super().__init__(*args, **kwargs)

        # Check if both gradient checkpointing and auxiliary loss are enabled
        use_aux_loss = getattr(self.args, 'use_aux_loss', False)
        gradient_checkpointing = getattr(self.args, 'gradient_checkpointing', False)

        if use_aux_loss and gradient_checkpointing:
            # Ensure gradient_checkpointing_kwargs exists
            if self.args.gradient_checkpointing_kwargs is None:
                self.args.gradient_checkpointing_kwargs = {}

            # Enforce use_reentrant=False for proper gradient flow through attention scores
            if self.args.gradient_checkpointing_kwargs.get('use_reentrant', True):
                print("[BlockRankAuxLossTrainer] Setting gradient_checkpointing_kwargs['use_reentrant']=False")
                print("  This is required for auxiliary loss to receive gradients with gradient checkpointing enabled.")
                self.args.gradient_checkpointing_kwargs['use_reentrant'] = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to add auxiliary contrastive loss.

        Args:
            model: The model being trained
            inputs: Input batch dictionary
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in batch (unused)

        Returns:
            loss: Combined loss (LM + auxiliary)
            outputs: Model outputs (if return_outputs=True)
        """
        # Check if auxiliary loss is enabled
        mode = "train" if self.model.training else "eval"
        use_aux_loss = getattr(self.args, 'use_aux_loss', False)

        if use_aux_loss:
            assert 'answer_ids' in inputs, "Input batch must contain 'answer_ids' for auxiliary loss."
            # Forward pass with attention outputs for the specified layer
            aux_layer_idx = self.args.aux_layer_idx

            # Prepare model inputs (exclude non-model keys)
            model_input_keys = {'input_ids', 'attention_mask', 'position_ids', 'labels'}
            model_inputs = {k: v for k, v in inputs.items() if k in model_input_keys}

            # Forward pass requesting attention scores from specific layer
            outputs = model(
                **model_inputs,
                output_attentions=True,
                layers_to_return_scores=[aux_layer_idx],
                num_last_queries=32,
            )

            assert outputs.attentions is not None, "Model did not return attention scores."
            assert len(outputs.attentions) > 0, "No attention scores returned from model."

            if (
                os.environ.get("BLOCKRANK_AUX_DEBUG", "0") == "1"
                and self.accelerator.is_main_process
                and not getattr(self, "_aux_debug_printed", False)
            ):
                self._aux_debug_printed = True
                attn0 = outputs.attentions[0]
                print(
                    "[BlockRankAuxDebug] outputs.attentions.len="
                    f"{len(outputs.attentions)} attn0.shape={tuple(attn0.shape)} "
                    f"dtype={attn0.dtype} device={attn0.device} "
                    f"requires_grad={attn0.requires_grad} "
                    f"grad_fn={type(attn0.grad_fn).__name__ if attn0.grad_fn is not None else 'None'}"
                )
                print(
                    "[BlockRankAuxDebug] input_ids.shape="
                    f"{tuple(inputs['input_ids'].shape)} attention_mask.shape="
                    f"{tuple(inputs['attention_mask'].shape)} labels.shape="
                    f"{tuple(inputs['labels'].shape)}"
                )
                print(
                    "[BlockRankAuxDebug] block_order="
                    f"{getattr(self.model.config, 'blockrank_block_order', 'instruction_first')} "
                    f"aux_layer_idx={aux_layer_idx} num_last_queries=32"
                )

            # Get standard LM loss from model outputs
            lm_loss = outputs.loss
            sft_weight = getattr(self.args, "sft_loss_weight", 1.0)

            aux_loss_type = getattr(self.args, "aux_loss_type", "infonce")
            aux_loss_fn = compute_auxiliary_attention_loss
            if aux_loss_type == "copynet":
                aux_loss_fn = compute_auxiliary_attention_loss_copynet
            elif aux_loss_type != "infonce":
                raise ValueError(f"Unknown aux_loss_type: {aux_loss_type}")

            aux_loss, attn_acc = aux_loss_fn(
                attention_scores=outputs.attentions[0],  # (B, N, 16, M*H)
                labels=inputs['labels'],  # (B, M*H)
                answer_ids=inputs['answer_ids'],  # (B, max_num_answers)
                attention_mask=inputs['attention_mask'],  # (B, M, H)
                temperature=self.args.aux_temperature,
                block_order=getattr(self.model.config, "blockrank_block_order", "instruction_first"),
                aux_norm_mode=getattr(self.args, "aux_norm_mode", "doc_plus_non_doc"),
            )

            # Combine losses with configured weight
            aux_weight = self.args.aux_loss_weight
            total_loss = sft_weight * lm_loss + aux_weight * aux_loss

            # Log individual loss components (every logging_steps)
            self._metrics[mode]["lm_loss"].append(self.accelerator.gather_for_metrics(lm_loss).mean().item())
            self._metrics[mode]["sft_loss_weight"].append(sft_weight)
            self._metrics[mode]["aux_loss"].append(self.accelerator.gather_for_metrics(aux_loss).mean().item())
            if aux_loss_type == "copynet":
                self._metrics[mode]["copynet_loss"].append(self.accelerator.gather_for_metrics(aux_loss).mean().item())
            self._metrics[mode]["total_loss"].append(self.accelerator.gather_for_metrics(total_loss).mean().item())
            self._metrics[mode]["aux_attn_accuracy"].append(self.accelerator.gather_for_metrics(attn_acc).mean().item())
        else:
            # Standard training without auxiliary loss
            outputs = model(**inputs)
            total_loss = outputs.loss

        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(total_loss.device)
        total_loss = total_loss / (num_items_in_batch if num_items_in_batch is not None else 1)

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            total_loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if not self.args.use_liger_kernel:
            with torch.no_grad():
                if "shift_labels" in inputs:
                    # When using CP, labels are pre-shifted. We must use these (and cannot manually shift) because:
                    # - The first discarded token from inputs["labels"] actually belongs to process n-1
                    # - The last logits require the label from process n+1
                    shift_logits = outputs.logits.contiguous()
                    shift_labels = inputs["shift_labels"]
                else:
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = inputs["labels"][..., 1:].contiguous()

                # Prompt Tuning and P-Tuning output logits for virtual tokens but Prefix-Tuning does not.
                if (
                    self.num_virtual_tokens > 0
                    and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
                ):
                    shift_logits = shift_logits[:, self.num_virtual_tokens :, :]

                # Get predictions
                predictions = shift_logits.argmax(dim=-1)

                # Create mask for non-padding tokens (assuming ignore_index is -100)
                mask = shift_labels != -100

                # Calculate accuracy only on non-padding tokens
                correct_predictions = (predictions == shift_labels) & mask
                total_tokens = mask.sum()
                correct_tokens = correct_predictions.sum()

                # Gather the correct_tokens and total_tokens across all processes
                correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
                total_tokens = self.accelerator.gather_for_metrics(total_tokens)

                # Compute the mean token accuracy and log it
                total_sum = total_tokens.sum()
                accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
                self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (total_loss, outputs) if return_outputs else total_loss
