"""
Dynamic Fine‑Tuning (DFT) loss plugin for Axolotl
=================================================

This module implements a plugin that swaps out the loss used by the
HuggingFace ``Trainer`` during an Axolotl supervised fine‑tuning run
with the DFT loss introduced in TRL v0.23.0.  The goal of DFT is to
scale gradients based on the probability of each token, encouraging
the model to focus on less confident predictions.  See the TRL
pull request for details:

    https://github.com/huggingface/trl/pull/4042

The plugin is written to be self‑contained and can be distributed as a
separate package.  It assumes that ``torch`` and ``transformers`` are
available at runtime.  If ``torch`` is unavailable the plugin will
initialise but will raise an informative error if the DFT loss is
actually invoked.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Dict, Optional
from axolotl.integrations.base import BasePlugin  # type: ignore

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - defer until used
    torch = None  # type: ignore
    nn = None  # type: ignore


def _selective_log_softmax(logits: "torch.Tensor", target_ids: "torch.Tensor") -> "torch.Tensor":
    """Return log probabilities for the target token at each position.

    This helper mirrors the internal TRL implementation and is factored out
    for readability.  ``logits`` is expected to have shape ``[B, T, V]``
    where ``V`` is the vocabulary size.  ``target_ids`` has shape
    ``[B, T]`` containing the token IDs whose log probabilities should
    be selected.  Entries in ``target_ids`` that are equal to ``-100``
    (the default ``ignore_index``) are ignored; however, callers must
    ensure that positions with ``-100`` are never gathered, perhaps by
    replacing them with a valid index before calling this function.

    :param logits: raw logits from the model
    :param target_ids: token IDs to select per position
    :returns: a tensor of shape ``[B, T]`` containing the log
              probability of the target token at each position
    """
    if torch is None or nn is None:
        raise RuntimeError(
            "DFT loss requested but PyTorch is not available. Please install torch to use this plugin."
        )
    # compute log softmax along the vocabulary dimension for numerical stability
    logprobs_all = nn.functional.log_softmax(logits, dim=-1)
    # gather per position the logprob of the target token
    return logprobs_all.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


def _dft_loss(outputs: Any, labels: "torch.Tensor", num_items_in_batch: int) -> "torch.Tensor":
    """Compute the Dynamic Fine‑Tuning (DFT) loss.

    The loss is computed per token as ``-stop_grad(p) * log p`` where
    ``p = exp(log p)`` and the gradient is stopped on ``p``.  Padding
    tokens (with value ``-100``) are masked out and the loss is
    normalised by the number of valid (non‑padding) tokens in the
    batch.  This formulation matches TRL v0.23.0.

    :param outputs: the model output object.  Must have a ``logits``
                    attribute with shape ``[B, T, V]``.
    :param labels: the target tokens used for teacher forcing.  Shape
                   ``[B, T]`` with ``-100`` marking positions to
                   ignore.  See ``ignore_index`` in PyTorch.
    :param num_items_in_batch: the total number of non‑padding tokens
                               across the batch.  Used to average
                               per token loss.
    :returns: the scalar DFT loss for the batch
    """
    if torch is None or nn is None:
        raise RuntimeError(
            "DFT loss requested but PyTorch is not available. Please install torch to use this plugin."
        )
    # shift labels right and pad so that each token is predicted based on
    # the previous token.  See TRL implementation for details.
    labels = nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous()
    # create a mask for positions that should contribute to the loss
    loss_mask = shift_labels != -100
    # ensure indices are valid; replace ignored positions with zero for gather
    safe_labels = shift_labels.clone()
    safe_labels[~loss_mask] = 0
    # compute log p for the selected token at each position
    logprobs = _selective_log_softmax(outputs.logits, safe_labels)
    # stop gradient on the probability (exp(log p)) and multiply by log p
    per_token_loss = -logprobs.exp().detach() * logprobs
    # sum over tokens and normalise by number of valid tokens
    if num_items_in_batch is None:
        num_items_in_batch = loss_mask.sum()
    loss = (per_token_loss * loss_mask).sum() / max(1, num_items_in_batch)
    return loss


@dataclass
class DFTPlugin(BasePlugin):
    """Plugin that injects the DFT loss into a HuggingFace ``Trainer``.

    The plugin expects to be invoked by Axolotl during trainer creation.
    When enabled (see ``dft_enabled``), it monkey patches the trainer's
    ``compute_loss`` method so that, when labels are present in the
    ``inputs`` dict, the DFT loss is computed instead of the default
    cross‑entropy.  If ``torch`` is unavailable and DFT is enabled, an
    exception will be raised when the loss is actually computed.  If
    disabled, the plugin has no effect.
    """

    dft_enabled: bool = True

    def __init__(self, dft: Optional[Dict[str, Any]] = None, **_: Any) -> None:
        """Initialise the plugin.

        :param dft: optional dictionary with configuration options.  If
                    provided and contains an ``enabled`` key, its
                    boolean value will override the default.
        """
        self.dft_enabled = True #if (dft is None) else bool(dft.get("enabled", True))
        self.first_log = True
        # if dft is not None:
        #     self.dft_enabled = bool(dft.get("enabled", True))

    # compatibility with older plugin interface; alias for dft_enabled
    def __bool__(self) -> bool:  # pragma: no cover - not used in tests
        return self.dft_enabled

    # # If you don’t add CLI flags, return an empty list.
    # def get_input_args(self) -> str:
    #     return "dft.args.DFTArgs"

    # # NEW: Axolotl calls this during config load
    # def register(self, cfg: dict):
    #     # read simple toggle from config: dft.enabled: true/false
    #     dft_cfg = (cfg or {}).get("dft", {}) if isinstance(cfg, dict) else {}
    #     self.dft_enabled = bool(dft_cfg.get("enabled", True))


    def _patch_trainer_loss(self, trainer: Any) -> None:
        """Internal helper to monkey‑patch a HF ``Trainer``'s ``compute_loss``.

        This method applies the DFT loss by wrapping the original
        ``trainer.compute_loss``.  When labels are present in the
        ``inputs`` dict, it computes the DFT loss; otherwise, it
        delegates back to the original method.  This helper is kept
        separate so it can be invoked from multiple hooks (e.g.,
        ``on_trainer_create`` or ``post_trainer_create``) without
        duplicating logic.
        """
        if not self.dft_enabled:
            return

        # capture the ignore index used for padding; default in
        # transformers is -100
        ignore_index = -100

        # keep a reference to the original compute_loss so we can
        # delegate when no labels are provided
        original_compute_loss = trainer.compute_loss
        def compute_loss_dft(this_trainer: Any, model: Any, inputs: Dict[str, Any], num_items_in_batch=None):
            # Determine whether to use our custom loss or fall back
            labels = inputs.get("labels")
            if self.first_log:
                print("[dft] Using DFT loss (like TRL SFT 'dft')")
                # print(inputs)
                self.first_log = False
            if labels is None and "input_ids" in inputs:
                # derive labels == input_ids (classic causal LM); mask padding as -100 if present
                labels = inputs["input_ids"]
                inputs = {**inputs, "labels": labels}
            if labels is None:
                return original_compute_loss(model, inputs, num_items_in_batch)
                # return original_compute_loss(model, inputs, return_outputs)
            if torch is None or nn is None:
                raise RuntimeError(
                    "DFT loss requested but PyTorch is not available. Please install torch to use this plugin."
                )
            # run the model to obtain logits; omit labels from inputs when
            # calling the model to avoid double application of the loss
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**model_inputs)
            # compute number of valid tokens to normalise the loss
            with torch.no_grad():
                valid = (labels != ignore_index).sum().item()
            loss = _dft_loss(outputs, labels, num_items_in_batch)
            return loss

        # bind our compute_loss function to the trainer instance
        trainer.compute_loss = types.MethodType(compute_loss_dft, trainer)

    def on_trainer_create(self, trainer: Any, *args: Any, **kwargs: Any) -> None:
        """Deprecated hook for older versions of Axolotl.

        Prior versions of Axolotl invoked ``on_trainer_create`` once
        the ``Trainer`` was instantiated.  Newer versions use
        ``post_trainer_create`` instead.  This method calls into
        :meth:`_patch_trainer_loss` for backward compatibility.
        """
        # patch the trainer if enabled
        self._patch_trainer_loss(trainer)

    def post_trainer_create(self, cfg: Any, trainer: Any) -> None:
        """Hook invoked by Axolotl after the HuggingFace ``Trainer`` is built.

        The plugin manager calls this method for each registered
        integration, passing in the Axolotl configuration and the
        trainer instance.  This method applies the DFT loss patch to
        the trainer.  ``cfg`` is unused here but accepted for API
        compatibility.
        """
        # apply our loss patch
        self._patch_trainer_loss(trainer)
