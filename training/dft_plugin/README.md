# Axolotl DFT Plugin

Bring **Dynamic Fine‑Tuning (DFT)** to your Axolotl SFT runs.

DFT was added to TRL v0.23.0 as an alternative loss to NLL and can be enabled. This plugin mirrors that behavior for **Axolotl** by swapping the HF Trainer’s `compute_loss` to use DFT during supervised fine‑tuning.

- TRL SFT Trainer docs: DFT aims to improve generalization by rectifying the reward signal and uses the usual one‑token shift with `ignore_index=-100` for masking.

> References  
> • TRL SFT Trainer docs – DFT description and `loss_type` values (`"nll"`, `"dft"`), label shifting and masking details.

---

## What this plugin does

- Hooks Axolotl’s HF `Trainer` **after** it is created (`on_trainer_create`).
- Replaces `Trainer.compute_loss` with a DFT loss variant when a batch contains `labels`.
- Leaves everything else (optimizer, schedulers, packing, PEFT/LoRA) **unchanged**.

### DFT loss (high level)
Let `log p_t` be the log‑probability of the target token at step *t* (after the one‑token shift). DFT computes:

```
loss_t = - stop_grad(exp(log p_t)) * log p_t
```

Mask out positions where `labels == -100` and average across **all valid tokens** in the batch. This mirrors TRL’s implementation.

---

## Requirements

- Python ≥ 3.10
- Axolotl (latest recommended)
- PyTorch ≥ 2.1 (CPU or CUDA)  
- transformers ≥ 4.40

> The unit tests included in the repo also provide a **torch‑free smoke test** that validates the trainer‑hooking logic (useful for constrained environments).

---

## Installation

### Option A — local editable install (recommended during development)
```bash
git clone <this-repo> axolotl-dft-plugin
cd axolotl-dft-plugin
pip install -e .
```

### Option B — vendor into an existing repo
Copy the `axolotl_dft_plugin/` folder into your source tree and ensure it’s on `PYTHONPATH`.

---

## Enable in Axolotl

Add this to your Axolotl YAML:

```yaml
plugins:
  - axolotl_dft_plugin.DFTPlugin

dft:
  enabled: true
```

Run training as usual, e.g.

```bash
axolotl train configs/your_sft_config.yaml
```

If `dft.enabled: true`, the plugin patches the trainer to compute the DFT loss automatically for SFT‑style runs.

---

## How it works (implementation notes)

1. **One‑token shift & masking**: Inputs are shifted by one to create labels; padding or masked positions use `ignore_index = -100` and are excluded from loss.  
2. **Selective log‑softmax**: Compute `log_softmax` over the vocabulary and gather the log‑prob only for the target token at each position.  
3. **Stop‑gradient on probabilities**: Convert log‑probs to probs with `exp(log p)` and **detach** before multiplying by `log p`.  
4. **Normalization**: Sum over valid positions and divide by the number of valid tokens across the batch.

The plugin overrides `Trainer.compute_loss` and falls back to the original method if `labels` are missing in a batch (e.g., during eval without labels).

---

## Configuration

```yaml
dft:
  enabled: true   # default True; disable to revert to the original NLL loss
```

No other knobs are required. The rest of your Axolotl config (packing, PEFT/LoRA, logging, etc.) remains the same.

---

## Testing

### 1) Hooking smoke test (no torch required)
Runs a dummy trainer + model and verifies that the plugin’s patched `compute_loss` path is taken when `labels` are present, and that it falls back otherwise.

```bash
python tests/test_dft_plugin.py
```

### 2) Numeric test (requires torch)
If PyTorch is available, you can temporarily un‑stub `_dft_loss` in the unit test and verify gradients flow and parameters update during a couple of steps with a tiny model (e.g., `sshleifer/tiny-gpt2`).

---

## Troubleshooting

- **“Loss didn’t change / identical to NLL”**: Ensure `dft.enabled: true` and that your dataset actually provides `labels` (Axolotl’s collators usually do).  
- **“Trainer compute_loss not patched”**: Confirm the plugin is listed under `plugins` in the YAML and that Axolotl logs show plugins being initialized.  
- **Shape or dtype errors**: Check that `labels` are `LongTensor` with `-100` for ignored tokens and align with the logits’ sequence length (after right‑shift).

---

## Notes on compatibility

- Designed for SFT‑style training (i.e., next‑token prediction). It doesn’t alter DPO/GRPO/RL trainers.  
- Works with PEFT/LoRA and common memory/perf integrations (packing, Liger, etc.) since the plugin only swaps the loss function.  
- If you also use loss‑altering integrations (e.g., Cut Cross‑Entropy), ensure only one component overrides `compute_loss` at a time.

---

## Acknowledgements

- **Hugging Face TRL team** for implementing DFT in TRL v0.23.0 and documenting it in the SFT Trainer docs.

---

## License

MIT (or the license of your choice).

