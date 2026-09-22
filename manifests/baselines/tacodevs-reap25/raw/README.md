---
license: mit
base_model: XiaomiMiMo/MiMo-V2.6-Flash-RL
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- apple-silicon
- mimo-v2
- mixture-of-experts
- gptq
- reap
- mtp
---

# MiMo-V2.6-Flash-RL · MLX · fits a 128 GB Mac

A compressed [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) (309B / 15B-active MoE) in **stock mlx-lm format**,
sized so the text model loads on a **128 GB Apple Silicon** machine (M3 Max / M4 Max / Ultra).
Text model on disk: **100.4 GB** (192 experts/layer, experts average 3.29 bits/weight). Bundled MTP / DFlash / vision / audio weights: 7.3 GB (not loaded for text inference).

## What was done

The official checkpoint's experts are *natively* MXFP4 (Xiaomi trained them with MXFP4 QAT; there is no BF16 release), so 4-bit MLX conversions are lossless and anything smaller is a real re-quantization. This model was produced by a calibrated, layer-wise pipeline (PyTorch, one H200):

1. **REAP expert pruning**: the 64 least salient experts per layer (25.0%) were removed (saliency = mean routing weight × ‖expert output‖ on calibration data, per [REAP, arXiv:2510.13999](https://arxiv.org/abs/2510.13999)); 192 of 256 experts remain in every MoE layer.
2. **Sensitivity-driven precision allocation** (GEMQ-style, [arXiv:2605.23078](https://arxiv.org/abs/2605.23078)): for every MoE layer and projection (gate/up/down), the Hessian-weighted output error of 2-bit / 3-bit candidates was measured on calibration activations and a MILP picked the mix under the size budget. Result: 93× 3-bit affine g128, 22× 4-bit mxfp4 (native), 18× 2-bit affine g128, 8× 3-bit affine g64.
3. **GPTQ** (sequential, error propagated through already-compressed layers) for every projection not kept at native MXFP4, using activation Hessians weighted by routing weights.
4. Attention, dense MLP (layer 0), embeddings and lm_head: **8-bit** affine, group 64. `attention_value_scale` is folded into `v_proj` (the mlx-lm class has none).
5. Native MTP head (`mtp/`), DFlash drafter (`dflash/`), vision & audio encoders (`omnimodal/`, `audio_tokenizer/`) are carried over from [Vontra's conversion](https://huggingface.co/Vontra/MiMo-V2.6-Flash-RL-MLX-4bit-MTP) unchanged; the checkpoint layout is theirs, so whatever loads that model loads this one.

Calibration: 256 sequences × 2048 tokens from evol-codealpaca, Mixture-of-Thoughts, SWE-smith trajectories, glaive function calling and UltraChat, rendered with the model's chat template. Held-out evaluation uses disjoint samples from the same mix.

## Quality

| Metric (held-out agentic/coding mix, 31×2048 tokens) | Original (MXFP4/FP8) | This model |
|---|---|---|
| Perplexity | 9.271 | 9.529 |
| KL(original ‖ this), mean per token | 0 | 0.8315 |
| Top-1 next-token agreement with original | 100% | 79.5% |

**On-policy** (40 responses sampled from the real MiMo-V2.6-Flash via API, only assistant tokens scored):

| group | tokens | original NLL | this model NLL | Δ | KL | top-1 agree |
|---|---|---|---|---|---|---|
| ALL | 24136 | 0.530 | 0.609 | +0.079 | 0.105 | 90.4% |
| code | 10627 | 0.511 | 0.589 | +0.078 | 0.106 | 90.8% |
| agent | 4786 | 0.695 | 0.779 | +0.084 | 0.126 | 88.5% |
| reasoning | 3815 | 0.313 | 0.358 | +0.045 | 0.052 | 94.2% |
| general | 4908 | 0.580 | 0.682 | +0.102 | 0.125 | 88.5% |

On-policy NLL is the most trustworthy number here: a compressed model that reproduces the original's own outputs has not drifted. Any perturbation of this MoE (even 8-bit attention) sits at KL≈0.4 on foreign text because top-8 routing flips, so only on-policy deltas are comparable across variants.

These are distribution-level numbers against the original model on the calibration domain; they are not benchmark scores. Expect a real capability loss versus the 4-bit original — measure on your task.

## Running it (128 GB Mac)

```bash
pip install -U mlx-lm
# default GPU wired limit is ~75% of RAM; allow the model + KV cache
sudo sysctl iogpu.wired_limit_mb=118000
hf download tacodevs/MiMo-V2.6-Flash-RL-MLX-REAP25-mixed3bit-GPTQ-MTP --local-dir MiMo-V2.6-Flash-RL-MLX-REAP25-mixed3bit-GPTQ-MTP
python -m mlx_lm generate --model MiMo-V2.6-Flash-RL-MLX-REAP25-mixed3bit-GPTQ-MTP --prompt "Write a Python function that checks whether an integer is prime." --max-tokens 256 --temp 0.6
```
Close other memory-hungry apps first. The MTP / DFlash payloads are packaged for MiMo-aware runtimes; stock mlx-lm decodes serially and does not use them yet.

## Files
- `model-*.safetensors`, `model.safetensors.index.json`, `config.json` — text model (this work)
- `compression_alloc.json`, `compression_eval.json` — per-layer precision map, pruned expert ids, evaluation
- `mtp/`, `dflash/`, `omnimodal/`, `audio_tokenizer/` — upstream auxiliary weights (Vontra)

Pipeline source: https://github.com/irvollo/mimo-mlx-compress (REAP saliency, sensitivity, MILP allocation, batched GPTQ, MLX packing).

## Credits
Xiaomi MiMo team (model, MIT license); Vontra (MLX layout, MTP packaging); Cerebras (REAP); Deng et al. (GEMQ); Frantar et al. (GPTQ).
