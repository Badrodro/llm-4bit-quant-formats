# 4-bit LLM Weight Formats: NVFP4, MXFP4, INT4, Lloyd-Max and Trellis Codes at Equal Perplexity

Hardware accelerators are moving to 4-bit weights. This repository measures, on three real LLMs, how much quality each 4-bit weight format keeps, and whether a trellis-coded quantizer (QTIP-style) saves enough bits over the best *scalar* format to be worth its decoder in hardware.

**Short answer:** at equal perplexity, a trellis code needs 11–17 % fewer bits than NVFP4 and about 15–20 % fewer than INT4 g128 and MXFP4. But a plain scalar 4-bit Lloyd-Max quantizer with a random Hadamard rotation already gets within **6–9 %** of the trellis. The decision rule fixed before the measurements (at least 10 % saving against the best scalar format, on every model and corpus) fails, so a non-uniform scalar format plus rotation captures most of the gain at a much lower decoding cost.

## Setup

| Item | Value |
| :--- | :--- |
| Models | Qwen2.5-0.5B, Qwen2.5-1.5B, SmolLM2-1.7B (base models, BF16 reference) |
| Quantized weights | every linear layer of the transformer blocks (q, k, v, o, gate, up, down); embeddings and LM head stay in BF16; activations stay in BF16 |
| Evaluation | perplexity on WikiText-2 (full test set) and C4 (128 validation windows), 2048-token windows |
| Rounding | round-to-nearest for scalar formats, Viterbi for the trellis; **no calibration data** for any format |
| Bits per weight (bpw) | include every scale and the initial trellis state |
| Hardware | NVIDIA RTX 5060 Ti 16 GB, PyTorch 2.14 (CUDA 13.0), transformers 5.17 |

### Formats

| Format | Values | Scale | bpw |
| :--- | :--- | :--- | ---: |
| INT4 g128 | symmetric INT4 | FP16 per 128 weights | 4.125 |
| MXFP4 (OCP MX) | E2M1 | power of two (E8M0) per 32 | 4.25 |
| NVFP4 | E2M1 | FP8 E4M3 per 16 + FP32 per tensor | 4.5 |
| NVFP4 + scale search | E2M1 | same, best of 8 scale multipliers per block | 4.5 |
| Lloyd-Max 4b / 3b g32 | MSE-optimal levels for a Gaussian | FP8 E4M3 per 32, searched per block | 4.25 / 3.25 |
| Trellis (k, V) | bitshift trellis, random Gaussian code, L = 14 | FP16 per row | ≈ 3.0 / 3.5 / 4.0 |

Every scalar format is evaluated **with and without** a random Hadamard rotation (RHT), and the better of the two is kept, so the comparison is fair to the scalar formats. The trellis always uses the rotation.

## Results

### Perplexity (best variant of each scalar format)

| Format | bpw | Qwen2.5-0.5B wiki / C4 | Qwen2.5-1.5B wiki / C4 | SmolLM2-1.7B wiki / C4 |
| :--- | ---: | ---: | ---: | ---: |
| BF16 | 16 | 13.08 / 19.69 | 9.27 / 14.58 | 8.12 / 12.57 |
| NVFP4 | 4.5 | 14.37 / 21.44 | 9.89 / 15.43 | 8.91 / 13.69 |
| NVFP4 + scale search | 4.5 | 14.31 / 21.25 | 9.80 / 15.33 | 8.88 / 13.63 |
| MXFP4 | 4.25 | 15.87 / 23.59 | 10.45 / 16.34 | 10.77 / 16.32 |
| INT4 g128 | 4.125 | 15.94 / 23.61 | 10.48 / 16.40 | 10.07 / 15.22 |
| Lloyd-Max 4b g32 | 4.25 | 14.25 / 21.44 | 9.79 / 15.36 | 9.08 / 13.89 |
| Lloyd-Max 3b g32 | 3.25 | 20.78 / 29.47 | 12.33 / 18.68 | 18.84 / 28.23 |
| Trellis k=4 V=1 | ≈ 4.02 | 14.18 / 21.12 | 9.67 / 15.17 | 8.85 / 13.58 |
| Trellis k=7 V=2 | ≈ 3.51 | 15.37 / 22.68 | 10.08 / 15.84 | 9.79 / 14.84 |
| Trellis k=3 V=1 | ≈ 3.02 | 18.04 / 26.38 | – | – |

Best variant: the rotation helps Lloyd-Max (all models), INT4 and MXFP4 (most cases), and slightly hurts NVFP4. The full table with both variants is in [`results/`](results).

### Bits saved by the trellis at equal perplexity

Bits per weight the trellis needs to match each scalar format, interpolated in log-perplexity between the two trellis points that bracket it. "≥" means that the lowest trellis point measured already beats the scalar format, so the saving is a lower bound.

| Against | Qwen2.5-0.5B wiki / C4 | Qwen2.5-1.5B wiki / C4 | SmolLM2-1.7B wiki / C4 |
| :--- | ---: | ---: | ---: |
| NVFP4 | 12.5 % / 13.0 % | 16.9 % / 15.1 % | 11.5 % / 11.8 % |
| NVFP4 + scale search | 11.9 % / 11.6 % | 14.5 % / 13.5 % | 11.1 % / 11.2 % |
| MXFP4 | 19.5 % / 20.2 % | ≥ 17.3 % / ≥ 17.3 % | ≥ 17.3 % / ≥ 17.3 % |
| INT4 g128 | 17.4 % / 17.9 % | ≥ 14.8 % / ≥ 14.8 % | ≥ 14.8 % / ≥ 14.8 % |
| **Lloyd-Max 4b g32 + RHT** | **6.0 % / 7.8 %** | **9.0 % / 8.9 %** | **8.5 % / 8.6 %** |
| Lloyd-Max 3b g32 + RHT | ≥ 6.9 % / ≥ 6.9 % | n/a | n/a |

(n/a: no 3-bit trellis point was run on these two models.)

### What this means for hardware

- **Uniform-grid 4-bit formats leave quality on the table.** MXFP4 and INT4 g128 lose clearly more perplexity than NVFP4. NVFP4's finer scales (one per 16 weights) are worth their extra 0.25 bpw.
- **Most of the trellis gain comes from rotation and non-uniform levels, not from the trellis itself.** A scalar Lloyd-Max 4-bit quantizer with rotation (4.25 bpw) matches or beats NVFP4 (4.5 bpw) on both Qwen models, but not on SmolLM2-1.7B.
- **The trellis adds 6–9 % on top**, below the 10 % threshold set in advance. That is not enough to justify a Viterbi-coded weight decoder next to every multiply-accumulate array.

## Limitations

- Round-to-nearest only: no GPTQ, AWQ or other calibration-based method, for any format. Calibration would raise every format, and the gaps between formats could change.
- Simulated quantization (quantize, then dequantize to BF16): no kernels, so no speed or energy measurement.
- Weights only; small models (0.5 to 1.7 B parameters); one seed for the random rotation and the trellis code.
- C4 uses 128 validation windows, not the full validation set.

## Reproduce

Needs an NVIDIA GPU (16 GB was enough for all three models). Quantizing one model with the trellis takes 3 to 21 minutes per configuration on an RTX 5060 Ti.

```bash
pip install -r requirements.txt
python download.py
python eval_perplexity.py Qwen/Qwen2.5-0.5B --extra
python eval_perplexity.py Qwen/Qwen2.5-1.5B
python eval_perplexity.py HuggingFaceTB/SmolLM2-1.7B
python analyze.py
```

`eval_perplexity.py` saves its results after each format and skips formats already present in `results/<model>.json`: delete that file to recompute from scratch. `--quick` evaluates only 16 windows per corpus, for a fast check.

| File | Contents |
| :--- | :--- |
| `formats.py` | scalar formats: INT4 g128, MXFP4, NVFP4, Lloyd-Max |
| `eval_perplexity.py` | random Hadamard rotation, trellis quantizer (Viterbi on GPU), perplexity evaluation |
| `analyze.py` | equal-perplexity comparison and decision rule |
| `download.py` | models and corpora from Hugging Face |
| `results/` | raw results for each model and the analysis |

## References

- A. Tseng et al., *QTIP: Quantization with Trellises and Incoherence Processing*, 2024, [arXiv:2406.11235](https://arxiv.org/abs/2406.11235).
- A. Tseng et al., *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks*, 2024, [arXiv:2402.04396](https://arxiv.org/abs/2402.04396).
- Open Compute Project, *OCP Microscaling Formats (MX) Specification v1.0*, 2023.
- S. Lloyd, *Least squares quantization in PCM*, IEEE Trans. Information Theory, 1982; J. Max, *Quantizing for minimum distortion*, IRE Trans. Information Theory, 1960.

## How this was made

I use AI assistants (Claude) in my projects; I choose the questions and check the results.

## Author

Badreddine Benfettouma, undergraduate in electronics at Université Bourgogne Europe. Website: [badrodro.github.io](https://badrodro.github.io). See also my RISC-V core: [RV32I-Alpha](https://github.com/Badrodro/RV32I-Alpha).
