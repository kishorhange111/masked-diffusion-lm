# masked-diffusion-lm

[![CI](https://github.com/kishorhange111/masked-diffusion-lm/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/masked-diffusion-lm/actions/workflows/ci.yml)

A masked (absorbing-state) discrete diffusion language model from scratch, in the style of
MDLM (Sahoo et al. 2024) and LLaDA (Nie et al. 2025). It is compared against an
autoregressive Transformer with **the same parameters, data and compute**.

## The model ([`model.py`](src/mdlm/model.py))

**Forward process.** Each token is independently replaced by `[MASK]` with probability
*t* ∈ [0, 1].

**Training.** A *bidirectional* Transformer predicts the original token at every masked
position. With the linear schedule, the continuous-time negative ELBO is

```
NELBO(x) = E_{t~U(0,1)} [ (1/t) · Σ_{i masked} −log p_θ(x_i | x_t) ]  ≥  −log p(x)
```

This is a cross-entropy reweighted by 1/t, so the diffusion model's likelihood bound is
directly comparable with the AR model's exact NLL. *t* is stratified across the batch to
reduce variance.

**Sampling (reverse process).** Start from all `[MASK]`. Going from *t* to *s*, each masked
token is revealed with probability (t − s)/t, drawn from the model (the MDLM ancestral
sampler). Revealed tokens never change. LLaDA's variant reveals the same number of tokens
but picks the most confident positions. Fewer steps means faster generation, since tokens
are generated in parallel, but at lower quality. Fixing known tokens turns the same sampler
into **infilling**, which a left-to-right model can't do natively.

The two models differ only in the attention mask and the extra `[MASK]` input embedding.

## Tests

There are 10, on CPU. They check:
- the masking rate equals *t*;
- **a uniform denoiser's NELBO is exactly 8 bits/byte**, as the 1/t weighting requires;
- an oracle denoiser has NELBO ≈ 0 and its sampler reconstructs the sequence exactly, for
  both unmasking orders;
- infilling never alters the given tokens;
- confidence order reveals the most confident position first;
- the AR model is causal, the diffusion model is not, and both have equal parameter counts;
- the end-to-end pipeline runs and resumes.

## Experiment ([`train.py`](src/mdlm/train.py))

Both models are 4.8M-parameter Transformers (6 layers, d = 256) trained on 256-byte windows
of Python standard-library source. They get the same 6,000 steps, batch, optimiser and
schedule. The script measures:
1. **Likelihood:** AR's exact bits/byte against the diffusion model's NELBO bound.
2. **Generation:** 32 samples of 256 bytes from AR (T = 1) and from diffusion at 8–256 steps
   with both unmasking orders. Each set is scored by the trained AR model (bits/byte), by
   byte entropy (to catch repetitive degenerate text) and by wall time.
3. **Infilling:** the middle 64 of 256 bytes are hidden, and the score is exact-byte
   accuracy. Diffusion sees both sides; AR sees the prefix only.

```
pytest -q                                          # 10 tests, CPU
python -m mdlm.train --out results --device cuda   # resumable (checkpoints + results.json)
```

## Results (Tesla T4, Colab)

Both models have 4,929,792 parameters and train for 6,000 steps × 32 × 256 bytes, about
3.5 minutes each. The output is in [`results/`](results/).

**Likelihood**

| model | val bits/byte |
|---|---|
| autoregressive (exact NLL) | **1.224** |
| masked diffusion (NELBO, an upper bound) | 2.745 ± 0.018 |

**Generation**: 32 samples × 256 bytes, judged by the trained AR model. Real validation
text has byte entropy 4.50 bits and judged 1.248 bits/byte.

| sampler | AR-judged bits/byte | byte entropy (bits) | seconds |
|---|---|---|---|
| AR, T = 1 (256 sequential steps) | 1.122 | 4.56 | 4.3 |
| diffusion, 8 steps, random order | 6.363 | 4.89 | 0.3 |
| diffusion, 32 steps, random order | 5.977 | 4.87 | 1.1 |
| diffusion, 128 steps, random order | 5.636 | 4.69 | 4.3 |
| diffusion, 256 steps, random order | 5.667 | 4.82 | 8.6 |
| diffusion, 8 steps, confidence order | 0.597 | **0.26** | 0.3 |
| diffusion, 32 steps, confidence order | 0.288 | 0.04 | 1.1 |
| diffusion, 128 steps, confidence order | 0.225 | 0.01 | 4.3 |
| diffusion, 256 steps, confidence order | 0.223 | 0.01 | 8.5 |

**Infilling** the middle 64 of 256 bytes, scored as exact-byte accuracy: **diffusion,
seeing both sides, 29.7%**; AR, seeing the prefix only, 20.7%.

What the numbers show:
- **At equal parameters and compute, masked diffusion is far behind on likelihood.** Its
  bound is 2.745 bits/byte against AR's 1.224, more than double. Part of that is the bound
  itself (an ELBO, not an exact NLL). Part is that the diffusion model must learn every
  conditional `p(x_i | any visible subset)`, not just left-to-right ones, from the same 6,000
  steps. The MDLM and LLaDA papers close much of this gap with much larger models and far
  longer training; this tiny budget does not.
- **Random-order sampling gives incoherent text** (judged 5.6–6.4 bits/byte, against 1.25
  for real text), and more steps barely help.
- **Confidence-order sampling looks excellent and is degenerate.** It is judged 0.22
  bits/byte, better than real text, but its byte entropy is 0.01 bits. The samples are
  essentially one repeated byte. Revealing the most confident position first snowballs into
  the most predictable possible text. This is why the run measures entropy: **judged
  likelihood alone would have declared the broken sampler the best one.**
- **Parallel decoding trades steps for time as advertised.** 8 steps take 0.3 s against AR's
  4.3 s for the same 256 bytes. At this model quality, though, the fast samples are not
  usable.
- **Infilling is the one task where diffusion wins**, 29.7% against 20.7%, because it
  conditions on the text after the hole as well as before it. A left-to-right model cannot
  do that without retraining (as fill-in-the-middle models are).
