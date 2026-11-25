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

**Results are pending** from the Colab run.
