Tesla T4; both models 4,929,792 params, 6000 steps × 32 × 256 bytes, identical optimiser and schedule.

| model | val bits/byte |
|---|---|
| autoregressive (exact) | 1.224 |
| masked diffusion (NELBO, upper bound) | 2.745 ± 0.018 |

| sampler | AR-judged bits/byte | byte entropy (bits) | seconds (32 × 256 B) |
|---|---|---|---|
| real val data | 1.248 | 4.50 | 0.0 |
| AR, T=1 (256 sequential steps) | 1.122 | 4.56 | 4.3 |
| diffusion, 8 steps, random order | 6.363 | 4.89 | 0.3 |
| diffusion, 32 steps, random order | 5.977 | 4.87 | 1.1 |
| diffusion, 128 steps, random order | 5.636 | 4.69 | 4.3 |
| diffusion, 256 steps, random order | 5.667 | 4.82 | 8.6 |
| diffusion, 8 steps, confidence order | 0.597 | 0.26 | 0.3 |
| diffusion, 32 steps, confidence order | 0.288 | 0.04 | 1.1 |
| diffusion, 128 steps, confidence order | 0.225 | 0.01 | 4.3 |
| diffusion, 256 steps, confidence order | 0.223 | 0.01 | 8.5 |

Infilling the middle 64 of 256 bytes, exact-byte accuracy: diffusion (sees both sides) **29.7%**, AR (prefix only) **20.7%**.
