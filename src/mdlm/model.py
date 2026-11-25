"""One Transformer, two ways to model text.

- **AR**: causal attention, predict token i+1 from tokens ≤ i. Exact likelihood.
- **Masked diffusion** (absorbing state; Austin et al. 2021, Sahoo et al. 2024 "MDLM",
  Nie et al. 2025 "LLaDA"). The forward process replaces each token independently by
  ``[MASK]`` with probability t ∈ [0, 1]. A *bidirectional* network predicts the original
  token at every masked position. With the linear schedule (α_t = 1 − t) the continuous-time
  negative ELBO is

      NELBO(x) = E_{t∼U(0,1)} [ (1/t) Σ_{i masked} −log p_θ(x_i | x_t) ]

  This upper-bounds −log p(x), so it is directly comparable with AR's exact NLL.

Both models have the same parameters apart from the attention mask and one extra input
token (``[MASK]``) for the diffusion model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

VOCAB = 256  # bytes
MASK = 256  # absorbing state (diffusion input only)


@dataclass
class Config:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    ctx: int = 256
    causal: bool = False


class Block(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.h, self.causal = c.n_heads, c.causal
        self.n1, self.n2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(c.d_model, 4 * c.d_model), nn.GELU(), nn.Linear(4 * c.d_model, c.d_model)
        )

    def forward(self, x):
        b, l, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, l, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        x = x + self.proj(y.transpose(1, 2).reshape(b, l, d))
        return x + self.mlp(self.n2(x))


class Transformer(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(VOCAB + 1, c.d_model)
        self.pos = nn.Embedding(c.ctx, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, VOCAB, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


# --------------------------------------------------------------------------------------- AR
def ar_loss(model, x):
    """Mean NLL (nats/token) of x[1:] given the prefix. The first token isn't scored."""
    logits = model(x[:, :-1])
    return F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten())


@torch.no_grad()
def ar_sample(model, n, length, device, temperature=1.0, gen=None, prefix=None):
    x = (
        prefix
        if prefix is not None
        else torch.randint(0, VOCAB, (n, 1), device=device, generator=gen)
    )
    while x.shape[1] < length:
        p = (model(x)[:, -1].float() / temperature).softmax(-1)
        x = torch.cat([x, torch.multinomial(p, 1, generator=gen)], 1)
    return x


# --------------------------------------------------------------------------------- diffusion
def forward_mask(x, t, gen=None):
    """Mask each token independently with probability t (b,). Returns (x_t, mask)."""
    u = torch.rand(x.shape, device=x.device, generator=gen)
    m = u < t[:, None]
    return torch.where(m, torch.full_like(x, MASK), x), m


def diffusion_loss(model, x, t=None, gen=None, eps=1e-3):
    """Monte-Carlo NELBO per token (nats) with weight 1/t on masked positions."""
    b = x.shape[0]
    if t is None:  # stratified t over the batch: same expectation, lower variance
        u = torch.rand((), device=x.device, generator=gen)
        t = ((torch.arange(b, device=x.device) + u) / b).clamp(min=eps)
    xt, m = forward_mask(x, t, gen)
    logits = model(xt).float()
    ce = F.cross_entropy(logits.transpose(1, 2), x, reduction="none")  # (b, l)
    return ((ce * m).sum(1) / t).mean() / x.shape[1]


@torch.no_grad()
def diffusion_sample(
    model, n, length, steps, device, gen=None, given=None, temperature=1.0, order="random"
):
    """Reverse process from all-[MASK] (MDLM ancestral sampler).

    Going from time t to s < t, each still-masked token is revealed with probability
    (t − s)/t, drawn from the model's prediction. Revealed tokens are never changed again.
    ``order="confidence"`` instead reveals the same *number* of tokens, choosing the most
    confident positions (LLaDA's low-confidence remasking). ``given`` (n, length) with
    −1 for unknown fixes known tokens, which gives infilling.
    """
    x = torch.full((n, length), MASK, dtype=torch.long, device=device)
    if given is not None:
        x = torch.where(given >= 0, given, x)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t, s = ts[i], ts[i + 1]
        masked = x == MASK
        if not masked.any():
            break
        p = (model(x).float() / temperature).softmax(-1)
        draw = torch.multinomial(p.flatten(0, 1), 1, generator=gen).view(n, length)
        frac = (t - s) / t
        if order == "random":
            reveal = masked & (torch.rand(x.shape, device=device, generator=gen) < frac)
        else:
            conf = p.gather(-1, draw[..., None])[..., 0].masked_fill(~masked, -1.0)
            k = torch.ceil(masked.sum(1).float() * frac).long()
            rank = conf.argsort(1, descending=True).argsort(1)
            reveal = masked & (rank < k[:, None])
        if i == steps - 1:
            reveal = masked
        x = torch.where(reveal, draw, x)
    return x
