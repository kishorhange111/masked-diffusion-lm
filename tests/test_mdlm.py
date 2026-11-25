import math

import pytest
import torch
from torch import nn

from mdlm.model import (
    MASK,
    VOCAB,
    Config,
    Transformer,
    ar_loss,
    diffusion_loss,
    diffusion_sample,
    forward_mask,
)
from mdlm.train import byte_entropy, main


class Oracle(nn.Module):
    """A denoiser that knows the clean sequence: near-one-hot logits on the true byte."""

    def __init__(self, x0, sharp=50.0):
        super().__init__()
        self.x0, self.sharp = x0, sharp

    def forward(self, xt):
        return torch.nn.functional.one_hot(self.x0, VOCAB).float() * self.sharp


class Uniform(nn.Module):
    def forward(self, xt):
        return torch.zeros(*xt.shape, VOCAB)


def _x(b=4, l=32, seed=0):
    return torch.randint(0, VOCAB, (b, l), generator=torch.Generator().manual_seed(seed))


def test_forward_mask_rate():
    x = _x(4, 4000)
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    xt, m = forward_mask(x, t, torch.Generator().manual_seed(1))
    rates = m.float().mean(1)
    torch.testing.assert_close(rates, t, atol=0.03, rtol=0)
    assert (xt[m] == MASK).all() and (xt[~m] == x[~m]).all()


def test_nelbo_of_uniform_denoiser_is_8_bits_per_byte():
    """Each masked byte costs ln 256 nats and E[(1/t)·#masked] = L: NELBO = ln 256 exactly
    in expectation (stratified t keeps the estimate close)."""
    x = _x(64, 64)
    est = diffusion_loss(Uniform(), x, gen=torch.Generator().manual_seed(2)).item()
    assert est / math.log(2) == pytest.approx(8.0, rel=0.05)


def test_nelbo_of_oracle_is_near_zero():
    x = _x()
    assert diffusion_loss(Oracle(x), x).item() < 1e-6


@pytest.mark.parametrize("order", ["random", "confidence"])
def test_oracle_sampling_reconstructs_and_fills_every_mask(order):
    x = _x(3, 40)
    s = diffusion_sample(Oracle(x), 3, 40, 7, "cpu", torch.Generator().manual_seed(0), order=order)
    assert (s != MASK).all()
    torch.testing.assert_close(s, x)


def test_infilling_keeps_given_tokens():
    x = _x(2, 30)
    given = x.clone()
    given[:, 10:20] = -1
    s = diffusion_sample(Uniform(), 2, 30, 5, "cpu", torch.Generator().manual_seed(0), given=given)
    torch.testing.assert_close(s[:, :10], x[:, :10])
    torch.testing.assert_close(s[:, 20:], x[:, 20:])
    assert (s != MASK).all()


def test_confidence_order_reveals_most_confident_first():
    class Conf(nn.Module):  # position 0 very confident, others flat
        def forward(self, xt):
            out = torch.zeros(*xt.shape, VOCAB)
            out[:, 0, 7] = 30.0
            return out

    s = diffusion_sample(
        Conf(), 1, 4, 4, "cpu", torch.Generator().manual_seed(0), order="confidence"
    )
    assert s[0, 0] == 7


def test_ar_is_causal_and_diffusion_is_not():
    torch.manual_seed(0)
    c = Config(d_model=32, n_layers=1, n_heads=2, ctx=16)
    ar = Transformer(Config(**{**c.__dict__, "causal": True})).eval()
    dm = Transformer(c).eval()
    x = _x(1, 16)
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % VOCAB
    assert torch.allclose(ar(x)[:, :10], ar(y)[:, :10])
    assert not torch.allclose(dm(x)[:, :10], dm(y)[:, :10])
    assert sum(p.numel() for p in ar.parameters()) == sum(p.numel() for p in dm.parameters())
    assert ar_loss(ar, x).item() == pytest.approx(math.log(256), rel=0.05)  # untrained ≈ uniform


def test_byte_entropy():
    assert byte_entropy(torch.zeros(100, dtype=torch.long)) == 0.0
    assert byte_entropy(torch.arange(256)) == pytest.approx(8.0)


def test_tiny_pipeline_and_resume(tmp_path):
    args = ["--tiny", "--device", "cpu", "--out", str(tmp_path), "--steps", "3", "--bs", "4"]
    main(args)
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "NELBO" in text and "Infilling" in text and "confidence order" in text
    assert (tmp_path / "ar.pt").exists() and (tmp_path / "diffusion.pt").exists()
    main(args)  # loads checkpoints, all stages cached
