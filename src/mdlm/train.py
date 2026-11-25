"""Train an AR model and a masked-diffusion model of the same size, then compare them.

    python -m mdlm.train --out results --device cuda

Both use the same data (Python standard-library source as bytes), steps, batch, optimiser
and schedule. Evaluation:

1. **Likelihood.** AR gets exact val bits/byte. Diffusion gets its NELBO, an *upper bound*
   on bits/byte, averaged over 8 passes with stratified t (± standard error).
2. **Unconditional generation** of 32 × 256 bytes. AR samples at T = 1; diffusion samples
   at 8–256 denoising steps, with random or confidence-ordered unmasking. Each set of
   samples is scored by the trained AR model's bits/byte (lower means more data-like) and
   by its byte entropy, since degenerate repetitive text also scores low bits/byte. Wall
   time is recorded too.
3. **Infilling.** In 256-byte val windows, the middle 64 bytes are hidden. Diffusion sees
   both sides; AR can only continue the prefix. The metric is the fraction of hidden bytes
   recovered exactly (near-greedy, T = 0.3).

Checkpoints and finished stages are kept in ``<out>`` so a reclaimed session resumes.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sysconfig
import time
import zlib

import numpy as np
import torch

from .model import (
    Config,
    Transformer,
    ar_loss,
    ar_sample,
    diffusion_loss,
    diffusion_sample,
)

LN2 = math.log(2)


def load_corpus():
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    if not files:
        raise FileNotFoundError(root)
    tr, va = [], []
    for f in files:
        rel = os.path.relpath(f, root).replace("\\", "/")
        with open(f, "rb") as fh:
            (va if zlib.crc32(rel.encode()) % 1000 < 50 else tr).append(fh.read())
    return (
        np.frombuffer(b"\n".join(tr), np.uint8).copy(),
        np.frombuffer(b"\n".join(va), np.uint8).copy(),
    )


def windows(data, n, length, rng, device):
    ix = rng.integers(0, len(data) - length, n)
    return torch.from_numpy(np.stack([data[i : i + length] for i in ix]).astype(np.int64)).to(
        device
    )


def train(kind: str, cfg: Config, data, steps, bs, lr, device, seed=0, log=print):
    torch.manual_seed(seed)
    model = Transformer(Config(**{**cfg.__dict__, "causal": kind == "ar"})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.1)
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    for step in range(1, steps + 1):
        f = min(1.0, step / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr * max(f, 0.02)
        x = windows(data, bs, cfg.ctx + (kind == "ar"), rng, device)
        with torch.autocast("cuda", torch.float16, enabled=amp):
            loss = ar_loss(model, x) if kind == "ar" else diffusion_loss(model, x, gen=gen)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % max(1, steps // 10) == 0:
            log(f"  {kind} step {step}: {loss.item() / LN2:.3f} bits/byte (train)")
    return model.eval()


@torch.no_grad()
def likelihood(ar, dm, val, device, n=128, passes=8, bs=32):
    rng = np.random.default_rng(7)
    x = windows(val, n, ar.c.ctx + 1, rng, device)
    ar_bpb = float(np.mean([ar_loss(ar, x[i : i + bs]).item() for i in range(0, n, bs)])) / LN2
    gen = torch.Generator(device=device).manual_seed(7)
    est = []
    for _ in range(passes):
        est.append(
            np.mean(
                [diffusion_loss(dm, x[i : i + bs, :-1], gen=gen).item() for i in range(0, n, bs)]
            )
            / LN2
        )
    return {
        "ar_bpb": ar_bpb,
        "diffusion_nelbo_bpb": float(np.mean(est)),
        "diffusion_nelbo_se": float(np.std(est, ddof=1) / math.sqrt(passes)),
    }


def byte_entropy(x: torch.Tensor) -> float:
    c = torch.bincount(x.flatten(), minlength=256).float()
    p = c[c > 0] / c.sum()
    return float(-(p * p.log2()).sum())


@torch.no_grad()
def judge(ar, x, bs=32) -> float:
    return float(np.mean([ar_loss(ar, x[i : i + bs]).item() for i in range(0, len(x), bs)])) / LN2


@torch.no_grad()
def generation(ar, dm, val, device, n=32, step_list=(8, 32, 128, 256)):
    L = ar.c.ctx
    gen = torch.Generator(device=device).manual_seed(11)
    ref = windows(val, n, L, np.random.default_rng(3), device)
    rows = [
        {
            "sampler": "real val data",
            "judge_bpb": judge(ar, ref),
            "entropy": byte_entropy(ref),
            "seconds": 0.0,
        }
    ]
    t0 = time.time()
    s = ar_sample(ar, n, L, device, gen=gen, prefix=torch.full((n, 1), 10, device=device))
    rows.append(
        {
            "sampler": "AR, T=1 (256 sequential steps)",
            "judge_bpb": judge(ar, s),
            "entropy": byte_entropy(s),
            "seconds": time.time() - t0,
        }
    )
    for order in ("random", "confidence"):
        for k in step_list:
            t0 = time.time()
            s = diffusion_sample(dm, n, L, k, device, gen=gen, order=order)
            rows.append(
                {
                    "sampler": f"diffusion, {k} steps, {order} order",
                    "judge_bpb": judge(ar, s),
                    "entropy": byte_entropy(s),
                    "seconds": time.time() - t0,
                }
            )
    return rows


@torch.no_grad()
def infill(ar, dm, val, device, n=64, hole=64, steps=64, temperature=0.3):
    L = ar.c.ctx
    x = windows(val, n, L, np.random.default_rng(5), device)
    a, b = (L - hole) // 2, (L + hole) // 2
    given = x.clone()
    given[:, a:b] = -1
    gen = torch.Generator(device=device).manual_seed(13)
    d = diffusion_sample(
        dm, n, L, steps, device, gen=gen, given=given, temperature=temperature, order="confidence"
    )
    r = ar_sample(ar, n, b, device, temperature=temperature, gen=gen, prefix=x[:, :a])
    return {
        "hole": hole,
        "diffusion_acc": float((d[:, a:b] == x[:, a:b]).float().mean()),
        "ar_prefix_only_acc": float((r[:, a:b] == x[:, a:b]).float().mean()),
    }


def report(res: dict) -> str:
    lk, inf = res.get("likelihood"), res.get("infill")
    lines = [res.get("meta", ""), ""]
    if lk:
        lines += [
            "| model | val bits/byte |",
            "|---|---|",
            f"| autoregressive (exact) | {lk['ar_bpb']:.3f} |",
            f"| masked diffusion (NELBO, upper bound) | {lk['diffusion_nelbo_bpb']:.3f} "
            f"± {lk['diffusion_nelbo_se']:.3f} |",
            "",
        ]
    if res.get("generation"):
        lines += [
            "| sampler | AR-judged bits/byte | byte entropy (bits) | seconds (32 × 256 B) |",
            "|---|---|---|---|",
        ]
        for r in res["generation"]:
            lines.append(
                f"| {r['sampler']} | {r['judge_bpb']:.3f} | {r['entropy']:.2f} | "
                f"{r['seconds']:.1f} |"
            )
        lines.append("")
    if inf:
        lines += [
            f"Infilling the middle {inf['hole']} of 256 bytes, exact-byte accuracy: "
            f"diffusion (sees both sides) **{100 * inf['diffusion_acc']:.1f}%**, "
            f"AR (prefix only) **{100 * inf['ar_prefix_only_acc']:.1f}%**."
        ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    cfg = Config(d_model=32, n_layers=1, n_heads=2, ctx=64) if a.tiny else Config()
    train_data, val = load_corpus()
    res_path = os.path.join(a.out, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    models = {}
    for kind in ("ar", "diffusion"):
        ck = os.path.join(a.out, f"{kind}.pt")
        m = Transformer(Config(**{**cfg.__dict__, "causal": kind == "ar"})).to(a.device)
        if os.path.exists(ck):
            m.load_state_dict(torch.load(ck, map_location=a.device))
            models[kind] = m.eval()
            log(f"loaded {kind}")
            continue
        t0 = time.time()
        models[kind] = train(kind, cfg, train_data, a.steps, a.bs, a.lr, a.device, log=log)
        torch.save(models[kind].state_dict(), ck)
        res[f"{kind}_train_minutes"] = (time.time() - t0) / 60
        json.dump(res, open(res_path, "w"), indent=1)
    ar, dm = models["ar"], models["diffusion"]
    n_par = sum(p.numel() for p in ar.parameters())
    dev = torch.cuda.get_device_name() if a.device == "cuda" else "cpu"
    res["meta"] = (
        f"{dev}; both models {n_par:,} params, {a.steps} steps × {a.bs} × "
        f"{cfg.ctx} bytes, identical optimiser and schedule."
    )
    small = dict(n=4) if a.tiny else {}
    for stage, fn in (
        (
            "likelihood",
            lambda: likelihood(
                ar, dm, val, a.device, **(dict(n=8, passes=2, bs=4) if a.tiny else {})
            ),
        ),
        (
            "generation",
            lambda: generation(
                ar, dm, val, a.device, step_list=(4,) if a.tiny else (8, 32, 128, 256), **small
            ),
        ),
        (
            "infill",
            lambda: infill(
                ar, dm, val, a.device, hole=16 if a.tiny else 64, steps=8 if a.tiny else 64, **small
            ),
        ),
    ):
        if stage not in res:
            res[stage] = fn()
            json.dump(res, open(res_path, "w"), indent=1)
            log(f"{stage} done")
    text = report(res)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
