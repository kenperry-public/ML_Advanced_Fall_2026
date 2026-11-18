"""Scenario generator for bs_reasoning: parameter sampler + NL renderer with
vol/rate confuser-bucket tags (given/build split: this is given, students
never write scenario generation).

Design principles this encodes:
  - True params live only on the Params object; the rendered text is the
    only thing that should ever land in an SFT file.
  - `S`/`K`/`T` always render as plain stated numbers -- no confuser
    convention on these three (D4-style scope control); only `sigma`/`r`
    draw from the resolvable/adversarial template buckets.
  - Every phrase ends with a period. This is a real, previously-hit bug:
    without a sentence boundary, a naive downstream parser's number search
    can bleed across clauses.
  - Held-out splits: hold out both scenarios AND template surface-forms.
    `make_scenarios(..., held_out=True)` reserves each bucket's LAST
    template for held-out use only (never drawn when `held_out=False`,
    the training default) -- a real template-level split, not just a
    fresh RNG seed drawing from the same pool (an earlier version
    only claimed this in a docstring and did not implement it). Continuous params
    (`S`/`K`/`T`/`r`/`sigma`) are never split -- a fresh seed already gives
    fresh draws, and text-dedup across splits isn't separately enforced
    (astronomically unlikely to collide given continuous sampling).
"""
from dataclasses import dataclass

import numpy as np

VOL_RESOLVABLE = [
    "Volatility is {pct}%.",
    "Vol of .{dec}.",
    "sigma = 0.{dec}.",
    "Annualized volatility of {pct} percent.",
    "Implied vol at {pct}%.",
]
VOL_ADVERSARIAL = [
    "Vol's running around {pts}.",
    "This is a {pts}-vol name.",
    "Priced for a choppy stretch, north of {pts}.",
    "The desk's marking it at {pts}.",
    "Realized has been printing near {pts} lately.",
]
RATE_RESOLVABLE = [
    "Risk-free rate of {pct}%.",
    "r = 0.{dec}.",
    "Funding at {bps} bps.",
    "The curve sits at {pct} percent.",
]
RATE_ADVERSARIAL = [
    "Financing costs about {pts} these days.",
    "Money market's paying {pts}.",
    "Cash earns roughly {pts} parked short-term.",
]


@dataclass(frozen=True)
class Params:
    S: float
    K: float
    T: float
    r: float
    sigma: float


@dataclass(frozen=True)
class Scenario:
    id: int
    text: str
    vol_bucket: str   # "resolvable" | "adversarial"
    rate_bucket: str
    vol_template_id: int
    rate_template_id: int
    params: Params


def sample_params(rng) -> Params:
    S = float(rng.uniform(50, 150))
    K = float(S * rng.uniform(0.8, 1.2))
    T = float(rng.uniform(0.1, 2.0))
    r = float(rng.uniform(0.01, 0.06))
    sigma = float(rng.uniform(0.10, 0.50))
    return Params(S=round(S, 2), K=round(K, 2), T=round(T, 2),
                  r=round(r, 3), sigma=round(sigma, 2))


def _render_vol(sigma, bucket, template_id):
    pct, dec, pts = f"{sigma*100:.0f}", f"{sigma:.2f}".split(".")[1], f"{sigma*100:.0f}"
    tpl = (VOL_RESOLVABLE if bucket == "resolvable" else VOL_ADVERSARIAL)[template_id]
    return tpl.format(pct=pct, dec=dec, pts=pts)


def _render_rate(r, bucket, template_id):
    pct, dec = f"{r*100:.1f}", f"{r:.3f}".split(".")[1]
    pts, bps = pct, f"{r*10000:.0f}"
    tpl = (RATE_RESOLVABLE if bucket == "resolvable" else RATE_ADVERSARIAL)[template_id]
    return tpl.format(pct=pct, dec=dec, pts=pts, bps=bps)


def render(params: Params, vol_bucket: str, rate_bucket: str, vol_template_id: int,
           rate_template_id: int) -> str:
    vol_phrase = _render_vol(params.sigma, vol_bucket, vol_template_id)
    rate_phrase = _render_rate(params.r, rate_bucket, rate_template_id)
    return (f"Spot is {params.S}, strike {params.K}, {params.T} years to expiry. "
            f"{vol_phrase} {rate_phrase}")


def make_scenarios(n: int, seed: int = 0, held_out: bool = False) -> list[Scenario]:
    """`held_out=True` restricts template draws to each bucket's reserved
    last template (never seen during `held_out=False` generation, e.g.
    your SFT-data generator, `grpo_train.py`, `base_rate.py`'s training-distribution
    calls) -- see module docstring."""
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        params = sample_params(rng)
        vol_bucket = "resolvable" if rng.random() < 0.5 else "adversarial"
        rate_bucket = "resolvable" if rng.random() < 0.5 else "adversarial"
        vol_n = len(VOL_RESOLVABLE if vol_bucket == "resolvable" else VOL_ADVERSARIAL)
        rate_n = len(RATE_RESOLVABLE if rate_bucket == "resolvable" else RATE_ADVERSARIAL)
        # Reserve each pool's last template index for held_out-only use.
        vol_lo, vol_hi = (vol_n - 1, vol_n) if held_out else (0, vol_n - 1)
        rate_lo, rate_hi = (rate_n - 1, rate_n) if held_out else (0, rate_n - 1)
        vol_tid = int(rng.randint(vol_lo, vol_hi))
        rate_tid = int(rng.randint(rate_lo, rate_hi))
        text = render(params, vol_bucket, rate_bucket, vol_tid, rate_tid)
        out.append(Scenario(id=i, text=text, vol_bucket=vol_bucket, rate_bucket=rate_bucket,
                             vol_template_id=vol_tid, rate_template_id=rate_tid, params=params))
    return out
