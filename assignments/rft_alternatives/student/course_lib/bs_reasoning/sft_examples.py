"""Given SFT training examples for the bs_reasoning_rft_alternatives
assignment's required Part 1.

Unlike a task where you write the SFT-data generator yourself, this
module's Part 1 gives you the SFT training examples directly -- you still build and run
`sft_train.py` against them, you just aren't designing the completion
format or the noising strategy yourself for the *required* part. Part 3,
once you've finished Part 2, invites you to build your own from scratch,
with a critical eye on this one.

Design rule (never leak the optimal answer into SFT data): every
completion below states a plausible-but-WRONG sigma/r, resampled
independently of the scenario's true params, with the shown price
computed from those wrong values -- never the scenario's true price. The
five `d1`/`d2`/`Nd1`/`Nd2`/`price` lines are always the same fixed,
correct SYMBOLIC formula text (the standard Black-Scholes
formula, not a leaked label) -- writing it
out never requires evaluating it for any particular scenario. Black-
Scholes math is duplicated here rather than imported from `reward.py`, so
this module's own import graph never reaches the verifier, a boundary
that is checked automatically, not just conventional.
"""
import math

import numpy as np

from ..formatting import format_prompt
from .scenarios import make_scenarios

MAX_DROP_RATE = 0.05
NOISE_TOL_REL = 0.10
NOISE_TOL_ABS = 0.02
MAX_RESAMPLE = 20

TEMPLATE = (
    "<think>\n"
    "sigma = `{sigma}`\n"
    "r = `{r}`\n"
    "d1 = `(log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))`\n"
    "d2 = `d1 - sigma*sqrt(T)`\n"
    "Nd1 = `ncdf(d1)`\n"
    "Nd2 = `ncdf(d2)`\n"
    "price = `S*Nd1 - K*exp(-r*T)*Nd2`\n"
    "</think>\n"
    "<answer>{price}</answer>"
)


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _close(a, b, rel=NOISE_TOL_REL, abs_tol=NOISE_TOL_ABS):
    return abs(a - b) <= max(abs_tol, rel * abs(b))


def make_noised_completion(sc, rng):
    """Returns `(completion_text, sigma_shown, r_shown, price_shown, ok)`.
    `ok=False` means every resample attempt still collided with the true
    value (vanishingly rare given continuous draws)."""
    p = sc.params
    for _ in range(MAX_RESAMPLE):
        sigma_shown = round(float(rng.uniform(0.10, 0.50)), 2)
        r_shown = round(float(rng.uniform(0.01, 0.06)), 3)
        if _close(sigma_shown, p.sigma) or _close(r_shown, p.r):
            continue
        price_shown = round(_bs_price(p.S, p.K, p.T, r_shown, sigma_shown), 2)
        true_price = _bs_price(p.S, p.K, p.T, p.r, p.sigma)
        if _close(price_shown, true_price):
            continue
        text = TEMPLATE.format(sigma=sigma_shown, r=r_shown, price=price_shown)
        return text, sigma_shown, r_shown, price_shown, True
    return None, None, None, None, False


def make_sft_examples(tokenizer, n=1500, seed=0):
    """Build `n` `{"prompt", "completion"}` rows -- same dict shape as
    `course_lib.sft_skeleton.make_sft_dataset`, though this task's own
    noised-derivation logic (not that generic JSON-params skeleton)
    produces the completions. Scenarios whose resample budget is
    exhausted are silently skipped (well under 5% in practice), so the returned list can be slightly shorter than `n`.

    Re-checks every returned row's noised values against the scenario's
    true params before returning -- belt and suspenders on top of
    `make_noised_completion`'s own check, same posture as every other
    SFT generator in this course."""
    rng = np.random.default_rng(seed)
    scenarios = make_scenarios(n, seed=seed)
    rows = []
    for sc in scenarios:
        completion, sigma_shown, r_shown, price_shown, ok = make_noised_completion(sc, rng)
        if not ok:
            continue
        p = sc.params
        true_price = _bs_price(p.S, p.K, p.T, p.r, p.sigma)
        assert not _close(sigma_shown, p.sigma)
        assert not _close(r_shown, p.r)
        assert not _close(price_shown, true_price)
        rows.append({
            "prompt": format_prompt(tokenizer, "bs_reasoning", sc.text),
            "completion": completion,
        })
    return rows
