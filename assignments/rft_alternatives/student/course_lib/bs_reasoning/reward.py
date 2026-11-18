"""Reward for bs_reasoning: process (recipe construction) + outcome
(harness-executed price accuracy),.

`bs_price` here is the given, closed-form verifier -- same status as
tool_calling's `reward.py::bs_price` (see that module's docstring for the
precedent this follows): Black-Scholes is the given formula the model is
being trained to *apply*, not a leaked optimal answer. It is called only
to score a submitted completion against ground truth, never to construct
an SFT completion (SFT-data code must never import the verifier).

**Reward-hacking guard, load-bearing:** `lines_resolved` counts ONLY the 7
canonical names in `course_lib.bs_reasoning.REQUIRED_STEPS`, matched
EXACTLY -- not `len(evaluate_chain(...))`, which would also count extra,
irrelevant-but-resolvable lines a GRPO-optimized policy could learn to pad
with under other names to inflate the process term without the chain
meaning anything. Exact-name matching (not the naming-drift-tolerant
substring matching the pre-SFT feasibility probe used for diagnostics) is
deliberate here: SFT's whole job is to make the canonical names reliable,
so by GRPO time the reward can and should require them exactly -- a
test guards this.

`w_process`/`w_outcome`/`tol`/`PENALTY_MALFORMED` were chosen so the price
credit is neither saturated nor always zero for an SFT-only policy.
"""
import math

from . import REQUIRED_STEPS
from .expr_harness import evaluate_chain

W_PROCESS = 0.3        # process weight
W_OUTCOME = 0.7        # outcome weight
PRICE_TOL = 0.25       # relative-error tolerance for full outcome credit
PENALTY_MALFORMED = -0.1  # below the worst valid completion (C2: -0.10 < 0.0429)


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def reward(completion_text, params, w_process=W_PROCESS, w_outcome=W_OUTCOME,
           tol=PRICE_TOL):
    """`params` is the scenario's true Params (S, K, T, r, sigma) -- reward
    computation, not an SFT label, so using the true values here is fine
    (see module docstring). Returns (reward, details) for logging."""
    prompt_vals = {"S": params.S, "K": params.K, "T": params.T}
    order = evaluate_chain(completion_text, prompt_vals)
    resolved = dict(order)  # last-write-wins if a name repeats; fine for this purpose

    lines_resolved = sum(1 for step in REQUIRED_STEPS if step in resolved)
    if lines_resolved == 0:
        return PENALTY_MALFORMED, {"lines_resolved": 0, "price_accuracy": 0.0,
                                    "malformed": True}

    process_term = lines_resolved / len(REQUIRED_STEPS)

    if "price" in resolved:
        true_price = bs_price(params.S, params.K, params.T, params.r, params.sigma)
        rel_err = abs(resolved["price"] - true_price) / max(abs(true_price), 1e-6)
        outcome_term = max(0.0, 1.0 - rel_err / tol)
    else:
        outcome_term = 0.0

    r_total = w_process * process_term + w_outcome * outcome_term
    return r_total, {"lines_resolved": lines_resolved, "process_term": process_term,
                      "outcome_term": outcome_term, "malformed": False}
