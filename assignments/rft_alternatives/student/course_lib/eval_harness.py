"""Eval harness and parse gate.

Every policy line — baselines, SFT checkpoint, GRPO checkpoint, benchmark,
grid-best — is scored on the *same* fixed seed per scenario, so differences
in the resulting CSV are attributable to the policy, not to luck. Greedy
decoding (temperature 0) is used at eval time for every policy, including
sampled ones at training time — comparing a sampled policy to deterministic
baselines is a subtle unfairness students would otherwise discover by
accident; state this in the harness docstring and in lecture (§8).
"""
import csv
from collections import Counter

from .formatting import extract_json, format_prompt


def evaluate_policies(scenarios, policies, score, m_eval=1024, seed0=12345,
                      score_params=None):
    """Score every policy on every scenario with matched seeds.

    `policies` is dict[name -> callable(scenario)] returning either a
    completion STRING (model policies — goes through `score`, i.e. the full
    extraction + range-check path) or a params DICT (analytic baselines —
    goes through `score_params`, bypassing range checks, because a textbook
    prescription outside the model's legal action range must be scored, not
    penalized). `score` is callable(scenario, completion, seed, M) -> float;
    `score_params` is the matching params-level entry point and is required
    only if some policy returns dicts.

    The seed for scenario `i` is `seed0 + i` — deterministic given the
    scenario's position in `scenarios`, and identical across all policies so
    a GRPO checkpoint and a naive baseline face the same draws (§8's "same
    fixed seeds"). `scenario_id` is `scenario.id` if the scenario object
    has one, else falls back to its index `i` (no per-project scenario
    schema exists yet at this layer, so we don't assume an `.id` field).

    Returns tidy rows: list of (scenario_id, policy_name, reward).
    """
    rows = []
    for i, sc in enumerate(scenarios):
        scenario_id = getattr(sc, "id", i)
        seed = seed0 + i
        for name, policy in policies.items():
            out = policy(sc)
            if isinstance(out, dict):
                if score_params is None:
                    raise ValueError(
                        f"policy {name!r} returned a params dict but no "
                        "score_params entry point was provided"
                    )
                reward = score_params(sc, out, seed, m_eval)
            else:
                reward = score(sc, out, seed, m_eval)
            rows.append((scenario_id, name, reward))
    return rows


def write_csv(rows, path):
    """Write tidy (scenario_id, policy, reward) rows to `path`."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "policy", "reward"])
        w.writerows(rows)


def parse_gate(generate_fn, heldout, tokenizer, project, keys, ranges, n=200):
    """Fraction of held-out prompts that parse cleanly, plus the error breakdown.

    `generate_fn` takes the list of formatted prompt strings and returns a
    list of completion strings, generated greedily (temperature 0) — the
    gate measures the deployable policy's format compliance, not a sampled
    one's. Returns (parse_rate, Counter(errs)).
    """
    prompts = [format_prompt(tokenizer, project, s.text) for s in heldout[:n]]
    texts = generate_fn(prompts)
    errs = [extract_json(t, keys, ranges)[1] for t in texts]
    parse_rate = sum(e is None for e in errs) / len(errs) if errs else 0.0
    return parse_rate, Counter(errs)
