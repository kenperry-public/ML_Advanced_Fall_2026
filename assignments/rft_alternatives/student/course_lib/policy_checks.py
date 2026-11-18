"""Behavioral and ordering checks on a trained policy (§8-§9, added 2026-07-17).

Two failures motivated this module:
a trained policy that emitted the *identical* answer for every scenario while
scoring within 0.4% of the oracle on the aggregate money plot, and an eval
whose money-plot ordering violated the project's stated expectations in a CSV
nobody machine-checked. The lesson both times: **a score is not a behavior.**
Every project's `evaluate.py` therefore does two things after producing its
tidy CSV, in this order:

1. **Behavioral check** — greedy-decode the trained checkpoint across a small,
   diverse prompt set and require the outputs to actually vary
   (`check_output_varies`), and to vary in the economically sensible
   direction where the handout states one (`rank_correlation`).
2. **Ordering check** — `assert_money_plot_ordering` with the handout's
   expected orderings. A run that fails either check *fails*, loudly, no
   matter how good any aggregate number looks.

Pure numpy/stdlib, like `callbacks.py`: no torch/transformers at import time.
"""
from collections import Counter, defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Money-plot ordering
# ---------------------------------------------------------------------------

def mean_rewards_from_rows(rows):
    """Collapse tidy (scenario_id, policy, reward) rows — the exact output of
    `eval_harness.evaluate_policies` — into {policy: mean_reward}."""
    acc = defaultdict(list)
    for _sid, name, reward in rows:
        acc[name].append(float(reward))
    return {name: float(np.mean(v)) for name, v in acc.items()}


def ordering_violations(means, constraints):
    """Check expected money-plot orderings; return a list of violation strings.

    `means` is {policy_name: mean_reward} (see `mean_rewards_from_rows`).
    `constraints` is a list of (better, worse) or (better, worse, min_gap)
    tuples, straight from the handout's "expected results" — e.g.
    [("grpo", "sft_only"), ("grpo", "benchmark"), ("grid_best", "grpo")].
    `min_gap` (default 0.0) is the margin by which
    `better` must exceed `worse`; use it when the handout states a decisive
    gap rather than a bare ordering.

    An empty return value means every stated ordering holds. A policy name
    missing from `means` is itself a violation (a silently dropped baseline
    must not make the check vacuously pass).
    """
    violations = []
    for c in constraints:
        better, worse = c[0], c[1]
        min_gap = float(c[2]) if len(c) > 2 else 0.0
        missing = [n for n in (better, worse) if n not in means]
        if missing:
            violations.append(f"policy missing from eval rows: {missing}")
            continue
        gap = means[better] - means[worse]
        if gap < min_gap:
            violations.append(
                f"expected {better!r} > {worse!r} by >= {min_gap:g}, "
                f"got gap {gap:+.6g} ({better}={means[better]:+.6g}, "
                f"{worse}={means[worse]:+.6g})"
            )
    return violations


def assert_money_plot_ordering(means, constraints):
    """Raise AssertionError listing *all* violated orderings; return `means`
    unchanged if every constraint holds.

    This is the machine-checked form of §9 item 5 ("reaches the expected
    money-plot ordering") — call it at the end of `evaluate.py` so a run that
    misses its target exits nonzero instead of quietly writing a CSV.
    """
    violations = ordering_violations(means, constraints)
    if violations:
        raise AssertionError(
            "money-plot ordering check FAILED:\n  " + "\n  ".join(violations)
        )
    return means


# ---------------------------------------------------------------------------
# Output diversity (does the policy condition on the prompt at all?)
# ---------------------------------------------------------------------------

def output_diversity(params_list, ranges):
    """Diversity statistics for decoded outputs across a diverse prompt set.

    `params_list` is one parsed params dict per probe prompt (the first
    element of `extract_json`; pass None for prompts that failed to parse —
    they are counted, not silently dropped). `ranges` is the project's legal
    {key: (lo, hi)} dict, used to express variation as a fraction of the
    legal range so "varies" means the same thing across differently-scaled
    parameters.

    Returns {"n_probes", "n_parsed", "n_unique", "per_key_relative_range",
    "per_key_std", "per_key_mode_fraction"} where `n_unique` counts distinct
    outputs after rounding each value to 3 significant digits (the course
    quantizes outputs to 2, so this never merges genuinely different
    answers).

    `per_key_mode_fraction[k]` is the fraction of parsed probes whose value
    for `k` rounds to the single most common rounded value — a distribution-
    SHAPE statistic, deliberately different from `per_key_relative_range`
    (min-max span). Added 2026-08-16 after a real miss: a policy whose `r`
    was the same constant on 90% of held-out probes still cleared
    `min_relative_range` easily, because the other 10% of (non-collapsed)
    values happened to span a wide enough range on their own — min/max is
    blind to how the BULK of a distribution is shaped, and a small noisy
    tail is enough to make a point-mass distribution look "spread out." This
    is the "hidden partial collapse" failure on a key such as r.
    """
    parsed = [p for p in params_list if p is not None]
    keys = list(ranges.keys())

    def _round3(x):
        x = float(x)
        return 0.0 if x == 0 else round(x, -int(np.floor(np.log10(abs(x)))) + 2)

    unique = {tuple(_round3(p[k]) for k in keys) for p in parsed}
    rel_range, std, mode_frac = {}, {}, {}
    for k in keys:
        lo, hi = ranges[k]
        width = hi - lo
        vals = np.array([float(p[k]) for p in parsed], dtype=float)
        if len(vals) == 0 or width <= 0:
            rel_range[k], std[k], mode_frac[k] = 0.0, 0.0, 0.0
        else:
            rel_range[k] = float((vals.max() - vals.min()) / width)
            std[k] = float(vals.std())
            rounded = [_round3(v) for v in vals]
            mode_frac[k] = max(Counter(rounded).values()) / len(rounded)
    return {
        "n_probes": len(params_list),
        "n_parsed": len(parsed),
        "n_unique": len(unique),
        "per_key_relative_range": rel_range,
        "per_key_std": std,
        "per_key_mode_fraction": mode_frac,
    }


def check_output_varies(params_list, ranges, min_unique=3,
                        min_relative_range=0.05, require_keys=None,
                        max_mode_fraction=0.5):
    """Gate form of `output_diversity`: (ok, report).

    Fails when the decoded outputs across the probe set are (near-)constant:
    fewer than `min_unique` distinct outputs, or no parameter spanning at
    least `min_relative_range` of its legal range. A trained policy that
    fails this has not learned to condition on the scenario — whatever its
    aggregate reward says (that is precisely how the failure hides; see the
    module docstring). Defaults are deliberately lenient: this is a
    collapsed-policy detector, not a quality bar.

    `require_keys`, if given, is a subset of `ranges`' keys that must EACH
    individually clear `min_relative_range` (in addition to the plain
    `min_unique` check), rather than any single key being enough. Use this
    when one key can vary "for free" from inputs the policy doesn't control
    (e.g. a derived price that still moves with prompt-given S/K/T even if
    the policy's own extracted sigma/r has collapsed to a constant) — with
    the default `require_keys=None`, that key's variation alone would let
    the gate pass while a real skill stayed dead. Omitting it keeps the
    original coarse "any key varies" behavior, unaffected here.

    For each key in `require_keys`, ALSO fails if `max_mode_fraction` of
    parsed probes share the same rounded value, regardless of
    `min_relative_range` — a second, independent masking mechanism found
    2026-08-16 (see `output_diversity`'s docstring): a policy can emit one
    constant value on the large majority of prompts while a small
    non-collapsed tail alone stretches the min-max span past
    `min_relative_range`, which the span check on its own cannot see. Only
    applied to `require_keys` (like the relative-range check above) to keep
    the plain any-key-varies default behavior unchanged.

    Probe-set design is the caller's job and it matters: use >= 10 held-out
    prompts chosen to span the scenario pool's extremes (cheapest/most
    expensive, least/most risk-averse, ...). A narrow probe set can make a
    healthy policy look collapsed.
    """
    report = output_diversity(params_list, ranges)
    reasons = []
    if report["n_parsed"] == 0:
        reasons.append("no probe output parsed")
    if report["n_unique"] < min_unique:
        reasons.append(
            f"only {report['n_unique']} distinct output(s) across "
            f"{report['n_parsed']} parsed probes (need >= {min_unique})"
        )
    if require_keys:
        for k in require_keys:
            rr = report["per_key_relative_range"].get(k, 0.0)
            if rr < min_relative_range:
                reasons.append(
                    f"required key {k!r} spans only {rr:.3f} of its legal "
                    f"range (need >= {min_relative_range:g}) — another key "
                    "varying is not sufficient for this key specifically"
                )
            mf = report["per_key_mode_fraction"].get(k, 0.0)
            if mf > max_mode_fraction:
                reasons.append(
                    f"required key {k!r} has {mf:.2f} of probes sharing its "
                    f"single most common value (need <= {max_mode_fraction:g}) "
                    "— a wide min-max range doesn't rule out a collapsed "
                    "majority hiding behind a small non-collapsed tail"
                )
    elif report["per_key_relative_range"] and max(
        report["per_key_relative_range"].values(), default=0.0
    ) < min_relative_range:
        reasons.append(
            f"no parameter spans >= {min_relative_range:g} of its legal "
            f"range: {report['per_key_relative_range']}"
        )
    report["ok"] = not reasons
    report["reasons"] = reasons
    return report["ok"], report


def paired_significance(rows, policy_a, policy_b, n_boot=10000, seed=0):
    """Bootstrap paired-difference significance test between two policies
    scored on the same held-out scenarios.

    `rows` is the same tidy `(scenario_id, policy, reward)` triples
    `mean_rewards_from_rows` takes. Pairs `policy_a`'s and `policy_b`'s
    reward by `scenario_id` -- both were scored on the *same* scenarios,
    so pair rather than treat as independent samples -- and computes
    `diff = reward[policy_a] - reward[policy_b]` per scenario.

    Returns `{"mean_diff", "t_stat", "ci_95": (lo, hi), "n_wins",
    "n_losses", "n_ties"}`. `ci_95` comes from a bootstrap (resample the
    paired diffs with replacement `n_boot` times, take the 2.5th/97.5th
    percentiles of the resampled means), not a textbook t-test formula,
    since it doesn't require the diffs to be normally distributed -- this
    course's reward is bounded and often lumpy, not a smooth bell curve. A
    CI that excludes zero supports "the gap is real"; one that straddles
    zero means the claim isn't backed by the data, whatever the two raw
    means say. `n_wins`/`n_losses`/`n_ties` (`policy_a` >/</== `policy_b`
    per scenario) answer the separate "how decisive" question -- "real"
    and "landslide" are different claims.

    Raises if any scenario has a reward for one policy but not the other
    (both must be scored on the identical held-out set to pair correctly).
    """
    by_scenario = defaultdict(dict)
    for sid, name, reward in rows:
        if name in (policy_a, policy_b):
            by_scenario[sid][name] = float(reward)
    missing = [sid for sid, d in by_scenario.items()
               if policy_a not in d or policy_b not in d]
    if missing:
        raise ValueError(
            f"{len(missing)} scenario(s) missing a reward for {policy_a!r} "
            f"or {policy_b!r} -- both policies must be scored on the same "
            f"held-out set to pair by scenario_id"
        )
    diffs = np.array(
        [by_scenario[sid][policy_a] - by_scenario[sid][policy_b]
         for sid in by_scenario], dtype=float)
    n = len(diffs)
    mean_diff = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else 0.0
    t_stat = float(mean_diff / (sd / np.sqrt(n))) if sd > 0 else float("nan")
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(diffs, size=(n_boot, n), replace=True).mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "ci_95": (float(ci_lo), float(ci_hi)),
        "n_wins": int((diffs > 0).sum()),
        "n_losses": int((diffs < 0).sum()),
        "n_ties": int((diffs == 0).sum()),
    }


def rank_correlation(x, y):
    """Spearman-style rank correlation in [-1, 1] (0.0 if either input is
    constant; ties get their sort-order rank, not the averaged rank — fine
    for the sign-and-rough-magnitude use this module puts it to).

    Use it for the directional half of the behavioral check when the handout
    states one — e.g. hedging-style "band width should widen with cost":
    `rank_correlation(costs, [p["w0"] for p in params_list]) > 0`. A policy
    can pass `check_output_varies` on noise alone; agreeing with the stated
    economics is the stronger evidence that conditioning was *learned*.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0
