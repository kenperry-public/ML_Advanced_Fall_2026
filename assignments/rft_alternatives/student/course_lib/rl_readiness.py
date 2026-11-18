"""RL-readiness pre-flight: pick a project's GRPO group size (G) by
measurement, not by inheriting another project's value.

Policy: num_generations is not a shared course-wide
default. Each project runs this sweep against its own SFT checkpoint and
reward function before committing to a full GRPO run, and picks the smallest
G that produces healthy within-group reward variance — bigger G costs more
compute/memory per step for the same wall-clock budget.

Framework-light on purpose, same convention as callbacks.py: this module
imports only numpy at module load time, so it's usable from a CPU-only
environment with no torch/transformers installed. `sample_fn` and
`reward_fn` are supplied by the caller's project-specific driver script,
which is where the actual model lives.
"""
from .callbacks import completion_metrics, group_stats


def readiness_sweep(sample_fn, reward_fn, prompts, candidate_Gs, keys, ranges,
                    extractor=None):
    """Measure within-group reward variance at each candidate G.

    `sample_fn(prompt, g)` -> list of `g` completion strings for `prompt`
    (sampled, e.g. temperature 1.0 — this is a signal-detection check, not a
    greedy eval). `reward_fn(prompts, completions, **kw)` -> list of floats,
    the project's real CRN-wrapped reward (never a proxy). `prompts` is a
    slice of real training prompts, not a single fixed one, so the sweep
    reflects the reward's behavior across the actual scenario distribution.

    `extractor`, if given, is passed through to `completion_metrics` (see
    callbacks.py) for projects whose completions aren't JSON -- e.g.
    tool_calling's `bs(...)` calls, or bs_reasoning's `` name = `expr` ``
    lines (ported from tool_calling's worktree, same signature).

    Returns {G: {"parse_rate", "err_counts", "within_group_std_mean",
    "frac_all_tied"}}. A candidate G is "ready" when frac_all_tied is low and
    within_group_std_mean is comfortably above the reward's noise floor —
    both are project-specific judgment calls the caller makes from this
    table, not a threshold this module hardcodes.
    """
    results = {}
    for g in candidate_Gs:
        flat_prompts, flat_completions = [], []
        for p in prompts:
            comps = sample_fn(p, g)
            flat_prompts += [p] * g
            flat_completions += list(comps)
        rewards = reward_fn(flat_prompts, flat_completions)
        m = completion_metrics(flat_completions, keys, ranges, extractor=extractor)
        gs = group_stats(rewards, g)
        results[g] = dict(
            parse_rate=m["parse_rate"],
            err_counts=dict(m["err_counts"]),
            within_group_std_mean=gs["within_group_std_mean"],
            frac_all_tied=gs["frac_all_tied"],
        )
    return results
