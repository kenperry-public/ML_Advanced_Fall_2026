"""Extra logged metrics for GRPO runs.

TRL already logs reward mean/std, KL, and entropy. This module supplies the
project-agnostic extras the doc calls out: parse rate per step, per-error-class
counts, within-group reward std, and fraction of all-tied groups.

Kept framework-light on purpose: everything below is a pure function of
plain Python/numpy values, so `course_lib.callbacks` imports fine with only
numpy installed (no trl, no transformers at module load time). The optional
`TrainerCallback` wrapper at the bottom imports `transformers` lazily, inside
a factory function, so importing this module never requires it.
"""
from collections import Counter

import numpy as np

from .formatting import extract_json


def completion_metrics(completions, keys, ranges, extractor=None):
    """Parse-rate and error-class breakdown for one batch of completions.

    Returns {"parse_rate": float, "err_counts": Counter} where `err_counts`
    tallies the non-None error labels from `extract_json` (see formatting.py
    for the {'no_json','bad_json','keys','range'} vocabulary). A completion
    with no error does not appear in `err_counts`.

    `extractor`, if given, replaces the default `extract_json(c, keys,
    ranges)` call with `extractor(c) -> (parsed, error)` -- added for
    tool_calling, whose completions are `bs(...)` calls, not JSON, so `keys`/
    `ranges` don't apply (ported from that project's worktree, same
    signature/docstring, for bs_reasoning's own non-JSON `` name = `expr` ``
    completions -- see `course_lib.bs_reasoning`'s grpo_train.py). `keys`/
    `ranges` are ignored when `extractor` is given but stay required
    positionally so the five JSON-based projects' call sites are unchanged.
    """
    parse = extractor if extractor is not None else (lambda c: extract_json(c, keys, ranges))
    errs = [parse(c)[1] for c in completions]
    n = len(errs)
    parse_rate = (sum(e is None for e in errs) / n) if n else 0.0
    err_counts = Counter(e for e in errs if e is not None)
    return {"parse_rate": parse_rate, "err_counts": err_counts}


def group_stats(rewards, num_generations):
    """Within-group reward stats for one GRPO step.

    `rewards` is the flat per-completion reward list TRL builds — each
    prompt's `num_generations` completions occupy consecutive slots (see
    crn.py's docstring: "TRL's reward function receives the prompts list
    with each prompt repeated G times"). Chunking into consecutive blocks of
    `num_generations` recovers the groups.

    Returns {"within_group_std_mean": float, "frac_all_tied": float}:
    the mean of each group's reward std (GRPO's advantage signal vanishes
    when this is 0), and the fraction of groups where every reward in the
    group is identical (an all-tied group also contributes zero advantage,
    for a different reason than "reward std is nonzero but uninformative").
    """
    rewards = np.asarray(rewards, dtype=float)
    n = len(rewards)
    if n == 0 or num_generations <= 0 or n % num_generations != 0:
        raise ValueError(
            f"rewards length {n} is not divisible by num_generations={num_generations}"
        )
    groups = rewards.reshape(-1, num_generations)
    stds = groups.std(axis=1)
    all_tied = np.all(groups == groups[:, :1], axis=1)
    return {
        "within_group_std_mean": float(stds.mean()),
        "frac_all_tied": float(all_tied.mean()),
    }


def make_metrics_callback(keys, ranges, num_generations):
    """Build a `transformers.TrainerCallback` that merges the metrics above
    into the trainer's logs.

    `transformers` is imported here, inside the factory, not at module top
    level — so `import course_lib.callbacks` never requires trl/transformers
    to be installed. Call this only from the GRPO training script.

    Usage: call `.record(completions, rewards)` from inside the reward
    function each step (it already has both lists in hand), then let the
    Trainer's normal `on_log` callback machinery pick up the merged dict.
    """
    from transformers import TrainerCallback

    class _MetricsCallback(TrainerCallback):
        def __init__(self):
            self._completions = None
            self._rewards = None

        def record(self, completions, rewards):
            self._completions = completions
            self._rewards = rewards

        def on_log(self, args, state, control, logs=None, **kw):
            if logs is None or self._completions is None:
                return control
            logs.update(completion_metrics(self._completions, keys, ranges))
            logs.update(group_stats(self._rewards, num_generations))
            return control

    return _MetricsCallback()
