"""SFT data generation skeleton shared by the five decision projects Tool-calling replaces the uniform
sampler with the weak parser and has its own generation module.

Leak-audit note (this file is checked as an entry point):
this module must import ONLY `numpy`, `json`, and `.formatting` — never a
project's `simulator` or `benchmark` module. The uniform-random sampler below
never looks at reward or optimal-answer code, so this is structurally easy to
keep true; do not "helpfully" import a simulator here to validate params.
"""
import json

import numpy as np

from .formatting import format_prompt, round_sig


def make_sft_dataset(scenarios, keys, ranges, tokenizer, project, n=1500, seed=0,
                      int_keys=()):
    """Build `n` (prompt, completion) rows of uniform-random legal params.

    `scenarios` is indexed, not passed to `rng.choice` directly — `rng.choice`
    tries to build a 2D array out of a list of namedtuples/dataclasses and
    either raises or silently mangles them, so we sample integer indices
    instead.

    `round_sig(x, 2)` quantization is load-bearing (see formatting.py):
    it caps number-token length and is a source of within-group diversity
    at RL time. Keys in `int_keys` (e.g. `max_inv`) are cast to `int` after
    quantization so `json.dumps` emits them without a decimal point.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scenarios), size=n)
    rows = []
    for i in idx:
        sc = scenarios[i]
        params = {}
        for k in keys:
            v = round_sig(rng.uniform(*ranges[k]), 2)
            if k in int_keys:
                v = int(round(v))
            params[k] = v
        rows.append({
            "prompt": format_prompt(tokenizer, project, sc.text),
            "completion": json.dumps(params),
        })
    return rows
