"""Prompt formatting and output extraction shared by every course project.

One `format_prompt` used identically at SFT, GRPO, and eval time — the #1
student failure mode is formatting the prompt differently across the three;
giving everyone this function removes it.

Extraction design decisions:
first-match extraction tolerates chatter around the answer; every error class
gets the same flat penalty (the class breakdown appears in logs, not in the
reward); the range check lives inside extraction so "malformed" and
"out-of-range" share one penalty path.
"""
import json
import math
import re

# Per-project system strings. Populated as each project is brought up; the
# text must match the project handout verbatim.
SYSTEM = {
    "bet_sizing": (
        "You are a risk manager sizing bets under uncertainty. "
        "Respond with a single JSON object and nothing else."
    ),
    "bs_reasoning": (
        "You price European call options, showing your work as a step-by-step "
        "derivation. Extract sigma and r from the brief, then compute d1, d2, "
        "Nd1, Nd2, and the price -- each as its own line, `name = `expr``, "
        "inside <think></think>, ending with <answer>price</answer>."
    ),
}

JSON_RE = re.compile(r"\{[^{}]*\}", re.S)   # first flat JSON object
CALL_RE = re.compile(r"bs\s*\(([^)]*)\)")   # tool-calling project


def format_prompt(tokenizer, project, scenario_text):
    """Render the chat-template prompt for `scenario_text`.

    Used at SFT-data time, GRPO time, and eval time — always via this
    function, never inline.
    """
    msgs = [
        {"role": "system", "content": SYSTEM[project]},
        {"role": "user", "content": scenario_text},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def extract_json(text, keys, ranges):
    """Return (params_dict, error). error in {None,'no_json','bad_json','keys','range'}."""
    m = JSON_RE.search(text)
    if not m:
        return None, "no_json"
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None, "bad_json"
    if set(d) != set(keys):
        return None, "keys"
    try:
        vals = {k: float(d[k]) for k in keys}
    except Exception:
        return None, "bad_json"
    for k, (lo, hi) in ranges.items():
        if not (lo <= vals[k] <= hi):
            return None, "range"
    return vals, None


def round_sig(x, sig=2):
    """Round to `sig` significant digits.

    The 2-significant-digit quantization is load-bearing (see
    the SFT skeleton): it caps number token length in SFT data and,
    at temperature 1.0, is a source of within-group diversity at RL time.
    """
    if x == 0:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))
