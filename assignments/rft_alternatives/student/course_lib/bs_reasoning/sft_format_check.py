"""Given format checker for your own SFT-data generator's generated completions
(the assignment's Part 1 Rules: the rule this enforces).

Every completion's `<think>` block must contain ONLY the seven bare
`` name = `expr` `` derivation lines -- no other text, before, between, or
after them. This is stated as a harness-clarity rule (a stray line breaks
nothing the reward computes, since `evaluate_chain` already ignores anything
it can't parse -- see `expr_harness.py`), and this checker is what makes it
enforceable rather than aspirational: run it against your own
SFT-data generator's output before training, the same way you'd run any other
check in this assignment.
"""
import re

_THINK_RE = re.compile(r"<think>\s*\n(.*?)</think>", re.S)
_LINE_RE = re.compile(r"^[A-Za-z_]\w*\s*=\s*`[^`]*`$")


def check_no_preamble(completions):
    """`completions`: an iterable of completion strings (e.g. the values of
    the dict your generator writes, or a list of raw completion text).

    Returns `(ok, violations)`. `violations` is a list of `(index, reason)`
    for every completion whose `<think>` block contains anything besides
    bare `` name = `expr` `` lines -- extra prose, a blank line's worth of
    commentary, anything. `ok` is `True` iff `violations` is empty.

    Does NOT check the seven names/order (that's `reward.py`'s job, scored
    at GRPO time, not an SFT-data structural requirement) -- only that
    nothing beyond bare derivation-line syntax appears inside `<think>`.
    """
    violations = []
    for i, text in enumerate(completions):
        m = _THINK_RE.search(text)
        if not m:
            violations.append((i, "no <think>...</think> block found"))
            continue
        lines = [ln for ln in m.group(1).split("\n") if ln.strip()]
        if not lines:
            violations.append((i, "<think> block is empty"))
            continue
        for j, ln in enumerate(lines):
            if not _LINE_RE.match(ln.strip()):
                violations.append((
                    i, f"line {j + 1} inside <think> isn't a bare "
                       f"`` name = `expr` `` line: {ln.strip()!r}"))
                break
    return len(violations) == 0, violations
