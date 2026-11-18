"""Shared constants for bs_reasoning: the required derivation step names and
the allowed expression functions. Single source of truth so the prompt, the
harness, and the reward never drift.
"""

# The 7 lines every completion must produce, in dependency order. Only these
# names count toward the process reward's `lines_resolved` term -- a policy
# padding extra, irrelevant-but-resolvable lines under other names must not
# inflate the score (a reward-hacking guard).
REQUIRED_STEPS = ["sigma", "r", "d1", "d2", "Nd1", "Nd2", "price"]

# Functions legal inside a backtick expression. Anything else is a parse
# failure for that line, by design -- expr_harness.py enforces this via a
# restricted-AST evaluator, never Python's eval().
ALLOWED_FUNCS = ["log", "sqrt", "exp", "ncdf"]
