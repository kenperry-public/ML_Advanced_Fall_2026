"""The expression harness -- the reward engine's core for bs_reasoning.

Reference implementation. Given/build split: this is given, students never modify it.

A restricted-AST evaluator -- NOT Python's eval() -- so a completion cannot
execute arbitrary code no matter what a policy learns to emit. Only
arithmetic (+ - * / **), unary negation, name lookups in the current
namespace, and calls to the allowlisted functions in
course_lib.bs_reasoning.ALLOWED_FUNCS are permitted; anything else raises
and that line simply fails to resolve.

`evaluate_chain` parses every `` name = `expr` `` line in a completion, in
order, and evaluates each one itself against a namespace that starts with
the prompt's true S/K/T (and, critically, whatever sigma/r the model
itself claimed -- never the true values, so a bad extraction propagates
into a bad price exactly the way a bad reading would on a real desk).
Tolerant of naming drift (e.g. a model writing `d1_ex` instead of `d1`):
resolution order is tracked, and each resolved value is stored under
whatever name the model used, available to later lines exactly as written.
"""
import ast
import math
import operator
import re


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


_FUNCS = {"log": math.log, "sqrt": math.sqrt, "exp": math.exp, "ncdf": norm_cdf}
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}

LINE_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*`+\s*([^`]+?)\s*`+")


def _eval_node(node, names):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ValueError(f"unknown name {node.id}")
        return names[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left, names), _eval_node(node.right, names))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand, names))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval_node(a, names) for a in node.args])
    raise ValueError("disallowed expression")


def safe_eval(expr: str, names: dict) -> float:
    """Evaluate `expr` against `names` via a restricted AST walk. Raises
    ValueError on anything not arithmetic / a known name / an allowlisted
    call -- never executes arbitrary code."""
    return _eval_node(ast.parse(expr, mode="eval").body, names)


def evaluate_chain(text: str, prompt_vals: dict) -> list[tuple[str, float]]:
    """Parse every `` name = `expr` `` line in `text`, in order, and
    safe-evaluate each one against a namespace seeded with `prompt_vals`
    (must include at least S, K, T; sigma/r come from the completion
    itself, never from ground truth). Returns [(name, value)] in
    resolution order -- unresolved lines are simply absent, not errors."""
    names = dict(prompt_vals)
    order = []
    for name, expr in LINE_RE.findall(text):
        try:
            val = safe_eval(expr, names)
        except Exception:
            continue
        names[name] = val
        order.append((name, val))
    return order


# Placeholders only -- chain_extractor is a format-completeness check for
# GRPO-step logging, never the reward (reward.py seeds the scenario's real
# S/K/T itself). Any positive values work; these just need to keep d1's
# log(S/K) and sqrt(T) well-defined.
_DUMMY_VALS = {"S": 100.0, "K": 100.0, "T": 1.0}


def chain_extractor(text: str):
    """`extractor(c) -> (parsed, error)` interface for
    `course_lib.callbacks.completion_metrics` / `course_lib.grpo_loop
    .run_grpo_stage`'s per-GRPO-step parse-rate logging -- same shape as
    tool_calling's `course_lib.tool_calling.calls.extract_call_args`, ported
    into this project via the `extractor` kwarg those two modules gained for
    exactly this purpose. `parsed` is the resolved-name dict; `error` is
    None iff every name in `course_lib.bs_reasoning.REQUIRED_STEPS` resolves,
    else `'incomplete_chain'`."""
    from . import REQUIRED_STEPS
    resolved = dict(evaluate_chain(text, _DUMMY_VALS))
    if all(step in resolved for step in REQUIRED_STEPS):
        return resolved, None
    return resolved, "incomplete_chain"
