"""The hand-rolled vol/rate disambiguation rule for bs_reasoning.

Reference implementation. Given to students as a comparison point for the
naturalistic-phrasing extension; never used as an SFT label source
(it must never leak a label) and never modified as part of the basic assignment.

Operates on an ALREADY-ISOLATED clause (the vol phrase or the rate phrase
alone, not the joined brief) -- the brief's field order is fixed and known
(S/K/T sentence, then vol clause, then rate clause), so clause-finding is
not the thing this rule is tested on; scale-disambiguation within a known
clause is (for example "23%" / ".23" / "23 vol").
"""
import re

NUM_RE = r"[-+]?\d*\.?\d+"


def rule_extract_vol(clause: str):
    m = re.search(NUM_RE, clause)
    if not m:
        return None
    val = float(m.group())
    tail = clause[m.end():m.end() + 12].lower()
    if re.search(r"sigma\s*=", clause[:m.start()], re.I):
        return val
    if "%" in tail or "percent" in tail:
        return val / 100
    if "." in m.group() and val < 1:
        return val
    return val / 100 if val >= 1 else val


def rule_extract_rate(clause: str):
    m = re.search(NUM_RE, clause)
    if not m:
        return None
    val = float(m.group())
    tail = clause[m.end():m.end() + 12].lower()
    if re.search(r"\br\s*=", clause[:m.start()], re.I):
        return val
    if "bp" in tail:
        return val / 10000
    if "%" in tail or "percent" in tail:
        return val / 100
    if "." in m.group() and val < 1:
        return val
    return val / 100 if val >= 1 else val
