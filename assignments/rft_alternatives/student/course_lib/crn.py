"""Common-random-numbers reward wrapper shared by every course project.

Requirement: all G completions of the
same prompt in the same optimizer step share randomness — policies inside a
GRPO group are compared on identical luck — and randomness is fresh across
steps. TRL's reward function receives the prompts list with each prompt
repeated G times, so keying the seed on (step, prompt) does both jobs.

Every `simulator.score` draws ALL of its randomness from
`np.random.default_rng(seed)`: identical seed ⇒ identical paths/uniforms
within a group. What "all its randomness" means per project (exogenous paths
vs. common uniforms) is defined in that project's instructor doc.
"""
import hashlib
import itertools


class CRNSeeder:
    def __init__(self):
        self.step = itertools.count()

    def begin_step(self):
        """Call exactly once per reward invocation (= once per GRPO step)."""
        self.cur = next(self.step)

    def seed(self, prompt):
        h = hashlib.md5(f"{self.cur}|{prompt}".encode()).digest()
        return int.from_bytes(h[:4], "little")


def make_reward_fn(simulator, scenario_from_prompt, seeder=None):
    """Wrap `simulator.score` as a TRL 0.14 reward function.

    `scenario_from_prompt` maps the exact prompt string TRL passes back to its
    Scenario object. Build it as a dict lookup at dataset-construction time —
    never re-parse natural language. A missing key must raise (loudly), not
    return a penalty: a KeyError here means the prompt formatting differs
    between the dataset and this map, which is a pipeline bug, not a model
    failure.
    """
    seeder = seeder or CRNSeeder()

    def reward_fn(prompts, completions, **kw):
        seeder.begin_step()
        return [
            simulator.score(scenario_from_prompt(p), c, seed=seeder.seed(p))
            for p, c in zip(prompts, completions)
        ]

    return reward_fn
