"""RLSD for bs_reasoning: alternating RL and self-distillation phases. See
RLLM-12.

Each cycle runs K GRPO steps (`course_lib.grpo_loop.run_grpo_stage`,
unmodified), then one self-distillation consolidation round (sample, filter by
reward, SFT; the same functions as self_distill.py, which must sit in the same
folder), and repeats. Each cycle's GRPO phase uses that cycle's starting
checkpoint as its KL reference. If a self-distillation phase keeps nothing, the
cycle continues from the GRPO-only checkpoint.

Logs one JSON line per cycle to <ckpt-dir>/metrics.jsonl (flushed, fsynced);
re-running with the same --ckpt-dir resumes after the last completed cycle.

Usage:
  python part2_scripts/rlsd_train.py --help
  python part2_scripts/rlsd_train.py --smoke
  python part2_scripts/rlsd_train.py --sft-ckpt <your SFT checkpoint> \
      --cycles 4 --grpo-steps-per-cycle 50 --ckpt-dir runs/rlsd
"""
import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

# The bundle root is the folder that contains course_lib/ (this script sits one
# level below it, in part2_scripts/). self_distill.py is imported from this folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                                       # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer      # noqa: E402

from course_lib.formatting import format_prompt                   # noqa: E402
from course_lib.grpo_loop import load_recommended_g, run_grpo_stage  # noqa: E402
from course_lib.bs_reasoning.expr_harness import chain_extractor   # noqa: E402
from course_lib.bs_reasoning.reward import reward as bs_reward     # noqa: E402
from course_lib.bs_reasoning.scenarios import make_scenarios       # noqa: E402

from self_distill import (append_jsonl, build_filtered_rows,  # noqa: E402
                          diversity_probe, sft_on_rows)  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROJECT = "bs_reasoning"


def grpo_phase(ckpt, ckpt_dir, train, G, max_steps, save_steps,
              microbatch_size=8, max_completion_length=160,
              drive_dir=None, drive_bundle_name=None):
    """K steps of course_lib.grpo_loop.run_grpo_stage, unmodified (same
    scenario-pool/reward_fn pattern as your own GRPO training). `drive_dir`/`drive_bundle_name` thread straight into
    run_grpo_stage's own mid-phase Drive-push.
    Caller is responsible for pulling any existing mid-phase Drive
    checkpoint into `ckpt_dir` BEFORE calling this, so run_grpo_stage's
    own local-checkpoint resume scan finds it."""
    tok = AutoTokenizer.from_pretrained(ckpt)
    scenario_map = {}

    def prompt_for_step(rng):
        sc = train[rng.integers(len(train))]
        prompt = format_prompt(tok, PROJECT, sc.text)
        scenario_map[prompt] = sc
        return prompt

    def reward_fn(prompts, completions, **kw):
        return [bs_reward(c, scenario_map[p].params)[0] for p, c in zip(prompts, completions)]

    return run_grpo_stage(
        ckpt, ckpt_dir, prompt_for_step, reward_fn, None, None,
        num_generations=G, max_steps=max_steps, save_steps=save_steps,
        max_completion_length=max_completion_length,
        microbatch_size=microbatch_size, extractor=chain_extractor,
        drive_dir=drive_dir,
        drive_bundle_name=drive_bundle_name or "rlsd_grpo_ckpt.tar.gz",
    )


def _pull_grpo_midphase_checkpoint(drive_dir, bundle_name, grpo_dir):
    """Pull a mid-GRPO-phase Drive checkpoint (pushed by run_grpo_stage's
    own make_drive_push) into `grpo_dir` before calling grpo_phase, so its
    local-checkpoint resume scan (ckpt_dir.glob("checkpoint-*")) finds it
    on a fresh VM with no local disk state. No-op if nothing's there yet
    (normal for a cycle's first attempt)."""
    if not drive_dir:
        return
    src = Path(drive_dir) / bundle_name
    if src.exists():
        print(f"[resume] pulling mid-GRPO-phase checkpoint from {src}", flush=True)
        with tarfile.open(src) as tf:
            tf.extractall(grpo_dir, filter="data")


def _drive_push_cycle(drive_dir, cycle_final_dir, cycle, metrics_path, bundle_name):
    """Push the JUST-COMPLETED cycle's final checkpoint to Drive, with a
    cycle-number marker so a fresh VM's pull knows which cycle to resume
    from -- same pattern as self_distill.py's _drive_push, one level up
    (per-cycle instead of per-round)."""
    if not drive_dir or not (Path(drive_dir) / ".mounted").exists():
        if drive_dir:
            print(f"[drive] WARNING: sentinel missing at {drive_dir} -- "
                  f"mount lost? skipping push", flush=True)
        return
    meta_path = Path("/tmp/rlsd_cycle.json")
    meta_path.write_text(json.dumps({"cycle": cycle}))
    bundle = "/tmp/rlsd_cycle_push.tar.gz"
    with tarfile.open(bundle, "w:gz") as tf:
        tf.add(cycle_final_dir, arcname="cycle_latest")
        tf.add(metrics_path, arcname="metrics.jsonl")
        tf.add(meta_path, arcname="cycle.json")
    Path(drive_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(drive_dir) / bundle_name
    shutil.copy(bundle, dest)
    src_size = Path(bundle).stat().st_size
    dest_size = dest.stat().st_size if dest.exists() else -1
    if dest_size != src_size:
        print(f"[drive] WARNING: size mismatch after push -- "
              f"local={src_size} drive={dest_size}; push may not have landed",
              flush=True)
    print(f"[drive] pushed cycle {cycle} -> {dest} ({dest_size} bytes)", flush=True)


def _resume_cycle_state(ckpt_dir, drive_dir, bundle_name, sft_ckpt):
    """Cycle-level resume: pull the latest Drive bundle first (covers a
    fresh VM with no local disk state), then check locally either way.
    Returns (start_cycle, load_from_ckpt_path)."""
    if drive_dir:
        src = Path(drive_dir) / bundle_name
        if src.exists():
            print(f"[resume] pulling {src} from Drive", flush=True)
            with tarfile.open(src) as tf:
                tf.extractall(ckpt_dir, filter="data")

    pulled = ckpt_dir / "cycle.json"
    if pulled.exists():
        n = json.loads(pulled.read_text())["cycle"]
        pulled_ckpt = ckpt_dir / "cycle_latest"
        if pulled_ckpt.exists():
            print(f"[resume] Drive pull gives completed cycle {n} at {pulled_ckpt}",
                  flush=True)
            return n + 1, str(pulled_ckpt)

    for n in range(1000, 0, -1):
        for cand_name in (f"cycle{n}_sd", f"cycle{n}_grpo"):
            cand = ckpt_dir / cand_name / "final"
            if cand.exists():
                print(f"[resume] found local completed cycle {n} at {cand}", flush=True)
                return n + 1, str(cand)

    return 1, sft_ckpt


def self_distill_phase(ckpt, ckpt_dir, train, n_prompts, g, temperature,
                       reward_threshold, epochs, max_new_tokens, smoke_steps=None):
    """One self-distillation consolidation round (a single sample-filter-SFT
    pass, not self_distill.py's own multi-round loop -- the alternation
    schedule here IS the outer loop)."""
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32)
    if torch.cuda.is_available():
        model = model.cuda()

    rows, stats = build_filtered_rows(model, tok, train, n_prompts, g, temperature,
                                      reward_threshold, max_new_tokens=max_new_tokens)
    if stats["n_kept"] == 0:
        return None, stats

    model = sft_on_rows(model, tok, rows, ckpt_dir, epochs=epochs, max_steps=smoke_steps)
    final = Path(ckpt_dir) / "final"
    model.save_pretrained(final)
    tok.save_pretrained(final)
    return final, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny plumbing check on the untrained base model -- "
                         "2 cycles, 2 GRPO steps + 1 tiny self-distill round "
                         "each, accept-everything filter threshold")
    ap.add_argument("--sft-ckpt", default=MODEL)
    ap.add_argument("--ckpt-dir",
                    default="runs/rlsd")
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--grpo-steps-per-cycle", type=int, default=50)
    ap.add_argument("--grpo-save-steps", type=int, default=25)
    ap.add_argument("--num-generations", type=int, default=None,
                    help="G for the GRPO phase; if not given, read from "
                         "rl_readiness.json next to --ckpt-dir's parent "
                         "(course_lib.grpo_loop.load_recommended_g), else "
                         "that function's default. Better: pass your own "
                         "measured G")
    ap.add_argument("--sd-n-prompts", type=int, default=100)
    ap.add_argument("--sd-g", type=int, default=8)
    ap.add_argument("--sd-temperature", type=float, default=1.0)
    ap.add_argument("--sd-reward-threshold", type=float, default=1.0)
    ap.add_argument("--sd-epochs", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--drive-dir", default=None,
                    help="if set: pulls a mid-GRPO-phase checkpoint into "
                         "the current cycle's GRPO dir before each GRPO "
                         "phase (run_grpo_stage pushes its own progress "
                         "there during that phase); pulls the latest "
                         "COMPLETED cycle on startup (restart-from-failure); "
                         "pushes each completed cycle immediately "
                         "(restartable-from-failure rule for long Colab runs)")
    ap.add_argument("--drive-cycle-bundle-name", default="rlsd_ckpt.tar.gz")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cycles = args.cycles
    grpo_steps, grpo_save = args.grpo_steps_per_cycle, args.grpo_save_steps
    sd_n_prompts, sd_g = args.sd_n_prompts, args.sd_g
    sd_threshold, max_new_tokens = args.sd_reward_threshold, args.max_new_tokens
    grpo_max_completion_length = 160
    smoke_sd_steps = None
    if args.smoke:
        cycles, grpo_steps = 2, 2
        grpo_save = grpo_steps  # only push once (at the final step), not
                                 # every step -- a real checkpoint is ~1.6GB
                                 # (135M weights + AdamW state) and gzip
                                 # compression of that is genuinely slow;
                                 # once is enough to smoke-test the push path
        sd_n_prompts, sd_g = 4, 2
        sd_threshold = -1.0  # accept everything -- plumbing only, see self_distill.py
        max_new_tokens = 16
        grpo_max_completion_length = 8
        smoke_sd_steps = 3

    G = args.num_generations or load_recommended_g(ckpt_dir.parent / "rl_readiness.json")
    if args.smoke:
        G = 2

    train = make_scenarios(1302, seed=0)
    probe = make_scenarios(20, seed=1, held_out=True)[:10]

    metrics_log = open(ckpt_dir / "metrics.jsonl", "a")

    start_cycle, ckpt = _resume_cycle_state(
        ckpt_dir, args.drive_dir, args.drive_cycle_bundle_name, args.sft_ckpt)
    if start_cycle > cycles:
        print(f"[resume] all {cycles} cycles already complete", flush=True)

    for cycle in range(start_cycle, cycles + 1):
        print(f"[cycle {cycle}/{cycles}] GRPO phase ({grpo_steps} steps)...", flush=True)
        grpo_dir = ckpt_dir / f"cycle{cycle}_grpo"
        grpo_bundle_name = f"rlsd_cycle{cycle}_grpo_ckpt.tar.gz"
        _pull_grpo_midphase_checkpoint(args.drive_dir, grpo_bundle_name, grpo_dir)
        ckpt = str(grpo_phase(ckpt, grpo_dir, train, G, grpo_steps, grpo_save,
                              max_completion_length=grpo_max_completion_length,
                              drive_dir=args.drive_dir,
                              drive_bundle_name=grpo_bundle_name))

        print(f"[cycle {cycle}/{cycles}] self-distillation phase...", flush=True)
        sd_dir = ckpt_dir / f"cycle{cycle}_sd"
        sd_ckpt, sd_stats = self_distill_phase(
            ckpt, sd_dir, train, sd_n_prompts, sd_g, args.sd_temperature,
            sd_threshold, args.sd_epochs, max_new_tokens, smoke_steps=smoke_sd_steps)
        print(f"[cycle {cycle}] self-distill stats: {json.dumps(sd_stats)}", flush=True)

        if sd_ckpt is None:
            print(f"[cycle {cycle}] self-distill phase kept 0 completions -- "
                  f"staying on the GRPO-only checkpoint for this cycle "
                  f"(the GRPO phase alone still ran; not a hard failure the "
                  f"way self_distill.py's own standalone gate is).", flush=True)
        else:
            ckpt = str(sd_ckpt)

        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32)
        if torch.cuda.is_available():
            model = model.cuda()
        ok, report = diversity_probe(model, tok, probe, max_new_tokens=max_new_tokens)
        print(f"[cycle {cycle}] check_output_varies: {'PASS' if ok else 'FAIL'} "
              f"per_key_relative_range={report['per_key_relative_range']}", flush=True)

        append_jsonl(metrics_log, dict(
            cycle=cycle, ckpt=ckpt, self_distill=sd_stats,
            diversity_ok=ok, per_key_relative_range=report["per_key_relative_range"]))

        # Cycle-level checkpoint: push whichever checkpoint this cycle
        # ended on (sd phase's output, or the GRPO-only one if sd kept 0
        # rows) so a fresh VM resumes from here, not from cycle 1.
        _drive_push_cycle(args.drive_dir, Path(ckpt), cycle,
                          ckpt_dir / "metrics.jsonl", args.drive_cycle_bundle_name)
        # Prune older cycles locally (keep only what this cycle needs --
        # resume only ever reads the latest via the Drive push above or the
        # local scan fallback). Narrow glob (cycle[0-9]*) so this never
        # touches cycle_latest/cycle.json, the resume-pull artifacts
        # cleaned up separately below.
        for old in ckpt_dir.glob("cycle[0-9]*"):
            if old.name not in (f"cycle{cycle}_grpo", f"cycle{cycle}_sd"):
                shutil.rmtree(old, ignore_errors=True)
        for stale in ("cycle_latest", "cycle.json"):
            p = ckpt_dir / stale
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()

    metrics_log.close()
    final = ckpt_dir / "final"
    shutil.copytree(ckpt, final, dirs_exist_ok=True)
    print(f"[done] -> {final}", flush=True)


if __name__ == "__main__":
    main()
