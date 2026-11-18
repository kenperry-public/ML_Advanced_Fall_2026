"""On-policy self-distillation for bs_reasoning (STaR / expert iteration).

Each round: sample G completions per prompt from the CURRENT policy, keep the
ones that clear a reward threshold (scored with the same reward GRPO uses),
then SFT on the kept (prompt, completion) pairs, and repeat from the updated
checkpoint. `--filter-mode topk` switches the filter to the RAFT shape (per
prompt, keep the top-k completions by reward, whatever their absolute reward);
only the filter differs. See RLLM-9.

Every round it prints and logs (metrics.jsonl, one JSON line per round, flushed
and fsynced): sampled/kept counts, keep rate, a cumulative reward histogram,
and `check_output_varies` on a small held-out probe decoded greedily (the same
way a held-out evaluation decodes). A run restarts from the last completed
round if re-run with the same --ckpt-dir.

Usage (from anywhere; the script finds the bundle root as its parent folder):
  python part2_scripts/self_distill.py --help
  python part2_scripts/self_distill.py --smoke
  python part2_scripts/self_distill.py --sft-ckpt <your SFT checkpoint> \
      --rounds 4 --n-prompts 200 --g 8 --ckpt-dir runs/self_distill
  ... --filter-mode topk --topk 1        # RAFT arm
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

# The bundle root is the folder that contains course_lib/ (this script sits one
# level below it, in part2_scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                     # noqa: E402
from datasets import Dataset                                     # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer      # noqa: E402
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer  # noqa: E402

from course_lib.formatting import format_prompt                  # noqa: E402
from course_lib.policy_checks import check_output_varies         # noqa: E402
from course_lib.bs_reasoning.expr_harness import evaluate_chain  # noqa: E402
from course_lib.bs_reasoning.reward import reward as bs_reward   # noqa: E402
from course_lib.bs_reasoning.scenarios import make_scenarios     # noqa: E402


MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROJECT = "bs_reasoning"
DIVERSITY_RANGES = dict(sigma=(0.10, 0.50), r=(0.01, 0.06), price=(0.0, 150.0))
REWARD_THRESHOLD = 0.7  # a starting point, not a recommendation: choose yours from
# the reward histogram this script prints (and check what it keeps).

def sample_completions(model, tok, prompt, g, max_new_tokens=160, temperature=1.0,
                       greedy=False):
    """g sampled completions of ONE prompt (decoded, prompt stripped).
    greedy=True decodes deterministically (one completion), the way a
    held-out evaluation decodes."""
    enc = tok([prompt], return_tensors="pt", padding=True,
              padding_side="left").to(model.device)
    with torch.no_grad():
        if greedy:
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 num_return_sequences=1,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        else:
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=temperature, num_return_sequences=g,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def append_jsonl(fh, rec):
    """One JSON line, flushed and fsynced so a crash loses at most this line."""
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


HISTOGRAM_BUCKETS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0)


def _reward_histogram(rewards):
    """Cumulative counts at each bucket ('>=B': how many rewards clear B) --
    for picking a reward_threshold from data instead of guessing one."""
    return {f">={b}": sum(1 for r in rewards if r >= b) for b in HISTOGRAM_BUCKETS}


def build_filtered_rows(model, tok, scenarios, n_prompts, g, temperature,
                        reward_threshold, max_new_tokens=160,
                        filter_mode="threshold", topk=1):
    """Sample g completions per prompt (n_prompts scenarios), then filter.

    filter_mode="threshold" (self-distillation/STaR shape, the original
    behavior): keep every completion, pooled across all prompts, whose
    reward clears reward_threshold -- a prompt whose best sample doesn't
    clear the bar contributes nothing this round.

    filter_mode="topk" (RAFT shape -- canonical best-of-G rejection
    sampling, topk=1): keep, PER PROMPT, only that
    prompt's top-`topk` completions by reward, UNCONDITIONALLY -- every
    prompt contributes exactly `topk` rows regardless of absolute reward
    level. reward_threshold is ignored in this mode.

    Returns (rows, stats) -- rows are {"prompt", "completion"} dicts, stats
    has n_sampled/n_kept/keep_rate/reward_histogram (cumulative counts at
    HISTOGRAM_BUCKETS, computed the same way regardless of filter_mode, so
    the two modes' sampled reward distributions stay directly comparable)."""
    rows = []
    n_sampled = n_kept = 0
    rewards = []
    for sc in scenarios[:n_prompts]:
        prompt = format_prompt(tok, PROJECT, sc.text)
        scored = []
        for c in sample_completions(model, tok, prompt, g,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature):
            n_sampled += 1
            r, _ = bs_reward(c, sc.params)
            rewards.append(r)
            scored.append((r, c))
        if filter_mode == "topk":
            scored.sort(key=lambda rc: rc[0], reverse=True)
            keep = scored[:topk]
        else:
            keep = [(r, c) for r, c in scored if r >= reward_threshold]
        for _, c in keep:
            n_kept += 1
            rows.append({"prompt": prompt, "completion": c})
    stats = dict(n_sampled=n_sampled, n_kept=n_kept,
                 keep_rate=n_kept / n_sampled if n_sampled else 0.0,
                 reward_histogram=_reward_histogram(rewards))
    return rows, stats


def sft_on_rows(model, tok, rows, ckpt_dir, epochs=1.0, max_steps=None):
    """One SFTTrainer pass on filtered self-samples -- same {"text":
    prompt+completion+eos} row shape and collator as sft_train.py."""
    ds = Dataset.from_list([
        {"text": r["prompt"] + r["completion"] + tok.eos_token} for r in rows
    ])
    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n", tokenizer=tok
    )
    cfg = SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=epochs,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        fp16=False, bf16=False,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        max_seq_length=320,
        dataset_text_field="text",
        **(dict(max_steps=max_steps) if max_steps else {}),
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         data_collator=collator, processing_class=tok)
    trainer.train()
    return model


def diversity_probe(model, tok, probe_scenarios, max_new_tokens=160):
    """check_output_varies on the CURRENT policy against real S/K/T --
    entropy-collapse detector, run every round. Decodes greedily, matching
    how a held-out evaluation decodes: a probe that samples at temperature
    can show spread that the policy's greedy output no longer has."""
    params_list = []
    for sc in probe_scenarios:
        prompt = format_prompt(tok, PROJECT, sc.text)
        completion = sample_completions(model, tok, prompt, 1,
                                        max_new_tokens=max_new_tokens,
                                        greedy=True)[0]
        resolved = dict(evaluate_chain(
            completion, {"S": sc.params.S, "K": sc.params.K, "T": sc.params.T}))
        if all(k in resolved for k in ("sigma", "r", "price")):
            params_list.append({k: resolved[k] for k in ("sigma", "r", "price")})
        else:
            params_list.append(None)
    return check_output_varies(params_list, DIVERSITY_RANGES,
                               require_keys=("sigma", "r"))


def _drive_push(drive_dir, round_final_dir, round_i, metrics_path, bundle_name):
    """Push the JUST-COMPLETED round's checkpoint to Drive -- same
    durability pattern as grpo_loop.make_drive_push (tar, copy, size-check
    readback), but round-count metadata travels alongside so a fresh VM's
    pull knows which round it's resuming from without depending on local
    directory names surviving. Always the SAME bundle name (overwritten
    each round), not one bundle per round -- matches grpo_loop's "prune
    older, push only current" durability-vs-quota tradeoff."""
    if not drive_dir or not (Path(drive_dir) / ".mounted").exists():
        if drive_dir:
            print(f"[drive] WARNING: sentinel missing at {drive_dir} -- "
                  f"mount lost? skipping push", flush=True)
        return
    meta_path = Path("/tmp/self_distill_round.json")
    meta_path.write_text(json.dumps({"round": round_i}))
    bundle = "/tmp/self_distill_push.tar.gz"
    with tarfile.open(bundle, "w:gz") as tf:
        tf.add(round_final_dir, arcname="round_latest")
        tf.add(metrics_path, arcname="metrics.jsonl")
        tf.add(meta_path, arcname="round.json")
    Path(drive_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(drive_dir) / bundle_name
    shutil.copy(bundle, dest)
    src_size = Path(bundle).stat().st_size
    dest_size = dest.stat().st_size if dest.exists() else -1
    if dest_size != src_size:
        print(f"[drive] WARNING: size mismatch after push -- "
              f"local={src_size} drive={dest_size}; push may not have landed",
              flush=True)
    print(f"[drive] pushed round {round_i} -> {dest} ({dest_size} bytes)", flush=True)


def _resume_state(ckpt_dir, drive_dir, bundle_name, sft_ckpt):
    """Resume-from-failure: pull the latest Drive bundle first (covers a
    fresh VM with no local disk state), then check locally either way for
    the highest completed round. Returns (start_round, load_from_ckpt_path,
    round_dir_to_reuse_or_None)."""
    if drive_dir:
        src = Path(drive_dir) / bundle_name
        if src.exists():
            print(f"[resume] pulling {src} from Drive", flush=True)
            with tarfile.open(src) as tf:
                tf.extractall(ckpt_dir, filter="data")

    pulled_round = ckpt_dir / "round.json"
    if pulled_round.exists():
        n = json.loads(pulled_round.read_text())["round"]
        pulled_ckpt = ckpt_dir / "round_latest"
        if pulled_ckpt.exists():
            print(f"[resume] Drive pull gives completed round {n} at {pulled_ckpt}",
                  flush=True)
            return n + 1, str(pulled_ckpt), None

    # Local-only fallback (same VM, no Drive pull needed/available): scan
    # for the highest round{N}/final that already exists.
    for n in range(1000, 0, -1):
        cand = ckpt_dir / f"round{n}" / "final"
        if cand.exists():
            print(f"[resume] found local completed round {n} at {cand}", flush=True)
            return n + 1, str(cand), None

    return 1, sft_ckpt, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny plumbing check on the untrained base model -- "
                         "does NOT test real self-distillation quality, just "
                         "that sampling/filtering/SFT/save don't crash")
    ap.add_argument("--sft-ckpt", default=MODEL,
                    help="starting checkpoint (the SFT checkpoint GRPO also "
                         "started from, for a fair comparison); defaults to "
                         "the untrained base model for --smoke only")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--n-prompts", type=int, default=200,
                    help="distinct training scenarios sampled per round")
    ap.add_argument("--g", type=int, default=8, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--reward-threshold", type=float, default=REWARD_THRESHOLD)
    ap.add_argument("--filter-mode", choices=("threshold", "topk"),
                    default="threshold",
                    help="'threshold' (default): self-distillation/STaR "
                         "shape, pooled reward_threshold filter. 'topk': "
                         "RAFT shape, per-prompt best-of-g rank filter "
                         "(reward_threshold ignored)")
    ap.add_argument("--topk", type=int, default=1,
                    help="only used with --filter-mode topk; RAFT "
                         "uses 1 (best-of-g)")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--sft-epochs", type=float, default=1.0,
                    help="epochs per round's SFT pass over that round's "
                         "filtered rows")
    ap.add_argument("--ckpt-dir",
                    default="runs/self_distill")
    ap.add_argument("--drive-dir", default=None,
                    help="if set, pull the latest round from here on startup "
                         "(restart-from-failure) and push each completed "
                         "round here immediately (for long Colab runs)")
    ap.add_argument("--drive-bundle-name", default="self_distill_ckpt.tar.gz")
    args = ap.parse_args()

    if args.smoke:
        args.n_prompts, args.g, args.rounds = 4, 2, 2
        args.reward_threshold = -1.0  # accept anything -- plumbing only, see docstring
        args.max_new_tokens = 16
        smoke_sft_steps = 3
    else:
        smoke_sft_steps = None

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_round, load_from, _ = _resume_state(
        ckpt_dir, args.drive_dir, args.drive_bundle_name, args.sft_ckpt)

    tok = AutoTokenizer.from_pretrained(load_from)
    model = AutoModelForCausalLM.from_pretrained(load_from, torch_dtype=torch.float32)
    if torch.cuda.is_available():
        model = model.cuda()

    train = make_scenarios(1302, seed=0)  # the same training scenarios as your SFT and GRPO runs
    probe = make_scenarios(20, seed=1, held_out=True)[:10]  # small held-out diversity probe

    metrics_log = open(ckpt_dir / "metrics.jsonl", "a")

    if start_round > args.rounds:
        print(f"[resume] all {args.rounds} rounds already complete", flush=True)

    for round_i in range(start_round, args.rounds + 1):
        print(f"[round {round_i}/{args.rounds}] sampling + filtering...", flush=True)
        rows, sample_stats = build_filtered_rows(
            model, tok, train, args.n_prompts, args.g, args.temperature,
            args.reward_threshold, max_new_tokens=args.max_new_tokens,
            filter_mode=args.filter_mode, topk=args.topk)
        print(f"[round {round_i}] {json.dumps(sample_stats)}", flush=True)

        if sample_stats["n_kept"] == 0:
            print(f"[round {round_i}] EXPLORATION GATE FAILED: zero kept "
                  f"completions -- turn a knob (temperature, g, threshold) "
                  f"before proceeding.", flush=True)
            append_jsonl(metrics_log, dict(round=round_i, **sample_stats,
                                           gate_failed=True))
            return 1

        model = sft_on_rows(model, tok, rows, ckpt_dir / f"round{round_i}",
                            epochs=args.sft_epochs, max_steps=smoke_sft_steps)

        ok, report = diversity_probe(model, tok, probe,
                                     max_new_tokens=args.max_new_tokens)
        print(f"[round {round_i}] check_output_varies: "
              f"{'PASS' if ok else 'FAIL'} "
              f"per_key_relative_range={report['per_key_relative_range']}",
              flush=True)
        append_jsonl(metrics_log, dict(
            round=round_i, **sample_stats, diversity_ok=ok,
            per_key_relative_range=report["per_key_relative_range"]))

        round_ckpt = ckpt_dir / f"round{round_i}" / "final"
        model.save_pretrained(round_ckpt)
        tok.save_pretrained(round_ckpt)
        # Prune older rounds locally (keep only the latest -- resume only
        # ever needs it, same disk-vs-durability tradeoff as
        # grpo_loop.make_drive_push's checkpoint pruning). Narrow glob
        # (round[0-9]*) so this never touches round.json/round_latest, the
        # resume-pull artifacts handled separately below.
        for old in ckpt_dir.glob("round[0-9]*"):
            if old != round_ckpt.parent:
                shutil.rmtree(old)
        for stale in ("round_latest", "round.json"):
            p = ckpt_dir / stale
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        _drive_push(args.drive_dir, round_ckpt, round_i,
                   ckpt_dir / "metrics.jsonl", args.drive_bundle_name)

    metrics_log.close()
    final = ckpt_dir / "final"
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"[done] -> {final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
