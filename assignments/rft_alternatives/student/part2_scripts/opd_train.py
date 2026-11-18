"""On-policy distillation (OPD) for bs_reasoning. See RLLM-11 and RLLM-12.

Structure mirrors `course_lib/grpo_loop.py::run_grpo_stage` (generate ->
loss -> optimizer step, checkpoint/resume, microbatching); the one new piece
is the loss:

  GRPO: -(group_relative_advantage * student_log_prob) + beta * KL(student ||
        frozen SFT reference): one scalar reward per completion.
  OPD:  forward-KL(teacher_distribution || student_distribution), summed over
        the vocab, at EVERY completion token position: dense, and needs no
        reward function, only a teacher's distribution over the SAME tokens
        the student sampled.

THE TEACHER IS YOURS TO CHOOSE; this script does not supply one. Pass
`--teacher MODEL_PATH_OR_HF_ID`, any frozen causal LM loadable with
transformers that shares the student's tokenizer vocabulary (checked at
startup). By default the teacher sees the same system message + scenario the
student sees. Optionally `--teacher-prompt-file FILE`: text you write, appended
to the teacher's system message (teacher-side context only; the student never
sees it). The teacher never generates; it only scores the student's tokens.

Logs one JSON line per step to <ckpt-dir>/metrics.jsonl (flushed, fsynced);
re-running with the same --ckpt-dir resumes from the latest checkpoint.

Usage:
  python part2_scripts/opd_train.py --help
  python part2_scripts/opd_train.py --smoke --teacher <model> --sft-ckpt <model>
  python part2_scripts/opd_train.py --sft-ckpt <your SFT checkpoint> \
      --teacher <your teacher> --num-generations 8 --max-steps 200
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

# The bundle root is the folder that contains course_lib/ (this script sits one
# level below it, in part2_scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
import torch.nn.functional as F                                   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer       # noqa: E402

from course_lib.formatting import SYSTEM, format_prompt                   # noqa: E402
from course_lib.grpo_loop import build_completion_mask             # noqa: E402
from course_lib.bs_reasoning.scenarios import make_scenarios       # noqa: E402


MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROJECT = "bs_reasoning"


def load_teacher(path_or_id, device):
    """Frozen, inference-only teacher: no gradients, no optimizer."""
    tok = AutoTokenizer.from_pretrained(path_or_id)
    model = AutoModelForCausalLM.from_pretrained(path_or_id, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tok


def teacher_prompt_ids(teacher_tok, scenario_text, device, extra_system=""):
    """The teacher's own context for this scenario: the course system message
    (plus any text you wrote in --teacher-prompt-file) and the scenario."""
    system = SYSTEM[PROJECT] + (("\n\n" + extra_system) if extra_system else "")
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": scenario_text}]
    prompt = teacher_tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
    return teacher_tok(prompt, return_tensors="pt").input_ids.to(device)


def teacher_completion_distribution(teacher_model, teacher_prompt_ids_row, completion_ids):
    """Teacher's softmax over the vocab at each position of `completion_ids`
    (the student's sampled token ids, scored under the teacher's own context).
    `teacher_prompt_ids_row` is (1, prompt_len); `completion_ids` is
    (batch, comp_len). Returns (batch, comp_len, vocab); no_grad."""
    batch = completion_ids.shape[0]
    prompt_ids = teacher_prompt_ids_row.expand(batch, -1)
    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    prompt_len = prompt_ids.shape[1]
    with torch.no_grad():
        logits = teacher_model(input_ids=full_ids).logits
    comp_logits = logits[:, prompt_len - 1:prompt_len - 1 + completion_ids.shape[1]]
    return F.softmax(comp_logits.float(), dim=-1)


def student_completion_log_probs(model, input_ids, attention_mask, prompt_len):
    """Full log_softmax over the vocab at each completion position (unlike
    grpo_loop.token_log_probs, which only gathers the sampled token's own
    log-prob -- OPD's loss needs the student's full distribution to compare
    against the teacher's)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    return logp[:, prompt_len - 1:]


def distillation_loss(student_log_probs, teacher_probs, completion_mask):
    """Forward-KL(teacher || student), summed over the vocab, masked to real
    (non-padding) completion positions, averaged per sequence then over the
    microbatch. `student_log_probs`/`teacher_probs` are (batch, comp_len,
    vocab); `completion_mask` is (batch, comp_len)."""
    per_tok_kl = (teacher_probs * (teacher_probs.clamp_min(1e-12).log()
                                    - student_log_probs)).sum(dim=-1)
    denom = completion_mask.sum(dim=1).clamp(min=1)
    seq_kl = (per_tok_kl * completion_mask).sum(dim=1) / denom
    return seq_kl


def run_opd_stage(sft_ckpt, ckpt_dir, prompt_for_step, *,
                  num_generations, max_steps, save_steps,
                  temperature=1.0, learning_rate=5e-6,
                  max_completion_length=48, microbatch_size=4,
                  drive_dir=None, drive_bundle_name="opd_ckpt.tar.gz", seed=0,
                  teacher=None, teacher_prompt_text=""):
    """Run OPD training. Returns the final checkpoint path. Structure mirrors
    `run_grpo_stage` (resume, checkpoint, metrics.jsonl); `teacher` is a model
    path or hub id, `teacher_prompt_text` optional teacher-side system text."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    G = num_generations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device} G={G} max_steps={max_steps} seed={seed}", flush=True)
    # Seed generation sampling as well as the scenario-index rng, so a different
    # --seed is a genuine replicate.
    torch.manual_seed(seed)

    tok = AutoTokenizer.from_pretrained(sft_ckpt)
    pad_id = tok.pad_token_id or tok.eos_token_id

    # Pull any existing mid-training checkpoint from Drive first (fresh VM).
    if drive_dir:
        drive_ckpt = Path(drive_dir) / drive_bundle_name
        if drive_ckpt.exists():
            print(f"[resume] pulling {drive_ckpt} from Drive", flush=True)
            with tarfile.open(drive_ckpt) as tf:
                tf.extractall(ckpt_dir, filter="data")

    resume_dir, start_step = None, 0
    existing = list(ckpt_dir.glob("checkpoint-*"))
    if existing:
        resume_dir = max(existing, key=lambda p: int(p.name.split("-")[1]))
        start_step = int(resume_dir.name.split("-")[1])
        print(f"[resume] loading {resume_dir} (step {start_step})", flush=True)

    load_from = str(resume_dir) if resume_dir else sft_ckpt
    model = AutoModelForCausalLM.from_pretrained(load_from, torch_dtype=torch.float32).to(device)

    teacher_model, teacher_tok = load_teacher(teacher, device)
    if teacher_tok.get_vocab() != tok.get_vocab():
        raise SystemExit("teacher and student tokenizer vocabularies differ; OPD "
                         "scores the student's token ids under the teacher, so they "
                         "must match")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if resume_dir and (resume_dir / "optimizer.pt").exists():
        state = torch.load(resume_dir / "optimizer.pt", map_location=device)
        optimizer.load_state_dict(state["optimizer"])

    metrics_log = open(ckpt_dir / "metrics.jsonl", "a")
    rng = np.random.default_rng(seed)

    for step in range(start_step + 1, max_steps + 1):
        t0 = time.perf_counter()
        prompt, scenario_text = prompt_for_step(rng)

        # 1. Generate on-policy from the STUDENT -- eval mode, no_grad, same
        # single generate() call GRPO uses.
        model.eval()
        enc = tok([prompt] * G, return_tensors="pt", padding=True, padding_side="left")
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_completion_length,
                                 do_sample=True, temperature=temperature,
                                 pad_token_id=pad_id)
        completion_ids = out[:, prompt_len:]

        # 2. Teacher scores the SAME sampled tokens under its own context
        # (no reward function anywhere).
        t_prompt_ids = teacher_prompt_ids(teacher_tok, scenario_text, device,
                                          extra_system=teacher_prompt_text)

        comp_mask = build_completion_mask(completion_ids, tok.eos_token_id, pad_id)
        attn_mask = torch.cat([enc["attention_mask"], comp_mask.long()], dim=1)

        model.train()
        optimizer.zero_grad()
        total_kl = 0.0
        for i in range(0, G, microbatch_size):
            sl = slice(i, min(i + microbatch_size, G))
            ids_mb, mask_mb, cmask_mb = out[sl], attn_mask[sl], comp_mask[sl]
            comp_ids_mb = completion_ids[sl]

            student_lp = student_completion_log_probs(model, ids_mb, mask_mb, prompt_len)
            teacher_p = teacher_completion_distribution(
                teacher_model, t_prompt_ids, comp_ids_mb)

            seq_kl = distillation_loss(student_lp, teacher_p, cmask_mb)
            (seq_kl.sum() / G).backward()
            total_kl += float(seq_kl.detach().sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        rec = dict(step=step, kl_loss=total_kl / G,
                   step_secs=time.perf_counter() - t0)
        print(f"[metrics] {json.dumps(rec)}", flush=True)
        metrics_log.write(json.dumps(rec) + "\n")
        metrics_log.flush()
        os.fsync(metrics_log.fileno())

        if step % save_steps == 0 or step == max_steps:
            ck = ckpt_dir / f"checkpoint-{step}"
            model.save_pretrained(ck)
            tok.save_pretrained(ck)
            torch.save({"step": step, "optimizer": optimizer.state_dict()}, ck / "optimizer.pt")
            for old in ckpt_dir.glob("checkpoint-*"):
                if old != ck:
                    shutil.rmtree(old)
            if drive_dir:
                _drive_push(drive_dir, ck, ckpt_dir / "metrics.jsonl", drive_bundle_name)

    final = ckpt_dir / "final"
    model.save_pretrained(final)
    tok.save_pretrained(final)
    metrics_log.close()
    print(f"[done] -> {final}", flush=True)
    return final


def _drive_push(drive_dir, ck, metrics_path, bundle_name):
    """Same durability pattern as grpo_loop.make_drive_push (tar the latest
    checkpoint + metrics, copy to Drive)."""
    if not (Path(drive_dir) / ".mounted").exists():
        print(f"[drive] WARNING: sentinel missing at {drive_dir} -- mount lost? skipping push",
              flush=True)
        return
    bundle = "/tmp/opd_ckpt_push.tar.gz"
    with tarfile.open(bundle, "w:gz") as tf:
        tf.add(ck, arcname=ck.name)
        if metrics_path.exists():
            tf.add(metrics_path, arcname="metrics.jsonl")
    Path(drive_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(drive_dir) / bundle_name
    shutil.copy(bundle, dest)
    print(f"[drive] pushed -> {dest}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny plumbing check -- G=2, 2 steps, 8 new tokens "
                         "(use a small --teacher; it is still loaded)")
    ap.add_argument("--sft-ckpt", default=MODEL)
    ap.add_argument("--teacher", required=True,
                    help="frozen teacher: a local path or Hugging Face id, "
                         "loaded with transformers; must share the student's "
                         "tokenizer vocabulary. Your choice; none is supplied")
    ap.add_argument("--teacher-prompt-file", default=None,
                    help="optional text file you wrote; its contents are "
                         "appended to the TEACHER's system message only")
    ap.add_argument("--ckpt-dir", default="runs/opd")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    ap.add_argument("--max-completion-length", type=int, default=160)
    ap.add_argument("--microbatch-size", type=int, default=4)
    ap.add_argument("--drive-dir", default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="scenario-index rng AND torch.manual_seed; vary it "
                         "for a genuine training-randomness replicate")
    args = ap.parse_args()

    G, max_steps, save_steps, max_completion_length = (
        args.num_generations, args.max_steps, args.save_steps,
        args.max_completion_length)
    if args.smoke:
        G, max_steps, save_steps, max_completion_length = 2, 2, 1, 8

    teacher_prompt_text = ""
    if args.teacher_prompt_file:
        teacher_prompt_text = Path(args.teacher_prompt_file).read_text().strip()

    tok_for_scenarios = AutoTokenizer.from_pretrained(args.sft_ckpt)
    train = make_scenarios(1302, seed=0)  # the same training scenarios as your SFT and GRPO runs

    def prompt_for_step(rng):
        sc = train[rng.integers(len(train))]
        prompt = format_prompt(tok_for_scenarios, PROJECT, sc.text)
        return prompt, sc.text

    run_opd_stage(
        args.sft_ckpt, args.ckpt_dir, prompt_for_step,
        num_generations=G, max_steps=max_steps, save_steps=save_steps,
        temperature=args.temperature, learning_rate=args.learning_rate,
        max_completion_length=max_completion_length,
        microbatch_size=args.microbatch_size, drive_dir=args.drive_dir,
        seed=args.seed, teacher=args.teacher,
        teacher_prompt_text=teacher_prompt_text,
    )


if __name__ == "__main__":
    main()
