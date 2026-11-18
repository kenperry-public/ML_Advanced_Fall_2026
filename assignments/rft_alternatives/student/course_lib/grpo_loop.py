"""Hand-rolled GRPO training loop — given course infrastructure.

This module IS the RL trainer for every SFT+GRPO project in the course. You
do not write a training loop; you read this one, and you interact with it
through three designated touch-points, all passed into `run_grpo_stage`:

  1. `reward_fn` — your project's CRN-wrapped reward (course_lib.crn).
  2. `num_generations` (G) — measured for YOUR project via
     `course_lib.rl_readiness.readiness_sweep`, never inherited.
  3. `advantage_fn` — defaults to group-relative (r - mean)/std; swap it only
     if your handout sets an experiment that says to.

Everything else (generation, teacher-forced loss, KL to the frozen SFT
reference, checkpoint/resume, Drive durability) is deliberately fixed, and
deliberately visible: ~200 lines of plain PyTorch instead of a framework
call, so that when a run misbehaves you can read the objective you are
actually optimizing.

Why not TRL's GRPOTrainer? Its `compute_loss` generates completions while
the model is still in `.train()` mode, and under `gradient_checkpointing=True`
(a plausible memory-saving move on a 16GB T4) that silently corrupts both
generation (Llama's forward() drops the KV cache when
`self.training and gradient_checkpointing`) and gradients
(`gradient_checkpointing_enable()` never calls `enable_input_require_grads()`
for a non-PEFT model). Here that failure class is structurally impossible:
generation (the ONLY `model.generate` call, eval mode, no_grad) and the
gradient-bearing forward pass (teacher-forced, over tokens already sampled)
are separate, explicit calls — nothing in generate() ever observes training
state, and nothing in the training forward pass ever generates. Memory is
tuned directly via `microbatch_size` instead of activation checkpointing.

Each optimizer step processes G sampled completions of ONE prompt (a single
GRPO group). The caller supplies `prompt_for_step(rng)` so fixed-scenario
(Stage 1/2) vs pool-sampled (Stage 3) is the driver's choice.
"""
import json
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .callbacks import completion_metrics, group_stats

DEFAULT_G = 16  # fallback when no rl_readiness.json is present (loud warning)


def load_recommended_g(readiness_path, tied_threshold=0.05, default_g=DEFAULT_G):
    """Read a readiness sweep's JSON output (rl_readiness_check driver) and
    pick the smallest candidate G with a healthy (low) frac_all_tied.
    Starting heuristic, not a hard rule — eyeball the table too."""
    path = Path(readiness_path)
    if not path.exists():
        print(f"[warn] no {path} — run your rl_readiness_check driver first; "
              f"falling back to G={default_g}", flush=True)
        return default_g
    results = json.loads(path.read_text())
    ready = [int(g) for g, r in results.items() if r["frac_all_tied"] <= tied_threshold]
    if not ready:
        print(f"[warn] no candidate G in {path} clears frac_all_tied<={tied_threshold}; "
              f"falling back to G={default_g}", flush=True)
        return default_g
    g = min(ready)
    print(f"[readiness] picked G={g} from {path}", flush=True)
    return g


def build_completion_mask(completion_ids, eos_id, pad_id):
    """Mask out padding after the first EOS/pad in each sampled completion
    (inclusive of the EOS token itself) — generated filler shouldn't count
    toward the policy-gradient or KL loss."""
    is_end = (completion_ids == eos_id) | (completion_ids == pad_id)
    L = completion_ids.shape[1]
    idx = torch.arange(L, device=completion_ids.device).unsqueeze(0).expand_as(completion_ids)
    fallback = torch.full((completion_ids.shape[0],), L, device=completion_ids.device)
    first_end = torch.where(is_end.any(dim=1), is_end.float().argmax(dim=1), fallback)
    return (idx <= first_end.unsqueeze(1)).float()


def token_log_probs(model, input_ids, attention_mask, prompt_len):
    """Per-token log-prob of the actually-sampled completion tokens (teacher
    forcing over the already-generated ids — no re-generation, so this
    forward pass never touches generation/caching machinery)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    tok_lp = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return tok_lp[:, prompt_len - 1:]


def group_advantage(rewards, eps=1e-4):
    """Group-relative advantage: (r - mean)/(std + eps) over one group.
    The designated touch-point for advantage experiments."""
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)


def make_drive_push(drive_dir, ckpt_dir, bundle_name):
    """Push a checkpoint bundle to Drive with the durability checks every
    long Colab run in this course needs (restartable-from-failure rule)."""
    def drive_push(step):
        if not drive_dir:
            return
        if not (Path(drive_dir) / ".mounted").exists():
            print(f"[drive] WARNING: sentinel missing at {drive_dir} — "
                  f"mount lost? skipping push", flush=True)
            return
        ck = ckpt_dir / f"checkpoint-{step}"
        bundle = "/tmp/ckpt_push.tar.gz"
        with tarfile.open(bundle, "w:gz") as tf:
            tf.add(ck, arcname=ck.name)
            mj = ckpt_dir / "metrics.jsonl"
            if mj.exists():
                tf.add(mj, arcname="metrics.jsonl")
        Path(drive_dir).mkdir(parents=True, exist_ok=True)
        dest = Path(drive_dir) / bundle_name
        shutil.copy(bundle, dest)
        # Read the size back immediately — shutil.copy onto the Drive FUSE
        # mount can report success while the real cloud sync is still
        # pending (or silently fails, e.g. on account-quota exhaustion);
        # this doesn't prove durability, but a size mismatch right here is
        # strong evidence the push didn't actually land (2026-07-10 incident:
        # a "successfully" pushed bundle was later confirmed missing from
        # Drive's own web UI).
        src_size = Path(bundle).stat().st_size
        dest_size = dest.stat().st_size if dest.exists() else -1
        if dest_size != src_size:
            print(f"[drive] WARNING: size mismatch after push — "
                  f"local={src_size} drive={dest_size}; push may not have "
                  f"landed", flush=True)
        print(f"[drive] checkpoint {step} -> {dest} ({dest_size} bytes)", flush=True)
    return drive_push


def run_grpo_stage(sft_ckpt, ckpt_dir, prompt_for_step, reward_fn, keys, ranges,
                   *, num_generations, max_steps, save_steps,
                   temperature=1.0, beta=0.04, learning_rate=5e-6,
                   max_completion_length=48, microbatch_size=8,
                   drive_dir=None, drive_bundle_name="grpo_ckpt.tar.gz",
                   advantage_fn=group_advantage, seed=0, extractor=None,
                   dynamic_sampling=False, dynamic_sampling_max_resamples=10,
                   snapshot_steps=()):
    """Run one GRPO ladder stage. Returns the final checkpoint path.

    `prompt_for_step(rng)` -> formatted prompt string for this step (the
    driver owns fixed-vs-pool scenario choice and the prompt->scenario map
    the reward closure reads). `reward_fn(prompts, completions)` -> list of
    floats — called exactly once per step with this step's G completions of
    one prompt, so the CRN contract (crn.py: all G completions of one prompt
    in one call share randomness) holds by construction.

    `extractor`, if given, is passed through to `completion_metrics` (see
    callbacks.py) for projects whose completions aren't JSON -- e.g.
    tool_calling's `bs(...)` calls, or bs_reasoning's `` name = `expr` ``
    lines (ported from tool_calling's worktree, same signature). `keys`/
    `ranges` are then unused but stay required positionally so the JSON-based
    projects' call sites are unchanged.

    `dynamic_sampling` (DAPO's third change, off by default -- no behavior
    change to any historical run): a group where every one of its G
    completions got the identical reward (`frac_all_tied == 1.0`) has zero
    within-group advantage, so the step's entire backward/optimizer-step
    cost is spent on a gradient that's exactly zero -- pure waste, not just
    a weak update. When enabled, such a group is discarded and a fresh
    prompt resampled instead, up to `dynamic_sampling_max_resamples` tries,
    before the step's loss/backward/optimizer-step actually runs. The
    number of resamples spent this step is logged in `metrics.jsonl` as
    `resamples` (0 when disabled or when the first sample already had
    signal) -- a systematically zero-signal setup shows up as `resamples`
    pinned at the cap every step, same diagnostic value as `frac_all_tied`
    itself already had, one level earlier in the pipeline.

    A stage always resumes from its own latest checkpoint when one exists
    (restartable-from-failure rule); delete the stage's ckpt_dir to start over.

    `snapshot_steps`, off by default -- no behavior change to any historical
    run: a set of step numbers whose checkpoint gets an ADDITIONAL push to
    Drive under a step-named bundle (`{drive_bundle_name} minus its
    extension}_snapshot_step{N}.tar.gz`) that is never overwritten or pruned
    by later saves, unlike the single rolling `checkpoint-N` this loop
    otherwise keeps (older ones are deleted at every save to protect Drive
    quota -- see the prune call below). For "paired-run" experiments that
    fork a later intervention from an earlier point in an in-progress run
    (e.g. testing whether a targeted lever changes the outcome relative to
    unassisted continuation from the identical starting weights) -- the
    fork point must be chosen and preserved WHILE the run is in progress,
    since only the run's own single blocking process can write to its own
    checkpoint directory.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    G = num_generations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device} G={G} max_steps={max_steps}", flush=True)

    tok = AutoTokenizer.from_pretrained(sft_ckpt)
    pad_id = tok.pad_token_id or tok.eos_token_id

    resume_dir, start_step = None, 0
    existing = list(ckpt_dir.glob("checkpoint-*"))
    if existing:
        resume_dir = max(existing, key=lambda p: int(p.name.split("-")[1]))
        start_step = int(resume_dir.name.split("-")[1])
        print(f"[resume] loading {resume_dir} (step {start_step})", flush=True)

    load_from = str(resume_dir) if resume_dir else sft_ckpt
    model = AutoModelForCausalLM.from_pretrained(load_from, torch_dtype=torch.float32).to(device)
    # Reference is fixed at this stage's starting checkpoint (the standard
    # GRPO setup: a frozen copy of the policy at stage start), independent of
    # any later resume.
    ref_model = AutoModelForCausalLM.from_pretrained(sft_ckpt, torch_dtype=torch.float32).to(device)
    ref_model.eval()
    ref_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if resume_dir and (resume_dir / "optimizer.pt").exists():
        # map_location=device directly, not "cpu" — load_state_dict does not
        # itself move state tensors to match the (already-on-device) params,
        # so loading to "cpu" then resuming on cuda would device-mismatch at
        # the first optimizer.step().
        state = torch.load(resume_dir / "optimizer.pt", map_location=device)
        optimizer.load_state_dict(state["optimizer"])

    metrics_log = open(ckpt_dir / "metrics.jsonl", "a")
    rng = np.random.default_rng(seed)
    # `seed` previously only controlled prompt order (the rng above) --
    # rollout generation (do_sample=True below) rode torch's global,
    # unseeded RNG, making stage-to-stage/run-to-run comparisons (e.g. a
    # G ablation) confounded by uncontrolled sampling noise. Fixed
    # 2026-08-17; any comparison involving a run from before this fix must
    # note that its rollout sampling was uncontrolled.
    torch.manual_seed(seed)
    drive_push = make_drive_push(drive_dir, ckpt_dir, drive_bundle_name)

    for step in range(start_step + 1, max_steps + 1):
        t0 = time.perf_counter()

        # 1-2. Generate + reward, wrapped in a resample loop when
        # dynamic_sampling is on -- see the docstring above. Disabled
        # (the default), this runs exactly once, identical to before.
        resamples = 0
        while True:
            prompt = prompt_for_step(rng)

            # Generate — eval mode, no_grad, the ONLY place generation
            # happens. Nothing about model.training is observed here.
            model.eval()
            enc = tok([prompt] * G, return_tensors="pt", padding=True, padding_side="left")
            enc = {k: v.to(device) for k, v in enc.items()}
            prompt_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_completion_length,
                                     do_sample=True, temperature=temperature,
                                     pad_token_id=pad_id)
            completion_ids = out[:, prompt_len:]
            completions = tok.batch_decode(completion_ids, skip_special_tokens=True)

            # Reward + logging (CRN-wrapped reward, course_lib metrics).
            rewards = reward_fn([prompt] * G, completions)
            m = completion_metrics(completions, keys, ranges, extractor=extractor)
            g_stats = group_stats(rewards, G)

            if not dynamic_sampling or g_stats["frac_all_tied"] < 1.0 \
                    or resamples >= dynamic_sampling_max_resamples:
                break
            resamples += 1

        # 3. Advantage — one group (this step's G completions of one prompt).
        advantage = advantage_fn(rewards)
        advantage_t = torch.tensor(np.asarray(advantage), dtype=torch.float32, device=device)

        # 4. Loss — teacher-forced forward pass over the already-sampled
        # tokens, micro-batched so peak activation memory is ours to tune
        # directly (no gradient_checkpointing/.training coupling involved).
        comp_mask = build_completion_mask(completion_ids, tok.eos_token_id, pad_id)
        attn_mask = torch.cat([enc["attention_mask"], comp_mask.long()], dim=1)

        model.train()
        optimizer.zero_grad()
        total_pg_loss = total_kl = 0.0
        for i in range(0, G, microbatch_size):
            sl = slice(i, min(i + microbatch_size, G))
            ids_mb, mask_mb, cmask_mb = out[sl], attn_mask[sl], comp_mask[sl]
            lp = token_log_probs(model, ids_mb, mask_mb, prompt_len)
            with torch.no_grad():
                ref_lp = token_log_probs(ref_model, ids_mb, mask_mb, prompt_len)
            denom = cmask_mb.sum(dim=1).clamp(min=1)
            seq_lp = (lp * cmask_mb).sum(dim=1) / denom
            kl_tok = (ref_lp - lp).exp() - (ref_lp - lp) - 1
            seq_kl = (kl_tok * cmask_mb).sum(dim=1) / denom
            adv_mb = advantage_t[sl]
            per_seq_loss = -(adv_mb * seq_lp) + beta * seq_kl
            (per_seq_loss.sum() / G).backward()
            total_pg_loss += float((-(adv_mb * seq_lp)).detach().sum())
            total_kl += float(seq_kl.detach().sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        r = np.asarray(rewards, dtype=float)
        rec = dict(step=step, parse_rate=m["parse_rate"], err=dict(m["err_counts"]),
                   reward_mean=float(r.mean()),
                   within_group_std=g_stats["within_group_std_mean"],
                   frac_all_tied=g_stats["frac_all_tied"],
                   resamples=resamples,
                   pg_loss=total_pg_loss / G, kl=total_kl / G,
                   step_secs=time.perf_counter() - t0)
        print(f"[metrics] {json.dumps(rec)}", flush=True)
        metrics_log.write(json.dumps(rec) + "\n")
        metrics_log.flush()

        if step % save_steps == 0 or step == max_steps:
            ck = ckpt_dir / f"checkpoint-{step}"
            model.save_pretrained(ck)
            tok.save_pretrained(ck)
            torch.save({"step": step, "optimizer": optimizer.state_dict()}, ck / "optimizer.pt")
            # Resume only ever reads the latest checkpoint-N (see the resume
            # scan above) — keeping older ones around just multiplies local
            # disk AND the Drive push size by however many saves have
            # happened; a long Stage-3 run would otherwise blow the account's
            # Drive quota (hit 2026-07-10) — prune every older one.
            for old in ckpt_dir.glob("checkpoint-*"):
                if old != ck:
                    shutil.rmtree(old)
            drive_push(step)
            if step in snapshot_steps and drive_dir:
                snap_name = drive_bundle_name.rsplit(".tar.gz", 1)[0] + f"_snapshot_step{step}.tar.gz"
                make_drive_push(drive_dir, ckpt_dir, snap_name)(step)

    final = ckpt_dir / "final"
    model.save_pretrained(final)
    tok.save_pretrained(final)
    metrics_log.close()
    print(f"[done] -> {final}", flush=True)
    return final
