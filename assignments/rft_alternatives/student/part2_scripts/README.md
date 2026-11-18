# Part 2 scripts

Three starting scripts for the guided Part 2 methods. Keep this folder inside
the bundle root (the folder that contains `course_lib/`): each script puts that
root on `sys.path` itself. `rlsd_train.py` imports `self_distill.py`, so keep
the files together. Run any of them with `--help`; `--smoke` is a tiny
plumbing check (it says nothing about quality).

- `self_distill.py` (RLLM-9): sample, filter by reward, SFT, repeat.
  `--filter-mode topk --topk 1` gives the RAFT arm.
- `opd_train.py` (RLLM-11): forward-KL from a teacher to the student on the
  student's own samples.
- `rlsd_train.py` (RLLM-12): cycles of GRPO steps then one self-distillation
  round.

**The OPD teacher is your choice; the script does not supply one.** Give it
`--teacher MODEL_PATH_OR_HF_ID` (a frozen model loaded with transformers, with
the same tokenizer vocabulary as your student). `--teacher-prompt-file FILE`
adds any teacher-side text you write to the teacher's system message. What
makes a good teacher, and what a teacher can and cannot do, is RLLM-12.

Each run writes `metrics.jsonl` (one JSON line per round, step or cycle) into
its `--ckpt-dir` and resumes from the latest checkpoint if re-run with the
same directory. Defaults such as the reward threshold, temperature and learning
rate are starting points, not recommendations; choose yours from evidence.
