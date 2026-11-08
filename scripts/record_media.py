"""Records the README GIFs (media/<name>.gif) from saved checkpoints; no training.

    python scripts/record_media.py                          # all four tasks
    python scripts/record_media.py --tasks hopper_medium    # selected tasks
    python scripts/record_media.py --seed-index 2           # another trained seed

For each task the fine-tuned IQL policy is used (checkpoints/<name>/finetuned.pkl) if it exists, otherwise offline
IQL (checkpoints/<name>/offline.pkl). Maze GIFs show the first successful evaluation episode (at most 20 environment
seeds are tried); MuJoCo GIFs show the first 8 seconds of an evaluation episode. Needs rendering (on Colab: a GPU
runtime with MUJOCO_GL=egl).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("MUJOCO_GL", "egl")

from offrl import Normalizer, Task, load_checkpoint  # noqa: E402
from offrl.video import record_gif  # noqa: E402

TASKS = ["halfcheetah_medium", "hopper_medium", "pointmaze_medium", "antmaze_medium_play"]
GIF_SETTINGS = {                                   # sized for a README (a few MB each)
    "mujoco": dict(width=480, height=270, fps=12.5, seconds=8.0),
    "maze": dict(size=400, every=4, require_success=True),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="*", default=TASKS)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--out-dir", default="media")
    p.add_argument("--seed-index", type=int, default=0, help="which trained seed to film")
    args = p.parse_args()
    for name in args.tasks:
        ck_path = next((f"{args.ckpt_dir}/{name}/{f}" for f in ("finetuned.pkl", "offline.pkl")
                        if os.path.exists(f"{args.ckpt_dir}/{name}/{f}")), None)
        if ck_path is None:
            print(f"{name}: no checkpoint in {args.ckpt_dir}/{name}/, skipped")
            continue
        ck = load_checkpoint(ck_path)
        task = Task(ck["config"], verbose=False)
        normalizer = Normalizer.from_state_dict(ck["normalizer"])
        policy = "fine-tuned IQL" if ck_path.endswith("finetuned.pkl") else "offline IQL"
        print(f"{name}: {policy}, trained seed {args.seed_index}")
        record_gif(task, ck["iql_states"], normalizer, f"{args.out_dir}/{name}.gif", seed_index=args.seed_index,
                   **GIF_SETTINGS["maze" if task.kind == "maze" else "mujoco"])


if __name__ == "__main__":
    main()
