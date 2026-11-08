# results/

Task notebooks and `scripts/run.py` write to `results/<config name>/`:

| File | Content |
|---|---|
| `offline_results.json` | Final evaluation of BC and IQL per seed, per-episode outcomes, training curves |
| `finetune_log.json` | Evaluation during online fine-tuning (step 0 is the offline policy) |
| `finetune_results.json` | Final evaluation of offline and fine-tuned IQL on the same episodes |
| `figures/*.png` | Training curves, final tests, per-episode outcomes, offline-to-online curve; mazes: success maps, value function, rollouts; MuJoCo: dataset returns, trajectories, filmstrips |
| `*.gif` | Episode recordings (not tracked by git, see `.gitignore`) |

The hyperparameter study (`notebooks/05_ablation_mujoco.ipynb`) writes to `results/ablation/`: one JSON file per task
and setting, `summary.json`, and `figures/`.

JSON and PNG files are tracked by git; checkpoints are written to `checkpoints/`, which is not tracked.
