# notebooks/

| Notebook | Content | Dataset | Outputs |
|---|---|---|---|
| `01_halfcheetah_medium.ipynb` | BC, IQL, online fine-tuning | `mujoco/halfcheetah/medium-v0` | executed (offrl 0.3.0) |
| `02_hopper_medium.ipynb` | BC, IQL, online fine-tuning | `mujoco/hopper/medium-v0` | executed (offrl 0.3.0) |
| `03_pointmaze_medium.ipynb` | BC, IQL | `D4RL/pointmaze/medium-v2` | executed (offrl 0.3.0) |
| `04_antmaze_medium_play.ipynb` | BC, IQL, online fine-tuning | `D4RL/antmaze/medium-play-v2` | not executed |
| `05_ablation_mujoco.ipynb` | IQL hyperparameter study ($\gamma$, $\beta$) | both MuJoCo datasets | executed (offrl 0.3.0) |

Each task notebook defines the problem, the dataset and the evaluation protocol, describes the methods, trains and
evaluates BC and IQL (Part 1) and, where applicable, fine-tunes IQL online (Part 2). Hyperparameters are read from
`configs/`. In notebooks with a Part 2, `RUN_OFFLINE = False` loads `checkpoints/<name>/offline.pkl` and runs Part 2
only. The study notebook writes one result file per setting to `results/ablation/` and resumes from existing files.
Animations are not stored in the executed notebooks; shortened copies are in `media/`.
