# RoboCasa-24 Evaluation

## Shared Settings

| Item | Setting |
|---|---|
| Tasks | All 24 RoboCasa tasks |
| Trials | 50 per task per evaluation seed |
| Evaluation seeds | `[0, 1]`: 2,400 rollouts per model |
| Control frequency | 20 Hz |
| Replanning interval | `K=32` |
| Inputs / outputs | Three-view RGB, EE state, language / 7D native environment actions |
| Metrics | Per-task and per-seed success rates; mean and standard deviation across seeds |

Defaults are defined in [configs/benchmark.json](configs/benchmark.json). All models use the same
test instances. Evaluation seeds produce nonoverlapping instance sets; GPU and worker counts do
not change seed generation.

## H20 Multi-GPU Execution (X-WAM)

Run from the repository root using the current `.venv` and NVIDIA EGL:

```bash
source evaluation/egl_env.sh
.venv/bin/python -m evaluation.multigpu \
  --gpus 0 1 2 3 4 5 6 7 \
  --workers-per-gpu 8 --output eval_results/xwam_h20_8gpu
```

Each GPU runs one complete model replica and up to 8 simulator workers. The default checkpoint is
`../X-WAM/checkpoints/robocasa_sft`. The managed multi-GPU launcher currently supports X-WAM;
other models use the shared client with their respective adapters.

- `--episodes 50`: trials per task per seed, not per GPU.
- `--seeds 0 1 2`: extend to 3 seeds and 3,600 rollouts, preserving existing test instances.
- `--dry-run`: inspect the evaluation plan without starting models.
- `--video --video-stride 4`: optionally record at 5 fps; recording is disabled by default.

The output directory must be new. `manifest.json` stores settings, `summary.json` stores aggregate
metrics, and each task's `episodes.json` stores individual results. Historical validation results
are in [BENCHMARKS.md](BENCHMARKS.md).
