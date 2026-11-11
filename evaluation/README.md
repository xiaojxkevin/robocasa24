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

## pi05 + D4RT RoboCasa checkpoint

The `pi05.example.json` configuration matches the training entry
`pi05_robocasa_3view_d4rt_full_unified_q96_bs64` in `../pi05-experiments`.
The server uses that project's Python (JAX/OpenPI); the simulator uses this
repository's pinned `.venv`. Neither script installs dependencies.

Start explicitly after a completed checkpoint is available. No evaluation was
run while the training job was active.

```bash
# Terminal 1: full step directory, not its params/ subdirectory.
CHECKPOINT=/absolute/path/to/checkpoint/20000 GPU_ID=0 \
  bash evaluation/run_pi05_server.sh

# Terminal 2: a small rollout when GPU resources become available.
bash evaluation/run_pi05_client.sh \
  --tasks CloseDoubleDoor --episodes 1 --seed 0 --output eval_results/pi05_one_episode

# Full shared benchmark: 24 tasks x 50 trials x 2 evaluation seeds.
bash evaluation/run_pi05_client.sh --output eval_results/pi05_robocasa24
```

`PI05_PROJECT`, `PI05_PYTHON`, `PI05_CONFIG`, `GPU_ID`, `CHECKPOINT` and `BIND`
configure the server; `ENDPOINT`, `TIMEOUT_MS`, `REPLAN_STEPS` and `ACTION_BOUNDS`
configure the client. Additional CLI arguments are forwarded. Output directories
must be new. GPU selection is explicit and no launcher is started automatically.

The inference contract `robocasa_pi05_raw7_v1` is:

- `state10`: base-frame xyz, rotation **columns 0 then 1**, normalized observed
  gripper opening. Rotation already includes `EEF_AXES`, exactly as the source
  training data; do not rotate it again. Existing controller observation/cache
  timing is preserved, without forcing an extra controller update.
- `image`: singular dictionary of HWC uint8 RGB resized to 224x224, in camera
  order left agentview -> `base_0_rgb`, right agentview -> `left_wrist_0_rgb`,
  eye-in-hand -> `right_wrist_0_rgb`. The second slot is an external camera.
  Rendering is vertically flipped once in `observe`; encoding does not flip again.
- Language is passed as `prompt`. Online observations contain no dataset frame
  indices, so training depth/CoTracker/geometry files are not read at inference.
- The trained policy performs checkpoint-local state/action normalization and
  unnormalization. The adapter verifies state/action stat dimensions 10/7 before
  loading the model. Checkpoint-local `assets/robocasa/robocasa24/norm_stats.json`
  is required. Workspace or base-model norm stats are not substituted.
- Policy output is **50x7 native raw_action**: base-frame OSC delta controls and
  the original gripper command. No extra delta transform, joint conversion,
  rotation conversion or gripper sign inversion is applied. The client executes
  the first K=32 actions before replanning at the 20 Hz control cadence.
- Native 12D simulator controls append `[0,0,0,0,-1]`, taken from server metadata.
  Other policies without that metadata keep legacy zero padding. Both mode 0
  and -1 select the non-positive/achieved branch in the pinned controller.
- Each episode resets the flow-sampling noise generator from its evaluation
  seed; inference uses 10 denoising steps. One simulator worker per server is
  retained. The managed `evaluation.multigpu` launcher is still X-WAM-specific.

The pi05 client explicitly selects `--action-bounds controller`: raw predictions
are passed to the controller's existing saturation logic, and bound violations
are recorded. Use `ACTION_BOUNDS=error` to stop on the first out-of-range value.
The existing generic pi0/Spatial Forcing input format remains available without
this named contract. Simulation sources and controller implementations are unchanged.

New interface tests are in `evaluation/tests/test_pi05_contract.py`. They have
not been executed during the active training job; runtime validation is deferred.
