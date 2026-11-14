# RoboCasa-24 Evaluation

## Shared Settings

| Item | Setting |
|---|---|
| Tasks | All 24 RoboCasa tasks |
| Trials | 50 per task per evaluation seed |
| Evaluation seeds | `[0, 1, 2]`: 3,600 rollouts per model |
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
- The default is `--seeds 0 1 2` (3,600 rollouts). Use `--seed 2` to supplement a completed `[0, 1]` run.
- `--dry-run`: inspect the evaluation plan without starting models.
- `--video --video-stride 4`: optionally record at 5 fps; recording is disabled by default.

The output directory must be new. `manifest.json` stores settings, `summary.json` stores aggregate
metrics, and each task's `episodes.json` stores individual results. Historical validation results
are in [BENCHMARKS.md](BENCHMARKS.md).

For pi05 + D4RT on eight GPUs, use its dedicated launcher:

```bash
source evaluation/egl_env.sh
.venv/bin/python -u -m evaluation.pi05_multigpu \
  --gpus 0 1 2 3 4 5 6 7 --workers-per-gpu 8 \
  --checkpoint ../pi05-experiments/checkpoints/pi05_robocasa_3view_d4rt_full_unified_q96_bs64/robocasa_3view_d4rt_q96_bs64/19999 \
  --episodes 50 --seeds 0 1 2 --output eval_results/pi05_d4rt_19999_full_8gpu_3seeds
```

This launcher gives each task to one GPU, balances groups using task horizons,
and caps simulator workers at eight per GPU and task. Each replica loads one
model and warms up before simulation starts. The launcher assigns disjoint CPU
cores to model and simulation process groups to bound JAX thread pools during
concurrent compilation. Video is disabled. `progress.json` tracks completed
rollouts; replica directories preserve individual results and
logs. On completion, the launcher validates that all requested task/seed pairs
appear exactly once, writes merged task results and `summary.json`, and stops
its model processes. A failure writes `error.json` and tears down owned processes.
Use `--dry-run` to inspect task ownership without loading models.

To add seed 2 to an existing completed `[0, 1]` evaluation, run the same launcher
with `--seed 2` and a new output directory, then combine the completed runs:

```bash
.venv/bin/python -m evaluation.merge_runs \
  --runs eval_results/pi05_d4rt_19999_full_8gpu eval_results/pi05_d4rt_19999_seed2_8gpu \
  --output eval_results/pi05_d4rt_19999_full_8gpu_3seeds
```

The merger defaults to `[0, 1, 2]`, verifies matching model/configuration and
simulation revisions, and rejects incomplete, duplicate or missing task/seed
records. It preserves existing episode records and writes combined task results,
three-seed metrics, and `task_success_rates.csv`. Its runtime totals sum the source
runs, including their separate startup periods. Historical evaluation manifests
remain unchanged.

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

# Full shared benchmark: 24 tasks x 50 trials x 3 evaluation seeds.
bash evaluation/run_pi05_client.sh --output eval_results/pi05_robocasa24

# Multiple simulation workers sharing one pi05 model (serialized inference).
WORKERS=4 bash evaluation/run_pi05_client.sh --output eval_results/pi05_robocasa24_w4
```

`PI05_PROJECT`, `PI05_PYTHON`, `PI05_CONFIG`, `GPU_ID`, `CHECKPOINT` and `BIND`
configure the server; `ENDPOINT`, `TIMEOUT_MS`, `REPLAN_STEPS` and `ACTION_BOUNDS`
configure the client; `WORKERS` (default 1) selects simulator processes. Additional CLI arguments are forwarded. Output directories
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
- Each episode resets its own flow-sampling noise generator from its evaluation
  seed; inference uses 10 denoising steps. Workers may interleave requests without
  sharing sampling streams. Use unique task/episode IDs per active rollout, as
  generated by the shared benchmark. The server processes inference requests
  serially while simulator processes run in parallel; inference is not batched.
  Small RNG objects are retained until server shutdown to preserve rollout streams.
  The managed `evaluation.multigpu` launcher is still X-WAM-specific.

The pi05 client explicitly selects `--action-bounds controller`: raw predictions
are passed to the controller's existing saturation logic, and bound violations
are recorded. Use `ACTION_BOUNDS=error` to stop on the first out-of-range value.
The existing generic pi0/Spatial Forcing input format remains available without
this named contract. Simulation sources and controller implementations are unchanged.

Interface tests are in `evaluation/tests/test_pi05_contract.py`, including
reproducible sampling under interleaved episode requests.

## pi0 and action-only pi05 baselines

The native adapter and managed multi-GPU evaluator support all four RoboCasa
training configurations in `../pi05-experiments`:

| Model | Evaluation JSON under `evaluation/configs/` | Training config |
| --- | --- | --- |
| pi05 + D4RT | `pi05.example.json` (existing) | `pi05_robocasa_3view_d4rt_full_unified_q96_bs64` |
| pi0 + D4RT | `pi0.example.json` | `pi0_robocasa_3view_d4rt_full_unified_q96_bs64` |
| pi0 | `pi0_baseline.example.json` | `pi0_robocasa_3view_bs64` |
| pi05 | `pi05_baseline.example.json` | `pi05_robocasa_3view_bs64` |

New configs use the family-independent `robocasa_openpi_raw7_v1` contract.
The original `robocasa_pi05_raw7_v1` remains supported with the same inputs,
outputs, normalization and per-episode noise isolation. The server checks that
`backend` agrees with the training model's `pi05` flag and rejects RTC models.
Each config must match the checkpoint that was trained; changing the evaluation
JSON does not convert weights between model families or add/remove a D4RT head.

For pi0 + D4RT (two terminals, from this repository):

```bash
CHECKPOINT=/absolute/path/to/trained/pi0_d4rt/step GPU_ID=0 \
  bash evaluation/run_pi0_server.sh
bash evaluation/run_pi0_client.sh \
  --tasks CloseDoubleDoor --episodes 1 --seed 0 --output eval_results/pi0_d4rt_one_episode
```

For action-only pi0, set `PI0_CONFIG=evaluation/configs/pi0_baseline.example.json`
on the server command. `PI0_PROJECT` and `PI0_PYTHON` optionally override the
model checkout and interpreter. For action-only pi05, use the existing pi05
server with `PI05_CONFIG=evaluation/configs/pi05_baseline.example.json`.
All servers require a completed RoboCasa training step containing both `params/`
and checkpoint-local `assets/robocasa/robocasa24/norm_stats.json`. A pretrained
base checkpoint alone is insufficient. State normalization and tokenization
follow the training config: pi0 uses continuous state/z-score, pi05 uses state
tokens/quantile normalization.

The pi0 multi-GPU entry shares the pi05 scheduler and benchmark settings:

```bash
source evaluation/egl_env.sh
.venv/bin/python -u -m evaluation.pi0_multigpu \
  --gpus 0 1 2 3 4 5 6 7 --workers-per-gpu 8 \
  --checkpoint /absolute/path/to/trained/pi0_d4rt/step \
  --output eval_results/pi0_d4rt_8gpu
```

For action-only pi0, add `--config evaluation/configs/pi0_baseline.example.json`.
For action-only pi05, use `evaluation.pi05_multigpu` with
`--config evaluation/configs/pi05_baseline.example.json`.
All four use the shared 24 tasks, 50 trials per task per seed, seeds `[0,1,2]`,
K=32, 10 denoising steps and controller action bounds. The manifest records the
selected backend. `--dry-run` prints task/seed ownership without model loading.

Validation for this adaptation: all 38 evaluation unit tests passed, including
four-config initialization, model/backend mismatch rejection, checkpoint-local
stats enforcement, unchanged native actions and identical 3,600-rollout dry-run
plans. The four JSON files were also checked against the real training configs
and workspace norm stats. A single-GPU pi0 smoke check loaded the local
`kai0/checkpoints/pi0_base/params` through the native adapter using temporary
checkpoint-local links to the RoboCasa stats. It returned finite 50x7 actions
and exactly reproducible actions after resetting the episode. This used base
weights and synthetic observations; no trained pi0/D4RT checkpoint or simulator
success rate was evaluated. Shell syntax and pi0 server argument forwarding
checks passed.


### Verified pi0 + D4RT checkpoint 19999

`configs/pi0_d4rt_19999.json` binds the native pi0 configuration to the completed
`pi0_robocasa_3view_d4rt_q96_bs64/19999` checkpoint in `../pi05-experiments`
(the directory name is plural). It uses the real
`pi0_robocasa_3view_d4rt_full_unified_q96_bs64` training config: pi0 continuous
state, z-score normalization, D4RT enabled, and no RTC. Online observations omit
training frame indices, so D4RT supervision files are not read. The shared
adapter loads checkpoint-local state10/action7 statistics and preserves native
raw actions and the fixed controller tail.

Validation loaded this trained checkpoint on one H20 and completed JIT inference
with synthetic RGB and checkpoint-mean state. Outputs were finite 50x7 actions;
episode reset and interleaved requests reproduced the original actions exactly.
All 46 evaluation CPU tests, shell syntax checks and the 8-GPU/8-worker dry-run
(3,600 rollouts) passed. The validation record is
`eval_results/pi0_d4rt_19999_adaptation/validation.json`. No simulator rollout or
full evaluation was started for this checkpoint during adaptation.

For a later full evaluation, from this repository root:

```bash
source evaluation/egl_env.sh
.venv/bin/python -u -m evaluation.pi0_multigpu \
  --config evaluation/configs/pi0_d4rt_19999.json \
  --checkpoint /2025233210/xiaojx/pi05-experiments/checkpoints/pi0_robocasa_3view_d4rt_full_unified_q96_bs64/pi0_robocasa_3view_d4rt_q96_bs64/19999 \
  --gpus 0 1 2 3 4 5 6 7 --workers-per-gpu 8 \
  --episodes 50 --seeds 0 1 2 --output eval_results/pi0_d4rt_19999_8gpu_3seeds
```

## Spatial Forcing RoboCasa checkpoint

`configs/spatial_forcing.example.json` now selects the native
`robocasa_spatial_forcing_raw7_v1` contract and training config
`pi05sf_jax_robocasa24_bs64` from `../spatial-forcing/policy/Spatial_Forcing/open_sf`.
The model server uses that project's `.venv`; simulation uses this repository's
`.venv`. Both launchers are explicit and do not install dependencies or run
while training is active.

After training completes and GPU resources become available, from this repository:

```bash
# Terminal 1: complete checkpoint step directory, including params/ and assets/.
CHECKPOINT=../spatial-forcing/outputs/checkpoints/pi05sf_jax_robocasa24_bs64/robocasa24_sf_bs64_20k/19999 \
  GPU_ID=0 bash evaluation/run_spatial_forcing_server.sh

# Terminal 2: check one episode first.
bash evaluation/run_spatial_forcing_client.sh \
  --tasks CloseDoubleDoor --episodes 1 --seed 0 --output eval_results/sf_one_episode

# Eight independent model replicas with the shared task/seed scheduler.
source evaluation/egl_env.sh
.venv/bin/python -u -m evaluation.spatial_forcing_multigpu \
  --checkpoint /absolute/path/to/trained/step --gpus 0 1 2 3 4 5 6 7 \
  --workers-per-gpu 8 --output eval_results/spatial_forcing_robocasa24

# Planning only, safe while GPUs are busy; no model/simulator/GPU query.
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venv/bin/python -m evaluation.spatial_forcing_multigpu \
  --checkpoint /absolute/path/to/step --output eval_results/sf_plan --dry-run
```

Defaults match the other native policies: state10, three HWC uint8 RGB images
resized to 224px, 50x7 native raw actions, 10 denoising steps, K=32, 20 Hz,
controller saturation, fixed action tail `[0,0,0,0,-1]`, 24 tasks, 50 trials per
task and evaluation seeds `[0,1,2]` (3,600 rollouts). No extra channel conversion,
vertical flip, axis rotation, delta transform or gripper sign inversion is added.
Each rollout owns its sampling stream, so simulator worker interleaving does not
change model noise.

The SF adapter requires a pi05 model with RoboCasaDataConfig and enabled VGGT
alignment in its training config. The projector is a training-only auxiliary
head: inference loads the ordinary pi05 backbone and discards extra projector
parameters. VGGT weights and feature caches are not loaded during inference.
Checkpoint-local `assets/robocasa/robocasa24/norm_stats.json` with finite state10
and action7 mean/std/quantiles is mandatory; training workspace/base statistics
are not substituted. The existing offline tokenizer is required under
`../spatial-forcing/outputs/cache/openpi/big_vision/`.

`SF_PROJECT`, `SF_PYTHON`, `SF_CONFIG`, `CHECKPOINT`, `GPU_ID`, `BIND` and
`OPENPI_DATA_HOME` configure the server. The client uses the same overrides as
`run_pi05_client.sh`. Managed replicas accept `--project` and `--policy-python`;
result directories must be new. No simulator or controller implementation changed.
CPU contract tests and dry-run planning cover this integration; trained SF GPU
loading and real simulation rollouts remain unverified until resources are free.
