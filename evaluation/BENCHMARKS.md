# Validated evaluation results

RoboCasa `756598a`, RoboSuite `232ce7d`, MuJoCo 3.2.6, PandaOmron, 20 Hz.
Checkpoint: `/2025233210/xiaojx/X-WAM/checkpoints/robocasa_sft`.
K=32; 50 video / 10 action denoising steps; CFG=0; native controller bounds mode.
These are single-task OpenDrawer tests, not a complete RoboCasa-24 benchmark.

## Rendering and initial parallel validation

Five identical seeds per run, warmed model, one shared policy service:

| Configuration | Wall seconds | Success |
|---|---:|---:|
| Mesa llvmpipe, full-rate video, one simulator | 271.07 | 5/5 |
| NVIDIA EGL, stride-4 video, one simulator | 142.13 | 5/5 |
| NVIDIA EGL, stride-4 video, two simulators | 89.83 | 5/5 |

A 20-step matched rendering probe measured three-camera acquisition at 340.01 ms on Mesa and
5.15 ms on NVIDIA H20; final EE states matched. Serial/parallel NVIDIA initial states, step counts,
and action-bound diagnostics matched. Switching rendering backends can change image pixels and
slightly change predicted trajectories; Mesa/NVIDIA used 996/1018 total control steps respectively.
The first model request included 85.83 s compilation; warmed generation averaged about 1.05 s.

## Worker sweep (16 episodes per group)

Same seeds 1200000–1200015, NVIDIA EGL, stride-4 video, one native numerical-library thread per
simulator. One model service on GPU 1 and all renderers on GPU 0. Model loading/compilation excluded;
worker spawn/import/teardown included. Order: 16, 8, 4; one run per setting.

| Workers | Seconds | Episodes/min | Mean request RTT | P95 request RTT | Render peak GiB | Sum RSS peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 150.73 | 6.37 | 11.13s | 15.78s | 47.3 | 95.3 |
| 8 | 141.38 | 6.79 | 4.63s | 7.34s | 23.1 | 51.0 |
| 4 | 166.04 | 5.78 | 2.02s | 3.16s | 12.5 | 28.4 |

All three groups: **15/16 successful**, 3544 control steps, 118 requests; seed 1200010 fails at
500 steps in all groups. Exact per-seed initial state hashes, outcomes, step counts and action-bound
diagnostics matched. Model generation was ~1.04 s in every group. Eight workers gave the best
throughput in this experiment; this is not a universal optimum for other policies or GPUs.
Resource peaks were sampled every ~2 s; summed RSS can double-count shared pages. Model process
memory peaked at 56160 MiB (54.8 GiB), measured separately from rendering. PyTorch peak allocated
memory in the earlier profiling run was 50.71 GiB. These are different memory metrics.

The 1/2-worker tests used five episodes and default native thread counts; they were not rerun and
are not matched-size controls for the 16-episode sweep. A failed first 16-worker attempt was excluded:
GL diagnostics queried an unbound context. The fix explicitly rebinds the environment context and
has a regression test. All final groups completed without infrastructure errors.

Raw experiment videos, logs, temporary configs, resource samples and sweep/profiling scripts were
removed after recording this summary. Core execution timing, smoke validation and tests remain.

## Managed eight-H20 validation

The production `evaluation.multigpu` launcher was validated with 8 independent X-WAM model/broker/
bridge stacks, GPUs 0–7, 16 globally sharded OpenDrawer episodes, 2 simulator workers per GPU,
K=32 and no video. All cards handled exactly 2 episodes, with no duplicate/missing seeds.
Result: **15/16 successful**, seed 1200010 fails at 500 steps as before.
Evaluation wall time: **52.58 s**; loading plus concurrent compilation/warmup:
**259.07 s**. This small two-episode-per-card run validates
orchestration and merging; it is not an 8x scaling claim or a 64-environment saturation benchmark.

The SFT checkpoint is now CPU-mapped instead of retained twice on GPU. Module tensor metadata
sizes: text encoder 10.582 GiB, VAE 1.313 GiB, core model 12.433 GiB (BF16 total 24.327 GiB).
Post-load live model allocation was about 24.33 GiB per card. GPU inventory after warmup
is retained separately; device total-minus-free was about 28.3 GiB per card, before simulators.

Identical warmup inputs on different cards showed maximum action difference
0.001528. Seeds are invariant, but compiled BF16 inference
is not claimed bitwise identical across devices. The multi-card run used 3550 control steps
versus the earlier single-service sweep's 3544; outcomes/failing seed agree. Both differences and
replica assignments are recorded rather than hidden.

Compact validation artifacts remain under `eval_results/h20_multigpu_smoke/`: global results,
manifest, startup/evaluation timing, warmup differences and memory inventory. Duplicated shard
outputs and successful service logs were deleted after verification. All owned processes exited
and all eight GPUs returned to idle. CPU test suite: 16 tests passed.

## pi05 + D4RT action integration and shared workers (2026-11-11)

Checkpoint: `../pi05-experiments/checkpoints/pi05_robocasa_3view_d4rt_full_unified_q96_bs64/robocasa_3view_d4rt_q96_bs64/19999`.
One pi05 service on H20 GPU 0, NVIDIA EGL, K=32, 10 denoising steps,
native controller bounds mode, stride-4 video. Task: CloseDoubleDoor,
evaluation seed 0, environment seeds 1100000 and 1100001.

| Environment seed | Success | Steps | Inference calls | Out-of-bounds components | Maximum bound excess |
|---|---|---:|---:|---:|---:|
| 1100000 | Yes | 374 | 12 | 290 | 0.0626191 |
| 1100001 | Yes | 401 | 13 | 250 | 0.1957552 |

Both episodes succeeded with one and two simulator workers. Across all repeats,
per-seed initial-state hashes, success, steps, inference counts, native action tail,
bound diagnostics, and encoded MP4 SHA-256 hashes matched exactly. Bounds excess
was delegated to the controller's existing saturation logic. This validates these
two instances; it is not an estimate of full benchmark success rate.

| Warm service, same two episodes and video settings | Wall seconds |
|---|---:|
| One simulator worker | 55.21 |
| Two simulator workers | 30.64 |

Observed speedup: 1.80x, one measurement per setting. Model loading and first-call
compilation excluded; client startup, environment creation, rendering and teardown
included. Inference remains serialized; simulator processes overlap. First-call
compilation in the initial single-episode run took approximately 13.69 s, with
subsequent request round trips around 0.11–0.12 s.

The native pi05 adapter isolates sampling RNGs by task/episode ID. Enable shared
workers with `WORKERS=2 bash evaluation/run_pi05_client.sh ...`. Other OpenPI
contracts retain their existing single-worker restriction. All 28 evaluation CPU
tests passed, including sampling invariance under interleaved episode requests.

Local detailed report: `eval_results/pi05_d4rt_19999_validation.json`.
Videos and manifests: `eval_results/pi05_d4rt_19999_single/`,
`eval_results/pi05_d4rt_19999_workers2/`, `eval_results/pi05_d4rt_19999_serial2_warm/`,
and `eval_results/pi05_d4rt_19999_workers2_warm/`.
