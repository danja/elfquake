# Sandpile Simulation

The sandpile simulator is a synthetic-data generator for ELFQuake. It should produce controlled avalanche-like sequences that are close enough in shape to real-world seismic data to support deep-learning pretraining experiments before fine-tuning on real seismic, VLF, and astronomical data.

This is an analogy, not a validated geological model. Its usefulness depends on measured structural similarity to real observations. Do not treat simulated avalanche targets as earthquake labels.

## Model

Use a 2D `x,y` lattice with a height or stress value per cell. The visible 3D terrain is the `z` value of each lattice column.

At each step:

1. choose a stochastic subset of source points
2. deposit grains or stress at those localized sources
3. find cells higher than a neighbour by at least the slope threshold
4. topple unstable slopes until the grid is stable
5. record avalanche statistics and sensor readings

Defaults:

* open boundaries, so material can leave the grid
* fixed random seed support for replay
* configurable grid size, source count, sensor count, threshold, deposition probability, and step count
* if the relaxation sweep limit is hit, unstable slopes are drained until stable
* mountain mode uses localized point-source deposition plus target refilling and periodic bottom-layer removal

## Outputs

Per-step summary CSV:

* `step`
* `deposition_count`
* `avalanche_count`
* `topple_count`
* `max_height`
* `mean_height`
* `released_mass`
* `relaxation_converged`
* `unstable_cell_count`
* `safety_released_mass`
* `target_fill_count`
* `bottom_layer_removed_mass`

Sensor CSV:

* `step`
* `sensor_id`
* `x`
* `y`
* `height`
* optional local activity counters

Piezo precursor sensor CSV:

* `step`
* `sensor_id`
* `x`
* `y`
* `piezo_signal`
* `piezo_total_source`
* `near_critical_cell_count`
* `critical_cell_count`
* `nearest_critical_distance`
* `max_stress_ratio`

Optional later outputs:

* height-grid snapshots at configured intervals
* avalanche masks or event maps
* chunked binary snapshots for larger runs

## Implementation

Use the CPU Numba-first implementation under `src/elfquake/sim/`.

Use NumPy arrays for simulation state and Numba-compiled kernels for hot loops:

* deposition
* unstable-cell scanning
* toppling propagation
* sensor sampling

Keep visualization separate from the simulation core. Batch simulation must run headlessly.

Initial milestone:

* `128 x 128` grid
* fixed-seed deterministic replay
* CSV outputs plus optional `.npy` grid snapshots
* small command-line runner
* JSON summary and benchmark reports
* PNG heatmap rendering for snapshot sanity checks

## Validation

Before using generated data for ML experiments, verify:

* same seed produces identical outputs
* no unstable neighbour slopes remain after relaxation
* mass accounting matches deposition minus open-boundary loss
* any `safety_released_mass` is treated as a corrective artifact, not a physical signal
* sensor table has `steps * sensor_count` rows
* summary and sensor CSV schemas are stable
* small benchmark reports steps per second

## ML Use

Use simulation outputs for:

* pretraining sequence encoders
* testing multimodal table assembly
* testing target labeling and backtesting infrastructure
* comparing synthetic avalanche targets with real event-label pipelines

Do not use simulation performance as evidence of earthquake prediction ability. Any useful claim must come from held-out real data and ablation comparisons.

## Piezo Precursor Analogue

The piezo channel is an analogue for electromagnetic precursors from quartz-like rock under stress. It is not a physical EM model.

The simulator creates a clustered susceptibility map to represent quartz-bearing regions. After deposition and background loading, but before relaxation/toppling, it measures cells whose local slope is near the failure threshold. Those cells emit a piezo-like source proportional to:

* local susceptibility
* positive stress/height change since the previous step
* closeness to the local slope threshold

Piezo sensors record a distance-weighted sum from nearby source cells. This creates a separate precursor time series that can appear before the avalanche-like toppling event. Keep this channel separate from seismic-like toppling outputs so later ML experiments can test whether precursor features add value.

## Mountain Mode

For terrain-like sanity checks, use mountain mode instead of the default low-threshold avalanche mode.

Mountain mode:

* treats `threshold` as a local slope limit, not a maximum height
* uses localized point-source deposition by default, so stress accumulates systematically around repeated locations
* fills toward `target_mean_height`, defaulting to `width / 2`, as broad background loading
* adds full uniform layers during target filling before adding any random remainder
* can limit target filling per step, so terrain builds visibly rather than appearing fully formed
* periodically removes one bottom layer from every nonzero cell

Example:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli run-sandpile-sim \
  --mountain-mode \
  --width 128 --height 128 --steps 1000 \
  --threshold 8 \
  --deposition-mode sources \
  --source-count 256 --sensor-count 16 \
  --deposition-probability 0.7 --seed 42 \
  --target-fill-limit 1024 \
  --bottom-layer-removal-interval 100 \
  --summary-out data/derived/sim/mountain_128x128_seed42_1000.summary.csv \
  --sensors-out data/derived/sim/mountain_128x128_seed42_1000.sensors.csv \
  --piezo-out data/derived/sim/mountain_128x128_seed42_1000.piezo.csv \
  --snapshot-dir data/derived/sim/mountain_128x128_seed42_1000.snapshots \
  --snapshot-interval 100 \
  --heatmap-dir data/derived/sim/mountain_128x128_seed42_1000.heatmaps \
  --heatmap-scale 4 \
  --heatmap-color-min 0 \
  --heatmap-color-max 128 \
  --heatmap-gamma 0.85 \
  --heatmap-workers 4 \
  --heatmap-progress-interval 50 \
  --progress-interval 100
```

Use `--heatmap-color-min` and `--heatmap-color-max` to keep colors comparable across frames. For mountain mode, set the max to the grid width unless you have a specific z-axis range. `--heatmap-gamma` adjusts visual contrast without changing the underlying values.

The root helper `./sim.sh` runs a parameterized mountain-mode simulation with fixed heatmap scaling:

```sh
./sim.sh
```

`sim.sh` defaults to `STEPS=10000`, `SNAPSHOT_INTERVAL=10`, `PROGRESS_INTERVAL=100`, `DEPOSITION_MODE=sources`, `SOURCE_COUNT=WIDTH * HEIGHT / 64`, `TARGET_FILL_LIMIT=WIDTH * HEIGHT / 16`, `PIEZO_SENSOR_COUNT=16`, `PIEZO_ACTIVATION_RATIO=0.75`, `HEATMAP_WORKERS=4`, `HEATMAP_PROGRESS_INTERVAL=50`, and `HEATMAP_GAMMA=0.85`, producing 1001 heatmap frames: steps `0, 10, ..., 9990, 9999`.

The default target fill limit adds at most one sixteenth of a full layer per step. This provides background loading without replacing the localized point-source stress pattern.

`sim.sh` sets `SLOPE_THRESHOLD` to `max(WIDTH / 16, 4)` unless overridden.

Create a video from generated PNG heatmaps:

```sh
./make-video.sh
```

## Install Notes

Required for the first implementation:

* `numpy` - already available in the current environment
* `numba` - install in the project venv

Optional:

* `matplotlib` for simple 2D heatmap animation
* `pyvista` for later interactive 3D visualization
* `h5py` or `zarr` for large chunked snapshot storage

The current system has no GPU. Keep this path CPU-only; do not add CUDA, CuPy, or GPU-only ML dependencies for current work.

First smoke command:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli run-sandpile-sim \
  --width 32 --height 32 --steps 20 \
  --source-count 8 --sensor-count 6 \
  --deposition-probability 0.7 --seed 42 \
  --summary-out data/derived/sim/sandpile_32x32_seed42.summary.csv \
  --sensors-out data/derived/sim/sandpile_32x32_seed42.sensors.csv \
  --snapshot-dir data/derived/sim/sandpile_32x32_seed42.snapshots \
  --snapshot-interval 5 \
  --heatmap-dir data/derived/sim/sandpile_32x32_seed42.heatmaps \
  --progress-interval 5
```

Summarize generated CSVs:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli summarize-sandpile-sim \
  --summary data/derived/sim/sandpile_32x32_seed42.summary.csv \
  --sensors data/derived/sim/sandpile_32x32_seed42.sensors.csv \
  --out data/derived/sim/sandpile_32x32_seed42.report.json
```

Run a CPU benchmark smoke:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli benchmark-sandpile-sim \
  --width 32 --height 32 --steps 20 \
  --source-count 8 --sensor-count 6 \
  --deposition-probability 0.7 --seed 42 \
  --out data/derived/sim/sandpile_32x32_seed42.benchmark.json
```

The benchmark report includes Numba first-call overhead, including compile or cache-load time.

Render a snapshot heatmap:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli render-sandpile-heatmap \
  --snapshot data/derived/sim/sandpile_32x32_seed42.snapshots/sandpile_step_000019.npy \
  --out data/derived/sim/sandpile_32x32_seed42.step_000019.png \
  --scale 8
```

Run 1000 steps with one PNG heatmap every 100 steps:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli run-sandpile-sim \
  --width 128 --height 128 --steps 1000 \
  --source-count 16 --sensor-count 16 \
  --deposition-probability 0.7 --seed 42 \
  --summary-out data/derived/sim/sandpile_128x128_seed42_1000.summary.csv \
  --sensors-out data/derived/sim/sandpile_128x128_seed42_1000.sensors.csv \
  --snapshot-dir data/derived/sim/sandpile_128x128_seed42_1000.snapshots \
  --snapshot-interval 100 \
  --heatmap-dir data/derived/sim/sandpile_128x128_seed42_1000.heatmaps \
  --heatmap-scale 4 \
  --progress-interval 100
```

Create an MP4 from generated heatmaps:

```sh
./make-video.sh \
  data/derived/sim/sandpile_128x128_seed42_1000.heatmaps \
  data/derived/sim/sandpile_128x128_seed42_1000.mp4 \
  4
```

The helper requires `ffmpeg` on `PATH`.
