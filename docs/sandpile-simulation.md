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

Direct avalanche signal CSV:

* `step`
* `sensor_id`
* `x`
* `y`
* `avalanche_signal`
* `avalanche_total_source`
* `active_topple_cell_count`
* `max_local_topple`
* `nearest_topple_distance`
* `stress_drop_total`
* `stress_drop_max`

Direct avalanche activity CSV:

* `step`
* `active_topple_cell_count`
* `topple_count`
* `centroid_x`
* `centroid_y`
* `weighted_centroid_x`
* `weighted_centroid_y`
* rupture bounding box and peak-topple cell

Optional later outputs:

* height-grid snapshots at configured intervals
* avalanche rupture masks
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

The simulator creates a clustered susceptibility map to represent quartz-bearing regions and a persistent `piezo_charge` grid. After deposition and background loading, but before relaxation/toppling, it updates the charge state:

1. decay existing charge by `PIEZO_CHARGE_DECAY`
2. measure each cell's steepest local downhill slope
3. add charge from positive stress/height change when the cell is near the failure threshold
4. cap charge with `PIEZO_SATURATION`, releasing any excess
5. release charge when strain is increasing near the failure threshold
6. release an additional fraction when a cell crosses into critical slope

The emitted source is therefore based on stored charge and strain-driven release, not only instantaneous height change. Sustained stress alone should not create a constant radio floor. Piezo sensors record a distance-weighted sum from nearby emitting cells. This creates a separate precursor time series sampled before the avalanche-like toppling event. Keep this channel separate from seismic-like toppling outputs so later ML experiments can test whether precursor features add value.

Piezo CSV diagnostics include total charge, maximum charge, and total release per step so the charge-store behavior can be audited.

The simulator writes separate VLF-like and seismic-like signal forms when configured by `sim.sh`:

* `*.piezo.csv` - piezo strain/charge sensor for VLF-like analogue outputs
* `*.avalanche_signal.csv` - direct toppling/stress-release signal sampled from avalanche relaxation
* `*.avalanche_activity.csv` - direct toppling footprint, centroid, bounding box, and peak cell
* `*.synthetic_events.csv` - INGV-like direct seismic event rows from summary/sensor tables
* `*.avalanche_events.csv` - INGV-like direct seismic event rows from the avalanche signal table

Keep these channels separate. The piezo channel is the VLF analogue and is sampled before relaxation; direct avalanche outputs are the seismic-event analogue and are sampled after toppling. Do not add display-time noise or artificial spikes to either path.

Default charge parameters in `sim.sh`:

* `PIEZO_CHARGE_DECAY=0.995`
* `PIEZO_CHARGE_COUPLING=1.0`
* `PIEZO_RELEASE_RATIO=0.15`
* `PIEZO_CRITICAL_RELEASE_RATIO=0.05`
* `PIEZO_SATURATION=1000`

A `32 x 32`, `200` step smoke run with these defaults produced a rough log-log PSD slope near `-1` for sensor `0`, which is the intended 1/f-like diagnostic target. Treat this as a simulation sanity check, not validation against real VLF data.

## Derived Synthetic Outputs

Convert avalanche-like steps into an INGV-like event list:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli build-synthetic-event-list \
  --summary data/derived/sim/mountain_128x128_seed42_1000.summary.csv \
  --sensors data/derived/sim/mountain_128x128_seed42_1000.sensors.csv \
  --grid-width 128 \
  --grid-height 128 \
  --out data/derived/sim/mountain_128x128_seed42_1000.synthetic_events.csv
```

The event list uses the normalized INGV-compatible fields first, then appends synthetic traceability fields such as `step`, `x`, `y`, `topple_count`, and `location_quality`. Grid coordinates are mapped to Central Italy by default.

For direct avalanche-signal events, use sparse peak extraction so small continuous relaxations do not become event-like rows:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli build-avalanche-signal-event-list \
  --avalanche data/derived/sim/mountain_128x128_seed42_1000.avalanche_signal.csv \
  --activity data/derived/sim/mountain_128x128_seed42_1000.avalanche_activity.csv \
  --grid-width 128 \
  --grid-height 128 \
  --min-signal-quantile 0.95 \
  --local-max-window 15 \
  --out data/derived/sim/mountain_128x128_seed42_1000.avalanche_events.csv
```

`run-all.sh` applies this peak extraction by default with `AVALANCHE_EVENT_QUANTILE=0.95` and `AVALANCHE_EVENT_WINDOW=15`.

By default, direct avalanche events use `--spatial-profile italy_apennines`. This keeps raw avalanche `x,y` coordinates in the CSV, but scales the selected event distribution onto an Apennine-style Italy belt for demo latitude/longitude fields. Use `--spatial-profile central_italy --no-fit-spatial-extent` for the older rectangular Central Italy projection.

Render a VLF-style spectrogram from the piezo sensor CSV:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli render-piezo-spectrogram \
  --piezo data/derived/sim/mountain_128x128_seed42_1000.piezo.csv \
  --out data/derived/sim/mountain_128x128_seed42_1000.piezo_spectrogram.png \
  --metadata-out data/derived/sim/mountain_128x128_seed42_1000.piezo_spectrogram.json
```

The spectrogram treats piezo sensor amplitudes as receiver time series and computes an FFT-based time-frequency view. The frequency axis is determined by `--step-seconds`; with the default `60` seconds per simulation step, the Nyquist frequency is only about `0.0083 Hz`. Use smaller `--step-seconds` values only when the simulation run is intended to represent faster physical sampling. This is an analogue display, not a physical EM spectrum.

Render a combined time-series and spectrogram PNG:

```sh
./piezo-summary.sh
```

This is an FFT diagnostic of the simulated receiver envelope. It is useful for checking drift, bursts, and rough spectral slope, but it is not expected to look like a real VLF receiver spectrogram because the simulation timestep is much slower than VLF carrier sampling.

By default the helper renders one receiver (`SENSOR_ID=0`) with a one-pole DC-blocking filter (`DC_BLOCK=0.995`). Set `SENSOR_ID` to another integer or clear the filter with `DC_BLOCK=0` when comparing sensors.

Render a VLF-shaped analogue summary:

```sh
./piezo-vlf-summary.sh
```

This maps the piezo strain envelope onto carrier-like bands from `0` to `24000` Hz. It is a display analogue for sanity checking the piezo-strain hypothesis, not a physical RF waveform or FFT of the simulation timestep.

Use `DISPLAY_COLOR_QUANTILE` to adjust display scaling when comparing against Cumiana images. This changes rendering only; it does not add signal.

Compare multiple seeds with:

```sh
REAL_EVENTS=data/derived/ingv/events_italy_2026-06-01_2026-06-29.normalized.csv ./compare-simulation-grid.sh
```

See [Simulation Time Scale](simulation-time-scale.md) before interpreting PSD metrics from simulated traces.

Render a WAV sonification of the summed piezo signal:

```sh
./piezo-audio.sh
```

The WAV is a time-compressed audio rendering for inspection, not a physical radio waveform. The default `SMOOTH_STEPS=64` suppresses step-to-step jitter before resampling to audio rate; reduce it to hear more high-frequency simulation jitter.

Render avalanche-derived seismic-like events over an offline Italy map:

```sh
./event-map.sh
```

The helper prefers the newest `*.avalanche_events.csv`, then falls back to `*.synthetic_events.csv` and normalized INGV CSVs. By default it uses a packaged Natural Earth 1:10m Italy GeoJSON as a realistic offline line basemap and still works without `geopandas`, `shapely`, or web tiles. Use `BASEMAP_GEOJSON=/path/to/map.geojson ./event-map.sh` to override the outline.

Current synthetic event locations use the weighted centroid from direct avalanche toppling activity when `*.avalanche_activity.csv` is available, falling back to direct avalanche-signal sensor locations for older runs. The default event-map projection scales those synthetic centroids over an Apennine-like belt and uses point size for synthetic magnitude. This is suitable for a demo overlay, but not yet for evaluating spatial realism. Add full rupture-mask output before using synthetic maps as spatial training data.

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

Run the full local demo pipeline:

```sh
./run-all.sh
```

This runs `sim.sh`, builds direct seismic synthetic events, renders the VLF-shaped piezo analogue, writes the piezo WAV sonification, builds the heatmap video, and renders the synthetic event map. It defaults to `mountain_256x256_seed42_10000`. Set `RUN_SIM=0` to reuse existing simulation files, `RUN_VIDEO=0` to skip MP4 generation, or `RUN_FFT=1` to also render the older FFT diagnostic PNG.

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
