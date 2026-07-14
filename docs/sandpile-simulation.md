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
* mountain mode uses localized point-source deposition plus target refilling and periodic bottom-layer removal; episode batches set target refill to reuse those persistent sources rather than introducing uniform random loading

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
* accumulated charge and release diagnostics

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

Piezo-like sensors are sampled before relaxation/toppling. They derive signal from near-critical stress gradients, local stress increase, susceptibility, accumulated charge, and distance attenuation. The optional `PIEZO_RELEASE_CHARGE_THRESHOLD` enables a stick-slip style release gate: charge must accumulate past the threshold before regular piezo release occurs. This remains derived from avalanche state; it must not inject artificial spikes.

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
* piezo sensor scans compare individual simulated sensors with Cumiana VLF image-column traces

## ML Use

Use simulation outputs for:

* pretraining sequence encoders
* testing multimodal table assembly
* testing target labeling and backtesting infrastructure
* comparing synthetic avalanche targets with real event-label pipelines

Do not use simulation performance as evidence of earthquake prediction ability. Any useful claim must come from held-out real data and ablation comparisons.

Current piezo note: thresholded charge release with a local receiver footprint is the current default after seed `40`-`42` validation. The best VLF-like sensor is seed-dependent, so preserve `sensor_id` and scan or pool sensors during model preparation.

Current episode-holdout diagnostics show that class-conditional piezo effects can change sign between simulation trajectories. Shortening the event target horizon, extending raw model context, and adding simple sensor max/std aggregates did not improve generalization. Before changing the sensor equations, measure event-centered lead-time profiles using only pre-relaxation piezo samples and post-relaxation avalanche events; do not inject synthetic precursor spikes.

The causal lead-time probe now compares each pre-event window with matched controls and an earlier local baseline. A stored-potential channel produced a short 1--5 step effect only when the diagnostic chose the sensor nearest to the *future* avalanche. Its spatial average failed the nine-episode confirmation, so it is not a model feature. `event_nearest` is an analysis-only oracle control; use it to guide sensor design, never as a predictive input.

The lead-time analyzer also supports causal `top_k` and `top_k_rise` pooling, which select the strongest current sensors without event coordinates. Neither passes the localized-default nine-episode check. A separate `SOURCE_COUNT=64` screen produced a short potential lead, but it failed nine-episode confirmation. The duration-aligned reduced-source profile is retained as a synthetic target baseline because it has usable class balance and drift, not because it supplies a validated precursor.

Pre-relaxation spatial-state diagnostics now include near-critical contact count, contact coherence, and stress-weighted criticality. These are derived directly from the grid before relaxation and do not alter it. Their three-episode screen did not survive the nine-episode check, confirming that the current instantaneous-relaxation dynamics do not provide a validated multi-step precursor. Further progress requires delayed failure or damage dynamics in the simulator itself, not another receiver transformation.

## Delayed Failure

`DamageConfig` is an opt-in state-evolution extension. Before relaxation, cells at or above `damage.activation_ratio` accumulate bounded damage; damage decays elsewhere. During relaxation, accumulated damage lowers only that cell's local failure threshold. A toppling cell then resets most of its damage. This changes avalanche timing through simulation state rather than adding an output-only signal. The pre-relaxation `damage_total`, `damage_max`, and active-cell count are written to piezo rows and summary rows. Each fixed piezo receiver also records local damage mean, maximum, active fraction, and standard deviation within its receiver footprint; these are causal receiver readouts, not future-event-derived features.

The initial nine-episode damage profile (`activation=0.85`, `decay=0.985`, `coupling=0.10`, threshold reduction `0.25`, reset `0.90`) supports `damage_total` at a 5--15 step lead. This is synthetic evidence only: it must improve leave-one-episode-out modeling relative to damage-disabled runs before being kept as a default model feature.

The fixed top-three local-damage receiver aggregation was tested on nine fresh control-dynamics episodes and 43 extracted events. None of the four local fields passed the causal lead rule. Localizing the existing damage readout therefore does not recover a transferable precursor; further work should separate damage accumulation from failure maturation rather than add more receiver pooling.

`MatureWeaknessConfig` implements that separation as an opt-in two-stage path. Stress first creates microdamage through `DamageConfig`. Only after microdamage remains above `mature_weakness.damage_threshold` for `dwell_steps` does a bounded mature-weakness field grow. In this mode microdamage does not lower the relaxation threshold; only mature weakness does. Toppling partially resets mature weakness and clears its dwell counter. The mature total, maximum, and active-cell count are recorded before relaxation in summary and piezo CSVs. This is a new mechanism and has only passed a small output smoke test, not causal validation.

The predeclared profile (`damage threshold=0.50`, dwell `5`, maturation rate `0.10`, decay `0.995`, threshold reduction `0.20`, reset `0.90`) has now completed a nine-episode test. Neither microdamage nor mature weakness passed the causal lead rule across 44 extracted events, despite stable target timing. Treat this mechanism as rejected for modeling at these settings; do not infer precursor value from its output smoke test.

## Piezo Precursor Analogue

The piezo channel is an analogue for electromagnetic precursors from quartz-like rock under stress. It is not a physical EM model.

The simulator creates a clustered susceptibility map to represent quartz-bearing regions and a persistent `piezo_charge` grid. After deposition and background loading, but before relaxation/toppling, it updates the charge state:

1. decay existing charge by `PIEZO_CHARGE_DECAY`
2. measure each cell's steepest local downhill slope
3. add charge from positive stress/height change when the cell is near the failure threshold
4. cap charge with `PIEZO_SATURATION`, releasing any excess
5. release charge when stored charge passes `PIEZO_RELEASE_CHARGE_THRESHOLD` and strain is increasing near the failure threshold
6. release an additional fraction when a cell crosses into critical slope

The emitted source is therefore based on stored charge and strain-driven release, not only instantaneous height change. Sustained stress alone should not create a constant radio floor. Piezo sensors record a distance-weighted sum from nearby emitting cells. This creates a separate precursor time series sampled before the avalanche-like toppling event. Keep this channel separate from seismic-like toppling outputs so later ML experiments can test whether precursor features add value.

Piezo CSV diagnostics include total charge, maximum charge, and total release per step so the charge-store behavior can be audited.

The direct avalanche signal uses its own receiver range controls. Piezo VLF-like tuning must not change the seismic-like avalanche channel.

The simulator writes separate VLF-like and seismic-like signal forms when configured by `sim.sh`:

* `*.piezo.csv` - piezo strain/charge sensor for VLF-like analogue outputs
* `*.avalanche_signal.csv` - direct toppling/stress-release signal sampled from avalanche relaxation
* `*.avalanche_activity.csv` - direct toppling footprint, centroid, bounding box, and peak cell
* `*.synthetic_events.csv` - INGV-like direct seismic event rows from summary/sensor tables
* `*.avalanche_events.csv` - INGV-like direct seismic event rows from the avalanche signal table

Keep these channels separate. The piezo channel is the VLF analogue and is sampled before relaxation; direct avalanche outputs are the seismic-event analogue and are sampled after toppling. Do not add display-time noise or artificial spikes to either path.

Default charge parameters in `sim.sh`:

* `PIEZO_ATTENUATION_RADIUS=16`
* `PIEZO_MAX_DISTANCE_RADIUS=48`
* `PIEZO_CHARGE_DECAY=0.995`
* `PIEZO_CHARGE_COUPLING=1.0`
* `PIEZO_RELEASE_CHARGE_THRESHOLD=40`
* `PIEZO_RELEASE_RATIO=0.25`
* `PIEZO_CRITICAL_RELEASE_RATIO=0.10`
* `PIEZO_SATURATION=1000`

These defaults come from the current seed `42`, `10000` step piezo tuning pass and were checked across seeds `40`, `41`, and `42`. Treat this as a simulation sanity check, not validation against real VLF data.

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

`run-all.sh` applies this peak extraction by default with `AVALANCHE_EVENT_QUANTILE=0.99` and `AVALANCHE_EVENT_WINDOW=30`.

To tune these extraction knobs against real seismic event-shape metrics:

```sh
PYTHONPATH=src python -m elfquake.cli tune-avalanche-event-extraction \
  --real-events data/derived/ingv/events_italy_2026-06-01_2026-06-29.normalized.csv \
  --avalanche data/derived/sim/mountain_256x256_seed42_10000.avalanche_signal.csv \
  --activity data/derived/sim/mountain_256x256_seed42_10000.avalanche_activity.csv \
  --grid-width 256 \
  --grid-height 256 \
  --quantile 0.90 --quantile 0.95 --quantile 0.975 --quantile 0.99 \
  --local-max-window 5 --local-max-window 15 --local-max-window 30 --local-max-window 60 \
  --out data/derived/sim/mountain_256x256_seed42_10000.avalanche_event_tuning.csv
```

The first full-size seed `40`-`42` tuning pass and longer `20000` step runs favour quantile `0.99`; window `30` is the most stable model-training default from that pass. Refined sparse tuning against the extended central-Italy catalog adds `--max-events` and a stable `shape_score`; the current sparse profile for `20000` step runs is quantile `0.999`, window `240`, max events `5`.

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
./scripts/piezo-summary.sh
```

This is an FFT diagnostic of the simulated receiver envelope. It is useful for checking drift, bursts, and rough spectral slope, but it is not expected to look like a real VLF receiver spectrogram because the simulation timestep is much slower than VLF carrier sampling.

By default the helper renders one receiver (`SENSOR_ID=5`) with a one-pole DC-blocking filter (`DC_BLOCK=0.995`). Set `SENSOR_ID` to another integer or clear the filter with `DC_BLOCK=0` when comparing sensors.

Render a VLF-shaped analogue summary:

```sh
./scripts/piezo-vlf-summary.sh
```

This maps the piezo strain envelope onto carrier-like bands from `0` to `24000` Hz. It is a display analogue for sanity checking the piezo-strain hypothesis, not a physical RF waveform or FFT of the simulation timestep.

Use `DISPLAY_COLOR_QUANTILE` to adjust display scaling when comparing against Cumiana images. This changes rendering only; it does not add signal.

Compare multiple seeds with:

```sh
REAL_EVENTS=data/derived/ingv/events_italy_2026-06-01_2026-06-29.normalized.csv ./scripts/compare-simulation-grid.sh
```

`compare-simulation-grid.sh` defaults to metrics-only runs with `RUN_HEATMAPS=0`, `RUN_VIDEO=0`, and `RUN_AUDIO=0`.

Tune piezo VLF-like sensor settings with:

```sh
./scripts/piezo-tune-grid.sh
```

Use `burst_run_rate`, not raw `burst_run_count`, when comparing VLF image-column traces with simulation traces of different lengths.

See [Simulation Time Scale](simulation-time-scale.md) before interpreting PSD metrics from simulated traces.

Render a WAV sonification of the summed piezo signal:

```sh
./scripts/piezo-audio.sh
```

The WAV is a time-compressed audio rendering for inspection, not a physical radio waveform. The default `SMOOTH_STEPS=64` suppresses step-to-step jitter before resampling to audio rate; reduce it to hear more high-frequency simulation jitter.

Render avalanche-derived seismic-like events over an offline Italy map:

```sh
./scripts/event-map.sh
```

The helper prefers the newest `*.avalanche_events.csv`, then falls back to `*.synthetic_events.csv` and normalized INGV CSVs. By default it uses a packaged Natural Earth 1:10m Italy GeoJSON as a realistic offline line basemap and still works without `geopandas`, `shapely`, or web tiles. Use `BASEMAP_GEOJSON=/path/to/map.geojson ./scripts/event-map.sh` to override the outline.

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

The root helper `./scripts/sim.sh` runs a parameterized mountain-mode simulation with fixed heatmap scaling:

```sh
./scripts/sim.sh
```

`sim.sh` defaults to `STEPS=10000`, `SNAPSHOT_INTERVAL=10`, `PROGRESS_INTERVAL=100`, `DEPOSITION_MODE=sources`, `SOURCE_COUNT=WIDTH * HEIGHT / 64`, `TARGET_FILL_LIMIT=WIDTH * HEIGHT / 16`, `PIEZO_SENSOR_COUNT=16`, `PIEZO_ACTIVATION_RATIO=0.75`, `HEATMAP_WORKERS=4`, `HEATMAP_PROGRESS_INTERVAL=50`, and `HEATMAP_GAMMA=0.85`, producing 1001 heatmap frames: steps `0, 10, ..., 9990, 9999`.

The default target fill limit adds at most one sixteenth of a full layer per step. This provides background loading without replacing the localized point-source stress pattern.

`sim.sh` sets `SLOPE_THRESHOLD` to `max(WIDTH / 16, 4)` unless overridden.

The current long-run h6 event-list targets show temporal drift: late rows are much more avalanche-positive than early rows. `diagnose-synthetic-event-list-drift.sh` should be run after target generation to measure this. Balanced and episode-balanced splits are useful learnability checks, but conservative validation still requires chronological splits.

For the next synthetic data pass, prefer multiple shorter stationarity-oriented episodes over a single longer run:

```sh
./scripts/run-synthetic-episode-batch.sh
```

That wrapper preserves localized point-source deposition, reduces background fill, removes bottom layers more frequently, and uses sparse event extraction defaults. Its purpose is to reduce simulation lifecycle bias while keeping systematic localized stress.

Current episode-batch defaults are intentionally conservative for mass balance: `TARGET_FILL_LIMIT=WIDTH * HEIGHT / 128`, `BOTTOM_LAYER_INTERVAL=25`, `DEPOSITION_PROBABILITY=0.45`, and `WARMUP_STEPS=3000`. The warm-up evolves each pile before recording rows, which better approximates using the tail of a longer run without storing cold-start data.

A three-episode, 3000-step recorded probe with `WARMUP_STEPS=3000` kept final mean height near `3.8` and reduced h6 train/test positive-rate drift to `0.048677`. A smaller `WARMUP_STEPS=1000` probe reached `0.017989` but failed when scaled to nine episodes (`0.294146`), so the longer warm-up is now preferred. The `WARMUP_STEPS=3000` profile scaled to nine episodes with acceptable h6 drift (`0.187025`). The earlier no-warm-up aggressive probe reached `0.065608`, and a previous 5000-step profile still accumulated mass to mean height about `111` and failed drift validation.

Sparse direct-event extraction remains the default. Denser profiles were tested on the nine-episode warmed run: `_q099_w60_m10` over-saturated h6 targets, and `_q0995_w120_m5` improved class balance but did not improve model smoke metrics.

Initial fill is available for cold-start experiments:

```sh
INITIAL_FILL_MODE=structured INITIAL_FILL_MEAN_HEIGHT=3 INITIAL_FILL_VARIATION=1.5 INITIAL_FILL_SMOOTH_PASSES=3 ./scripts/sim.sh
```

The first structured-fill probe did not beat the warmed aggressive profile: h6 positive-rate drift was `0.307937`. The likely issue is that bottom-layer removal starts immediately and strips much of the initial low terrain, while the event process still organizes later. Prefer unrecorded warm-up over structured initial fill unless delayed-removal experiments improve the result.

Run the full local demo pipeline:

```sh
./scripts/run-all.sh
```

This runs `sim.sh`, builds direct seismic synthetic events, renders the VLF-shaped piezo analogue, writes the piezo WAV sonification, builds the heatmap video, and renders the synthetic event map. It defaults to `mountain_256x256_seed42_10000`. Set `RUN_SIM=0` to reuse existing simulation files, `RUN_HEATMAPS=0` to skip snapshot and heatmap PNG generation, `RUN_VIDEO=0` to skip MP4 generation, `RUN_AUDIO=0` to skip WAV output, or `RUN_FFT=1` to also render the older FFT diagnostic PNG.

Create a video from generated PNG heatmaps:

```sh
./scripts/make-video.sh
```

## Install Notes

Required for the first implementation:

* `numpy>=1.22,<2.5` - constrained because Numba `0.65.1` does not accept NumPy `2.5.x`
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
./scripts/make-video.sh \
  data/derived/sim/sandpile_128x128_seed42_1000.heatmaps \
  data/derived/sim/sandpile_128x128_seed42_1000.mp4 \
  4
```

The helper requires `ffmpeg` on `PATH`.
