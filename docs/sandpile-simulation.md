# Sandpile Simulation

The sandpile simulator is a synthetic-data generator for ELFQuake. It should produce controlled avalanche-like sequences that can be used for pipeline testing, representation learning, and pretraining experiments before fine-tuning on real seismic, VLF, and astronomical data.

This is an analogy, not a validated geological model. Do not treat simulated avalanche targets as earthquake labels.

## Model

Use a 2D `x,y` lattice with a height or stress value per cell. The visible 3D terrain is the `z` value of each lattice column.

At each step:

1. choose a stochastic subset of source points
2. deposit grains or stress at those sources
3. find cells above the critical threshold
4. topple unstable cells until the grid is stable
5. record avalanche statistics and sensor readings

Defaults:

* open boundaries, so material can leave the grid
* fixed random seed support for replay
* configurable grid size, source count, sensor count, threshold, deposition probability, and step count

## Outputs

Per-step summary CSV:

* `step`
* `deposition_count`
* `avalanche_count`
* `topple_count`
* `max_height`
* `mean_height`
* `released_mass`

Sensor CSV:

* `step`
* `sensor_id`
* `x`
* `y`
* `height`
* optional local activity counters

Optional later outputs:

* height-grid snapshots at configured intervals
* avalanche masks or event maps
* chunked binary snapshots for larger runs

## Implementation

Target a CPU Numba-first implementation under a future `src/elfquake/sim/` package.

Use NumPy arrays for simulation state and Numba-compiled kernels for hot loops:

* deposition
* unstable-cell scanning
* toppling propagation
* sensor sampling

Keep visualization separate from the simulation core. Batch simulation must run headlessly.

Initial milestone:

* `128 x 128` grid
* fixed-seed deterministic replay
* CSV outputs only
* small command-line runner
* optional near-real-time 2D heatmap visualization

## Validation

Before using generated data for ML experiments, verify:

* same seed produces identical outputs
* no unstable cells remain after relaxation
* mass accounting matches deposition minus open-boundary loss
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

## Install Notes

Required for the first implementation:

* `numpy` - already available in the current environment
* `numba` - needs installing

Optional:

* `matplotlib` for simple 2D heatmap animation
* `pyvista` for later interactive 3D visualization
* `h5py` or `zarr` for large chunked snapshot storage

GPU libraries are out of scope for the first version.

First smoke command:

```sh
PYTHONPATH=src python3 -m elfquake.cli run-sandpile-sim \
  --width 32 --height 32 --steps 20 \
  --source-count 8 --sensor-count 6 \
  --deposition-probability 0.7 --seed 42 \
  --summary-out data/derived/sim/sandpile_32x32_seed42.summary.csv \
  --sensors-out data/derived/sim/sandpile_32x32_seed42.sensors.csv
```
