---
name: elfquake-simulation-pipeline
description: Use when running or regenerating ELFQuake sandpile simulations, synthetic seismic events, piezo/VLF-like signals, maps, videos, or simulation-derived model artifacts.
---

# ELFQuake Simulation Pipeline

Run from the repository root with the project virtual environment active when optional dependencies are needed. Keep the workflow CPU-only on this system.

## Usual Order

1. Generate the default synthetic simulation bundle:

```sh
./run-all.sh
```

2. Refresh synthetic model artifacts after simulation output changes:

```sh
./refresh-synthetic-model-artifacts.sh
```

3. Create demo media when needed:

```sh
./sim.sh
./make-video.sh
./prediction-event-map.sh
```

## Validation Scripts

Use these to check whether synthetic signals remain plausible and comparable:

```sh
./compare-simulation-grid.sh
./compare-piezo-vlf.sh
./compare-signal-shapes.sh
./piezo-sensor-scan.sh
```

Keep piezo/VLF-like precursor signals separate from direct avalanche/seismic-like event signals. Optional sensors must not change the core sandpile trajectory or deterministic seed behavior.

Only run `./tidy.sh` when deletion of old derived artifacts is explicitly intended.
