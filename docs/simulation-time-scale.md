# Simulation Time Scale

Simulation time is an explicit modeling assumption.

Current default:

* `1 simulation step = 60 seconds`
* event traces are binned with `EVENT_BIN_SECONDS=3600`
* signal-shape comparisons use `SIM_STEP_SECONDS=60`

This mapping is convenient for comparing simulated avalanche sequences with hourly INGV event-energy traces. It is not a claim that one model step physically represents one minute of crustal evolution.

Frequency-domain interpretation:

* PSD slopes and band-power ratios are shape diagnostics under the declared mapping.
* Absolute frequencies are not physically meaningful until a separate calibration is justified.
* VLF carrier displays are visual analogues derived from piezo envelope timing, not RF waveform reconstructions.

Override examples:

```sh
SIM_STEP_SECONDS=30 EVENT_BIN_SECONDS=1800 ./compare-signal-shapes.sh
```

Use the same declared mapping across all simulation seeds in a tuning run.
