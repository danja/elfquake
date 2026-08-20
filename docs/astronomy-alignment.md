# Astronomy and Geomagnetic Feature Alignment (2026-08-17)

Implements [Next Actions](next-actions.md) item 100. Before this, astronomy was
present in the multimodal table in name only: the audit in item 99 found two
channels reaching the transformer, one of them a monthly constant repeated on
all 11,020 rows and the other a count of files the collector happened to write.
No astronomy ablation run before this date measured astronomy.

**Nothing here is evidence that astronomical or geomagnetic data helps predict
earthquakes.** It made the question askable, and the answer came back negative:
the held-out ablation in [next-actions item 102](next-actions.md) shows these
features add nothing at a 7-day horizon in the fixed-cell design. What this work
establishes is that the null is now a real measurement rather than an artifact
of a constant channel.

## What the channels are now

Two kinds, with different missingness contracts.

**Ephemeris** — deterministic functions of UTC, computed in
`src/elfquake/features/ephemeris.py`. Never missing, so their masks are
honestly always clear.

| Channel | Meaning |
| --- | --- |
| `astro_moon_phase_angle_deg` | Moon-Sun elongation, `0` at new moon, `180` at full |
| `astro_moon_phase_sin`, `astro_moon_phase_cos` | Wraparound-free encoding of the same angle |
| `astro_moon_illuminated_fraction` | Illuminated disc fraction |
| `astro_moon_distance_km` | Geocentric lunar distance |
| `astro_tidal_potential` | Degree-two lunar + solar tidal potential at the anchor |
| `astro_tidal_potential_min`, `_max`, `_range` | Same, summarized over the lookback window |

**Observed** — normalized from the archives in `data/derived/astronomy/`, each
with an age in hours and its own missing flag.

| Channel | Source | Cadence |
| --- | --- | --- |
| `astro_kp`, `astro_ap`, `astro_kp_max`, `astro_ap_max` | GFZ `Kp_ap_since_1932` | 3-hourly |
| `astro_dst_nt`, `astro_dst_min_nt` | Kyoto WDC Dst | hourly |
| `astro_f107` | Spaceweather Canada daily flux table | 3 readings per day |

The named-phase category `astro_usno_next_phase` is gone. A next-event label is
a sawtooth countdown over an arbitrary ordering, not a physical state.

## The alignment rule

Anchors are the window end, which is the VLF capture time — a 30-minute grid
against source cadences of 3 hours, 1 hour and 1 day.

1. **Zero-order hold, never interpolation.** Interpolating between the readings
   either side of an anchor mixes a future reading into a feature that is
   supposed to be causal. Each channel takes the most recent reading whose
   observation interval has *closed* at or before the anchor. A Kp bin covering
   09:00–12:00 is not usable at 09:57.
2. **Staleness is a field, not a footnote.** `astro_*_age_hours` is the gap from
   the end of that interval to the anchor. A daily F10.7 value presenting as a
   30-minute observation was the failure this replaces.
3. **Held values expire.** Past `6 h` (Kp, Dst) or `72 h` (F10.7) the hold is
   abandoned: the value is blank and `quality_missing_*` is `1`. Holding a
   three-day-old Kp reading forward would manufacture exactly the constant this
   work removes.
4. **`quality_missing_astro` reports observation, not computability.** It is set
   only when Kp, Dst *and* F10.7 are all missing. Ephemeris channels are
   deliberately excluded from the test — they are always computable, so letting
   them satisfy it would pin the flag to `0` forever, which is what the old
   implementation did.

Window aggregates cover only readings that closed inside the window.
`astro_dst_min_nt` is the storm depth, since Dst goes negative during a storm.

## The tidal potential

The degree-two potential at a reference site, in units of the lunar term at
mean distance with the Moon overhead:

```
V = (r̄_moon/d_moon)³ · P₂(cos ψ_moon) + 0.4599 · (AU/d_sun)³ · P₂(cos ψ_sun)
```

The reference site is the centre of the project's Italy bounding box
(`41.4°N`, `12.5°E`). Italy spans about 13 degrees of longitude, so the
semi-diurnal phase at the extremes differs from the centre by under an hour.
Positions are geocentric; topocentric parallax moves the lunar zenith angle by
up to a degree, well under a percent of the term's range.

The `0.4599` solar coefficient is `(M_sun/M_moon)·(r̄_moon/AU)³`, which
reproduces the textbook ratio of the solar to lunar tide.

## Accuracy

`ephemeris.py` uses the truncated periodic series from Meeus, *Astronomical
Algorithms*, chapters 12, 25 and 47. Checked against the worked examples there
and against published 2026 lunation times:

| Quantity | Reference | Computed | Error |
| --- | --- | --- | --- |
| Moon longitude, 1992-04-12 | `133.162655°` | `133.1354°` | `1.6'` |
| Moon latitude, 1992-04-12 | `-3.229126°` | `-3.2292°` | `<1"` |
| Moon distance, 1992-04-12 | `368409.7 km` | `368399.3 km` | `10 km` |
| Sun longitude, 1992-10-13 | `199.90988°` | `199.90987°` | `<0.1"` |
| GMST, 1987-04-10 00:00 UT | `197.693195°` | `197.6932°` | `<0.1"` |
| New moon | `2026-08-12 17:37Z` | within `1°` of elongation `0` | — |
| Full moon | `2026-08-28 04:18Z` | within `1°` of elongation `180` | — |

Meeus's examples are in dynamical time; UTC differs by about a minute, which is
itself roughly half an arcminute of lunar motion. Two arcminutes is far more
than a tidal proxy needs.

## Observed range in the rebuilt table

Over the 761 anchors of `all_italy.prospective_vlf_windows.csv`, every channel
now varies:

| Channel | Range | Distinct values |
| --- | --- | --- |
| `astro_kp` | `0` – `7.333` | 20 |
| `astro_ap` | `0` – `154` | 20 |
| `astro_dst_nt` | `-150` – `+53` nT | 101 |
| `astro_f107` | `94.8` – `258.1` | 65 |
| `astro_moon_phase_angle_deg` | `0.19` – `359.9` | 760 |
| `astro_tidal_potential` | `-0.804` – `+1.278` | 761 |

Compare the audited state: `astro_noaa_solar_cycle_f107_value` constant at
`125.69`, `astro_capture_count` nonzero on 3 of 17 days. The window covers a
real geomagnetic storm (`Dst` to `-150 nT`, `ap` to `154`), so the geomagnetic
channels have something to be tested against.

## The channel gate

`src/elfquake/models/channel_gate.py` runs inside the fixture builder and
raises rather than warns on three defects:

* `constant_channel` — every value identical across at least 8 rows. This is the
  item-94 failure generalized from the standardizer to the builder, where it
  would have been caught five weeks earlier. A channel that is constant by
  design has to be named with `--allow-constant-channel`, so the decision is
  recorded rather than inferred.
* `unmasked_missing_channel` — blank on some rows with no `quality_missing_*`
  flag set on all of them. Such a channel is imputed silently downstream and the
  model cannot tell a measured zero from no measurement.
* `empty_channel` — never populated at all.

`--no-strict-channels` downgrades the gate to a report entry for exploratory
runs. The defect list is written into the fixture report either way.

## Reproducing

```sh
./scripts/refresh-space-weather.sh          # fetch + normalize the archives
./scripts/refresh-prospective-labels.sh     # rebuild the window tables
./scripts/build-common-transformer-fixture.sh
```

`refresh-space-weather.sh` is deliberately on its own daily timer
(`deploy/systemd/elfquake-space-weather.timer`) rather than the prospective
job's 30-minute cadence: the Kp/ap and F10.7 sources are whole-history archives
of roughly 16 MB and 2 MB, and both publish at most once a day. It skips a
refetch when a copy under 20 hours old is already on disk.

## Source notes

* **GFZ Kp/ap** — CC BY 4.0. Whole history from 1932; the file ends at the last
  completed 3-hour bin, so the newest anchors legitimately report Kp missing.
* **Kyoto Dst** — non-commercial use. Recent months are served only by the
  `realtime` tier: probed on 2026-08-14, `final` and `provisional` both 404 for
  2026-06 through 2026-08 while `realtime` returns 200, and `provisional` serves
  2025-12. **Realtime values are provisional and get revised**, which is why
  `dst_tier` is carried through normalization and the previous month is
  refetched on every run.
* **Spaceweather Canada F10.7** — three Penticton observations per day at 17,
  20 and 23 UT. The 20 UT `fluxadjflux` reading is the conventional daily
  F10.7, so the observation time is preserved rather than collapsed to a date.

## Known gaps

* **The ablation has now been run, and it is negative.** See next-actions item
  102. Per-era, held out, thresholds calibrated on training rows only,
  `seismic_only` → `seismic_astronomy` moves `0.500000` → `0.500000` in `era_0`
  and `0.514024` → `0.518123` in `era_3`. Against a `0.5` majority baseline on
  1,368 test rows that is noise. These features are correct, varying, and
  aligned; they do not improve prediction at a 7-day horizon in the fixed-cell
  design. `full_multimodal` is worse than `seismic_only`.
* `src/elfquake/features/astronomy.py` and `multimodal_smoke.py` still emit the
  old columns for the older `build-multimodal-smoke` path. They are marked
  superseded; the channel gate will reject any fixture built from them.
* The tidal potential is computed at one reference point, not per target cell.
  For a country the size of Italy that is a sub-hour phase approximation, but it
  is an approximation.
