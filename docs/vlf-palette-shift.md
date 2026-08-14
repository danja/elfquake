# Cumiana Colour-Scale Change (2026-08-14)

Diagnostic of [Next Actions](next-actions.md) item 95(a): decide from **raw
Cumiana spectrograms**, not derived features, whether the level change across
the July collector outage is instrumental or physical. Item 95(a) set the
operational consequence in advance — an instrument change would mean `era_0`
and `era_3` image features are not on a common scale and must not be pooled.

**Result: the receiver's colour scale was changed during the outage.**
`era_0` and `era_3` pixel-derived image features are **not on a common scale
and must not be pooled.**

Reproduce with `./scripts/diagnose-vlf-palette-shift.sh`.

## The colourbar moved; the dB ruler did not

Every `last_E_VLF` capture embeds its own 96-step colourbar above the
spectrogram, with a fixed `-100 dB … 0 dB` tick ruler printed beneath it. The
ruler is a plotting constant; the ramp inside the bar is a receiver-software
setting. They can move independently, and here exactly one of them did.

Across all 699 captures from `2026-06-29` to `2026-08-14`:

| Variant | Solid-red onset | Displayed dB window | Captures | Span |
| --- | --- | --- | --- | --- |
| `palette_0` | step `59` | `-80.0` … `-37.9` | 277 | `2026-06-29T09:45Z` – `2026-07-06T21:45Z` |
| `palette_1` | step `48` | `-91.6` … `-49.5` | 422 | `2026-07-11T06:00Z` – `2026-08-14T09:45Z` |

The change is a single step, with no intermediate values: every capture on or
before `2026-07-06T21:45Z` reads `59`, every capture from `2026-07-11T06:00Z`
onward reads `48`. The ramp moved **11 steps = `11.58 dB`**, downward — the
later palette assigns hot colours to *lower* absolute levels.

Everything else in the image is unchanged. The frequency tick rows on the
right-hand axis are pixel-identical across the two variants (`90, 112, 135,
157, 179, 201, 223, 246, 268, 290, 312, 334, 357, 379` for 14000 Hz down to
1000 Hz), as is the image geometry (`842x573`) and the dB ruler. This is a
colour-scale setting, not a re-plot and not a layout change.

The change point falls **inside the collector outage**: captures stop on
`2026-07-06` and the next one, on `2026-07-11`, already carries the new
palette. It is therefore invisible to any diagnostic that only compares dense
`era_0` against dense `era_3`.

## Why this settles the pooling question on its own

A palette change is sufficient for the item-95(a) conclusion regardless of what
prompted it. `vlf_intensity_mean`, `vlf_hot_color_ratio`,
`vlf_high_intensity_ratio`, and the `vlf_band_*_mean` family are all functions
of pixel colour. The same colour denotes a level `11.58 dB` lower after the
change than before it. Those features are readings on two different rulers, and
pooling them across `2026-07-06` measures the ruler.

This is the mechanism behind the item-93 observation that image-content
features dominate the era shift while cadence-derived features move far less.
It also explains the reported band asymmetry: the earlier "bands 0-3 fell,
bands 4-5 unchanged" split does not reflect frequency-selective physics, since
bands 4-5 read the separate sub-1500 Hz zoom panel and sit deep in the
palette's saturated region in both eras.

## Decoding to absolute dB

Because each image carries its own colourbar, pixels can be inverted back to
absolute dB by nearest-colour lookup against that image's ramp, which makes the
two eras comparable again. The diagnostic does this over the upper panel's
rightmost 45 columns — the most recent sweep, so successive captures are close
to independent rather than re-reads of the same scrollback — restricted to
`11:00`–`13:00` UTC, because the VLF record has a strong diurnal cycle and an
unmatched hour comparison would mostly measure time of day.

Two limits are reported rather than hidden:

* **Saturation.** The palette clips to black below its floor and to solid red
  above its top. Only the window resolvable under *both* variants supports a
  comparison: `-80.0 … -49.5 dB`. Levels outside it are marked censored, not
  compared.
* **Inversion residual.** JPEG compression, gridlines, and overlaid
  annotations push pixels off the palette curve. Pixels further than `30` in
  RGB distance from any ramp colour are masked. This drops `20.5%` of pixels in
  `era_0` and `20.1%` in `era_1`, with median residuals `16.4` and `15.3` — a
  symmetric loss, so it does not bias the comparison in either direction.

Hour-matched, 21 early captures against 51 late ones:

| Band | `palette_0` median | `palette_1` median | Delta | |
| --- | --- | --- | --- | --- |
| 12–15 kHz | `-57.9` | `-73.7` | `-15.8` | |
| 9–12 kHz | `-55.8` | `-71.6` | `-15.8` | |
| 6–9 kHz | `-54.7` | `-72.6` | `-17.9` | |
| 4–6 kHz | `-55.8` | `-80.5` | `-24.7` | censored |
| 2.5–4 kHz | `-69.5` | `-83.2` | `-13.7` | censored |
| 1.5–2.5 kHz | `-70.5` | `-83.2` | `-12.6` | censored |
| 0.7–1.5 kHz | `-63.2` | `-82.1` | `-19.0` | censored |
| 0.2–0.7 kHz | `-55.8` | `-79.0` | `-23.2` | |

Four of eight bands are resolvable under both palettes. Their deltas span
`-15.8` to `-23.2 dB`, a spread of `7.4 dB` across the whole `200 Hz`–`15 kHz`
range.

## What this does and does not settle

Settled:

* The colour scale changed once, during the outage, by `11.58 dB`. The dB
  ruler and image geometry did not change.
* `era_0` and `era_3` image features are not on a common scale. **Do not pool
  them, and do not read any cross-era feature comparison built from raw pixel
  statistics as a physical result.** This is the item-95(a) condition, met.
* The change is invisible to era-boundary diagnostics, because it happened
  between the two dense eras rather than at either edge of one.
* A recoverable path exists. Each image embeds the colourbar it was drawn
  with, so palette-inverted dB features would be era-invariant by construction
  and would not need the eras kept apart.

Not settled:

* **Whether the underlying level change is instrumental or atmospheric.** The
  palette move proves the *features* are incomparable; it does not prove what
  the receiver was seeing. After decoding to absolute dB the late era still
  reads `16`–`23 dB` lower, and the images alone cannot separate a front-end
  gain reduction from a genuinely quieter period. The evidence leans
  instrumental — the shift is broadband and roughly uniform across two decades
  of frequency, is step-like rather than gradual, and coincides with an
  operator changing a display setting, which is what an operator does after a
  gain change. That is circumstantial, not conclusive. Station metadata or
  operator contact would settle it; pixels will not.
* Whether palette-inverted features carry signal. Nothing here tests that. It
  removes a known scale artifact; it does not create evidence.

This diagnostic does not support any claim about earthquake-related VLF
signals. It identifies and quantifies an instrument-side artifact in the
capture record.
