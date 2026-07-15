# WALDO Broadband VLF

**Project status: secondary historical fallback, not a primary acquisition source.**

WALDO is a historical archive, not a live receiver. It separates:

* **Broadband VLF**: raw receiver samples, normally 100 kHz sampling, suitable
  for natural-radio features.
* **Broadband LF**: normally 1 MHz sampling and much larger files.
* **Narrowband**: amplitude and phase of selected transmitters; keep this out of
  the passive natural-radio input unless explicitly used as a separate modality.

## Access

Browsing is available without an account. Downloading requires a free verified
account. The site provides daily availability calendars, maps, quick-look
plots, and a JavaScript file browser. WALDO also offers larger-scale access by
arrangement.

## File contract

Broadband files use a MATLAB v4-like format and contain station metadata plus
`data`. The filename encodes station, UTC start time, sampling-rate class, and
antenna channel. Broadband VLF is normally `int16`, with two orthogonal loop
antenna channels. The format page documents a Python toolset and the field
layout, so a small native reader can avoid loading full files into memory.

## Japan relevance

WALDO explicitly lists broadband data from a Japan receiver within 100 km of
the 2011 Tohoku earthquake. The data are old, station/date discovery is awkward,
and the download workflow is account- and browser-dependent. This makes WALDO
useful for a historical case study or optional self-supervised pretraining, but
not a good primary source for a current Japan/seismic comparison.

## Onboarding rule

Do not add WALDO to the live capture manifest by default. If it is used for a
historical experiment, record the station
ID, coordinates, UTC coverage, file URL, account/licensing terms, sampling rate,
channel orientation, and a nonempty sample checksum. Prefer one short sample
and its quick-look plot before requesting bulk data.

Reference: [WALDO](https://waldo.world/), [data description](https://waldo.world/description-of-data/), and [availability page](https://waldo.world/whats-available/).
