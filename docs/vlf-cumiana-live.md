# Cumiana Live VLF

Confirmed live VLF candidate source.

## Page

`http://www.vlf.it/cumiana/livedata.html`

Fetch note: the page required `curl --http1.1 --raw` during testing.

## Station

* name: Cumiana VLF Monitoring Station
* location: Cumiana, Torino, NW Italy
* coordinates: `44.95609`, `7.42123` approximate decimal degrees
* maintainer: Renato Romero

## Live Image Endpoints

| Endpoint | Content | Cadence |
| --- | --- | --- |
| `http://www.vlf.it/cumiana/last_E-VLF.jpg` | last 8 hours VLF electric-field spectrogram | 30 minutes |
| `http://www.vlf.it/cumiana/last-geomar.jpg` | geophone plus electric-field spectrogram | 30 minutes |
| `http://www.vlf.it/cumiana/last-marconi-multistrip-slow.jpg` | electric-field daily multistrip | 30 minutes |
| `http://www.vlf.it/cumiana/last-geophone-multistrip-slow.jpg` | geophone multistrip | 60 minutes |
| `http://www.vlf.it/cumiana/last-plotted.jpg` | last 30 hours plotted traces | 30 minutes |

The server returns `Last-Modified` headers for the JPG endpoints. Use those as capture metadata, not as a substitute for timestamps visible inside the images.

## Signal Notes

* VLF activity image: last 8 hours from Marconi antenna vertical electric field.
* Combined image: geophone channel `1-30 Hz`; electric-field channel `1-105 Hz`.
* Combined image timing: UTC, 40 second scroll time, FFT frequency resolution `21 mHz`.
* Electric-field multistrip: 110 second scroll time, FFT frequency resolution `10.5 mHz`.
* Geophone multistrip: 4.6 second scroll time, FFT frequency resolution `112 mHz`.
* Plotted traces: last 30 hours, values detected every `150 s`.

## Use In Project

Treat these as image-derived VLF/ULF/ELF features unless raw waveform exports are found. The first feature pass should capture images on cadence, store response headers, and extract coarse image statistics by time band.

Capture plan: [VLF Capture Manifest](vlf-capture-manifest.md).

## Open Questions

* Confirm permission for automated periodic capture.
* Confirm whether raw SpectrumLab output files are available.
* Confirm whether the plotted trace image can be replaced by numeric trace data.
* Define image parsing and timestamp extraction rules before model training.
