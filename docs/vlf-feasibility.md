# VLF Feasibility

Early feasibility check for Italy-relevant VLF radio data.

## Candidate

`http://www.vlf.it/`

The site is reachable over HTTP and is focused on reception and study of radio waves below 22 kHz.

Confirmed live candidates: [Cumiana Live VLF](vlf-cumiana-live.md) and [Abelian VLF](vlf-abelian.md).

## Current Findings

* The site appears to be a static collection of articles and experiments, not a documented data API.
* The Open Lab rules page says published intellectual property is free for non-commercial use, while commercial rights remain with original authors.
* Cumiana live JPG spectrogram and plot endpoints are machine-fetchable and include HTTP `Last-Modified` headers.
* Abelian exposes a Cumiana live Ogg stream and a `retrieve.php` archive form for short WAV/VT/spectrogram/time-domain requests.
* The first Abelian live and archive smoke requests returned zero audio/data bytes for Cumiana `vlf15`.
* No nonempty Abelian live or historical pull has been confirmed yet.
* No numeric trace endpoint has been confirmed yet.
* HTTPS returned no usable body during this pass; HTTP was usable for at least the homepage and rules page.

## Required Before Modeling

For modeling-quality use, confirm at least one VLF source with:

* timestamped observations or recordings
* station or receiver location metadata
* sampling rate or aggregation cadence
* explicit reuse terms for derived features
* enough coverage to align with seismic windows

## Decision

Treat Cumiana live imagery as the first usable VLF capture candidate. Treat Abelian live audio and archive retrieval as high-priority candidates for raw VLF samples only after nonempty pulls are reproducible.
