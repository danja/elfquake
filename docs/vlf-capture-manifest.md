# VLF Capture Manifest

Capture plan for Cumiana live VLF images.

## Manifest

Machine-readable endpoint manifest:

`data/raw/vlf/cumiana/manifest.csv`

## Storage Layout

Store raw captures by station, endpoint, and capture date:

```text
data/raw/vlf/cumiana/captures/YYYY-MM-DD/<endpoint_id>_<last_modified_utc>.jpg
data/raw/vlf/cumiana/captures/YYYY-MM-DD/<endpoint_id>_<last_modified_utc>.headers.txt
```

Use UTC in filenames. Replace `:` with `-`.

## Capture Rules

* Use HTTP endpoints from [Cumiana Live VLF](vlf-cumiana-live.md).
* Save the image unchanged.
* Save response headers alongside the image.
* Record `Date`, `Last-Modified`, `ETag`, `Content-Length`, and `Content-Type`.
* Do not overwrite an existing file with the same endpoint and `Last-Modified`.
* Respect the published cadence: poll no more often than every 30 minutes for 30-minute images and every 60 minutes for the geophone multistrip.

Use `capture-vlf-cumiana-loop` for short cadence checks. It defaults to 30-minute spacing and rejects repeated polling below 60 seconds.

## First Feature Pass

Before model training, derive only coarse image features:

* image availability by window
* mean and variance by broad time-frequency bands
* simple anomaly scores relative to recent captures
* missing or stale image flags

Raw waveform or numeric trace exports should replace image-derived features if they become available.

Current coarse feature stub: [Feature Extraction](feature-extraction.md).
