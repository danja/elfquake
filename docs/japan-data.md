# Japan Data Path

Japan is a parallel regional experiment for the same multimodal model contract.
It uses the public USGS FDSN GeoJSON event service for the initial seismic
source, restricted to a Japan bounding box. Raw JSON is retained unchanged;
normalized rows use the shared event fields plus `region_id=japan` and
`country=JP`.

The passive ELF/VLF connector is manifest-driven and deliberately separate from
transmitter-path measurements. Nagoya University ISEE documents two Japanese
passive candidates: Moshiri (44.37 N, 142.27 E) and Kagoshima (31.48 N,
130.72 E). Both use loop-antenna ELF/VLF receivers; Moshiri is documented at
20 kHz and later 40 kHz sampling, while Kagoshima has digital data from 2008
and later 40 kHz operation. Their raw digital data are currently available by
request, so they are recorded in the station registry but not yet in the fetch
manifest. [ISEE station details](https://stdb2.isee.nagoya-u.ac.jp/vlf/vlf_stations.html)

hirahara@nagoya-u.jp, otsuka@isee.nagoya-u.ac.jp, shiokawa@nagoya-u.jp

WALDO is retained only as a secondary historical fallback. Its data are old,
the station/date discovery process is awkward, and downloads require a browser
account. The primary Japan route is now to request a current or recent digital
sample directly from the ISEE Moshiri or Kagoshima station owners. See
[waldo.md](waldo.md).

## Commands

```sh
START=2024-01-01T00:00:00Z END=2026-07-08T00:00:00Z ./scripts/backfill-japan-history.sh
./scripts/capture-japan-vlf-loop.sh
./scripts/build-japan-vlf-features.sh
```

The resulting Japan seismic windows are compatible with the existing training
window interface. Do not merge Japan and Italy rows until region-aware splits,
station provenance, and time overlap checks are in place.

`build-japan-vlf-features.sh` is ready for image-based passive spectrograms. It
will produce no useful rows until the Japan manifest contains verified image
endpoints; audio and other broadband formats require a dedicated feature
adapter rather than being silently treated as images.
