# Source Inventory

Track only Italy-relevant data. Mark a source usable only after a sample pull is reproducible.

| Source | Use | Access | Format | Status |
| --- | --- | --- | --- | --- |
| INGV FDSN event | Italian earthquake catalog | `https://webservices.ingv.it/fdsnws/event/1/` | text | Usable for recent Italy text export; see connector notes |
| INGV FDSN station | Station metadata | `https://webservices.ingv.it/fdsnws/station/1/` | StationXML, text | Candidate |
| INGV FDSN dataselect | Waveform signals | `https://webservices.ingv.it/fdsnws/dataselect/1/` | MiniSEED | Candidate |
| vlf.it | Natural radio context | `http://www.vlf.it/` | Site-specific | Unverified; confirm usable datasets and license |
| NOAA SWPC | Solar and geomagnetic indexes | `https://services.swpc.noaa.gov/json/` | JSON | Candidate; sample Kp endpoint verified |

## Italy Filter

Use an explicit Italy region filter for every source. Start with a bounding box, then replace it with region polygons when the project needs finer spatial labels.

Initial bounding box:

* latitude: `35` to `48`
* longitude: `6` to `19`

## Notes

INGV states its web services follow FDSN specifications and lists event, station, and dataselect services. The INGV web services page also lists Creative Commons Attribution 4.0 licensing.

See [Connector Notes](connector-notes.md) for tested request shapes and known failures.
