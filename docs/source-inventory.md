# Source Inventory

Track only Italy-relevant data. Mark a source usable only after a sample pull is reproducible.

| Source | Use | Access | Format | Status |
| --- | --- | --- | --- | --- |
| INGV FDSN event | Italian earthquake catalog | `https://webservices.ingv.it/fdsnws/event/1/` | text | Usable for recent Italy text export; see connector notes |
| INGV FDSN station | Station metadata | `https://webservices.ingv.it/fdsnws/station/1/` | StationXML, text | Candidate |
| INGV FDSN dataselect | Waveform signals | `https://webservices.ingv.it/fdsnws/dataselect/1/` | MiniSEED | Candidate |
| vlf.it Cumiana live | VLF radio context | `http://www.vlf.it/cumiana/livedata.html` | JPG spectrograms and plots | Live image endpoints confirmed; raw waveform not confirmed |
| Abelian Cumiana VLF | VLF audio and archive candidate | `http://abelian.org/vlf/` | Ogg live stream, retrieve form, WAV/VT candidates | Endpoints confirmed; nonempty live/archive pull pending |
| NOAA SWPC | Solar and geomagnetic indexes | `https://services.swpc.noaa.gov/json/` | JSON | Monthly solar-cycle history confirmed; Kp, Dst, GOES X-ray rolling feeds only so far |
| NOAA/NCEI GOES SEM | GOES XRS archive | `https://www.ncei.noaa.gov/instruments/solar-space-observing/particle-detectors/sem/goes/access/` | HTML, text, NetCDF | Historical archive tree confirmed |
| GFZ Kp | Planetary Kp/ap archive | `https://kp.gfz.de/app/files/Kp_ap_since_1932.txt` | text | Historical Kp/ap candidate confirmed |
| Kyoto WDC Dst | Dst hourly index | `https://wdc.kugi.kyoto-u.ac.jp/dstdir/` | HTML, text | Final and provisional archive pages confirmed; non-commercial notice |
| Space Weather Canada | Daily F10.7 solar radio flux | `https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt` | text | Daily archive candidate confirmed |
| USNO | Lunar phase events | `https://aa.usno.navy.mil/api/` | JSON | Moon phase endpoint confirmed |

## Italy Filter

Use an explicit Italy region filter for every source. Start with a bounding box, then replace it with region polygons when the project needs finer spatial labels.

Initial bounding box:

* latitude: `35` to `48`
* longitude: `6` to `19`

## Notes

INGV states its web services follow FDSN specifications and lists event, station, and dataselect services. The INGV web services page also lists Creative Commons Attribution 4.0 licensing.

See [Connector Notes](connector-notes.md) for tested request shapes and known failures.

See [VLF Feasibility](vlf-feasibility.md), [Cumiana Live VLF](vlf-cumiana-live.md), and [Abelian VLF](vlf-abelian.md) for current VLF source findings.

See [Astronomical Feasibility](astronomical-feasibility.md) and [NOAA Archive Feasibility](noaa-archive-feasibility.md) for lunar, solar, and geomagnetic source findings.
