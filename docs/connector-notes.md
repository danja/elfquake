# Connector Notes

Record exact request shapes that have been tested.

## INGV Event Text Export

Working recent Italy text request:

```text
https://webservices.ingv.it/fdsnws/event/1/query?starttime=2026-06-22T00%3A00%3A00&endtime=2026-06-29T23%3A59%3A59&minmag=2&maxmag=10&mindepth=-10&maxdepth=1000&minlat=35&maxlat=48&minlon=6&maxlon=19&minversion=100&orderby=time-asc&format=text&limit=10000
```

Observed fields:

```text
#EventID|Time|Latitude|Longitude|Depth/Km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName|EventType
```

## Known Failures

* `format=geojson` returned server error for the same Italy request.
* Narrow Central Italy bounding boxes returned server error.
* Some historical 2016 requests returned server error.

## Next Connector Work

Use the working text export for the first smoke dataset. Derive smaller regions locally until INGV narrow bounding-box behavior is understood.

Connector implementation note: accept UTC input in CLI metadata, but omit trailing `Z` in the INGV request query.
