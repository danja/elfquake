# Pre-Real-Data Checklist

Before live backfill or service start:

* install and smoke-test `deploy/systemd/elfquake.service`
* confirm the target host clock uses UTC or reliable NTP
* run one manual `fetch-vlf-cumiana --only last_E_VLF`
* run live tests with `ELFQUAKE_LIVE_TESTS=1`
* review `data/derived/backfill/ingv_italy_2026-06.plan.csv`
* verify `netCDF4` import on the runtime host
* confirm disk retention policy for raw VLF images
