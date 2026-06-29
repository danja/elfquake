"""Archive-oriented space weather acquisition helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from elfquake.http import HttpCapture, fetch_bytes
from elfquake.storage import StoredCapture, filename_timestamp, write_capture


GFZ_KP_AP_URL = "https://kp.gfz.de/app/files/Kp_ap_since_1932.txt"
SPACEWEATHER_CANADA_F107_DAILY_URL = (
    "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt"
)
KYOTO_DST_FINAL_URL = "https://wdc.kugi.kyoto-u.ac.jp/dst_final/{year_month}/index.html"
KYOTO_DST_PROVISIONAL_URL = "https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{year_month}/index.html"
NCEI_GOES15_XRS_AVG1M_YEAR_URL = (
    "https://www.ncei.noaa.gov/instruments/solar-space-observing/particle-detectors/sem/goes/"
    "access/science/xrs/goes15/xrsf-l2-avg1m_science/"
    "sci_xrsf-l2-avg1m_g15_y{year}_v2-2-1.nc"
)


def build_kyoto_dst_url(year_month: str, *, provisional: bool = False) -> str:
    template = KYOTO_DST_PROVISIONAL_URL if provisional else KYOTO_DST_FINAL_URL
    return template.format(year_month=year_month)


def build_ncei_goes15_xrs_year_url(year: int) -> str:
    return NCEI_GOES15_XRS_AVG1M_YEAR_URL.format(year=year)


def fetch_gfz_kp_ap(*, out_root: Path, fetcher: Callable[[str], HttpCapture] = fetch_bytes) -> StoredCapture:
    capture = fetcher(GFZ_KP_AP_URL)
    return _write_archive_capture(
        capture,
        out_root=out_root,
        source_id="gfz_kp_ap_since_1932",
        filename=f"gfz_kp_ap_since_1932_{filename_timestamp(capture.captured_at_utc)}.txt",
        content="Kp and ap geomagnetic indexes",
        cadence="3 hour",
    )


def fetch_spaceweather_canada_f107_daily(
    *,
    out_root: Path,
    fetcher: Callable[[str], HttpCapture] = fetch_bytes,
) -> StoredCapture:
    capture = fetcher(SPACEWEATHER_CANADA_F107_DAILY_URL)
    return _write_archive_capture(
        capture,
        out_root=out_root,
        source_id="spaceweather_canada_f107_daily",
        filename=f"spaceweather_canada_f107_daily_{filename_timestamp(capture.captured_at_utc)}.txt",
        content="F10.7 daily solar radio flux",
        cadence="daily",
    )


def fetch_kyoto_dst_month(
    year_month: str,
    *,
    out_root: Path,
    provisional: bool = False,
    fetcher: Callable[[str], HttpCapture] = fetch_bytes,
) -> StoredCapture:
    source_id = "kyoto_dst_provisional" if provisional else "kyoto_dst_final"
    capture = fetcher(build_kyoto_dst_url(year_month, provisional=provisional))
    return _write_archive_capture(
        capture,
        out_root=out_root,
        source_id=source_id,
        filename=f"{source_id}_{year_month}_{filename_timestamp(capture.captured_at_utc)}.html",
        content="Dst hourly index",
        cadence="monthly page",
        extra_metadata={"year_month": year_month},
    )


def fetch_ncei_goes15_xrs_year(
    year: int,
    *,
    out_root: Path,
    fetcher: Callable[[str], HttpCapture] = fetch_bytes,
) -> StoredCapture:
    capture = fetcher(build_ncei_goes15_xrs_year_url(year))
    return _write_archive_capture(
        capture,
        out_root=out_root,
        source_id="ncei_goes_xrs_g15_avg1m",
        filename=f"ncei_goes_xrs_g15_avg1m_{year}_{filename_timestamp(capture.captured_at_utc)}.nc",
        content="GOES-15 XRS 1-minute irradiance",
        cadence="yearly NetCDF",
        extra_metadata={"year": str(year)},
    )


def _write_archive_capture(
    capture: HttpCapture,
    *,
    out_root: Path,
    source_id: str,
    filename: str,
    content: str,
    cadence: str,
    extra_metadata: dict[str, str] | None = None,
) -> StoredCapture:
    metadata = {"content": content, "cadence": cadence}
    if extra_metadata:
        metadata.update(extra_metadata)
    return write_capture(
        out_root / "captures" / capture.captured_at_utc.date().isoformat() / filename,
        capture.body,
        url=capture.url,
        status=capture.status,
        captured_at_utc=capture.captured_at_utc,
        headers=capture.headers,
        source_id=source_id,
        extra_metadata=metadata,
        skip_existing=True,
    )
