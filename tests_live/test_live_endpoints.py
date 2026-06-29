from __future__ import annotations

import json
import os
import unittest
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from elfquake.connectors.ingv import build_event_url
from elfquake.connectors.space_archives import (
    GFZ_KP_AP_URL,
    SPACEWEATHER_CANADA_F107_DAILY_URL,
    build_kyoto_dst_url,
    build_ncei_goes15_xrs_year_url,
)


LIVE_ENABLED = os.environ.get("ELFQUAKE_LIVE_TESTS") == "1"


@unittest.skipUnless(LIVE_ENABLED, "set ELFQUAKE_LIVE_TESTS=1 to run live endpoint tests")
class LiveEndpointTests(unittest.TestCase):
    def test_usno_moon_phase_endpoint_returns_json(self) -> None:
        payload = _get_text(
            "https://aa.usno.navy.mil/api/moon/phases/date?date=2026-06-29&nump=1"
        )
        parsed = json.loads(payload)

        self.assertEqual(parsed["year"], 2026)
        self.assertEqual(parsed["numphases"], 1)
        self.assertIn("phasedata", parsed)

    def test_ingv_event_endpoint_accepts_italy_text_query(self) -> None:
        url = build_event_url(
            "2026-06-22T00:00:00Z",
            "2026-06-29T23:59:59Z",
            limit=1,
        )
        try:
            payload = _get_text(url)
        except HTTPError as error:
            if error.code == 500:
                self.skipTest("INGV endpoint returned HTTP 500 for known-valid Italy text query")
            raise

        query = parse_qs(urlparse(url).query)
        self.assertEqual(query["format"], ["text"])
        self.assertTrue(payload.startswith("#EventID|Time|Latitude|Longitude|Depth/Km"))

    def test_cumiana_vlf_image_endpoint_has_jpeg_headers(self) -> None:
        headers = _head("http://www.vlf.it/cumiana/last_E-VLF.jpg")

        self.assertIn("image/jpeg", headers.get("Content-Type", ""))
        self.assertIn("Last-Modified", headers)

    def test_gfz_kp_ap_archive_head(self) -> None:
        headers = _head(GFZ_KP_AP_URL)

        self.assertIn("text/plain", headers.get("Content-Type", ""))
        self.assertGreater(int(headers.get("Content-Length", "0")), 1_000_000)

    def test_kyoto_dst_month_page_returns_final_dst_table(self) -> None:
        payload = _get_text(build_kyoto_dst_url("201601"))

        self.assertIn("Hourly Equatorial Dst Values", payload)
        self.assertIn("JANUARY   2016", payload)

    def test_ncei_goes_xrs_archive_head(self) -> None:
        headers = _head(build_ncei_goes15_xrs_year_url(2016))

        self.assertEqual(headers["_status"], "200")
        self.assertTrue(
            "netcdf" in headers.get("Content-Type", "").lower()
            or build_ncei_goes15_xrs_year_url(2016).endswith(".nc")
        )

    def test_spaceweather_canada_f107_daily_archive_head(self) -> None:
        headers = _head(SPACEWEATHER_CANADA_F107_DAILY_URL)

        self.assertIn("text/plain", headers.get("Content-Type", ""))
        self.assertGreater(int(headers.get("Content-Length", "0")), 1_000_000)


def _get_text(url: str, timeout_seconds: int = 30) -> str:
    request = Request(url, headers={"User-Agent": "ELFQuake/0.1 live-tests"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode(_charset(response.headers.get("Content-Type")), errors="replace")


def _head(url: str, timeout_seconds: int = 30) -> dict[str, str]:
    request = Request(url, method="HEAD", headers={"User-Agent": "ELFQuake/0.1 live-tests"})
    with urlopen(request, timeout=timeout_seconds) as response:
        headers = {key: value for key, value in response.headers.items()}
        headers["_status"] = str(response.status)
        return headers


def _charset(content_type: str | None) -> str:
    if not content_type:
        return "utf-8"
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            return part.split("=", 1)[1]
    return "utf-8"


if __name__ == "__main__":
    unittest.main()
