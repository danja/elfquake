from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from elfquake.connectors.astronomy import _materialize_url
from elfquake.connectors.ingv import build_event_url
from elfquake.storage import write_capture


class AcquisitionScaffoldTests(unittest.TestCase):
    def test_ingv_event_url_uses_italy_bounds_and_text_format(self) -> None:
        url = build_event_url("2026-06-22T00:00:00Z", "2026-06-29T23:59:59Z")
        query = parse_qs(urlparse(url).query)

        self.assertEqual(query["format"], ["text"])
        self.assertEqual(query["minlat"], ["35"])
        self.assertEqual(query["maxlat"], ["48"])
        self.assertEqual(query["minlon"], ["6"])
        self.assertEqual(query["maxlon"], ["19"])
        self.assertEqual(query["orderby"], ["time-asc"])

    def test_astronomy_url_materializes_moon_placeholders(self) -> None:
        url = _materialize_url(
            "https://aa.usno.navy.mil/api/moon/phases/date?date=YYYY-MM-DD&nump=N",
            date="2026-06-29",
            moon_phase_count=4,
        )

        self.assertEqual(
            url,
            "https://aa.usno.navy.mil/api/moon/phases/date?date=2026-06-29&nump=4",
        )

    def test_write_capture_writes_payload_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload_path = Path(directory) / "capture.bin"
            stored = write_capture(
                payload_path,
                b"payload",
                url="https://example.test/data",
                status=200,
                captured_at_utc=datetime(2026, 6, 29, 9, 45, tzinfo=timezone.utc),
                headers={"Content-Type": "application/octet-stream"},
                source_id="example",
            )

            self.assertFalse(stored.skipped_existing)
            self.assertEqual(payload_path.read_bytes(), b"payload")
            metadata = json.loads(stored.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_id"], "example")
            self.assertEqual(metadata["captured_at_utc"], "2026-06-29T09:45:00Z")


if __name__ == "__main__":
    unittest.main()
