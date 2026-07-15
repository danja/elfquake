from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from elfquake.connectors.japan import build_japan_event_url, fetch_japan_events
from elfquake.connectors.vlf_manifest import fetch_manifest_captures
from elfquake.http import HttpCapture
from elfquake.normalize.japan import normalize_japan_event_json


def _capture(url: str, body: bytes) -> HttpCapture:
    return HttpCapture(url, 200, datetime(2026, 7, 15, tzinfo=timezone.utc), {}, body)


class JapanPathTests(unittest.TestCase):
    def test_japan_url_is_bounded(self) -> None:
        query = parse_qs(urlparse(build_japan_event_url(
            "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"
        )).query)
        self.assertEqual(query["minlatitude"], ["30"])
        self.assertEqual(query["maxlatitude"], ["46"])
        self.assertEqual(query["minlongitude"], ["129"])
        self.assertEqual(query["maxlongitude"], ["146"])
        self.assertEqual(query["format"], ["geojson"])

    def test_japan_fetch_and_normalize_share_event_contract(self) -> None:
        payload = {"features": [{
            "id": "jp-test-1",
            "properties": {"time": 1784073600000, "mag": 3.2, "magType": "ml", "place": "Japan", "type": "earthquake"},
            "geometry": {"coordinates": [140.1, 35.2, 20.0]},
        }]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stored = fetch_japan_events(
                "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", out_root=root,
                fetcher=lambda url: _capture(url, json.dumps(payload).encode()),
            )
            out = root / "normalized.csv"
            self.assertEqual(normalize_japan_event_json(stored.payload_path, out), 1)
            row = out.read_text(encoding="utf-8").splitlines()[1].split(",")
            self.assertIn("japan", row)
            self.assertIn("JP", row)

    def test_manifest_capture_preserves_passive_receiver_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "endpoint_id,url,station,latitude,longitude,receiver_mode,region_id,expected_content_type\n"
                "station1,https://example.test/station.ogg,Lulin,23.46,120.87,passive_broadband_elf_vlf,japan,audio/ogg\n",
                encoding="utf-8",
            )
            stored = fetch_manifest_captures(
                manifest, out_root=root / "out", source_namespace="japan",
                fetcher=lambda url: _capture(url, b"raw-radio"),
            )
            metadata = json.loads(stored[0].metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["receiver_mode"], "passive_broadband_elf_vlf")
            self.assertEqual(metadata["region_id"], "japan")


if __name__ == "__main__":
    unittest.main()
