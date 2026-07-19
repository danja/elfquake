import unittest
from datetime import datetime, timedelta, timezone

from elfquake.models.real_transfer_trial import Event, _multiscale_features


class MultiscaleFeatureTests(unittest.TestCase):
    def test_feature_width_is_stable(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [Event(start + timedelta(days=1), 42.0, 12.0, 2.5)]

        values = _multiscale_features(events, start + timedelta(days=7), 42.0, 12.0, 0.5)

        self.assertEqual(len(values), 29)

    def test_future_events_do_not_change_features(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        week = start + timedelta(days=7)
        past = [Event(start + timedelta(days=1), 42.0, 12.0, 2.5)]
        future = past + [Event(week + timedelta(days=1), 42.0, 12.0, 6.0)]

        self.assertEqual(
            _multiscale_features(past, week, 42.0, 12.0, 0.5),
            _multiscale_features(future, week, 42.0, 12.0, 0.5),
        )


if __name__ == "__main__":
    unittest.main()
