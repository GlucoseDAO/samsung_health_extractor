"""Body measurements (weight, height, blood pressure) on a synthetic export in the export's own conventions."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from conftest import OFFSET, STAMP, _fmt, write_csv
from samsung_health import SamsungExport, body

UTC = lambda local: local - timedelta(hours=3)  # noqa: E731  (OFFSET is UTC+0300)


@pytest.fixture()
def body_dir(tmp_path: Path) -> Path:
    root = tmp_path / f"samsunghealth_body_{STAMP}"
    root.mkdir()
    write_csv(root, body.HEIGHT, ["start_time", "height", "time_offset", "datauuid"], [
        [_fmt(UTC(datetime(2026, 1, 1, 9))), 180.0, OFFSET, "h1"],
        [_fmt(UTC(datetime(2026, 6, 1, 9))), 200.0, OFFSET, "h2"],
    ])
    # a manual weighing before any height, a scale weighing, an entry error, and the same weighing twice
    write_csv(root, body.WEIGHT, ["start_time", "weight", "body_fat", "skeletal_muscle_mass", "time_offset",
                                  "datauuid"], [
        [_fmt(UTC(datetime(2025, 12, 1, 8))), 81.0, None, None, OFFSET, "w0"],
        [_fmt(UTC(datetime(2026, 3, 1, 8))), 72.9, 20.5, 33.0, OFFSET, "w1"],
        [_fmt(UTC(datetime(2026, 3, 2, 8))), 0.5, None, None, OFFSET, "w2"],
        [_fmt(UTC(datetime(2026, 7, 1, 8))), 80.0, None, None, OFFSET, "w3"],
        [_fmt(UTC(datetime(2026, 7, 1, 8))), 80.0, None, None, OFFSET, "w3"],
    ])
    write_csv(root, body.BLOOD_PRESSURE, ["start_time", "systolic", "diastolic", "mean", "pulse", "time_offset",
                                          "!datauuid"], [
        [_fmt(UTC(datetime(2026, 5, 1, 7, 30))), 121, 79, 93, 58, OFFSET, "b1"],
        [_fmt(UTC(datetime(2026, 5, 2, 7, 30))), None, None, None, 60, OFFSET, "b2"],
    ], prefix="com.samsung.health.blood_pressure.")
    return root


def test_weight_uses_the_height_known_at_the_time(body_dir: Path) -> None:
    w = body.weight(SamsungExport(body_dir))
    assert w["start"].to_list() == [datetime(2025, 12, 1, 8), datetime(2026, 3, 1, 8), datetime(2026, 7, 1, 8)]
    assert w["height_cm"].to_list() == [180.0, 180.0, 200.0]  # the first height backfills the earliest weighing
    assert w["bmi"].to_list() == pytest.approx([25.0, 22.5, 20.0])
    assert w["body_fat_pct"].to_list()[1] == pytest.approx(20.5)
    assert w["skeletal_muscle_mass_kg"].null_count() == 2 and w["fat_free_mass_kg"].null_count() == 3


def test_blood_pressure_keeps_complete_cuff_readings(body_dir: Path) -> None:
    bp = body.blood_pressure(SamsungExport(body_dir))
    assert bp.height == 1
    assert bp.row(0, named=True) == {"start": datetime(2026, 5, 1, 7, 30), "systolic": 121.0, "diastolic": 79.0,
                                     "mean": 93.0, "pulse": 58.0}


def test_body_streams_read_empty_without_files(tmp_path: Path) -> None:
    export = SamsungExport(tmp_path)
    assert body.weight(export).is_empty() and "bmi" in body.weight(export).columns
    assert body.height(export).is_empty() and body.blood_pressure(export).is_empty()
