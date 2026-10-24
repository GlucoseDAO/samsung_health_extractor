"""A tiny synthetic Samsung Health export, written in the export's own file conventions.

Every CSV starts with a metadata line, carries prefixed column names and ends each row with a trailing
comma; payloads live under ``jsons/<stream>/<first character>/``. Times are UTC with a ``UTC+0300``
offset, so local wall-clock time is three hours later than the stored time.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

STAMP = "20261024125757"
OFFSET = "UTC+0300"


def _fmt(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S.000")


def _ms(t: datetime) -> int:
    return int((t - datetime(1970, 1, 1)).total_seconds() * 1000)


def write_csv(root: Path, stream: str, columns: list[str], rows: list[list[object]], prefix: str = "") -> None:
    header = [f"{prefix}{c}" if prefix and not c.startswith("!") else c.lstrip("!") for c in columns]
    lines = [f"{stream},7006011,3", ",".join(header)]
    lines += [",".join("" if v is None else str(v) for v in row) + "," for row in rows]
    (root / f"{stream}.{STAMP}.csv").write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")


def write_json(root: Path, stream: str, name: str, payload: object) -> None:
    folder = root / "jsons" / stream / name[0]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def export_dir(tmp_path: Path) -> Path:
    root = tmp_path / f"samsunghealth_test_{STAMP}"
    root.mkdir()
    utc = lambda local: local - timedelta(hours=3)  # noqa: E731

    # Heart rate: a spot reading at 08:00:30 local, and a record of minute bins 09:00-09:03 local
    # whose own summary value (999) must be ignored.
    hr = "com.samsung.shealth.tracker.heart_rate"
    write_csv(root, hr, ["start_time", "end_time", "heart_rate", "binning_data", "time_offset", "!datauuid"], [
        [_fmt(utc(datetime(2026, 7, 1, 8, 0, 30))), _fmt(utc(datetime(2026, 7, 1, 8, 0, 30))), 70.0, None, OFFSET, "a"],
        [_fmt(utc(datetime(2026, 7, 1, 9))), _fmt(utc(datetime(2026, 7, 1, 9, 3))), 999.0, "b1.binning.json", OFFSET, "b"],
    ], prefix="com.samsung.health.heart_rate.")
    write_json(root, hr, "b1.binning.json", [
        {"heart_rate": 60.0 + i, "heart_rate_min": 59.0, "heart_rate_max": 70.0,
         "start_time": _ms(utc(datetime(2026, 7, 1, 9, i))), "end_time": _ms(utc(datetime(2026, 7, 1, 9, i, 59)))}
        for i in range(3)
    ])

    # Steps: one day of 144 ten-minute bins for all devices (-2) and a phone-only row (0) that must be
    # ignored. 100 steps from 10:00 to 10:10 local, 40 from 10:10 to 10:20.
    trend = "com.samsung.shealth.step_daily_trend"
    write_csv(root, trend, ["binning_data", "update_time", "source_type", "count", "datauuid", "day_time"], [
        ["all.binning_data.json", _fmt(datetime(2026, 7, 1, 23)), -2, 140, "all", _fmt(datetime(2026, 7, 1))],
        ["phone.binning_data.json", _fmt(datetime(2026, 7, 1, 23)), 0, 9999, "phone", _fmt(datetime(2026, 7, 1))],
    ])
    bins = [{"count": 0, "calorie": 0.0, "distance": 0.0, "speed": 0.0} for _ in range(144)]
    bins[60] = {"count": 100, "calorie": 4.0, "distance": 70.0, "speed": 1.2}
    bins[61] = {"count": 40, "calorie": 2.0, "distance": 30.0, "speed": 1.0}
    write_json(root, trend, "all.binning_data.json", bins)
    write_json(root, trend, "phone.binning_data.json", [{"count": 500, "calorie": 9.0} for _ in range(144)])

    # Exercise: a 20-minute walk from 10:02 local; a weather file whose name extends the stream's.
    ex = "com.samsung.shealth.exercise"
    write_csv(root, ex, ["start_time", "end_time", "exercise_type", "duration", "calorie", "mean_heart_rate",
                         "max_heart_rate", "distance", "time_offset", "datauuid"], [
        [_fmt(utc(datetime(2026, 7, 1, 10, 2))), _fmt(utc(datetime(2026, 7, 1, 10, 22))), 1001, 1_200_000, 80.0,
         95.0, 110.0, 1500.0, OFFSET, "w"],
    ], prefix="com.samsung.health.exercise.")
    write_csv(root, f"{ex}.weather", ["start_time", "temperature"], [[_fmt(datetime(2026, 7, 1)), 20.0]])

    # Sleep: the night of 1 to 2 July, 23:00 to 07:00 local, with a deep stage 02:00-03:00 and an awake
    # stage 03:00-03:10; the night of 2 to 3 July has no sleep record.
    write_csv(root, "com.samsung.shealth.sleep", ["start_time", "end_time", "time_offset", "datauuid"], [
        [_fmt(utc(datetime(2026, 7, 1, 23))), _fmt(utc(datetime(2026, 7, 2, 7))), OFFSET, "s1"],
    ], prefix="com.samsung.health.sleep.")
    write_csv(root, "com.samsung.health.sleep_stage", ["start_time", "end_time", "stage", "time_offset",
                                                       "sleep_id"], [
        [_fmt(utc(datetime(2026, 7, 2, 2))), _fmt(utc(datetime(2026, 7, 2, 3))), 40003, OFFSET, "s1"],
        [_fmt(utc(datetime(2026, 7, 2, 3))), _fmt(utc(datetime(2026, 7, 2, 3, 10))), 40001, OFFSET, "s1"],
    ])

    # Periods: 1-4 June and 29 June-2 July (a 28-day cycle), stored as local midnight in UTC.
    days = [datetime(2026, 6, d) for d in range(1, 5)] + [datetime(2026, 6, 29), datetime(2026, 6, 30),
                                                          datetime(2026, 7, 1), datetime(2026, 7, 2)]
    write_csv(root, "com.samsung.health.cycle.flow", ["spotting", "start_time", "amount", "time_offset"],
              [[0, _fmt(utc(d)), 1, OFFSET] for d in days])
    return root
