from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from samsung_health import SamsungExport, Streams, add_wearables, wearable_columns
from samsung_health import streams


def at(*times: datetime) -> pl.Series:
    return pl.Series("date", list(times), dtype=pl.Datetime("us"))


def row(export_dir: Path, t: datetime) -> dict:
    return wearable_columns(at(t), Streams.read(SamsungExport(export_dir))).row(0, named=True)


def test_exact_stream_match_and_prefixes(export_dir: Path) -> None:
    export = SamsungExport.locate(export_dir.parent)
    assert export.csv_path("com.samsung.shealth.exercise").name.startswith("com.samsung.shealth.exercise.2026")
    frame = export.read("com.samsung.shealth.exercise")
    assert frame.height == 1 and "exercise_type" in frame.columns and frame["datauuid"][0] == "w"
    assert export.read("com.samsung.shealth.missing").is_empty()


def test_heart_rate_minutes_replace_the_summary(export_dir: Path) -> None:
    hr = streams.heart_rate(SamsungExport(export_dir))
    assert 999.0 not in hr["heart_rate"].to_list()
    assert hr["available"].to_list() == [datetime(2026, 7, 1, 8, 0, 30), datetime(2026, 7, 1, 9, 1),
                                         datetime(2026, 7, 1, 9, 2), datetime(2026, 7, 1, 9, 3)]


def test_heart_rate_reads_only_the_past(export_dir: Path) -> None:
    assert row(export_dir, datetime(2026, 7, 1, 9, 2))["heartrate"] == pytest.approx(60.5)  # bins ending 9:01, 9:02
    assert row(export_dir, datetime(2026, 7, 1, 9, 5))["heartrate"] == pytest.approx(61.0)
    assert row(export_dir, datetime(2026, 7, 1, 8, 0))["heartrate"] is None  # the 8:00:30 reading is ahead
    assert row(export_dir, datetime(2026, 7, 1, 8, 10))["heartrate"] == 70.0
    assert row(export_dir, datetime(2026, 7, 1, 8, 11))["heartrate"] is None  # older than ten minutes


def test_steps_lag_to_the_latest_complete_bin(export_dir: Path) -> None:
    bins = streams.step_bins(SamsungExport(export_dir))
    assert bins.height == 144 and bins["steps"].sum() == 140  # the phone-only row is ignored
    assert row(export_dir, datetime(2026, 7, 1, 10, 5))["steps"] == 0.0  # 10:00-10:10 not complete yet
    r = row(export_dir, datetime(2026, 7, 1, 10, 10))
    assert r["steps"] == 50.0 and r["calories_burned"] == 2.0  # per five minutes
    assert row(export_dir, datetime(2026, 7, 1, 10, 25))["steps"] == 20.0
    assert row(export_dir, datetime(2026, 7, 2, 0, 15))["steps"] is None  # no record for 2 July


def test_exercise_minutes_and_metabonet_event_row(export_dir: Path) -> None:
    data = Streams.read(SamsungExport(export_dir))
    times = [datetime(2026, 7, 1, 10, m) for m in range(0, 35, 5)]
    frame = wearable_columns(at(*times), data)
    assert frame["exercise_minutes"].to_list() == [0.0, 3.0, 5.0, 5.0, 5.0, 2.0, 0.0]
    assert frame["exercise_label"].to_list() == [None, "Walking", "Walking", "Walking", "Walking", "Walking", None]
    assert frame["workout_label"].to_list() == [None, "Walking", None, None, None, None, None]
    assert frame["workout_duration"].to_list()[1] == 20.0


def test_sleep_stage_and_unrecorded_nights(export_dir: Path) -> None:
    assert row(export_dir, datetime(2026, 7, 2, 2, 30))["sleep_stage"] == "deep"
    awake = row(export_dir, datetime(2026, 7, 2, 3, 5))
    assert awake["sleep_stage"] == "awake" and awake["asleep"] == 0.0
    unstaged = row(export_dir, datetime(2026, 7, 2, 0, 30))
    assert unstaged["sleep_stage"] == "unstaged" and unstaged["asleep"] == 1.0
    assert row(export_dir, datetime(2026, 7, 2, 9, 0))["asleep"] == 0.0  # same night, after waking
    assert row(export_dir, datetime(2026, 7, 3, 2, 0))["asleep"] is None  # a night with no record


def test_cycle_uses_only_periods_already_begun(export_dir: Path) -> None:
    r = row(export_dir, datetime(2026, 6, 28, 12))
    assert r["cycle_day"] == 28 and r["menstruation"] == 0.0 and r["cycle_phase"] == "luteal"
    r = row(export_dir, datetime(2026, 6, 29, 0, 5))
    assert r["cycle_day"] == 1 and r["menstruation"] == 1.0 and r["cycle_phase"] == "menstrual"
    assert row(export_dir, datetime(2026, 6, 8))["cycle_phase"] == "follicular"
    assert row(export_dir, datetime(2026, 6, 14))["cycle_phase"] == "ovulatory"  # day 14 of an expected 28
    assert row(export_dir, datetime(2026, 5, 31))["cycle_day"] is None  # before the first logged period


def test_add_wearables_fills_empty_columns_and_refuses_full_ones(export_dir: Path) -> None:
    data = Streams.read(SamsungExport(export_dir))
    frame = pl.DataFrame({"date": at(datetime(2026, 7, 1, 10, 10)), "CGM": [120.0],
                          "steps": pl.Series([None], dtype=pl.Float64)})
    out = add_wearables(frame, "date", data)
    assert out.columns[:3] == ["date", "CGM", "steps"] and out["steps"][0] == 50.0 and "asleep" in out.columns
    with pytest.raises(ValueError):
        add_wearables(out, "date", data)
    assert add_wearables(out, "date", data, overwrite=True)["steps"][0] == 50.0


def test_rows_keep_their_order(export_dir: Path) -> None:
    data = Streams.read(SamsungExport(export_dir))
    t = [datetime(2026, 7, 1, 10, 25), datetime(2026, 7, 1, 10, 10)]
    assert wearable_columns(at(*t), data)["steps"].to_list() == [20.0, 50.0]
    assert wearable_columns(at(*t), data)["exercise_minutes"].to_list() == [2.0, 5.0]


def test_local_time_uses_each_records_offset(export_dir: Path) -> None:
    sessions = streams.exercise_sessions(SamsungExport(export_dir))
    assert sessions["start"][0] == datetime(2026, 7, 1, 10, 2)
    assert sessions["end"][0] - sessions["start"][0] == timedelta(minutes=20)
