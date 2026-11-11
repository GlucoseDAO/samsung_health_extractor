"""Vitals beyond heart rate and steps, on a synthetic export in the export's own file conventions."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from conftest import OFFSET, STAMP, _fmt, _ms, write_csv, write_json
from samsung_health import SamsungExport, vitals

UTC = lambda local: local - timedelta(hours=3)  # noqa: E731  (OFFSET is UTC+0300)


@pytest.fixture()
def vitals_dir(tmp_path: Path) -> Path:
    root = tmp_path / f"samsunghealth_vitals_{STAMP}"
    root.mkdir()
    night = datetime(2026, 7, 1, 23, 0)
    write_csv(root, vitals.SLEEP, ["start_time", "end_time", "time_offset", "datauuid"],
              [[_fmt(UTC(night)), _fmt(UTC(night + timedelta(hours=1))), OFFSET, "s1"]],
              prefix="com.samsung.health.sleep.")
    # 120 epochs of 30 s: 1 h of beats at 1000 ms +/- 30 ms, with one artefact in epoch 0
    rng = np.random.default_rng(0)
    epochs = []
    for e in range(120):
        rr = [int(round(v)) for v in 1000 + rng.normal(0, 30, 30)]
        if e == 0:
            rr[5] = 2400
        epochs.append({"acc": [0] * 8, "rri_info": [[1, 200, v] for v in rr]})
    write_csv(root, vitals.SLEEP_RAW, ["sleep_uuid", "data", "datauuid"], [["s1", "r1.data.json", "r1"]])
    write_json(root, vitals.SLEEP_RAW, "r1.data.json", {"data": epochs, "version": 1})
    write_csv(root, vitals.HRV, ["start_time", "time_offset", "binning_data", "datauuid"],
              [[_fmt(UTC(night)), OFFSET, "h1.binning_data.json", "h1"]])
    write_json(root, vitals.HRV, "h1.binning_data.json",
               [{"start_time": _ms(UTC(night + timedelta(seconds=30 * i))), "end_time": 0, "sdnn": 40.0, "rmssd": 42.0}
                for i in range(4)])
    write_csv(root, vitals.ECG, ["start_time", "time_offset", "sample_frequency", "mean_heart_rate", "chart_data",
                                 "classification", "datauuid"],
              [[_fmt(UTC(datetime(2026, 7, 2, 8))), OFFSET, 500, 61.0, "e1.chart_data.json", 1, "e1"],
               [_fmt(UTC(datetime(2026, 7, 2, 9))), OFFSET, 500, 62.0, None, None, "e2"]])
    write_json(root, vitals.ECG, "e1.chart_data.json", {"chart_values": [0.0, 0.1, 0.2]})
    return root


def test_binned_places_each_bin_on_the_wall_clock(vitals_dir: Path) -> None:
    bins = vitals.hrv_bins(SamsungExport.locate(vitals_dir))
    assert bins.height == 4 and bins["rmssd"].to_list() == [42.0] * 4
    assert bins["start"][0] == datetime(2026, 7, 1, 23, 0) and bins["start"][1] == datetime(2026, 7, 1, 23, 0, 30)


def test_sleep_rr_places_beats_in_their_epochs(vitals_dir: Path) -> None:
    rr = vitals.sleep_rr(SamsungExport.locate(vitals_dir))
    assert rr.height == 3600 and rr["epoch"].max() == 119
    second = rr.filter(pl.col("epoch") == 1)
    assert second["beat"][0] == datetime(2026, 7, 1, 23, 0, 30)


def test_night_hrv_drops_artefacts_and_reads_white_noise(vitals_dir: Path) -> None:
    rr = vitals.sleep_rr(SamsungExport.locate(vitals_dir))
    night = vitals.night_hrv(rr)
    assert night.height == 1 and night["segments"][0] >= 10
    assert abs(night["rmssd_ms"][0] - 42.4) < 5 and abs(night["sdnn_ms"][0] - 30) < 4   # sqrt(2) * 30 and 30
    assert abs(night["heart_rate"][0] - 60) < 1 and night["kept_fraction"][0] < 1.0


def test_normal_beats_refuses_out_of_range_and_jumps() -> None:
    rr = np.full(50, 1000.0)
    rr[[3, 10, 20]] = [250.0, 2500.0, 1500.0]
    keep = vitals.normal_beats(rr)
    assert not keep[[3, 10, 20]].any() and keep.sum() == 47


def test_ecg_skips_a_record_without_waveform(vitals_dir: Path) -> None:
    e = vitals.ecg(SamsungExport.locate(vitals_dir))
    assert e.height == 1 and e["samples"][0].to_list() == [0.0, 0.1, 0.2] and e["mean_heart_rate"][0] == 61.0


def test_resting_heart_rate_takes_the_lowest_five_minutes() -> None:
    t0 = datetime(2026, 7, 1, 23, 0)
    times = pl.datetime_range(t0, t0 + timedelta(hours=7), "1m", eager=True)
    hr = np.full(times.len(), 60.0)
    hr[200:206] = 48.0
    stream = pl.DataFrame({"available": times, "heart_rate": hr})
    sleeps = pl.DataFrame({"start": [t0, t0 + timedelta(days=1)], "end": [t0 + timedelta(hours=7), t0 + timedelta(days=1, hours=1)],
                           "sleep_id": ["s", "nap"]})
    out = vitals.resting_heart_rate(stream, sleeps)
    assert out["sleep_id"].to_list() == ["s"] and out["resting_heart_rate"][0] == 48.0


def test_missing_streams_read_empty(tmp_path: Path) -> None:
    root = tmp_path / f"samsunghealth_empty_{STAMP}"
    root.mkdir()
    write_csv(root, "com.samsung.shealth.tracker.heart_rate", ["start_time"], [])
    export = SamsungExport.locate(root)
    assert vitals.hrv_bins(export).is_empty() and vitals.sleep_rr(export).is_empty() and vitals.ecg(export).is_empty()
