"""The watch's vitals beyond heart rate and steps: HRV, RR intervals in sleep, SpO2, skin temperature,
respiratory rate, stress and the ECG app's recordings; and nightly HRV and resting heart rate from them.

Each stream below keeps its values in JSON payloads named by a CSV cell (``binning_data``, ``data``,
``chart_data``), one payload per record:

* ``com.samsung.health.hrv``: Samsung's own SDNN and RMSSD per 30 s bin (its algorithm);
* ``com.samsung.health.respiratory_rate``, ``com.samsung.health.skin_temperature``,
  ``com.samsung.shealth.stress``, ``com.samsung.health.oxygen_saturation.raw``: binned values, each bin
  with its own ``start_time`` (or ``time``) in UTC milliseconds;
* ``com.samsung.shealth.sleep_raw_data``: **per-beat RR intervals in sleep**. One payload per sleep (its
  ``sleep_uuid`` is the sleep's ``datauuid``), a list of 30 s epochs from the sleep's start, each with
  ``rri_info`` triples ``[flag, confidence 1-255, RR in ms]`` and accelerometer counts. On real exports the
  RR intervals of a night add up to the sleep's length within a few per cent;
* ``com.samsung.health.ecg``: the ECG app's 30 s single-lead recordings (``chart_data``, 500 Hz).

An export holds no PPG waveform: the optical sensor appears only through these summaries.

Every time is local wall-clock time, as in :mod:`samsung_health.streams`. Nightly HRV here is computed from
the RR intervals, never from Samsung's bins, and the two are kept apart (on two real exports they agreed at
r 0.97-0.98 over 15 nights, the RR-derived RMSSD a few per cent lower).
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl

from samsung_health.export import SamsungExport, local_time

HRV = "com.samsung.health.hrv"
RESPIRATORY_RATE = "com.samsung.health.respiratory_rate"
SKIN_TEMPERATURE = "com.samsung.health.skin_temperature"
STRESS = "com.samsung.shealth.stress"
SPO2_RAW = "com.samsung.health.oxygen_saturation.raw"
SLEEP_RAW = "com.samsung.shealth.sleep_raw_data"
SLEEP = "com.samsung.shealth.sleep"
ECG = "com.samsung.health.ecg"

LOCAL = pl.Datetime("us")
#: RR intervals outside this range are not normal beats (40-200 bpm).
RR_RANGE_MS = (300.0, 2000.0)


def binned(export: SamsungExport, stream: str, column: str = "binning_data") -> pl.DataFrame:
    """Every bin of every record: ``start`` (local) and each numeric field of the bin.

    The record's ``time_offset`` places the bins on the wall clock. A record without its payload is skipped.
    """
    raw = export.read(stream)
    if raw.is_empty() or column not in raw.columns:
        return pl.DataFrame(schema={"start": LOCAL})
    offsets = raw.select(local_time("start_time").alias("local"), "start_time").with_columns(
        (pl.col("local") - pl.col("start_time").str.to_datetime("%Y-%m-%d %H:%M:%S%.f", time_unit="us",
                                                                  strict=False)).alias("shift"))["shift"].to_list()
    rows = []
    for name, shift in zip(raw[column].to_list(), offsets):
        if not name or shift is None:
            continue
        payload = export.read_json(stream, name)
        if isinstance(payload, dict):
            payload = next((v for v in payload.values() if isinstance(v, list)), None)
        for b in payload or []:
            if not isinstance(b, dict):
                continue
            t = b.get("start_time", b.get("time"))
            if t is None:
                continue
            row = {k: float(v) for k, v in b.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("start_time", "time", "end_time")}
            row["start_ms"] = int(t)
            row["shift_us"] = int(shift.total_seconds() * 1e6)
            rows.append(row)
    if not rows:
        return pl.DataFrame(schema={"start": LOCAL})
    frame = pl.DataFrame(rows, infer_schema_length=None)
    start = pl.from_epoch("start_ms", time_unit="ms").cast(LOCAL) + pl.duration(microseconds="shift_us")
    return (frame.with_columns(start.alias("start")).drop("start_ms", "shift_us")
            .select("start", pl.exclude("start")).sort("start"))


def hrv_bins(export: SamsungExport) -> pl.DataFrame:
    """Samsung's own HRV: ``start``, ``sdnn``, ``rmssd`` (ms) per 30 s bin."""
    return binned(export, HRV)


def respiratory_rate(export: SamsungExport) -> pl.DataFrame:
    """``start``, ``respiratory_rate`` (breaths per minute), per bin."""
    return binned(export, RESPIRATORY_RATE)


def skin_temperature(export: SamsungExport) -> pl.DataFrame:
    """``start``, ``mean``, ``min``, ``max`` (degrees C at the wrist, not core temperature), per bin."""
    return binned(export, SKIN_TEMPERATURE)


def stress(export: SamsungExport) -> pl.DataFrame:
    """``start``, ``score`` and Samsung's other stress fields, per bin."""
    return binned(export, STRESS)


def spo2_raw(export: SamsungExport) -> pl.DataFrame:
    """``start``, ``channel``: the overnight raw oxygen-saturation series (about one value a second)."""
    return binned(export, SPO2_RAW)


def sleep_rr(export: SamsungExport) -> pl.DataFrame:
    """Per-beat RR intervals in sleep: ``sleep_id``, ``beat`` (local time), ``epoch``, ``rr_ms``, ``confidence``.

    A beat's time is its 30 s epoch's start from the sleep's start plus the RR intervals before it within the
    epoch. A sleep the sleep table does not list is skipped (no start to place it on).
    """
    schema = {"sleep_id": pl.String, "beat": LOCAL, "epoch": pl.Int64, "rr_ms": pl.Float64, "confidence": pl.Int64}
    raw = export.read(SLEEP_RAW)
    sleeps = export.read(SLEEP)
    if raw.is_empty() or sleeps.is_empty():
        return pl.DataFrame(schema=schema)
    starts = dict(sleeps.select("datauuid", local_time("start_time").alias("start")).iter_rows())
    rows = []
    for sleep_id, name in raw.select("sleep_uuid", "data").iter_rows():
        start = starts.get(sleep_id)
        payload = export.read_json(SLEEP_RAW, name) if name else None
        if start is None or not payload:
            continue
        for epoch, record in enumerate(payload.get("data", [])):
            t = start + timedelta(seconds=30 * epoch)
            for _, confidence, rr in record.get("rri_info", []):
                rows.append((sleep_id, t, epoch, float(rr), int(confidence)))
                t = t + timedelta(milliseconds=rr)
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row").sort("beat")


def ecg(export: SamsungExport) -> pl.DataFrame:
    """The ECG app's recordings: ``start`` (local), ``sample_frequency``, ``samples`` (as stored),
    ``mean_heart_rate`` and ``classification`` (the app's own values)."""
    schema = {"start": LOCAL, "sample_frequency": pl.Float64, "samples": pl.List(pl.Float64),
              "mean_heart_rate": pl.Float64, "classification": pl.String}
    raw = export.read(ECG)
    if raw.is_empty() or "chart_data" not in raw.columns:
        return pl.DataFrame(schema=schema)
    rows = []
    for r in raw.with_columns(local_time("start_time").alias("local")).iter_rows(named=True):
        payload = export.read_json(ECG, r["chart_data"]) if r.get("chart_data") else None
        if not payload or not payload.get("chart_values"):
            continue
        rows.append((r["local"], float(r.get("sample_frequency") or 500), [float(v) for v in payload["chart_values"]],
                     float(r["mean_heart_rate"]) if r.get("mean_heart_rate") else None, r.get("classification")))
    return pl.DataFrame(rows, schema=schema, orient="row") if rows else pl.DataFrame(schema=schema)


# ---- nightly summaries ------------------------------------------------------------------------------
def normal_beats(rr_ms: np.ndarray, jump: float = 0.2, window: int = 11) -> np.ndarray:
    """A mask of normal-to-normal beats: within :data:`RR_RANGE_MS` and within ``jump`` of the local median."""
    rr = np.asarray(rr_ms, float)
    ok = (rr >= RR_RANGE_MS[0]) & (rr <= RR_RANGE_MS[1])
    if len(rr) == 0:
        return ok
    half = window // 2
    padded = np.pad(np.where(ok, rr, np.nan), half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(np.isfinite(windows).any(1, keepdims=True), windows, 0), axis=1)
    return ok & (np.abs(rr - med) <= jump * med)


def night_hrv(rr: pl.DataFrame, segment_minutes: int = 5, min_beats: int = 150) -> pl.DataFrame:
    """Per sleep, from :func:`sleep_rr`: RMSSD and SDNN of each ``segment_minutes`` window of normal beats, and
    their median over the night; the mean heart rate and the lowest segment's; the share of beats kept.

    Successive differences are taken only between neighbouring beats both kept. A window with fewer than
    ``min_beats`` normal beats is left out.
    """
    out = []
    for (sleep_id,), night in rr.sort("beat").group_by("sleep_id", maintain_order=True):
        x = night["rr_ms"].to_numpy()
        keep = normal_beats(x)
        t0 = night["beat"][0]
        seg = ((night["beat"] - t0).dt.total_seconds() // (segment_minutes * 60)).to_numpy()
        rmssd, sdnn, hr = [], [], []
        for s in np.unique(seg):
            m = seg == s
            if (keep & m).sum() < min_beats:
                continue
            xs, ks = x[m], keep[m]
            d = np.diff(xs)[ks[1:] & ks[:-1]]
            if len(d) < min_beats // 2:
                continue
            rmssd.append(float(np.sqrt(np.mean(d ** 2))))
            sdnn.append(float(np.std(xs[ks], ddof=1)))
            hr.append(60000.0 / float(np.mean(xs[ks])))
        out.append({"sleep_id": sleep_id, "start": t0, "hours": float((night["beat"][-1] - t0).total_seconds() / 3600),
                    "segments": len(rmssd), "kept_fraction": float(keep.mean()),
                    "rmssd_ms": float(np.median(rmssd)) if rmssd else None,
                    "sdnn_ms": float(np.median(sdnn)) if sdnn else None,
                    "heart_rate": float(np.median(hr)) if hr else None,
                    "lowest_segment_heart_rate": float(np.min(hr)) if hr else None})
    schema = {"sleep_id": pl.String, "start": LOCAL, "hours": pl.Float64, "segments": pl.Int64,
              "kept_fraction": pl.Float64, "rmssd_ms": pl.Float64, "sdnn_ms": pl.Float64, "heart_rate": pl.Float64,
              "lowest_segment_heart_rate": pl.Float64}
    return pl.DataFrame(out, schema=schema) if out else pl.DataFrame(schema=schema)


def resting_heart_rate(heart_rate: pl.DataFrame, sleeps: pl.DataFrame, minutes: int = 5,
                       min_hours: float = 3.0) -> pl.DataFrame:
    """Per sleep of ``min_hours`` or more: the lowest ``minutes``-minute mean of the heart rate inside it.

    ``heart_rate`` and ``sleeps`` are :func:`samsung_health.streams.heart_rate` and ``sleep_sessions``. On two
    real exports this agreed with the RR intervals' lowest 5-minute heart rate within about 2 bpm.
    """
    schema = {"sleep_id": pl.String, "start": LOCAL, "hours": pl.Float64, "readings": pl.Int64,
              "resting_heart_rate": pl.Float64}
    hr = heart_rate.select(pl.col("available").alias("t"), "heart_rate").drop_nulls().sort("t")
    out = []
    for start, end, sleep_id in sleeps.select("start", "end", "sleep_id").iter_rows():
        if end - start < timedelta(hours=min_hours):
            continue
        part = hr.filter(pl.col("t").is_between(start, end))
        if part.height < 3:
            continue
        rolled = part.group_by_dynamic("t", every="1m", period=f"{minutes}m").agg(pl.col("heart_rate").mean())
        out.append((sleep_id, start, (end - start).total_seconds() / 3600, part.height,
                    float(rolled["heart_rate"].min())))
    return pl.DataFrame(out, schema=schema, orient="row") if out else pl.DataFrame(schema=schema)
