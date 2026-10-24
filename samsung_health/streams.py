"""One tidy table per Samsung Health stream, every time in naive local wall-clock time.

Each function reads one stream of a :class:`~samsung_health.export.SamsungExport` and returns a frame
with fixed columns, sorted by time. The frames are the intermediate layer: :mod:`samsung_health.align`
turns them into columns on any timeline, and ``samsung-health streams`` writes them to parquet as they are.

Every reading is placed at the moment it became complete, ``available``, rather than when it began:
a one-minute heart-rate bin from 08:00 to 08:01 is available at 08:01, a ten-minute step bin from 08:00 to
08:10 at 08:10. That is what lets the aligned columns describe only the past of each row.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from samsung_health.exercise_types import exercise_name
from samsung_health.export import SamsungExport, local_time, offset_minutes, utc_time

HEART_RATE = "com.samsung.shealth.tracker.heart_rate"
STEP_TREND = "com.samsung.shealth.step_daily_trend"
SLEEP = "com.samsung.shealth.sleep"
SLEEP_STAGE = "com.samsung.health.sleep_stage"
EXERCISE = "com.samsung.shealth.exercise"
CYCLE_FLOW = "com.samsung.health.cycle.flow"

#: ``source_type`` of the step trend rows that merge every device (phone and watch) without double
#: counting. Rows of one device carry 0 (the phone) or the watch's own code (107 on a Galaxy Watch).
ALL_DEVICES = -2
#: Samsung's sleep stage codes.
SLEEP_STAGES = {40001: "awake", 40002: "light", 40003: "deep", 40004: "rem"}
#: Heart rates outside this range are sensor errors, not physiology.
HEART_RATE_RANGE = (25.0, 230.0)

LOCAL = pl.Datetime("us")


def heart_rate(export: SamsungExport) -> pl.DataFrame:
    """Every heart-rate reading: ``available`` (local), ``heart_rate``, ``heart_rate_min``, ``heart_rate_max``, ``kind``.

    The watch writes two kinds of record. A spot reading (one value, start equal to end; by default every
    ten minutes) is available at its own time. A record with ``binning_data`` holds a JSON list of
    one-minute bins (continuous measurement, during exercise or when set so), each available at the end
    of its minute; its own summary value is dropped so the minutes are not counted twice. Duplicate
    readings, the same value at the same moment, are kept once.
    """
    raw = export.read(HEART_RATE)
    empty = pl.DataFrame(schema={"available": LOCAL, "heart_rate": pl.Float64, "heart_rate_min": pl.Float64,
                                 "heart_rate_max": pl.Float64, "kind": pl.String})
    if raw.is_empty():
        return empty
    raw = raw.with_columns(offset_minutes().alias("offset_min"))
    # A spot reading is its own minimum and maximum.
    spots = raw.filter(pl.col("binning_data").is_null()).select(
        (utc_time("start_time") + pl.duration(minutes=pl.col("offset_min"))).alias("available"),
        pl.col("heart_rate").cast(pl.Float64),
        pl.col("heart_rate").cast(pl.Float64).alias("heart_rate_min"),
        pl.col("heart_rate").cast(pl.Float64).alias("heart_rate_max"),
        pl.lit("spot").alias("kind"),
    )
    minutes: list[dict] = []
    for name, offset in raw.filter(pl.col("binning_data").is_not_null()).select(
            "binning_data", "offset_min").iter_rows():
        for row in export.read_json(HEART_RATE, name) or []:
            if row.get("heart_rate") is None or row.get("start_time") is None:
                continue
            minutes.append({
                "start_ms": int(row["start_time"]), "offset": offset,
                "heart_rate": float(row["heart_rate"]),
                "heart_rate_min": _float(row.get("heart_rate_min")),
                "heart_rate_max": _float(row.get("heart_rate_max")),
            })
    binned = empty
    if minutes:
        binned = pl.DataFrame(minutes).select(
            # A bin is complete at the end of its minute; start + 60 s is that end on Samsung's 1-min grid.
            (pl.from_epoch(pl.col("start_ms") + 60_000, time_unit="ms").cast(LOCAL)
             + pl.duration(minutes=pl.col("offset"))).alias("available"),
            "heart_rate", "heart_rate_min", "heart_rate_max", pl.lit("minute").alias("kind"),
        )
    low, high = HEART_RATE_RANGE
    return (pl.concat([spots.cast(empty.schema), binned.cast(empty.schema)])
            .filter(pl.col("heart_rate").is_between(low, high))
            .unique(subset=["available", "heart_rate"], keep="first", maintain_order=True)
            .sort("available"))


def step_bins(export: SamsungExport, source_type: int = ALL_DEVICES) -> pl.DataFrame:
    """The day's step bins: ``start``, ``end`` (local), ``steps``, ``calories``, ``distance_m``.

    ``step_daily_trend`` holds one row per local day and device group, with the day split into bins (144
    of ten minutes) in a JSON payload. The default ``source_type`` -2 is Samsung's merge of every device,
    which counts a step once when the phone and the watch both saw it; the minute-level
    ``pedometer_step_count`` records do not, and the export keeps only its last month anyway. ``calories``
    is the active energy Samsung attributes to the steps, in kilocalories. Where a day was written more
    than once, the latest write wins.
    """
    raw = export.read(STEP_TREND)
    empty = pl.DataFrame(schema={"start": LOCAL, "end": LOCAL, "steps": pl.Float64,
                                 "calories": pl.Float64, "distance_m": pl.Float64})
    if raw.is_empty():
        return empty
    days = (raw.filter(pl.col("source_type").cast(pl.Int64) == source_type)
            .with_columns(utc_time("day_time").alias("day"), utc_time("update_time").alias("updated"))
            .sort("updated").unique(subset=["day"], keep="last").sort("day"))
    rows: list[dict] = []
    for day, name in days.select("day", "binning_data").iter_rows():
        bins = export.read_json(STEP_TREND, name) if name else None
        if not bins:
            continue
        width = timedelta(minutes=1440 / len(bins))
        for i, b in enumerate(bins):
            start = day + i * width
            rows.append({"start": start, "end": start + width, "steps": _float(b.get("count")) or 0.0,
                         "calories": _float(b.get("calorie")) or 0.0,
                         "distance_m": _float(b.get("distance")) or 0.0})
    if not rows:
        return empty
    return pl.DataFrame(rows).cast(empty.schema).sort("start")


def sleep_sessions(export: SamsungExport) -> pl.DataFrame:
    """Every recorded sleep, night or nap: ``start``, ``end`` (local), ``sleep_id``."""
    raw = export.read(SLEEP)
    empty = pl.DataFrame(schema={"start": LOCAL, "end": LOCAL, "sleep_id": pl.String})
    if raw.is_empty():
        return empty
    return (raw.select(local_time("start_time").alias("start"), local_time("end_time").alias("end"),
                    pl.col("datauuid").alias("sleep_id"))
            .filter(pl.col("start").is_not_null() & (pl.col("end") > pl.col("start")))
            .unique(subset=["start", "end"], keep="first", maintain_order=True)
            .cast(empty.schema).sort("start"))


def sleep_stages(export: SamsungExport) -> pl.DataFrame:
    """Staged intervals inside the sleeps: ``start``, ``end`` (local), ``stage`` (awake, light, deep, rem), ``sleep_id``."""
    raw = export.read(SLEEP_STAGE)
    empty = pl.DataFrame(schema={"start": LOCAL, "end": LOCAL, "stage": pl.String, "sleep_id": pl.String})
    if raw.is_empty():
        return empty
    return (raw.select(local_time("start_time").alias("start"), local_time("end_time").alias("end"),
                    pl.col("stage").cast(pl.Int64).replace_strict(SLEEP_STAGES, default=None,
                                                                  return_dtype=pl.String).alias("stage"),
                    "sleep_id")
            .filter(pl.col("stage").is_not_null() & (pl.col("end") > pl.col("start")))
            .unique(subset=["start", "end"], keep="first", maintain_order=True)
            .cast(empty.schema).sort("start"))


def exercise_sessions(export: SamsungExport) -> pl.DataFrame:
    """Every exercise, recorded or auto-detected.

    Columns: ``start``, ``end`` (local), ``exercise_type`` (Samsung's code), ``label`` (its name),
    ``duration_min`` (active minutes, pauses excluded), ``calories`` (kcal), ``mean_heart_rate``,
    ``max_heart_rate``, ``distance_m``. Most rows on an ordinary phone are walks of ten minutes or more
    that Samsung detects on its own.
    """
    raw = export.read(EXERCISE)
    schema = {"start": LOCAL, "end": LOCAL, "exercise_type": pl.Int64, "label": pl.String,
              "duration_min": pl.Float64, "calories": pl.Float64, "mean_heart_rate": pl.Float64,
              "max_heart_rate": pl.Float64, "distance_m": pl.Float64}
    if raw.is_empty():
        return pl.DataFrame(schema=schema)
    frame = (raw.select(local_time("start_time").alias("start"), local_time("end_time").alias("end"),
                     pl.col("exercise_type").cast(pl.Int64),
                     (pl.col("duration").cast(pl.Float64) / 60_000).alias("duration_min"),
                     pl.col("calorie").cast(pl.Float64).alias("calories"),
                     pl.col("mean_heart_rate").cast(pl.Float64), pl.col("max_heart_rate").cast(pl.Float64),
                     pl.col("distance").cast(pl.Float64).alias("distance_m"))
             .filter(pl.col("start").is_not_null() & (pl.col("end") > pl.col("start"))))
    frame = frame.with_columns(
        pl.col("exercise_type").map_elements(exercise_name, return_dtype=pl.String).alias("label"),
        pl.col("mean_heart_rate").replace(0.0, None), pl.col("max_heart_rate").replace(0.0, None),
    )
    return (frame.unique(subset=["start", "exercise_type"], keep="first", maintain_order=True)
            .select(list(schema)).cast(schema).sort("start"))


def menstruation_days(export: SamsungExport) -> pl.DataFrame:
    """Every local day with a logged period: ``date``, ``spotting`` (True where only spotting was logged).

    The app stores a period day as local midnight in UTC, so the day is the local date of ``start_time``.
    Its ``amount`` field is 1 on every row unless the user grades the flow, so it is not kept.
    """
    raw = export.read(CYCLE_FLOW)
    schema = {"date": pl.Date, "spotting": pl.Boolean}
    if raw.is_empty():
        return pl.DataFrame(schema=schema)
    return (raw.select(local_time("start_time").dt.date().alias("date"),
                    (pl.col("spotting").cast(pl.Int64) == 1).alias("spotting"))
            .group_by("date").agg(pl.col("spotting").all())
            .cast(schema).sort("date"))


def _float(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
