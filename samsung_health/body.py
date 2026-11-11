"""Body measurements: weight and body composition, height, and blood pressure, in local wall-clock time.

* ``com.samsung.health.weight``: one row per weighing; a smart scale adds body fat, skeletal muscle, fat-free
  mass, total body water and basal metabolic rate (bioimpedance estimates), a manual entry only the weight.
* ``com.samsung.health.height``: one row per height the user entered.
* ``com.samsung.shealth.blood_pressure``: one row per cuff reading (systolic, diastolic, mean, pulse).

Times are stored in UTC with the row's ``time_offset``, as everywhere in the export. BMI is computed from
each weighing and the latest height entered at or before it (or the first height, when every height came
later), so a height corrected afterwards does not rewrite earlier weighings' BMI beyond what was known.
"""

from __future__ import annotations

import polars as pl

from samsung_health.export import SamsungExport, local_time

WEIGHT = "com.samsung.health.weight"
HEIGHT = "com.samsung.health.height"
BLOOD_PRESSURE = "com.samsung.shealth.blood_pressure"

LOCAL = pl.Datetime("us")
#: Weighings outside this range (kg) are entry errors.
WEIGHT_RANGE = (20.0, 400.0)
#: Heights outside this range (cm) are entry errors.
HEIGHT_RANGE = (100.0, 250.0)

WEIGHT_FIELDS = {
    "weight": "weight_kg", "body_fat": "body_fat_pct", "body_fat_mass": "body_fat_mass_kg",
    "skeletal_muscle_mass": "skeletal_muscle_mass_kg", "fat_free_mass": "fat_free_mass_kg",
    "total_body_water": "total_body_water_kg", "basal_metabolic_rate": "basal_metabolic_rate_kcal",
}


def _num(frame: pl.DataFrame, column: str, alias: str) -> pl.Expr:
    if column in frame.columns:
        return pl.col(column).cast(pl.Float64, strict=False).alias(alias)
    return pl.lit(None, dtype=pl.Float64).alias(alias)


def height(export: SamsungExport) -> pl.DataFrame:
    """Every height entered: ``start`` (local), ``height_cm``."""
    raw = export.read(HEIGHT)
    if raw.is_empty() or "height" not in raw.columns:
        return pl.DataFrame(schema={"start": LOCAL, "height_cm": pl.Float64})
    return (
        raw.select(local_time("start_time").alias("start"), _num(raw, "height", "height_cm"))
        .drop_nulls()
        .filter(pl.col("height_cm").is_between(*HEIGHT_RANGE))
        .unique()
        .sort("start")
    )


def weight(export: SamsungExport) -> pl.DataFrame:
    """Every weighing: ``start`` (local), ``weight_kg``, the scale's composition columns (null when absent),
    ``height_cm`` (latest at or before, else the first) and ``bmi``."""
    raw = export.read(WEIGHT)
    schema = {"start": LOCAL, **{v: pl.Float64 for v in WEIGHT_FIELDS.values()}, "height_cm": pl.Float64,
              "bmi": pl.Float64}
    if raw.is_empty() or "weight" not in raw.columns:
        return pl.DataFrame(schema=schema)
    frame = (
        raw.select(local_time("start_time").alias("start"), *[_num(raw, k, v) for k, v in WEIGHT_FIELDS.items()])
        .drop_nulls(["start", "weight_kg"])
        .filter(pl.col("weight_kg").is_between(*WEIGHT_RANGE))
        .unique()
        .sort("start")
    )
    heights = height(export)
    if heights.is_empty():
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("height_cm"))
    else:
        frame = frame.join_asof(heights, on="start", strategy="backward").with_columns(
            pl.col("height_cm").fill_null(heights["height_cm"][0]))
    return frame.with_columns((pl.col("weight_kg") / (pl.col("height_cm") / 100) ** 2).alias("bmi")).select(
        list(schema))


def blood_pressure(export: SamsungExport) -> pl.DataFrame:
    """Every cuff reading: ``start`` (local), ``systolic``, ``diastolic``, ``mean``, ``pulse``."""
    raw = export.read(BLOOD_PRESSURE)
    schema = {"start": LOCAL, "systolic": pl.Float64, "diastolic": pl.Float64, "mean": pl.Float64,
              "pulse": pl.Float64}
    if raw.is_empty() or "systolic" not in raw.columns:
        return pl.DataFrame(schema=schema)
    return (
        raw.select(local_time("start_time").alias("start"), *[_num(raw, c, c) for c in list(schema)[1:]])
        .drop_nulls(["start", "systolic", "diastolic"])
        .unique()
        .sort("start")
    )
