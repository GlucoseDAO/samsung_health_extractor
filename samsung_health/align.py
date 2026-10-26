"""Wearable columns on any timeline, each describing only the past of its row.

:func:`wearable_columns` takes the timestamps of another record, typically a CGM track on a five-minute
grid in naive local time, and returns one row of wearable features per timestamp. Every value at time
``t`` is built from what was complete by ``t``, so a forecaster that reads row ``t`` sees nothing after it:

=====================  ============================================================================
Column                 What it holds at row ``t``
=====================  ============================================================================
``heartrate``          mean heart rate (bpm) of the readings available in ``(t - 10 min, t]``, ten
                       minutes because the watch spot-reads every ten minutes by default; null when
                       there were none
``steps``              steps per five minutes over the latest complete step bin (ten minutes), so it
                       lags by up to ten minutes; 0 where the day was recorded and no step was taken;
                       null where the day has no record
``calories_burned``    the active kilocalories Samsung attributes to those steps, per five minutes,
                       read the same way; exercise without steps (swimming, cycling) is not in it
``workout_label``      at the row whose window holds an exercise's start: its name. MetaboNet's
                       convention, one row per workout
``workout_duration``   on the same row: the workout's active minutes. Known only once the workout
                       ended, so unlike every other column it is not past-only
``exercise_label``     name of the exercise under way during the window, null when none
``exercise_minutes``   minutes of the window spent in an exercise (0 to 5); null where steps are null
``sleep_stage``        ``awake``, ``light``, ``deep``, ``rem`` or ``unstaged`` (a sleep with no stages)
                       at the window's middle; null outside a recorded sleep
``asleep``             1 inside a sleep and not in an awake stage, 0 otherwise; null for a night
                       (noon to noon) with no main sleep of three hours or more, because a night
                       without the watch is unknown, not awake
``menstruation``       1 on a day with a logged period, 0 on other days of the tracked span (from the
                       first logged day to 60 days after the last), null outside it
``cycle_day``          days since the latest period began, the first day being 1; null before the
                       first logged period or past day 60 (a gap in logging)
``cycle_phase``        ``menstrual``, ``follicular``, ``ovulatory`` or ``luteal``, estimated only from
                       periods that have already begun (see :func:`cycle_columns`)
``hr_zone``            the heart-rate zone of ``heartrate``, 0 to 5, on the person's own heart-rate
                       reserve (:class:`Zones`); null without a heart rate or without zones
``effort_30min``       minutes in the 30 minutes up to ``t`` whose heart rate reached zone 3; null
                       when no heart rate was read in them or without zones
``effort_120min``      the same over 120 minutes, because a hard effort keeps acting on glucose for
                       hours after it stops
=====================  ============================================================================

Heart rate includes the workouts' own samples (``streams.workout_heart_rate``); without them the general
stream holds almost none of a person's hard effort, and the zone columns would rarely fire.

The sleep stages are written by Samsung after waking, so a live system would only have last night's
stages in the morning; as a description of the past they are exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from samsung_health import streams
from samsung_health.export import SamsungExport

WINDOW = timedelta(minutes=5)
NS_PER_MIN = 60_000_000_000
WINDOW_NS = 5 * NS_PER_MIN
#: The heart-rate look-back: twice the row spacing, because the watch spot-reads every ten minutes by
#: default and a five-minute window would leave every other row empty.
HEART_RATE_WINDOW_NS = 10 * NS_PER_MIN
#: A sleep of at least this long makes its night (noon to noon) a recorded night.
MAIN_SLEEP = timedelta(hours=3)
#: A period day this far after the previous one starts a new period.
NEW_PERIOD_GAP_DAYS = 8
#: Past this many days since the last period start, the cycle is treated as unlogged.
MAX_CYCLE_DAY = 60
#: Plausible cycle lengths for the running average; anything outside is a logging gap.
CYCLE_LENGTH_RANGE = (18, 45)
DEFAULT_CYCLE_LENGTH = 28
#: How many past cycles the expected length averages.
CYCLE_HISTORY = 6

COLUMNS: dict[str, pl.DataType] = {
    "heartrate": pl.Float64(), "steps": pl.Float64(), "calories_burned": pl.Float64(),
    "workout_label": pl.String(), "workout_duration": pl.Float64(),
    "exercise_label": pl.String(), "exercise_minutes": pl.Float64(),
    "sleep_stage": pl.String(), "asleep": pl.Float64(),
    "menstruation": pl.Float64(), "cycle_day": pl.Float64(), "cycle_phase": pl.String(),
    "hr_zone": pl.Float64(), "effort_30min": pl.Float64(), "effort_120min": pl.Float64(),
}
#: Karvonen zone floors as fractions of the heart-rate reserve: zone 1 from 50 per cent, zone 5 from 90.
ZONE_FRACTIONS = (0.5, 0.6, 0.7, 0.8, 0.9)
#: The zone from which effort is counted as hard.
HARD_ZONE = 3
#: The columns MetaboNet's schema already has; the rest are additions.
METABONET_COLUMNS = ("heartrate", "steps", "calories_burned", "workout_label", "workout_duration")


@dataclass(frozen=True)
class Zones:
    """A person's heart-rate zones, from their resting and maximum heart rate (Karvonen, on the reserve).

    The resting rate is best read off sleep (a low percentile of the heart rate while asleep) and the
    maximum, without a test, as 220 minus the person's age; both are given explicitly, so the zones never
    depend on data from after the row they describe.
    """

    resting: float
    maximum: float

    def __post_init__(self) -> None:
        if not 0 < self.resting < self.maximum:
            raise ValueError(f"resting heart rate {self.resting} must lie below the maximum {self.maximum}")

    @property
    def floors(self) -> np.ndarray:
        """The lowest heart rate of zones 1 to 5."""
        return self.resting + np.array(ZONE_FRACTIONS) * (self.maximum - self.resting)

    def zone(self, heart_rate: np.ndarray) -> np.ndarray:
        """0 to 5 for each heart rate, NaN where it is NaN."""
        heart_rate = np.asarray(heart_rate, dtype=float)
        out = np.searchsorted(self.floors, heart_rate, side="right").astype(float)
        out[~np.isfinite(heart_rate)] = np.nan
        return out


@dataclass(frozen=True)
class Streams:
    """Every stream :func:`wearable_columns` reads, already parsed."""

    heart_rate: pl.DataFrame
    step_bins: pl.DataFrame
    sleep_sessions: pl.DataFrame
    sleep_stages: pl.DataFrame
    exercise_sessions: pl.DataFrame
    menstruation_days: pl.DataFrame

    @classmethod
    def read(cls, export: SamsungExport) -> "Streams":
        return cls(
            heart_rate=streams.heart_rate(export),
            step_bins=streams.step_bins(export),
            sleep_sessions=streams.sleep_sessions(export),
            sleep_stages=streams.sleep_stages(export),
            exercise_sessions=streams.exercise_sessions(export),
            menstruation_days=streams.menstruation_days(export),
        )


def wearable_columns(times: pl.Series, data: Streams, zones: Zones | None = None) -> pl.DataFrame:
    """One row of :data:`COLUMNS` per timestamp in ``times`` (naive local datetimes, any order).

    Without ``zones`` the three zone columns are null.
    """
    t = _ns(times)
    out: dict[str, object] = {}
    out["heartrate"] = _window_mean(t, data.heart_rate, "available", "heart_rate", HEART_RATE_WINDOW_NS)
    empty = np.full(t.shape, np.nan)
    out["hr_zone"] = empty if zones is None else zones.zone(out["heartrate"])
    out["effort_30min"] = empty if zones is None else _hard_minutes(t, data.heart_rate, zones, 30 * NS_PER_MIN)
    out["effort_120min"] = empty if zones is None else _hard_minutes(t, data.heart_rate, zones, 120 * NS_PER_MIN)
    steps, calories = _latest_bin(t, data.step_bins)
    out["steps"], out["calories_burned"] = steps, calories
    out.update(_exercise(t, data.exercise_sessions, covered=~np.isnan(steps)))
    out.update(_sleep(t, data.sleep_sessions, data.sleep_stages))
    out.update(cycle_columns(times, data.menstruation_days))
    return pl.DataFrame([_series(name, out[name], dtype) for name, dtype in COLUMNS.items()])


def _series(name: str, values: object, dtype: pl.DataType) -> pl.Series:
    """A column with nulls where the array holds NaN or None."""
    if isinstance(values, np.ndarray) and values.dtype == object:
        return pl.Series(name, values.tolist(), dtype=dtype)
    return pl.Series(name, values, dtype=dtype, nan_to_null=True)


def _ns(column: pl.Series) -> np.ndarray:
    return column.cast(pl.Datetime("ns")).dt.epoch("ns").to_numpy().astype(np.int64)


def _window_mean(t: np.ndarray, frame: pl.DataFrame, time: str, value: str, window: int) -> np.ndarray:
    """Mean of the readings available in ``(t - window, t]`` (nanoseconds), NaN where there were none."""
    result = np.full(t.shape, np.nan)
    if frame.is_empty():
        return result
    frame = frame.sort(time)
    at = _ns(frame[time])
    cumulative = np.concatenate([[0.0], np.cumsum(frame[value].to_numpy().astype(float))])
    lo = np.searchsorted(at, t - window, side="right")
    hi = np.searchsorted(at, t, side="right")
    count = hi - lo
    has = count > 0
    result[has] = (cumulative[hi[has]] - cumulative[lo[has]]) / count[has]
    return result


def _hard_minutes(t: np.ndarray, frame: pl.DataFrame, zones: Zones, window: int) -> np.ndarray:
    """Minutes in ``(t - window, t]`` whose heart rate reached zone 3, NaN where no heart rate was read in them.

    Each reading belongs to the minute holding it, and a minute counts once it is complete; a minute is hard
    when its highest reading is at zone 3 or above.
    """
    result = np.full(t.shape, np.nan)
    if frame.is_empty():
        return result
    at = _ns(frame["available"])
    minutes = (pl.DataFrame({"m": (at - 1) // NS_PER_MIN, "v": frame["heart_rate"].to_numpy()})
               .group_by("m").agg(pl.col("v").max()).sort("m"))
    end = (minutes["m"].to_numpy() + 1) * NS_PER_MIN
    hard = np.concatenate([[0], np.cumsum(minutes["v"].to_numpy() >= zones.floors[HARD_ZONE - 1])])
    lo = np.searchsorted(end, t - window, side="right")
    hi = np.searchsorted(end, t, side="right")
    seen = hi > lo
    result[seen] = (hard[hi] - hard[lo])[seen]
    return result


def _latest_bin(t: np.ndarray, bins: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Steps and calories per WINDOW over the latest bin that ended by ``t``, if it ended within one bin width."""
    steps = np.full(t.shape, np.nan)
    calories = np.full(t.shape, np.nan)
    if bins.is_empty():
        return steps, calories
    bins = bins.sort("end")
    end = _ns(bins["end"])
    width = end - _ns(bins["start"])
    idx = np.searchsorted(end, t, side="right") - 1
    ok = idx >= 0
    ok[ok] = (t[ok] - end[idx[ok]]) < width[idx[ok]]
    scale = WINDOW_NS / width[idx[ok]]
    steps[ok] = bins["steps"].to_numpy()[idx[ok]] * scale
    calories[ok] = bins["calories"].to_numpy()[idx[ok]] * scale
    return steps, calories


def _exercise(t: np.ndarray, sessions: pl.DataFrame, covered: np.ndarray) -> dict[str, object]:
    window = WINDOW_NS
    minutes = np.where(covered, 0.0, np.nan)
    best = np.zeros(t.shape)
    active = np.full(t.shape, None, dtype=object)
    start_label = np.full(t.shape, None, dtype=object)
    start_duration = np.full(t.shape, np.nan)
    if sessions.is_empty():
        return {"exercise_minutes": minutes, "exercise_label": active,
                "workout_label": start_label, "workout_duration": start_duration}
    order = np.argsort(t, kind="stable")
    ts = t[order]
    for start, end, label, duration in zip(_ns(sessions["start"]), _ns(sessions["end"]),
                                           sessions["label"].to_list(), sessions["duration_min"].to_list()):
        # Rows whose window (t - 5 min, t] overlaps [start, end): start < t and t - 5 min < end.
        lo = np.searchsorted(ts, start, side="right")
        hi = np.searchsorted(ts, end + window, side="left")
        if hi <= lo:
            continue
        rows = order[lo:hi]
        overlap = (np.minimum(t[rows], end) - np.maximum(t[rows] - window, start)) / NS_PER_MIN
        overlap = np.clip(overlap, 0.0, None)
        minutes[rows] = np.nan_to_num(minutes[rows]) + overlap
        better = overlap > best[rows]
        active[rows[better]] = label
        best[rows[better]] = overlap[better]
        # MetaboNet's event row: the first row at or after the start, if the start lies in its window.
        first = np.searchsorted(ts, start, side="left")
        if first < len(ts) and ts[first] - start < window and start_label[order[first]] is None:
            start_label[order[first]] = label
            start_duration[order[first]] = duration if duration is not None else np.nan
    return {"exercise_minutes": np.minimum(minutes, window / NS_PER_MIN), "exercise_label": active,
            "workout_label": start_label, "workout_duration": start_duration}


def _night(ns: np.ndarray) -> np.ndarray:
    """The night a moment belongs to, noon to noon, as the date of the evening (days since epoch)."""
    return (ns - 12 * 60 * NS_PER_MIN) // (24 * 60 * NS_PER_MIN)


def _sleep(t: np.ndarray, sessions: pl.DataFrame, stages: pl.DataFrame) -> dict[str, object]:
    middle = t - WINDOW_NS // 2
    stage = np.full(t.shape, None, dtype=object)
    asleep = np.full(t.shape, np.nan)
    if sessions.is_empty():
        return {"sleep_stage": stage, "asleep": asleep}
    s_start, s_end = _ns(sessions["start"]), _ns(sessions["end"])
    recorded: set[int] = set()
    for a, b in zip(s_start, s_end):
        if b - a >= int(MAIN_SLEEP.total_seconds()) * 1_000_000_000:
            recorded.update(range(int(_night(np.array([a]))[0]), int(_night(np.array([b]))[0]) + 1))
    nights = _night(middle)
    known = np.isin(nights, np.fromiter(recorded, dtype=np.int64, count=len(recorded)))
    asleep[known] = 0.0
    inside = _covering(middle, s_start, s_end)
    stage[inside >= 0] = "unstaged"
    asleep[inside >= 0] = 1.0
    if not stages.is_empty():
        g_start, g_end = _ns(stages["start"]), _ns(stages["end"])
        names = np.array(stages["stage"].to_list(), dtype=object)
        hit = _covering(middle, g_start, g_end)
        sel = hit >= 0
        stage[sel] = names[hit[sel]]
        asleep[sel] = np.where(names[hit[sel]] == "awake", 0.0, 1.0)
    return {"sleep_stage": stage, "asleep": asleep}


def _covering(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Index of an interval ``[start, end)`` holding each point (the latest-starting one), -1 if none."""
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    # Running maximum of ends, so a long interval is not hidden by a short one that starts inside it.
    reach = np.maximum.accumulate(ends)
    idx = np.searchsorted(starts, points, side="right") - 1
    out = np.full(points.shape, -1)
    ok = idx >= 0
    ok[ok] = points[ok] < reach[idx[ok]]
    # Walk back to the latest-starting interval that really contains the point.
    for i in np.flatnonzero(ok):
        j = idx[i]
        while j >= 0 and not (starts[j] <= points[i] < ends[j]):
            j -= 1
        out[i] = order[j] if j >= 0 else -1
    return out


def cycle_columns(times: pl.Series, days: pl.DataFrame) -> dict[str, object]:
    """``menstruation``, ``cycle_day`` and ``cycle_phase`` for each timestamp, from logged period days.

    A period begins on a logged day with no logged day in the :data:`NEW_PERIOD_GAP_DAYS` before it. The
    expected cycle length at a period start is the mean of the up to :data:`CYCLE_HISTORY` previous cycles
    of plausible length (28 days with no history), and ovulation is expected 14 days before the next
    period, on cycle day ``length - 14``. So the phase is ``menstrual`` on a logged day, ``ovulatory``
    from the day before to the day after the expected ovulation, ``follicular`` before that and
    ``luteal`` after. It uses only periods that began on or before the row's day, so it never reads
    the next period's date; a cycle that runs long simply stays ``luteal``.
    """
    n = len(times)
    menstruation = np.full(n, np.nan)
    cycle_day = np.full(n, np.nan)
    phase = np.full(n, None, dtype=object)
    if days.is_empty():
        return {"menstruation": menstruation, "cycle_day": cycle_day, "cycle_phase": phase}
    logged = sorted(days["date"].to_list())
    logged_set = set(logged)
    starts: list[date] = [d for i, d in enumerate(logged)
                          if i == 0 or (d - logged[i - 1]).days >= NEW_PERIOD_GAP_DAYS]
    lengths = [(b - a).days for a, b in zip(starts, starts[1:])]
    expected: list[float] = []
    for k in range(len(starts)):
        past = [x for x in lengths[:k] if CYCLE_LENGTH_RANGE[0] <= x <= CYCLE_LENGTH_RANGE[1]]
        past = past[-CYCLE_HISTORY:]
        expected.append(float(np.mean(past)) if past else float(DEFAULT_CYCLE_LENGTH))
    first, last = logged[0], logged[-1] + timedelta(days=MAX_CYCLE_DAY)
    day_of = times.cast(pl.Datetime("us")).dt.date().to_list()
    start_ord = np.array([d.toordinal() for d in starts])
    for i, d in enumerate(day_of):
        if d is None or d < first or d > last:
            continue
        flow = d in logged_set
        menstruation[i] = 1.0 if flow else 0.0
        k = int(np.searchsorted(start_ord, d.toordinal(), side="right")) - 1
        if k < 0:
            continue
        day = d.toordinal() - start_ord[k] + 1
        if day > MAX_CYCLE_DAY:
            continue
        cycle_day[i] = day
        ovulation = expected[k] - 14
        if flow:
            phase[i] = "menstrual"
        elif day < ovulation - 1:
            phase[i] = "follicular"
        elif day <= ovulation + 1:
            phase[i] = "ovulatory"
        else:
            phase[i] = "luteal"
    return {"menstruation": menstruation, "cycle_day": cycle_day, "cycle_phase": phase}


def add_wearables(frame: pl.DataFrame, time_column: str, data: Streams, overwrite: bool = False,
                  zones: Zones | None = None) -> pl.DataFrame:
    """``frame`` with the wearable columns filled in place where it has them, appended where it does not.

    A column the frame already holds must be empty (all null), as MetaboNet's wearable columns are for a
    person whose study recorded none, unless ``overwrite`` is set. A string time column (a CSV read
    without type inference) is parsed for the alignment and left as it was in the output.
    """
    times = frame[time_column]
    if times.dtype == pl.String:
        times = times.str.to_datetime(strict=False)
    block = wearable_columns(times, data, zones)
    for name in block.columns:
        if name in frame.columns and frame[name].null_count() < frame.height and not overwrite:
            raise ValueError(f"column {name!r} already holds data; pass overwrite=True to replace it")
    replaced = [pl.Series(name, block[name]).cast(frame.schema[name]) if name in frame.columns
                else block[name] for name in block.columns]
    return frame.with_columns(replaced)
