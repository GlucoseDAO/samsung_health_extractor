"""Reading a Samsung Health export folder.

An export is a folder named ``samsunghealth_<user>_<YYYYMMDDHHMMSS>`` holding one CSV per data stream,
``com.samsung.<group>.<stream>.<stamp>.csv``, and a ``jsons/`` tree with the large per-record payloads
(minute heart rate, the day's ten-minute step bins, exercise live data). Three properties of these files
break naive readers, and each is handled here:

* The first line of every CSV is a metadata line (``com.samsung.shealth.tracker.heart_rate,7006011,3``),
  and the header is the second line.
* Every data row ends with a trailing comma, so it carries one field more than the header.
* Stream names are prefixes of each other (``com.samsung.shealth.exercise`` and
  ``com.samsung.shealth.exercise.weather``), so a glob on ``<stream>.*.csv`` silently picks the wrong file.
  :meth:`SamsungExport.csv_path` matches the stream name and the stamp exactly.

Times in the CSVs are UTC, and the ``time_offset`` column (``UTC+0300``) says what the wall clock read
where the record was made. :func:`local_time` turns the pair into a naive local datetime, which is how CGM
exports record time.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

STAMP_PATTERN = r"\.\d{14}\.csv"


@dataclass(frozen=True)
class SamsungExport:
    """One unpacked export folder (the one holding the ``com.samsung.*.csv`` files)."""

    root: Path

    @classmethod
    def locate(cls, path: Path | str) -> "SamsungExport":
        """Accept the export folder itself or a folder that contains exactly one export folder."""
        path = Path(path)
        if any(path.glob("com.samsung.*.csv")):
            return cls(path)
        candidates = [p for p in path.iterdir() if p.is_dir() and any(p.glob("com.samsung.*.csv"))]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"{path}: expected one Samsung Health export folder, found {len(candidates)}"
            )
        return cls(candidates[0])

    def csv_path(self, stream: str) -> Path | None:
        """The CSV of exactly this stream, e.g. ``com.samsung.shealth.exercise``, or None if absent."""
        pattern = re.compile(re.escape(stream) + STAMP_PATTERN)
        matches = sorted(p for p in self.root.glob(f"{stream}.*.csv") if pattern.fullmatch(p.name))
        return matches[-1] if matches else None

    def read(self, stream: str) -> pl.DataFrame:
        """The stream's rows as strings, empty cells as null, column prefixes stripped where unambiguous.

        ``com.samsung.health.heart_rate.start_time`` becomes ``start_time``; a short name that two
        columns would share keeps its full form. A missing stream reads as an empty frame.
        """
        path = self.csv_path(stream)
        if path is None:
            return pl.DataFrame()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            handle.readline()  # the metadata line
            rows = list(csv.reader(handle))
        if not rows:
            return pl.DataFrame()
        header = rows[0]
        width = len(header)
        body = [(row + [""] * width)[:width] for row in rows[1:] if any(cell for cell in row)]
        short = [name.rsplit(".", 1)[-1] if name.startswith("com.samsung") else name for name in header]
        names = [s if short.count(s) == 1 else full for s, full in zip(short, header)]
        frame = pl.DataFrame(body, schema={n: pl.String for n in names}, orient="row")
        return frame.with_columns(pl.all().replace("", None))

    def json_path(self, stream: str, filename: str) -> Path | None:
        """A payload file named in a CSV cell, stored under ``jsons/<stream>/<first character>/``."""
        for path in (self.root / "jsons" / stream / filename[:1] / filename,
                     self.root / "jsons" / stream / filename):
            if path.is_file():
                return path
        return None

    def read_json(self, stream: str, filename: str) -> Any:
        path = self.json_path(stream, filename)
        if path is None:
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def offset_minutes(column: str = "time_offset") -> pl.Expr:
    """``UTC+0300`` as +180 minutes, ``UTC-0430`` as -270."""
    parts = pl.col(column).str.extract_groups(r"UTC([+-])(\d{2})(\d{2})")
    sign = pl.when(parts.struct.field("1") == "-").then(-1).otherwise(1)
    hours = parts.struct.field("2").cast(pl.Int32)
    minutes = parts.struct.field("3").cast(pl.Int32)
    return (sign * (hours * 60 + minutes)).alias(column)


def utc_time(column: str) -> pl.Expr:
    """A Samsung timestamp string (``2024-04-11 18:25:56.661``) as a naive UTC datetime in microseconds."""
    return pl.col(column).str.to_datetime("%Y-%m-%d %H:%M:%S%.f", time_unit="us", strict=False)


def local_time(column: str, offset: str = "time_offset") -> pl.Expr:
    """The wall-clock time of a UTC column, shifted by the record's own offset (naive, no zone)."""
    return utc_time(column) + pl.duration(minutes=offset_minutes(offset))
