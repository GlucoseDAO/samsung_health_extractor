# Samsung Health Extractor

A modern, type-safe script to generate CSV files from Samsung Health data exports.

## Features

- 🚀 Built with modern Python tooling (uv, polars, typer)
- 🔒 Fully type-hinted for better IDE support and code safety
- ⚡ Fast data processing with Polars
- 📊 Configurable data combinations and filtering
- 🪵 Comprehensive logging with Eliot
- 🎯 CLI interface with Typer

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Install uv (if not already installed)

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install dependencies

```bash
# Create a virtual environment and install dependencies
uv sync
```

## Usage

### Step 1: Download Your Data

Download your data from the Samsung Health app in the settings. It will generate a folder on your phone's storage (usually named like `samsunghealth_username_YYYYMMDDHHMMSS`).

### Step 2: Place the Data

Place the exported folder inside a `Samsung Health` directory at the root of this project:

```
samsung_health_extractor/
├── Samsung Health/
│   └── samsunghealth_username_20241009172448/
│       ├── *.csv files
│       └── ...
├── samsung_health_extractor.py
├── data_combination.json
├── ignore_csvs.txt
└── ...
```

### Step 3: Run the Script

```bash
# Run with default settings
uv run samsung-health extract

# Specify a custom data path
uv run samsung-health extract "./path/to/Samsung Health"

# Specify a custom log file
uv run samsung-health extract --log-file "./my-log.log"
```

`samsung-health-extract` is kept as an alias of `samsung-health`, so `uv run samsung-health-extract extract` works too.

Logs are automatically saved to the `logs/` directory with timestamps (e.g., `logs/samsung_health_extract_20241015_143022.log`). Each log file contains detailed structured logging in JSON format for easy parsing and debugging.

## Wearable tracks for glucose data

The `samsung_health` package reads an export into tidy tables and turns them into columns on any timeline,
typically a continuous glucose monitor (CGM) track on a five-minute grid. Three commands:

```bash
# One parquet per stream: heart rate, step bins, sleep sessions and stages, exercise sessions, period days
uv run samsung-health streams "./Samsung Health" ./streams

# Add the wearable columns to a CGM file (parquet or CSV) whose timestamp column is naive local time;
# --resting-hr and --max-hr (both or neither) add the person's heart-rate zone and minutes of hard effort
uv run samsung-health align "./Samsung Health" cgm.parquet cgm_with_wearables.parquet --time-column date \
    --resting-hr 62 --max-hr 185

# The wearable columns alone on a regular five-minute grid
uv run samsung-health grid "./Samsung Health" wearables_5min.parquet --start 2024-01-01T00:00 --end 2024-12-31T23:55
```

`align` keeps every original column and, for a CSV, every original cell and line ending; it fills the
columns it adds where the file already has them empty and refuses to overwrite data unless given
`--overwrite`. Every value at a row with time `t` is built from what was complete by `t`, so a forecaster
that reads a history ending at a row sees nothing after it:

| Column | What it holds at a row with time `t` |
|---|---|
| `heartrate` | mean heart rate (bpm) of the readings in the ten minutes up to `t`: the general stream's spot readings (every ten minutes by default) and minute bins, and always the workouts' own heart rate, about one sample a second, which Samsung keeps in each workout's file rather than in the general stream |
| `steps` | steps per five minutes over the most recent finished ten-minute step block of Samsung's all-device merge (it lags by up to ten minutes); 0 where the day was recorded and nobody walked, null where the day has no record |
| `calories_burned` | the active kilocalories Samsung attributes to those steps, per five minutes |
| `workout_label`, `workout_duration` | at the row whose five minutes contain a workout's start: its name and active minutes, one row per workout. The duration is known only when the workout ends, so it is the one column that is not past-only |
| `exercise_label`, `exercise_minutes` | the exercise under way during the five minutes up to `t`, and how many of those minutes it covered |
| `sleep_stage` | `awake`, `light`, `deep`, `rem`, or `unstaged` inside a sleep without stages, at the middle of the five minutes; null outside sleep |
| `asleep` | 1 inside a sleep and not awake, 0 otherwise; null for a night (noon to noon) with no sleep of three hours or more, because a night without the watch is unknown rather than awake |
| `menstruation`, `cycle_day`, `cycle_phase` | a logged period day; days since the latest period began; and the phase (`menstrual`, `follicular`, `ovulatory`, `luteal`) estimated only from periods that have already begun |
| `hr_zone` | the zone, 0 to 5, of `heartrate` on the person's heart-rate reserve (Karvonen: zone 1 from 50 per cent of the way from resting to maximum, zone 5 from 90), with `--resting-hr` and `--max-hr` |
| `effort_30min`, `effort_120min` | minutes in the last 30 and 120 whose heart rate reached zone 3; a hard effort keeps acting on glucose for hours after it stops |

The column names `heartrate`, `steps`, `calories_burned`, `workout_label` and `workout_duration` are the
ones the MetaboNet diabetes dataset uses, so a file in its schema gets those columns filled in place.

Three properties of the export break naive readers, and `samsung_health.export.SamsungExport` handles
each: every CSV opens with a metadata line before its header; every row ends with a trailing comma; and
stream names are prefixes of each other (`com.samsung.shealth.exercise` and
`com.samsung.shealth.exercise.weather`), so a glob on the stream name opens the wrong file. Stored times are
UTC, and each record's `time_offset` gives the local wall clock, which is what CGM exports record.

**Workout heart rate is not in the general heart-rate stream.** During a workout, recorded or detected by
the watch on its own, Samsung writes the heart rate about once a second into that workout's `live_data`
file, and the general stream (`tracker.heart_rate`) holds almost nothing for those minutes. The parser always
reads both, so the heart rate covers exercise as well as rest.

## Vitals: HRV, RR intervals in sleep, SpO2, skin temperature, respiratory rate, stress, ECG

```bash
# One parquet per table, every time in local wall-clock time
uv run samsung-health vitals "./Samsung Health" ./vitals
```

| Table | What it holds |
|---|---|
| `hrv_bins` | Samsung's own SDNN and RMSSD per 30 s bin (its algorithm) |
| `sleep_rr` | **per-beat RR intervals in sleep** (`sleep_raw_data`): `beat` time, 30 s `epoch`, `rr_ms`, `confidence` |
| `night_hrv` | per sleep, from `sleep_rr`: RMSSD and SDNN per 5 min of normal beats, their median, mean and lowest heart rate |
| `resting_heart_rate` | per sleep of 3 h or more: the lowest 5-minute mean of the heart-rate stream |
| `spo2_raw`, `skin_temperature`, `respiratory_rate`, `stress` | the watch's binned values |
| `ecg` | the ECG app's 30 s recordings (500 Hz samples) with the app's own mean heart rate and classification |

`sleep_raw_data` keeps one payload per sleep: 30 s epochs from the sleep's start, each with `rri_info` triples
`[flag, confidence 1-255, RR in ms]`. On two real exports the RR intervals of a night add up to the sleep's
length within a few per cent, the nightly RMSSD from them agrees with Samsung's own bins at r 0.97-0.98, and
the resting heart rate with the RR-derived lowest 5-minute rate within about 2 bpm. Only the last week or so of
nights keeps its RR intervals. An export holds no PPG waveform.

## Configuration

### Ignore List (`ignore_csvs.txt`)

This file contains CSV file names to ignore during processing. Files listed here are typically:
- Empty or summary files
- Samsung's own evaluations
- Data you don't need

Edit this file to include or exclude specific CSV files. Lines starting with `#` are treated as comments.

### Data Combination (`data_combination.json`)

This JSON file describes how to combine multiple CSV files into unified datasets. You can:
- Define which CSV files to process
- Specify columns to include from each source
- Rename columns for consistency
- Configure merge keys for combining data
- Set output formats and sorting

#### Example Configuration Structure:

```json
{
  "csv_filtering": {
    "enabled_csvs": [
      ".cycle.daily_temperature",
      ".cycle.flow"
    ]
  },
  "data_combinations": {
    "cycle_complete": {
      "output_file": "complete_cycle_data.csv",
      "sources": [
        {
          "csv_name": ".cycle.daily_temperature",
          "priority": 1,
          "merge_key": "update_time",
          "merge_key_rename": "day_time",
          "required": true,
          "columns_to_include": ["create_time", "update_time"],
          "rename_columns": {
            "create_time": "temperature"
          }
        }
      ],
      "output_structure": {
        "primary_sort": "day_time",
        "sort_ascending": true,
        "final_columns": ["day_time", "temperature"],
        "data_processing": {
          "fill_missing_values": "N/A"
        }
      }
    }
  }
}
```

## Technical Details

### Technology Stack

- **[Polars](https://pola.rs/)**: Fast DataFrame library for efficient data processing
- **[Typer](https://typer.tiangolo.com/)**: Modern CLI framework
- **[Eliot](https://eliot.readthedocs.io/)**: Structured logging for better debugging
- **[uv](https://github.com/astral-sh/uv)**: Fast Python package installer and resolver
- **Type Hints**: Full type annotations for better code quality and IDE support
- **Pathlib**: Modern path handling instead of os.path

### How It Works

1. **File Discovery**: Scans the Samsung Health export directory for CSV files
2. **Filtering**: Applies ignore list and data availability checks
3. **Processing**: Reads enabled CSV files and combines them according to configuration
4. **Output**: Generates combined CSV files with processed data
5. **Logging**: All operations are logged to timestamped files in the `logs/` directory

The Samsung Health export structure is complex, but this tool extracts the essential timestamps and values in readable formats, then concatenates data from various sources based on your configuration.

### Logging

The script uses Eliot for structured logging. Each run creates a timestamped log file in the `logs/` directory:
- **Location**: `logs/samsung_health_extract_YYYYMMDD_HHMMSS.log`
- **Format**: JSON (one JSON object per line for easy parsing)
- **Content**: Detailed information about file processing, filtering, data combination, and any errors

You can use tools like `jq` to filter and analyze logs:
```bash
# View all error messages
cat logs/samsung_health_extract_*.log | jq 'select(.message_type == "error")'

# View filtered files
cat logs/samsung_health_extract_*.log | jq 'select(.message_type | startswith("filtered"))'
```

## Development

### Tests

```bash
uv run pytest -q
```

The tests build a small synthetic export in the export's own file conventions, so they need no real data.

### Type Checking

```bash
uv run mypy samsung_health_extractor.py
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

