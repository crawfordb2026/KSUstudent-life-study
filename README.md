# KSUstudent-life-study

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Cleaning a Garmin export

`clean_garmin.py` turns one participant's raw Garmin CSV into a regular time series of steps
and heart rate:

```bash
.venv/bin/python clean_garmin.py "raw_data/SLS_Garmin step count_for time series.csv"          # 1 row per second
.venv/bin/python clean_garmin.py "raw_data/SLS_Garmin step count_for time series.csv" --bin 1  # 1 row per minute
.venv/bin/python clean_garmin.py "raw_data/SLS_Garmin step count_for time series.csv" --bin 2  # 1 row per 5 minutes
```

Output goes to `cleaned_data/<rsp_id>_1s.csv`, `_1min.csv` or `_5min.csv` (or pass `-o path.csv`). Missing data is written
as `NA` and the `status` column says why. Run `clean_garmin.py --help` for the column
definitions and options, and see `notes.txt` for known data issues.
