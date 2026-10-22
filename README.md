# KSUstudent-life-study

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Cleaning a Garmin export

`clean_garmin.py` turns one participant's raw Garmin CSV into a regular time series of steps
and/or heart rate:

```bash
.venv/bin/python clean_garmin.py "raw_data/SLS_Garmin step count_for time series.csv"
```

What goes in the output (steps, heart rate, or both) and the row size (1 second, 1 minute or
5 minutes) are set in the `SETTINGS` block at the top of `clean_garmin.py`.

Output goes to `cleaned_data/<rsp_id>_1s.csv`, `_1min.csv` or `_5min.csv` (or pass `-o path.csv`). Missing data is written
as `-1` and the `status` column says why. Run `clean_garmin.py --help` for the column
definitions and options, and see `notes.txt` for known data issues.
