#!/usr/bin/env python3
"""Convert one participant's raw Garmin export into a regular time series.

The raw export is event-based: the watch only sends a value when it changes, and
sends nothing at all while it is disconnected from the phone. This script lays
the step and/or heart-rate data onto a fixed one-row-per-second grid and labels
every second with whether its data can be trusted. It can then group those
seconds into 1-minute or 5-minute rows.

Usage:
    python clean_garmin.py "raw_data/SLS_Garmin step count_for time series.csv"
    python clean_garmin.py INPUT.csv -o OUTPUT.csv

What goes in the output and the row size are set in the SETTINGS block below:
    INCLUDE_STEPS  True/False   step columns
    INCLUDE_HR     True/False   heart-rate column
    BIN_SIZE       '1s'   one row per second      -> cleaned_data/<rsp_id>_1s.csv
                   '1min' one row per minute      -> cleaned_data/<rsp_id>_1min.csv
                   '5min' one row per 5 minutes   -> cleaned_data/<rsp_id>_5min.csv
Bins start on whole minutes / 5-minute marks (e.g. 14:05:00-14:09:59), so the
first and last bins of a recording are usually partial.

Output columns, BIN_SIZE '1s' (per second):
    rsp_id             participant id
    timestamp_utc      start of the second, UTC
    timestamp_local    start of the second, US Eastern. When DST ends, one hour
                       appears twice; use timestamp_utc to tell them apart.
    local_date         Eastern calendar date (the day the step counter belongs to)
    status             ok | no_data | steps_unknown | catchup (see below)
  with INCLUDE_STEPS:
    steps_today        the watch's running step total for local_date
    steps_this_second  steps taken during this second
    unplaced_steps     steps that happened at an unknown earlier time and were
                       delivered in this second (0 on all other rows)
  with INCLUDE_HR:
    hr                 heart rate in bpm

Status values:
    ok             watch connected; steps_this_second is real (0 means no steps)
    no_data        watch sent nothing; steps_today, steps_this_second and hr are NA
    steps_unknown  watch is sending again after a gap, but hasn't re-sent the
                   step counter yet; step columns are NA (hr may be present)
    catchup        the step counter jumped after a gap, at the start of the
                   recording, or after the watch restarted; the jump is in
                   unplaced_steps and steps_this_second is NA
Without steps, only ok and no_data are used (watch connected or not).
hr can be NA on an ok row: the watch was connected but had no pulse reading,
usually because it wasn't being worn.

Output columns, BIN_SIZE '1min' and '5min':
    rsp_id, timestamp_utc, timestamp_local, local_date
                       as above, for the start of the bin
    status             ok | partial | missing (see below)
  with INCLUDE_STEPS:
    known_seconds      seconds in the bin with known steps (status ok above);
                       out of 60 for '1min', 300 for '5min'
    steps_today        last known running total in the bin
    steps              steps taken during the bin, summed over known_seconds.
                       In a partial bin this is a lower bound.
    unplaced_steps     unplaced steps delivered during the bin
  with INCLUDE_HR:
    hr                 mean heart rate over the seconds that had one

Binned status values (with steps: seconds with known steps; without: connected seconds):
    ok        every second in the bin is covered
    partial   some seconds are covered
    missing   no second is covered; steps, steps_today and hr are NA

Summing steps + unplaced_steps over a local_date gives the watch's daily total
at every bin size.

Missing values are written as NA (change with --na-rep). NA never means zero.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ============================== SETTINGS ==============================
# what data to include
# in the output (True or False; at least one must be True)
INCLUDE_STEPS = True
INCLUDE_HR = True # heart rate

# rows per output: '1s' (1 second), '1min' (1 minute) or '5min' (5 minutes)
BIN_SIZE = '1s'
# ======================================================================

LOCAL_TZ = 'America/New_York'
REQUIRED_COLUMNS = ('ts', 'rsp_id', 'data_type', 'val')
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'cleaned_data'
# BIN_SIZE -> seconds per row
BIN_SECONDS = {'1s': 1, '1min': 60, '5min': 300}


# stop early with a clear message if the settings block has an invalid value
def check_settings() -> None:
    if BIN_SIZE not in BIN_SECONDS:
        sys.exit(f'BIN_SIZE must be one of {list(BIN_SECONDS)}, not {BIN_SIZE!r}')
    if not (INCLUDE_STEPS or INCLUDE_HR):
        sys.exit('INCLUDE_STEPS and INCLUDE_HR are both False; nothing to output')


# read the raw export, check its format, and tidy it into time order
def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f'{path}: missing column(s) {missing}. Only the standard export format is supported (see notes.txt).')
    if df['rsp_id'].nunique() != 1:
        sys.exit(f'{path}: expected one participant, found rsp_id values {df["rsp_id"].unique().tolist()}')

    # the export contains a few exact duplicate rows and rows sent up to ~1 s out of order
    df = df.drop_duplicates(['ts', 'data_type', 'val'])
    df = df.sort_values('ts', kind='stable').reset_index(drop=True)
    # whole second each record falls in; this is the row it maps to on the output grid
    df['sec'] = np.floor(df['ts']).astype('int64')
    return df


# flag every grid second inside a silence (no record of any type) longer than max_silence
def no_data_mask(ts: np.ndarray, start: int, n: int, max_silence: float) -> np.ndarray:
    gaps = np.flatnonzero(np.diff(ts) > max_silence)
    first = np.floor(ts[gaps]).astype('int64') + 1 - start
    end = np.floor(ts[gaps + 1]).astype('int64') - start
    # +1 where each silence starts and -1 where it ends; a running total > 0 means "inside a silence"
    edges = np.zeros(n + 1, dtype='int64')
    np.add.at(edges, np.clip(first, 0, n), 1)
    np.add.at(edges, np.clip(end, 0, n), -1)
    return np.cumsum(edges[:-1]) > 0


# Eastern clock time with the timezone dropped (formats ~35x faster than tz-aware times)
def wall_clock(utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return utc.tz_convert(LOCAL_TZ).tz_localize(None)


# epoch seconds -> Eastern clock time
def seconds_to_local(secs: np.ndarray) -> pd.DatetimeIndex:
    return wall_clock(pd.to_datetime(secs, unit='s', utc=True))


# each step-counter reading, how many steps it added, and whether those steps can be placed in time
def step_observations(events: pd.DataFrame, start: int, no_data: np.ndarray) -> pd.DataFrame:
    # if several readings land in one second, keep the latest
    counter = events[events['data_type'] == 'steps'].groupby('sec')['val'].last()
    obs = pd.DataFrame({'sec': counter.index.to_numpy(), 'val': counter.to_numpy()})
    obs['idx'] = obs['sec'] - start
    obs['day'] = seconds_to_local(obs['sec'].to_numpy()).strftime('%Y-%m-%d')

    prev_val = obs['val'].shift()
    first = prev_val.isna()
    new_day = ~first & (obs['day'] != obs['day'].shift())
    # a drop within the same local day means the watch restarted its counter
    restart = ~first & ~new_day & (obs['val'] < prev_val)
    # steps added = rise in the counter, or the whole value if the counter started over from 0
    counts_from_zero = first | new_day | restart
    obs['delta'] = np.where(counts_from_zero, obs['val'], obs['val'] - prev_val).astype('int64')

    # true when a no_data stretch sits between this reading and the previous one
    nd_cum = np.cumsum(no_data)
    idx = obs['idx'].to_numpy()
    gap_before = np.zeros(len(obs), dtype=bool)
    gap_before[1:] = (nd_cum[idx[1:] - 1] - nd_cum[idx[:-1]]) > 0

    # steps get a timestamp only if the watch was connected the whole time since the last reading
    obs['placed'] = (~first & ~restart & ~gap_before) | (obs['delta'] == 0)
    # only these two lose data in a way worth reporting in the run summary
    obs['event'] = np.select([restart, new_day & gap_before], ['watch_restart', 'new_day_after_gap'], default='')
    return obs


# add the step columns to the per-second table and mark step-specific statuses; returns the readings
def add_step_columns(series: pd.DataFrame, events: pd.DataFrame, start: int, no_data: np.ndarray) -> pd.DataFrame:
    if not (events['data_type'] == 'steps').any():
        sys.exit('input has no step records (set INCLUDE_STEPS = False to output heart rate only)')
    n = len(series)
    obs = step_observations(events, start, no_data)

    steps_today = np.full(n, np.nan)
    steps_today[obs['idx']] = obs['val']
    # carry the counter forward, but never across a no_data stretch
    steps_today = pd.Series(steps_today).groupby(np.cumsum(no_data)).ffill().to_numpy(copy=True)
    steps_today[no_data] = np.nan

    # connected seconds start at 0 steps, unknown seconds stay NA; readings then fill in the steps
    steps_this_second = np.where(np.isnan(steps_today), np.nan, 0.0)
    unplaced_steps = np.zeros(n, dtype='int64')
    placed = obs[obs['placed']]
    unplaced = obs[~obs['placed']]
    steps_this_second[placed['idx']] = placed['delta']
    steps_this_second[unplaced['idx']] = np.nan
    unplaced_steps[unplaced['idx']] = unplaced['delta']

    status = series['status'].to_numpy(copy=True)
    status[np.isnan(steps_today) & ~no_data] = 'steps_unknown'
    status[unplaced['idx']] = 'catchup'

    series['status'] = status
    series['steps_today'] = pd.array(steps_today).astype('Int64')
    series['steps_this_second'] = pd.array(steps_this_second).astype('Int64')
    series['unplaced_steps'] = unplaced_steps
    return obs


# one heart-rate value per grid second, NA where there's no valid reading
def heart_rate(events: pd.DataFrame, start: int, n: int, no_data: np.ndarray, hold: int) -> np.ndarray:
    readings = events[events['data_type'] == 'hr'].groupby('sec')['val'].last()
    idx = readings.index.to_numpy() - start
    keep = (idx >= 0) & (idx < n)
    hr = np.full(n, np.nan)
    hr[idx[keep]] = readings.to_numpy()[keep]
    # fill the short gaps between beats with the last reading
    hr = pd.Series(hr).ffill(limit=hold).to_numpy(copy=True)
    # the watch reports hr=0 when it reconnects or can't read a pulse
    hr[(hr == 0) | no_data] = np.nan
    return hr


# build the one-row-per-second table (and return the step readings for the summary, if any)
def build_series(events: pd.DataFrame, max_silence: float, hr_hold: int) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    # grid runs from the first record to the last record of any type
    start = int(events['sec'].min())
    n = int(events['sec'].max()) - start + 1
    no_data = no_data_mask(events['ts'].to_numpy(), start, n, max_silence)

    # status starts as connected/not; the step columns refine it
    series = pd.DataFrame({
        'rsp_id': events['rsp_id'].iloc[0],
        'timestamp_utc': pd.to_datetime(np.arange(start, start + n), unit='s', utc=True),
        'status': np.where(no_data, 'no_data', 'ok').astype(object),
    })
    obs = add_step_columns(series, events, start, no_data) if INCLUDE_STEPS else None
    if INCLUDE_HR:
        series['hr'] = pd.array(heart_rate(events, start, n, no_data, hr_hold)).astype('Int64')
    return series, obs


# group the per-second table into clock-aligned bins of `seconds` (60 or 300)
def bin_series(series: pd.DataFrame, seconds: int) -> pd.DataFrame:
    bin_start = series['timestamp_utc'].dt.floor(f'{seconds}s')
    grouped = series.groupby(bin_start)
    # covered seconds: known steps if steps are included, otherwise connected seconds
    if INCLUDE_STEPS:
        covered = grouped['steps_this_second'].count()
    else:
        covered = (series['status'] != 'no_data').groupby(bin_start).sum()

    binned = pd.DataFrame({
        'rsp_id': grouped['rsp_id'].first(),
        'status': np.select([covered == seconds, covered == 0], ['ok', 'missing'], default='partial'),
    })
    if INCLUDE_STEPS:
        binned['known_seconds'] = covered
        # last() skips NA, so this is the last known total in the bin
        binned['steps_today'] = grouped['steps_today'].last()
        binned['steps'] = grouped['steps_this_second'].sum(min_count=1)
        binned['unplaced_steps'] = grouped['unplaced_steps'].sum()
    if INCLUDE_HR:
        binned['hr'] = grouped['hr'].mean().round(1)
    binned = binned.rename_axis('timestamp_utc').reset_index()
    return binned[['rsp_id', *binned.columns.drop('rsp_id')]]


# turn the UTC datetime column into text and add Eastern time and date columns after it
def with_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    utc = pd.DatetimeIndex(out['timestamp_utc'])
    local = wall_clock(utc)
    out['timestamp_utc'] = utc.tz_localize(None).strftime('%Y-%m-%d %H:%M:%S')
    out.insert(2, 'timestamp_local', local.strftime('%Y-%m-%d %H:%M:%S'))
    out.insert(3, 'local_date', local.strftime('%Y-%m-%d'))
    return out


# print coverage, data-losing step events, and a per-day table to the terminal
def print_summary(series: pd.DataFrame, obs: pd.DataFrame | None) -> None:
    counts = series['status'].value_counts()
    local = wall_clock(pd.DatetimeIndex(series['timestamp_utc']))
    print(f'participant {series["rsp_id"].iloc[0]}: '
          f'{local[0]:%Y-%m-%d %H:%M:%S} to {local[-1]:%Y-%m-%d %H:%M:%S} Eastern')
    print('hours by status: ' + ', '.join(f'{s}={n / 3600:.1f}' for s, n in counts.items() if s != 'catchup')
          + (f' | catchup rows: {counts.get("catchup", 0)}' if INCLUDE_STEPS else ''))

    daily_columns = {}
    if INCLUDE_STEPS:
        descriptions = {
            'new_day_after_gap': 'a new day started while disconnected; the previous day\'s final total was never received',
            'watch_restart': 'the counter dropped mid-day (watch restart?); steps before the drop are kept, timing after it unknown',
        }
        notable = obs[(obs['event'] != '') & (obs['delta'] > 0)]
        whens = seconds_to_local(notable['sec'].to_numpy()).strftime('%Y-%m-%d %H:%M:%S')
        for when, event, delta in zip(whens, notable['event'], notable['delta']):
            print(f'  {when}: {descriptions[event]} ({delta} unplaced steps)')
        daily_columns |= {
            'hours_with_steps': ('steps_this_second', 'count'),
            'placed_steps': ('steps_this_second', 'sum'),
            'unplaced_steps': ('unplaced_steps', 'sum'),
            'last_counter': ('steps_today', 'last'),
        }
    if INCLUDE_HR:
        daily_columns['hours_with_hr'] = ('hr', 'count')

    daily = series.groupby(local.strftime('%Y-%m-%d').rename('local_date')).agg(**daily_columns)
    # the hours_* columns are counts of seconds until converted here
    hour_columns = [c for c in daily.columns if c.startswith('hours_')]
    daily[hour_columns] /= 3600
    print(daily.round(1).to_string())


# parse the command line, clean one file, and write the CSV
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, help='raw Garmin export CSV for one participant')
    parser.add_argument('-o', '--output', type=Path,
                        help=f'output CSV (default: {DEFAULT_OUTPUT_DIR.name}/<rsp_id>_<BIN_SIZE>.csv)')
    # even when idle or off-wrist, a connected watch sends calories about once a minute
    parser.add_argument('--max-silence', type=float, default=150,
                        help='seconds without any record before a stretch counts as no_data (default: 150)')
    parser.add_argument('--hr-hold', type=int, default=5,
                        help='seconds to carry a heart-rate reading forward between beats (default: 5)')
    parser.add_argument('--na-rep', default='NA', help='text written for missing values (default: NA)')
    args = parser.parse_args()
    check_settings()

    events = load_events(args.input)
    series, obs = build_series(events, args.max_silence, args.hr_hold)
    print_summary(series, obs)

    # binning always starts from the per-second series so the gap rules are identical at every size
    seconds = BIN_SECONDS[BIN_SIZE]
    output = series if seconds == 1 else bin_series(series, seconds)

    out_path = args.output or DEFAULT_OUTPUT_DIR / f'{series["rsp_id"].iloc[0]}_{BIN_SIZE}.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with_time_columns(output).to_csv(out_path, index=False, na_rep=args.na_rep)
    print(f'wrote {len(output):,} rows to {out_path}')


if __name__ == '__main__':
    main()
