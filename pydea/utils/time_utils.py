import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pydea.utils.check import  _check_dt

def interpolate_time_index(time_index, line_number, time_shift= timedelta(seconds=0)):
    t_start = pd.to_datetime(time_index[0]).to_pydatetime()
    t_end = pd.to_datetime(time_index[-1]).to_pydatetime()
    print(f'start:{t_start}, end {t_end}')
    delta_line_number = line_number[-1] - line_number[0]
    line_number_0ind= line_number - line_number[0]
    dt = (t_end - t_start).total_seconds() / delta_line_number
    time_shifted = t_start + time_shift
    new_time_index = [time_shifted + timedelta(seconds=dt * line_number) for line_number in line_number_0ind]
    return new_time_index

def convert_linenumber_to_time_index(line_number, time_index):
    t_start = pd.to_datetime(time_index[0]).to_pydatetime()
    t_end = pd.to_datetime(time_index[-1]).to_pydatetime()
    delta_line_number = line_number[-1] - line_number[0]
    dt = (t_end - t_start).total_seconds() / delta_line_number
    # TODO
    new_timeindex = time_index
    return new_timeindex

def convert_to_utc(timestamp, in_timezone, out_timezone):
    return

def _stamp_to_dt(utc_stamp):
    """Convert timestamp to datetime object in Windows-friendly way."""
    # The min on windows is 86400
    stamp = [int(s) for s in utc_stamp]
    if len(stamp) == 1:  # In case there is no microseconds information
        stamp.append(0)
    return (datetime.fromtimestamp(0, tz=timezone.utc) +
            timedelta(0, stamp[0], stamp[1]))  # day, sec, µs


def _dt_to_stamp(inp_date):
    """Convert a datetime object to a timestamp."""
    _check_dt(inp_date)
    return int(inp_date.timestamp() // 1), inp_date.microsecond

