import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

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
    new_timeindex = timeindex
    return new_timeindex

def convert_to_utc(timestamp, in_timezone, out_timezone):
    return
