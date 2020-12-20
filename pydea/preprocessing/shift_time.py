from datetime import datetime, timedelta, timezone
import pytz
import pandas as pd
import numpy as np
from pydea.data_validation import check_monotonic
from loguru import logger

M80_TIME_SHIFT_TO_UTC = timedelta(hours=3,minutes=53,seconds=9) #M80

def interpolate_time_index_from_line_number(time_index, line_number, time_shift=None, out_time_zone = timezone.utc):
    t_start = pd.to_datetime(time_index[0]).to_pydatetime()
    t_end = pd.to_datetime(time_index[-1]).to_pydatetime()
    print(f'start:{t_start}, end {t_end}')

    # check line numer monotonic
    delta_line_number = line_number[-1] - line_number[0]
    cond1 = (delta_line_number+1 >= len(line_number))
    cond2 = check_monotonic.check_monotonic_increase(line_number)
    if not (cond1 and cond2):
        logger.warning('FBGS line number not monotonic. Time index not changed.')
        return time_index
        # raise ValueError('FBGS line number not monotonic!')

    line_number0 = line_number - line_number[0]
    dt = (t_end - t_start).total_seconds() / delta_line_number
    # t_start_melbourne = t_start + time_shift
    t_start_shifted = t_start + time_shift
    t_start_utc = t_start_shifted.replace(tzinfo=out_time_zone)
    new_time_index = [t_start_utc  + timedelta(seconds=dt * line_number) for line_number in line_number0]
    return new_time_index

