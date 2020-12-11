import pandas as pd
from scipy import signal

def detrend(arr):
    median1000 = signal.medfilt(arr, kernel_size=1001)
    arr = arr - median1000
    return arr

def detrend_df(df, window_size=1000):
    moving_median = df.rolling(window_size).median()
    # moving_median.fillna(moving_median.iloc[])
    moving_median.iloc[0:window_size,:].fillna(moving_median.iloc[0:window_size,:].median(),inplace=True)
    # = [df.iloc[0:1000,:].median()]*25
    detrended = df - moving_median
    return detrended
