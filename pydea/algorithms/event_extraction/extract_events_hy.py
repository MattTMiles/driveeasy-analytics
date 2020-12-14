from datetime import datetime, timedelta, timezone
from scipy import signal
import numpy as np
import pandas as pd
from typing import List, Union, Dict

def extract_events(wls, sample_frequence=200,
                   event_distance=timedelta(seconds=1),
                   event_length=timedelta(seconds=3),
                   height=0.5):
    if isinstance(wls, np.ndarray):
        arr = wls
    elif isinstance(wls, pd.Series):
        arr = abs(wls.values.flatten())
    elif isinstance(wls, List):
        arr = np.array(wls)
    else:
        print("input data format not supported.")
    arr = abs(arr)

    event_distance = int(event_distance.total_seconds() * sample_frequence)
    event_length = int(event_length.total_seconds() * sample_frequence)
    peaks, prominence = signal.find_peaks(arr, height=height, distance=event_distance)
    index_list = []
    half_window_size = int(event_length / 2)
    for pk in peaks:
        left = max(0, pk - half_window_size)
        right = min(len(arr), pk + half_window_size)
        index_list.append((left, right))
    return peaks, index_list

def extract_events_algo1():
    pass


def two_stage_event_extraction(wls, large_threshold=20, small_threshold=2.5):
    # extract_events = EventExtractionHY.extract_events
    if isinstance(wls, np.ndarray):
        arr = wls
    elif isinstance(wls, pd.Series):
        arr = abs(wls.values.flatten())
    elif isinstance(wls, List):
        arr = np.array(wls)
    else:
        print("input data format not supported.")
    min_agg = arr.copy()
    # first big events
    large_peaks, large_index_range = extract_events(min_agg, sample_frequence=200,
                                                    event_distance=timedelta(seconds=1),
                                                    event_length=timedelta(seconds=3), height=large_threshold)
    # set big events data to 0 (remove)
    for start, end in large_index_range:
        # min_agg.iloc[start:end] = 0
        min_agg[start:end] = 0

    # detect small vehicles
    small_peaks, small_index_range = extract_events(min_agg, sample_frequence=200,
                                                    event_distance=timedelta(seconds=0.5),
                                                    event_length=timedelta(seconds=0.8), height=small_threshold)

    return large_peaks, large_index_range, small_peaks, small_index_range

