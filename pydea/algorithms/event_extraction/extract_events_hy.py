from datetime import datetime, timedelta, timezone
from scipy import signal
import numpy as np
import pandas as pd
from typing import List, Union, Dict
from pydea.datamodels.datamodel import  Event

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

def create_event_data(wav1, wav2, index_range_list:List, event_info='large'):
    events = []
    event_id = 0
    for start, end in index_range_list:
        event = Event()
        event.timestamp = wav1.timestamp[start:end]
        event.wav1 = wav1.wav[start:end]
        event.wav2 = wav2.wav[start:end]
        event.fiber1_id = wav1.fiber_id
        event.fiber2_id = wav2.fiber_id
        event.info = event_info
        event.event_id = event_id
        event_id += 1


        if (wav1.wls is not None) and (wav2.wls is not None):
            event.wls1 = wav1.wls
            event.wls2 = wav2.wls
        events.append(event)
    return events


