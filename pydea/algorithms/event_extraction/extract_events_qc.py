from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import scipy.signal as signal

from pydea.datamodels.datamodel import Event


def define_baseline_alg1(data_seg, threshold):
    if np.max(np.abs(data_seg - np.median(data_seg))) > threshold:
        return 1
    else:
        return 0


# look for large slope in the time sequence
def event_detection(data_trace, threshold=0.001, seg_length=3):
    event_flag = []
    for i in range(int(len(data_trace) / seg_length)):
        event_flag.append(define_baseline_alg1(data_trace[i * seg_length:(i + 1) * seg_length], threshold=threshold))
        # event_flag = define_baseline_alg2(data_trace, moving_ave, threshold=0.001)
    return event_flag


def extract_events(df: pd.DataFrame,
                   lane_sensor: List =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   threshold: float = 0.001,  # picometer
                   save_to_file: bool = True,
                   out_dir: Path = None,
                   bin_size: int = 25,
                   filename: Union[str, Path] = 'events.npz') -> List:
    """

    Args:
        df: sensor dataframe, index is datetime index, columns are sensors (25 sensors for example)
        threshold:
        save_to_file: Bool.
        out_dir:
        bin_size:
        filename:

    Returns:
        list of extracted events. Element in the list are of Event type.

    """

    # lane_sensor = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_data_points, num_sensors = df.shape
    event_vis = np.zeros([num_data_points, num_sensors])


    for j in range(num_sensors):
        #     trace_temp = de_trend(df.iloc[:, j]) # detrend or not for initial detection
        # trace_temp = df.iloc[:, j + 1]
        trace_temp = df.iloc[:, j]
        event_vis_temp = event_detection(trace_temp, threshold=threshold, seg_length=3)
        event_vis[:len(event_vis_temp), j] = np.asarray(event_vis_temp)
        print('Processing sensor #{}'.format(j))

    # only choose sensors within the lane; then aggregate sensors
    combined_event_vis = np.sum(event_vis[:, lane_sensor], axis=1)
    combined_event_vis = combined_event_vis[:len(event_vis_temp)]

    event_list = []
    ap_bin = 1  # TODO: what is this parameter?

    peaks, peak_properties = signal.find_peaks(combined_event_vis, width=1)

    for i in range(len(peaks)):
        event_temp = Event()
        event_temp.sensor = lane_sensor
        event_temp.start = int(peak_properties['left_bases'][i] * bin_size)
        event_temp.end = int(peak_properties['right_bases'][i] * bin_size)
        if len(event_list) > 0:
            if event_temp.start < event_list[-1].end - bin_size:
                print('overlength event #{}'.format(i))
                continue

        event_temp.data['leading'] = df.iloc[event_temp.start:event_temp.end + bin_size * ap_bin, :].values
        # event_temp.data['leading'] = event_temp.data['leading'].reset_index(drop=True)

        event_temp.data['trailing'] = df.iloc[event_temp.start:event_temp.end + bin_size * ap_bin, :].values
        # event_temp.data['trailing'] = event_temp.data['trailing'].reset_index(drop=True)
        event_temp.data['timestamp'] = df.index[event_temp.start:event_temp.end + bin_size * ap_bin].values
        # event_temp.data['timestamp'] = event_temp.data['timestamp'].reset_index(drop=True)
        event_temp.timestamp = df.index[event_temp.start].values.flatten()[0]
        # event_temp.timestamp = df.iloc[event_temp.start, 0]

        # event_temp.calculate_speed(event_vis, sn=25, offset=0)
        # if event_temp.speed == 0:
        #     print('speed error at {}'.format(i))
        #     continue
        #
        # if event_temp.speed < 0:
        #     print('vehicle at opposite lane')
        #     continue
        event_temp.index = combined_event_vis[peaks[i]]
        event_list.append(event_temp)

    if save_to_file:
        if out_dir is None:
            out_dir = Path('.')
        if filename is None:
            filename = 'events.npz'
        np.savez_compressed(filename,
                            events=event_list)
    return event_list


def extract_events_algo1(raw,
                         lane_sensor=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         threshold: float = 0.001,  # picometer
                         save_to_file: bool = True,
                         out_dir: Path = None,
                         bin_size: int = 25,
                         filename: Union[str, Path] = 'events.npz'
                         ):
    pass


if __name__ == "__main__":
    from pydea.datamodels.datamodel import Wav

    data_dir = Path(
        r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\HighFrequency333Hz_20201213\HighFreq_333Hz\wav')
    wav_file = data_dir / 'wav_20201212_195900_F03_UTC.npz'
    wav = Wav(wav_file)
