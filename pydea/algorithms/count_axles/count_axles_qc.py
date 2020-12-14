from typing import List, Union

import numpy as np
import scipy.signal as signal

# from pydea.datamodels.datamodel import Event, EventFeatures
from ...datamodels.datamodel import Event, EventFeatures
from ..speed_estimation.calculate_speed import calculate_speed_alg1, calculate_speed_alg2, calculate_speed_ensemble

SAMPLING_RATE = 200
FIBER_DISTANCE = 2.5

def find_axle_location(wav1, wav2, promin_1=0.001, promin_2=0.001):
    peaks_1 = signal.find_peaks(wav1, prominence=promin_1)
    peaks_2 = signal.find_peaks(wav2, prominence=promin_2)
    axle_count = 0
    axle_list = []
    if len(peaks_1[0]) + len(peaks_2[0]) == 0:
        print('No peaks found!')
        return axle_count, axle_list

    if len(peaks_1[0]) == len(peaks_2[0]):
        axle_list = np.zeros(len(peaks_1))
        axle_count = len(peaks_1)
        shift = peaks_2[0][0] - peaks_1[0][0]
        axle_fiber1 = np.asarray(peaks_1[0]) - peaks_1[0][0]
        axle_fiber2 = np.asarray(peaks_2[0]) - shift - peaks_2[0][0]
        if len(peaks_1[0]) > 1:
            for i in range(1, axle_count):
                try:
                    axle_list[i] = (axle_fiber1[i] + axle_fiber2[i]) / 2
                except:
                    print('i={}, fiber1={}, fiber2={}'.format(i, len(axle_fiber1), len(axle_fiber2)))
                    axle_list = axle_fiber1
    # Peaks in two channels are different. Choose the channel with more peaks found
    else:
        axle_count = np.max([len(peaks_1[0]), len(peaks_2[0])])
        if axle_count == len(peaks_1[0]):
            axle_list = np.asarray(peaks_1[0]) - peaks_1[0][0]
        else:
            axle_list = np.asarray(peaks_2[0]) - peaks_2[0][0]
    return axle_count, axle_list


def calculate_axle_distance(axle_list: List,
                            speed: Union[float, int],
                            sampling_rate: Union[float, int]):
    axle_length_list = np.zeros(len(axle_list) - 1)
    for i in range(len(axle_length_list)):
        axle_length_list[i] = (axle_list[i + 1] - axle_list[i]) / sampling_rate * speed / 3.6
    return axle_length_list


def calculate_axle_number_distance(event: Event,
                                   lane_sensor: List,
                                   promin_1: float = 0.001,
                                   promin_2: float = 0.001,
                                   sampling_rate: Union[float, int] = 200):
    wav1 = event.wav1
    wav2 = event.wav2
    trace_temp2 = np.abs(np.min(wav2[:, lane_sensor], axis=1))
    trace_temp1 = np.abs(np.min(wav1[:, lane_sensor], axis=1))
    axle_count, axle_list = find_axle_location(trace_temp1, trace_temp2, promin_1=promin_1, promin_2=promin_2)

    speed_list = calculate_speed_ensemble(event=event, lane_sensor=lane_sensor)

    if np.min(speed_list) > 0:
        speed = np.min(speed_list)
    else:
        if np.max(speed_list) < 80:
            speed = np.max(speed_list)
        else:
            speed = np.median(speed_list)

    axle_distance_list = []
    num_groups = 0

    if axle_count > 1:
        axle_length_temp = calculate_axle_distance(axle_list, speed, sampling_rate)
        if axle_count > 4:
            check_value = 1
            group_check_value = 2
        else:
            check_value = 0.8
            group_check_value = 1.5
        check = axle_length_temp > check_value
        axle_length_temp = axle_length_temp[check]
        axle_count = len(axle_length_temp) + 1
        num_groups = (axle_length_temp > group_check_value).sum()
        # event_features.group = (axle_length_temp > group_check_value).sum()
        axle_distance_list = axle_length_temp
    # axle_distance_list = calculate_axle_distance(axle_list, speed, sampling_rate)
    return axle_count, axle_distance_list, num_groups

if __name__ == '__main__':
    # load extracted events
    from pathlib import Path

    data_dir = Path(r'/data/FrancisSt')
    filename = data_dir / r'extracted_Francis_1201_09_10_lane3_cleaned_event.npz'
    events = np.load(filename, allow_pickle=True)
    lane_sensor_1 = np.arange(2, 15)
    event_features_list = []
    for i in range(len(events['events'])):
        event_features = EventFeatures()
        axle_length_temp = []
        ax_count = 0
        ax_list = []
        # kk = np.min(events['events'][i].wav1[:, lane_sensor_1], axis=0).argmin()
        # trace_temp1 = np.max(np.abs(event_list[j].wav1[:,lane_sensor_1]), axis=1)
        # trace_temp1 = np.abs(events['events'][i].wav1[:, kk])
        # trace_temp2 = np.abs(events['events'][i].wav2[:, kk])
        trace_temp2 = np.abs(np.min(events['events'][i].wav2[:, lane_sensor_1], axis=1))
        trace_temp1 = np.abs(np.min(events['events'][i].wav1[:, lane_sensor_1], axis=1))
        ax_count, ax_list = find_axle_location(trace_temp1, trace_temp2, promin_1=0.0025, promin_2=0.0025)

        speed_temp1 = calculate_speed_alg1(events['events'][i], lane_sensor_1)
        speed_temp2, speed_temp3, axle_index = calculate_speed_alg2(events['events'][i], lane_sensor_1)
        speed_list = [speed_temp1, speed_temp2, speed_temp3]

        if np.min(speed_list) > 0:
            speed = np.min(speed_list)
        else:
            if np.max(speed_list) < 80:
                speed = np.max(speed_list)
            else:
                speed = np.median(speed_list)

        if ax_count > 1:
            axle_length_temp = calculate_axle_distance(ax_list, speed, SAMPLING_RATE)
            if ax_count > 4:
                check_value = 1
                group_check_value = 2
            else:
                check_value = 0.8
                group_check_value = 1.5
            check = axle_length_temp > check_value
            axle_length_temp = axle_length_temp[check]
            ax_count = len(axle_length_temp) + 1
            event_features.group = (axle_length_temp > group_check_value).sum()

        event_features.speed = speed
        event_features.axles = ax_count
        event_features.axle_lengths = axle_length_temp
        event_features_list.append(event_features)

    save_results = False
    if save_results:
        save_filename = filename[:-4] + 'features.npz'
        np.savez_compressed(save_filename, data=event_features_list)
