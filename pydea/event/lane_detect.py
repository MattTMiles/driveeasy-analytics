import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import scipy.signal as signal
import datetime


'''
Takes in a df and returns events and lane positions
'''

def event_detection(data_trace, threshold=0.001, seg_length=BIN_SIZE):
    event_flag = []
    for i in range(int(len(data_trace)/seg_length)):
        event_flag.append(define_baseline_alg1(data_trace[i*seg_length:(i+1)*seg_length], threshold=threshold))
    return event_flag


def identify_lane(event, _lane_sensor_1):

#     start = np.max([wave_center-BIN_SIZE, 0])
#     end = np.min([wave_center+BIN_SIZE, len(wav)])
#     _axle_trace_1 = np.std(np.abs(event.wav1), axis=0)
    _axle_trace_1 = np.max(np.abs(event.wav1), axis=0)
    _axle_peaks_1 = signal.find_peaks(_axle_trace_1, prominence=0.01, distance=2)
    
    _axle_trace_2 = np.max(np.abs(event.wav2), axis=0)
    _axle_peaks_2 = signal.find_peaks(_axle_trace_2, prominence=0.01, distance=2)
    
    if len(_axle_peaks_1[0]) > len(_axle_peaks_2[0]):
        axle_peaks = _axle_peaks_1
    else:
        axle_peaks = _axle_peaks_2
    
    if len(axle_peaks[0]) == 0:
        print('No wheel found!')
        return 0
    _cog = np.average(axle_peaks[0])
# Francis street, Westbound
    if len(axle_peaks[0]) <= 2:
        if axle_peaks[0][-1] in _lane_sensor_1:
            return 1
        else:
            return 0
            # specific to Francis street lane_3
#             if int(_cog) < _lane_sensor_1[-1] + 0:
#                 print('cross lane')
#                 return 1
#             else:
#                 return 0
        
    else:
        if ((axle_peaks[0]>=_lane_sensor_1[0]) & (axle_peaks[0]<=_lane_sensor_1[-1])).sum()>=2:
            return 1
        else:
            return 0
        
        
def identify_lane2(event, _lane_sensor, threshold=0.0015, total_sensorn=25, left_wheel_index=0):
    # left_wheel_index: 0: left is closer to smaller index sensors; -1: left is closer to larger index sensors
    lane_trace = pd.DataFrame()
    for j in range(total_sensorn):
        lane_trace['sensor{}'.format(j+1)]=event_detection(event.wav1[:,j], threshold=threshold, seg_length=3)
    peaks_lane = signal.find_peaks(np.sum(lane_trace, axis=0), prominence=1, distance=3)
    _cog = np.average(peaks_lane[0])
    if len(peaks_lane[0]) == 0:
        print('No wheel found!')
        return 0
    if len(peaks_lane[0]) <= 2:
#         if peaks_lane[0][left_wheel_index] in _lane_sensor: 
        if int(_cog) in _lane_sensor:
            return 1
        else:
            return 0
    else:
        if ((peaks_lane[0]>=_lane_sensor[0]) & (peaks_lane[0]<=_lane_sensor[-1])).sum()>=2:
            return 1
        else:
            return 0
        
        
def identify_lane3(event, _lane_sensor, threshold=0.0015, total_sensorn=25, left_wheel_index=0):
    sos = signal.butter(2, 5, 'hp', fs=SAMPLING_RATE, output='sos')
    filtered_wav1 = signal.sosfilt(sos, event.wav1, axis=0)
    filtered_wav2 = signal.sosfilt(sos, event.wav2, axis=0)
    lane_combined = np.max(np.abs(filtered_wav1), axis=0)
    peaks_lane = signal.find_peaks(lane_combined, prominence=threshold, distance=4)
    _cog = np.average(peaks_lane[0])
#     print(peaks_lane[0])
    if len(peaks_lane[0]) == 0:
        print('No wheel found!')
        return 0
    if len(peaks_lane[0]) <= 2:
        if peaks_lane[0][left_wheel_index] in _lane_sensor: 
#         if int(_cog) in _lane_sensor:
            return 1
        else:
            return 0
    else:
        if ((peaks_lane[0]>=_lane_sensor[0]) & (peaks_lane[0]<=_lane_sensor[-1])).sum()>=2:
            return 1
        else:
            # account for the edge case
            if np.max(lane_combined[left_wheel_index:2]) > threshold*2:
                return 1
            else:
                return 0