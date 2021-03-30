import numpy as np
import scipy.signal as signal
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import time


class Event:
    def __init__(self):
        self.timestamp = datetime.datetime.now()
        self.event_id = 0
        self.fiber1_id = 0
        self.fiber2_id = 0
        self.fiber1_sensors = []
        self.fiber2_sensors = []
        self.info = ''
        self.wav1 = []
        self.wav2 = []


class EventFeatures:
    def __init__(self):
        self.timestamp = datetime.datetime.now()
        self.event_id = 0
        self.fiber1_id = 0
        self.fiber2_id = 0
        self.info = ''
        self.speed = 0
        self.axles = 0
        self.axle_lengths = []
        self.group = 0
        self.class_id = 0


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
                    axle_list[i] = (axle_fiber1[i]+axle_fiber2[i])/2
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


def calculate_axle_length(axle_list, speed, sampling_rate):
    axle_length_list = np.zeros(len(axle_list)-1)
    for i in range(len(axle_length_list)):
        axle_length_list[i] = (axle_list[i+1] - axle_list[i])/sampling_rate*speed/3.6
    return axle_length_list


def calculate_speed_qc_alg1(event2, lane_sensor):
    trace_temp1 = np.min(event2.wav1[:, lane_sensor], axis=1)
    trace_temp2 = np.min(event2.wav2[:, lane_sensor], axis=1)

    speed_valid = signal.correlate(trace_temp1, trace_temp2)
    try:
        speed_corr = -1 * 2.5 / (speed_valid.argmax() - len(event2.wav1)) * SAMPLING_RATE * 3.6
    except:
        speed_corr = 0

    return speed_corr


# work with event2 with wls. Use max of absolute value to aggregate data from difference sensors;
def calculate_speed_qc_alg2(event2, lane_sensor):
    kk = np.min(event2.wav1[:, lane_sensor_1], axis=0).argmin()
    # trace_temp1 = np.max(np.abs(event_list[j].wav1[:,lane_sensor_1]), axis=1)
    trace_temp1 = np.abs(event2.wav1[:, kk])
    trace_temp2 = np.abs(event2.wav2[:, kk])

    #     trace_temp2 = np.max(np.abs(event2.wav2[:, lane_sensor]), axis=1)
    #     trace_temp1 = np.max(np.abs(event2.wav1[:, lane_sensor]), axis=1)
    speed_valid = signal.correlate(trace_temp1, trace_temp2)
    try:
        speed_corr = -1 * 2.5 / (speed_valid.argmax() - len(event2.wav1)) * SAMPLING_RATE * 3.6
    except:
        speed_corr = 0
    peaks_temp2 = signal.find_peaks(trace_temp2, prominence=0.002)
    peaks_temp1 = signal.find_peaks(trace_temp1, prominence=0.002)
    if len(peaks_temp1[0]) == 0 or len(peaks_temp2[0]) == 0:
        return speed_corr, 0, 0
    else:
        speed_agg_peak = -1 * 2.5 / (peaks_temp1[0][0] - peaks_temp2[0][0]) * SAMPLING_RATE * 3.6
        return speed_corr, speed_agg_peak, np.max([len(peaks_temp1[0]), len(peaks_temp1[0])])


def event_detection(data_trace, threshold=0.001, seg_length=3):
    event_flag = []
    for i in range(int(len(data_trace) / seg_length)):
        event_flag.append(define_baseline_alg1(data_trace[i * seg_length:(i + 1) * seg_length], threshold=threshold))
        # event_flag = define_baseline_alg2(data_trace, moving_ave, threshold=0.001)
    return event_flag


def define_baseline_alg1(data_seg, threshold):
    if np.max(np.abs(data_seg - np.median(data_seg))) > threshold:
        return 1
    else:
        return 0





def normalize_tomax(wav):
    wav = (wav-np.min(wav))/(np.max(wav)-np.min(wav))
    return wav

def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


def de_trend(data, n_ave=200):
    trend = moving_average(data, periods=n_ave)
    return data[:-(n_ave - 1)] - trend


def subtract_firstrow(df):
    first_row = df.iloc[[0]].values[0]
    df_temp = df.apply(lambda row: row - first_row, axis=1)
    return df_temp


def read_npz_file(filename):
    data = np.load(filename, allow_pickle=True)
    df = pd.DataFrame(data=data['wav'], index=data['timestamp'])

    df.columns = [f'sensor{i+1}' for i in range(25)]
    try:
        df['linenum'] = data['linenum']
    except:
        print('no linenum found')
    return df


def clean_wav(fiber):
    df_fiber = fiber
    check = df_fiber.index.to_series().diff().dt.total_seconds()
    df = pd.DataFrame(index=(df_fiber[check.values>0.006].index-datetime.timedelta(seconds=0.005)), columns=df_fiber[check.values>0.006].columns)
    df_fiber = df_fiber.append(df, ignore_index=False)
    df_fiber = df_fiber.sort_index()
    df_fiber.reset_index(inplace=True)
    #df_fiber.columns = ['timestamp'] + [f'sensor{i+1}' for i in range(25)] + ['linenum']
    df_fiber.columns = ['timestamp'] + [f'sensor{i+1}' for i in range(25)]
    return df_fiber


def find_outliers(df, total_sensorn=25, percent_low=0.001, percent_high=0.99999):

    y = df['sensor24']
    removed_outliers = y.between(y.quantile(0.5)-0.2, y.quantile(0.5)+0.2)

    for i in range(total_sensorn):
        y = df[f'sensor{i+1}']
        if np.max([y.max() - y.mean(), y.mean() - y.min()])> 0.5:
            removed_outliers = y.between(y.quantile(0.5)-0.5, y.quantile(0.5)+0.5) & removed_outliers
    index_names = df[~removed_outliers].index
    return index_names

def remove_outliers_from_paired_fibers(outlier_ch1, outlier_ch2, df_1, df_2):
    total_outliers = list(dict.fromkeys(outlier_ch1.to_list() + outlier_ch2.to_list()))
    df_1.loc[total_outliers, [f'sensor{i + 1}' for i in range(25)]] = np.nan
    df_2.loc[total_outliers, [f'sensor{i + 1}' for i in range(25)]] = np.nan

    # fix missing data in the middle using interpolate
    df_1.interpolate(inplace=True)
    df_2.interpolate(inplace=True)

    # fix missing data in the head
    df_1 = df_1.fillna(method='bfill')
    df_2 = df_2.fillna(method='bfill')
    return df_1, df_2


#Lane identification functions
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

