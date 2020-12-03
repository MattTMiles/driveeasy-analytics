import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import scipy.signal as signal
import datetime
from pydea.ETL import tirtle, wav
from pathlib import Path

FIBER_DISTANCE = 2.5
SAMPLING_RATE = 200
THRESHOLD = 0.001  # picometer
BIN_SIZE = 50


class Event:
    def __init__(self):
        self.timestamp = 0
        self.start = 0
        self.end = 1
        self.speed = 0
        self.index = 1
        self.ch = 1
        self.location = 1
        self.event = {}
        self.sensor = []
        self.sensor_active = []

    def calculate_speed(self, event_vis_list, sn=25, offset=0):
        speed_temp = []
        speed = 0
        self.sensor_active = []
        #         self.speed_alg3([1,2])
        if (self.speed > 130) or (self.speed < 50):
            for i in self.sensor:
                if np.max(event_vis_list[int(self.start / BIN_SIZE) - 1:int(self.end / BIN_SIZE) + 1, i]) > 0:
                    self.sensor_active.append(i)
                speed = signal.correlate(np.asarray(self.data['leading'].iloc[:, i]) - self.data['leading'].iloc[0, i],
                                         np.asarray(self.data['trailing'].iloc[:, i - offset]) -
                                         self.data['trailing'].iloc[0, i - offset])
                speed_temp.append(speed.argmax())
            if len(speed_temp) < 1:
                print('Speed detection error_1!')
                self.speed = 0
            else:
                self.speed = -1 * FIBER_DISTANCE / (
                            np.median(speed_temp) - int(len(self.data['leading']))) * SAMPLING_RATE * 3.6
                if np.abs(self.speed) > 180:
                    print('Speed detection error_2!')
                    self.speed_alg2(self.sensor)

    def speed_alg2(self, active_sensor):
        leading = np.sum(self.data['leading'].iloc[:, active_sensor], axis=1)
        leading = np.abs(leading - leading[0])
        trailing = np.sum(self.data['trailing'].iloc[:, active_sensor], axis=1)
        trailing = np.abs(trailing - trailing[0])
        pk1 = signal.find_peaks(leading, height=0.005)

        pk2 = signal.find_peaks(trailing, height=0.005)
        if np.min([len(pk1[0]), len(pk2[0])]) == 0:
            print('Speed detection error_3!')
            self.speed = 0
        else:
            self.speed = FIBER_DISTANCE / (pk2[0][0] - pk1[0][0]) * SAMPLING_RATE * 3.6
        if np.abs(self.speed) > 200:
            print('Speed detection error_4!')
            self.speed = 0

    def speed_alg3(self, active_sensor):
        leading = np.abs(self.data['leading'].iloc[:, active_sensor] - self.data['leading'].iloc[0, active_sensor])
        pk1 = signal.find_peaks(leading, height=0.002)
        trailing = np.abs(self.data['trailing'].iloc[:, active_sensor] - self.data['trailing'].iloc[0, active_sensor])
        pk2 = signal.find_peaks(trailing, height=0.002)
        if np.min([len(pk1[0]), len(pk2[0])]) == 0:
            print('Speed detection error_3!')
            self.speed = 0
        else:
            self.speed = FIBER_DISTANCE / (pk2[0][0] - pk1[0][0]) * SAMPLING_RATE * 3.6

    def calculate_location(self):
        pass

    def calculate_index(self):
        pass


class Event2:
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


# work with event2 with wls. Use speed determine by axle first; if axle method fails, use corr of aggregate signal
def calculate_speed_qc_alg1(event2, lane_sensor):
    trace_temp1 = np.sum(np.abs(event2.wav1[:, lane_sensor]), axis=1)
    peaks_temp1 = signal.find_peaks(trace_temp1, prominence=0.008)

    trace_temp2 = np.sum(np.abs(event2.wav2[:, lane_sensor]), axis=1)
    peaks_temp2 = signal.find_peaks(trace_temp2, prominence=0.008)

    speed_valid = signal.correlate(trace_temp1, trace_temp2)
    speed_corr = -1 * 2.5 / (speed_valid.argmax() - len(event2.wav1)) * SAMPLING_RATE * 3.6

    if len(peaks_temp1[0]) == 0 or len(peaks_temp2[0]) != len(peaks_temp1[0]):
        # print('Aggregate peak method fails!')
        return speed_corr, 0
    else:
        #  print(len(peaks_temp1[0]))
        #         speed_agg_peak = -1*2.5 / (peaks_temp1[0][0] - peaks_temp2[0][0]) * SAMPLING_RATE * 3.6
        return speed_corr, len(peaks_temp1[0])


def define_baseline_alg2(data_seg, moving_ave, threshold):
    return np.asarray(data_seg[:len(moving_ave)]) - np.asarray(moving_ave) > threshold


def define_baseline_alg1(data_seg, threshold):
    if np.max(np.abs(data_seg - np.median(data_seg))) > threshold:
        return 1
    else:
        return 0


# define the minimum distance in time between two vehicles: BIN_SIZE
def event_detection(data_trace, threshold=0.001, seg_length=BIN_SIZE):
    #     print('Process start')
    #     start = time.process_time_ns()
    event_flag = []
    for i in range(int(len(data_trace) / seg_length)):
        event_flag.append(define_baseline_alg1(data_trace[i * seg_length:(i + 1) * seg_length], threshold=threshold))
        # event_flag = define_baseline_alg2(data_trace, moving_ave, threshold=0.001)
    #     print('Process end. Takes s: ')

    #     print((time.process_time_ns()-start)/1e9)
    return event_flag


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

data_dir = Path(r'C:\Users\qchen\PARC\Fibridge-PARC - Drive Easy\AustraliaDeploy\Francis\VIPER VIM validation\1120-1122\driveeasy_wav\melbourne_time_20201120_0900AM')

ch1_file = 'wav_20201119_215814_F01_UTC.npz'
ch2_file = 'wav_20201119_215814_F02_UTC.npz'
ch3_file = 'wav_20201119_215814_F03_UTC.npz'
ch4_file = 'wav_20201119_215814_F04_UTC.npz'

### npz
df_1 = wav.load_wav_into_dataframe(data_dir/ch1_file)
df_2 = wav.load_wav_into_dataframe(data_dir/ch2_file)

# remove outliers; There should be better way to do it.
y = df_1.iloc[:, 0]
removed_outliers_1 = y.between(y.quantile(.001), y.quantile(.999))
y = df_2.iloc[:, 0]
removed_outliers_2 = y.between(y.quantile(.001), y.quantile(.999))
index_names = df_1[~removed_outliers_1 or ~removed_outliers_2].index

df_1.drop(index=index_names, inplace=True)
df_2.drop(index=index_names, inplace=True)

df_1.plot()
df_2.plot()