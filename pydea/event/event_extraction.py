import numpy as np
from scipy import signal
import pandas as pd
from datetime import datetime, timedelta
from typing import List

FIBER_DISTANCE = 2.5

def plot_events(event_list):
    pass

class EventExtractionJY:
    pass

class Event:
    def __init__(self):
        self.timestamp = 0
        self.start = 0
        self.end = 1
        self.speed = 0
        self.index = 1
        self.ch = 1
        self.location = 1
        self.data = {}
        self.sensor = []
        self.sensor_active = []

    SAMPLING_RATE = 200
    def calculate_speed(self, event_vis_list, sn=25, offset=0):
        SAMPLING_RATE = 200
        BIN_SIZE = 25

        speed_temp = []
        speed = 0
        self.active_sensor = []
#         self.speed_alg3([1,2])
        if (self.speed > 130) or (self.speed < 50):
            for i in self.sensor:
                if np.max(event_vis_list[int(self.start/BIN_SIZE)-1:int(self.end/BIN_SIZE)+1, i]) > 0:
                    self.active_sensor.append(i)
                speed = signal.correlate(np.asarray(self.data['leading'].iloc[:, i])-self.data['leading'].iloc[0,i],
                                         np.asarray(self.data['trailing'].iloc[:, i-offset])-self.data['trailing'].iloc[0,i-offset])
                speed_temp.append(speed.argmax())
            if len(speed_temp) < 1:
                print('Speed detection error_1!')
                self.speed = 0
            else:
                self.speed = -1*FIBER_DISTANCE/(np.median(speed_temp)-int(len(self.data['leading'])))*SAMPLING_RATE*3.6
                if np.abs(self.speed)>180:
                    print('Speed detection error_2!')
                    self.speed_alg2(self.sensor)

    def speed_alg2(self, active_sensor):
        leading = np.sum(self.data['leading'].iloc[:,active_sensor], axis=1)
        leading = np.abs(leading-leading[0])
        trailing = np.sum(self.data['trailing'].iloc[:,active_sensor], axis=1)
        trailing = np.abs(trailing-trailing[0])
        pk1 = signal.find_peaks(leading, height=0.005)

        pk2 = signal.find_peaks(trailing, height=0.005)
        if np.min([len(pk1[0]),len(pk2[0])]) == 0:
            print('Speed detection error_3!')
            self.speed = 0
        else:
            self.speed = FIBER_DISTANCE/(pk2[0][0]-pk1[0][0])*SAMPLING_RATE*3.6
        if np.abs(self.speed)>200:
                print('Speed detection error_4!')
                self.speed = 0

    def speed_alg3(self, active_sensor):
        leading = np.abs(self.data['leading'].iloc[:,active_sensor]-self.data['leading'].iloc[0,active_sensor])
        pk1 = signal.find_peaks(leading, height=0.002)
        trailing = np.abs(self.data['trailing'].iloc[:,active_sensor]-self.data['trailing'].iloc[0,active_sensor])
        pk2 = signal.find_peaks(trailing, height=0.002)
        if np.min([len(pk1[0]),len(pk2[0])]) == 0:
            print('Speed detection error_3!')
            self.speed = 0
        else:
            self.speed = FIBER_DISTANCE/(pk2[0][0]-pk1[0][0])*SAMPLING_RATE*3.6

    def calculate_location(self):
        pass

    def calculate_index(self):
        pass


class EventExtractionQC:

    @staticmethod
    def event_detection(data_trace, threshold=0.001, seg_length=25):

        def define_baseline_alg1(data_seg, threshold):
            if np.max(np.abs(data_seg - np.median(data_seg))) > threshold:
                return 1
            else:
                return 0

        def define_baseline_alg2(data_seg, moving_ave, threshold):
            return np.asarray(data_seg[:len(moving_ave)]) - np.asarray(moving_ave) > threshold

        #     print('Process start')
        #     start = time.process_time_ns()
        event_flag = []
        for i in range(int(len(data_trace) / seg_length)):
            event_flag.append(
                define_baseline_alg1(data_trace[i * seg_length:(i + 1) * seg_length], threshold=threshold))
            # event_flag = define_baseline_alg2(data_trace, moving_ave, threshold=0.001)
        #     print('Process end. Takes s: ')

        #     print((time.process_time_ns()-start)/1e9)
        return event_flag

    def extract_events(self, df, threshold):
        THRESHOLD = 0.001  # picometer
        BIN_SIZE = 25
        event_detection = EventExtractionQC.event_detection

        lane_sensor = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


        event_vis = np.zeros([11000, 25])
        for j in range(25):
            #     trace_temp = de_trend(df.iloc[:, j]) # detrend or not for initial detection
            trace_temp = df.iloc[:, j + 1]
            event_vis_temp = event_detection(trace_temp, threshold=threshold)
            event_vis[:len(event_vis_temp), j] = np.asarray(event_vis_temp)
            print('Processing sensor #{}'.format(j))

        combined_event_vis = np.sum(event_vis[:, lane_sensor], axis=1)
        combined_event_vis = combined_event_vis[:len(event_vis_temp)]

        event_list = []
        ap_bin = 1

        peaks = signal.find_peaks(combined_event_vis, width=1)

        for i in range(len(peaks[0])):
            event_temp = Event()
            event_temp.sensor = lane_sensor
            event_temp.start = int(peaks[1]['left_bases'][i] * BIN_SIZE)
            event_temp.end = int(peaks[1]['right_bases'][i] * BIN_SIZE)
            if len(event_list) > 0:
                if event_temp.start < event_list[-1].end - BIN_SIZE:
                    print('overlength event #{}'.format(i))
                    continue
            event_temp.data['leading'] = df.iloc[event_temp.start:event_temp.end + BIN_SIZE * ap_bin, 1:26]
            event_temp.data['leading'] = event_temp.data['leading'].reset_index(drop=True)
            event_temp.data['trailing'] = df2.iloc[event_temp.start:event_temp.end + BIN_SIZE * ap_bin, 1:26]
            event_temp.data['trailing'] = event_temp.data['trailing'].reset_index(drop=True)
            event_temp.data['timestamp'] = df.iloc[event_temp.start:event_temp.end + BIN_SIZE * ap_bin, 0]
            event_temp.data['timestamp'] = event_temp.data['timestamp'].reset_index(drop=True)
            event_temp.timestamp = df.iloc[event_temp.start, 0]
            event_temp.calculate_speed(event_vis, sn=25, offset=0)
            if event_temp.speed == 0:
                print('speed error at {}'.format(i))
                continue
            if event_temp.speed < 0:
                print('vehicle at opposite lane')
                continue
            event_temp.index = combined_event_vis[peaks[0][i]]
            event_list.append(event_temp)

        pass

def merge_peaks():
    pass

def conv_events(wls):
    pass

class EventExtractionHY:
    @staticmethod
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


    @staticmethod
    def two_stage_event_extraction(wls, large_threshold=20, small_threshold=2.5):
        extract_events = EventExtractionHY.extract_events
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


