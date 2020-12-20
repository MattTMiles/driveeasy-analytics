# plot VIPER data for comparison
import pandas as pd
import numpy as np
import re
import glob
from pathlib import Path
import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
import datetime
import itertools
import importlib

# class Event:
#     def __init__(self):
#         self.timestamp = 0
#         self.start = 0
#         self.end = 1
#         self.speed = 0
#         self.index = 1
#         self.ch = 1
#         self.location = 1
#         self.data = {}
#         self.sensor = []
#         self.sensor_active = []

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


def compare_geocount(df_geocount, event_list, axle_count=[]):
    geo_match_counter = 0
    geo_mismatch_list = []
    geo_mismatch_axle_counter = 0
    mis_matched_axle_ind = []
    list_temp = 0
    fibridge_matched = []
    for i in range(len(df_geocount)):
        geo_time = df_geocount['Timestamp'][i]
        print(geo_time)
        for j in list(range(list_temp, len(event_list))):
            if np.abs(event_list[j].timestamp - geo_time) < datetime.timedelta(seconds=1.8):
                geo_match_counter = geo_match_counter + 1
                list_temp = j + 1
                fibridge_matched.append(j)
                print('matched, {}'.format(event_list[j].timestamp))
                # if np.abs(int(df_viper['Axles'][i])-axle_count[j]) > 0:
                if len(axle_count) > 0:
                    if np.abs(int(df_geocount['naxles'][i]) - axle_count[j]) > 0:
                        geo_mismatch_axle_counter = geo_mismatch_axle_counter + 1
                        mis_matched_axle_ind.append(i)
                break
            if (event_list[j].timestamp - geo_time) >= datetime.timedelta(seconds=1.8):
                geo_mismatch_list.append(i)
                print('mis-matched, {}'.format(event_list[j].timestamp))
                break
    return geo_match_counter, geo_mismatch_list, geo_mismatch_axle_counter, mis_matched_axle_ind, fibridge_matched


def read_geocount_to_df(filename):
    df_geo = pd.read_csv(filename)
    df_geo['Timestamp'] = pd.to_datetime(df_geo['Date']+' '+df_geo['Time']) + datetime.timedelta(seconds=0)
    return df_geo


def load_datafile(filename):
    """Load data dictionary and convert to class

    Allows dot lookup instead of string lookup.

    """

    class Dict2Class(object):
        """
        turns a dictionary into a class
        """

        def __init__(self, dictionary):
            """Constructor"""
            for key in dictionary:
                setattr(self, key, dictionary[key])

    datatemp = np.load(filename, allow_pickle=True)
    data = Dict2Class(datatemp)
    return data
# class EventProperties:
#     def __init__(self):
#         self.timestamp = 0
#         self.start = 0
#         self.end = 1
#         self.speed = 0
#         self.index = 1
#         self.ch = 1
#         self.location = 1
#         self.sensor = []


if __name__ == '__main__':
    filename_geocount = r'C:\Users\qchen\PARC\Fibridge-PARC - Drive Easy\AustraliaDeploy\Francis\raw_data\Melbourne_time_20201201\Geocount\042-000432.csv'
    df_geo = read_geocount_to_df(filename_geocount)
    df_geo_lan = df_geo[df_geo['LaneTo'].isin(['1'])].reset_index(drop=True)

    filename_extracted_raw = r'C:\driveeasy2020\driveeasy-analytics\driveeasy-analytics\explore\1_qc\Francis_1201_0930_lane_1_trial_1219_filtered_triggered_events.npz'
    events = load_datafile(filename_extracted_raw)

    # Change the timestamp to UTC+11 local time
    for i in range(len(events.events)):
        events.events[i].timestamp = events.events[i].timestamp + datetime.timedelta(hours=11, seconds=1)

    event_time = [events.events[k].timestamp for k in range(len(events.events))]
    aa, bb, cc, dd, ee = compare_geocount(df_geo_lan, events.events, [])
    plt.figure(1)
    plt.clf()
    plt.stem(event_time, [1,]*len(event_time), use_line_collection=True)
    plt.stem(df_geo_lan['Timestamp'] + datetime.timedelta(seconds=0), [1, ] * len(df_geo_lan), markerfmt='C6o',
             linefmt='C6-', use_line_collection=True)
    plt.stem(df_geo_lan['Timestamp'][bb]+datetime.timedelta(seconds=0), [1, ]*len(bb), markerfmt='C7o', linefmt='C7-', use_line_collection=True)
    plt.xticks(rotation=20)
    plt.ylabel('Detected Event')

    # count total event in given time range, find false positive event index
    df_fibridge = pd.DataFrame(event_time, columns=['Timestamp'])

    print('Total Fibridge count {}'.format(len(df_fibridge[df_fibridge['Timestamp'] < df_geo['Timestamp'].iloc[-1]])))

    false_positive = pd.DataFrame(data=list(range(len(events.events))), columns=['index'])
    false_positive = false_positive[~false_positive['index'].isin(ee)]