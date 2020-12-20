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


class EventProperties:
    def __init__(self):
        self.timestamp = 0
        self.start = 0
        self.end = 1
        self.speed = 0
        self.index = 1
        self.ch = 1
        self.location = 1
        self.sensor = []


# read the VIPER data in given time frame
def read_viper_to_df(file, time_range=['2020-11-09 13:45:29', '2020-11-09 15:45:29']):
    filename = file
    time_start = datetime.datetime.strptime(time_range[0], "%Y-%m-%d %H:%M:%S")
    time_end = datetime.datetime.strptime(time_range[1], "%Y-%m-%d %H:%M:%S")
    with open(filename, 'r') as f:
        count = 0

        while True:
            b = f.readline()
            b = re.split(r'\t+', b)

            if count == 0:
                columns = b[:35]
                count = count + 1
                # b = f.readline()
                # b = re.split(r'\t+', b)
                # bseries = b[:35]
                df = pd.DataFrame(columns=columns)
                continue
            try:
                current_time = datetime.datetime.strptime(b[0], "%Y/%m/%d %H:%M:%S:%f")
            except:
                print('failed to decode timestamp')
                break

            if current_time < time_start:
                continue
            dftemp = pd.DataFrame([b[:35]], columns=columns)
            df = df.append(dftemp, ignore_index=False)
            # print('length of df:' + str(len(df)))
            count = count + 1

            if current_time > time_end:
                break
            # ui = input('Continue? Y/N: (Y)')
            # if ui == 'N':
            #     print(count)
            #     break
            # else:
            #     continue
    return df


def compare_viper(df_viper, event_list):

    viper_matched_counter = 0
    viper_mismatched_counter = 0
    speed_error_counter = 0
    axle_error_counter = 0
    mis_matched_ind = []
    mis_matched_sp_ind = []
    mis_matched_axle_ind = []

    list_temp = 0
    for i in range(len(df_viper)):
        viper_time = datetime.datetime.strptime(df_viper.iloc[i, 0], "%Y/%m/%d %H:%M:%S:%f")
        viper_time = viper_time + datetime.timedelta(hours=1)
        for j in list(range(list_temp, len(event_list))):
            if (event_list[j].timestamp - viper_time) < datetime.timedelta(seconds=1):
                if (event_list[j].timestamp - viper_time) > datetime.timedelta(seconds=-1):
                    viper_matched_counter = viper_matched_counter + 1
                    list_temp = j + 1
                    if np.abs(int(df_viper['Speed KPH'][i])-event_list[j].speed) > 5:
                        speed_error_counter = speed_error_counter + 1
                        mis_matched_sp_ind.append(i)
                    if np.abs(int(df_viper['Axles'][i])-event_list[j].index) > 0:
                        axle_error_counter = axle_error_counter + 1
                        mis_matched_axle_ind.append(i)
                    break
            if (event_list[j].timestamp - viper_time) >= datetime.timedelta(seconds=1):
                viper_mismatched_counter = viper_mismatched_counter + 1
                mis_matched_ind.append(i)
                break
    return viper_matched_counter, viper_mismatched_counter, mis_matched_ind, speed_error_counter, mis_matched_sp_ind, axle_error_counter, mis_matched_axle_ind


filename = r'C:\Users\qchen\PARC\Fibridge-PARC - Drive Easy\AustraliaDeploy\Francis\VIPER VIM validation\43_20201201.txt'
df_vim = read_viper_to_df(filename, time_range=['2020-12-01 08:33:26', '2020-12-01 9:35:00'])

#Specify the lane to compare
df_lan2 = df_vim[df_vim['Lane'].isin(['2'])].reset_index(drop=True)

# event_dir = Path(
#     r'C:\driveeasy2020\driveeasy-analytics\driveeasy-analytics\explore\1_qc\data')
# event_files = list(event_dir.glob('*event1.npz'))

event_files = [r'C:\driveeasy2020\driveeasy-analytics\driveeasy-analytics\explore\1_qc\extracted_Francis_1201_09_10_lane3_cleaned_eventprop.npz', ]

# work with multiple files
event_list = []
for file in event_files:
    data = np.load(file, allow_pickle=True)
    event_list.append(data['events'].tolist())

event_list = list(itertools.chain.from_iterable(x for x in event_list))

# Change the timestamp to UTC+11 local time
for i in range(len(event_list)):
    event_list[i].timestamp = event_list[i].timestamp + datetime.timedelta(hours=11, seconds=6)

ref_time_1 = [datetime.datetime.strptime(df_lan2.iloc[i, 0], "%Y/%m/%d %H:%M:%S:%f") for i in range(len(df_lan2))]
# Change the VIPER time to local time
ref_time_1 = np.asarray(ref_time_1) + datetime.timedelta(hours=1)
ref_speed_1 = np.asarray(df_lan2['Speed KPH']).astype('int')
ref_index = df_lan2['Axles'].astype(int)

# Count the number of speed errors in extracted events (negative is considered error here)
speed_err = 0
for i in range(len(event_list)):
    if (event_list[i].speed < 0) or (event_list[i].speed > 200):
        speed_err = speed_err + 1
        event_list[i].speed = 0

event_speed = [event_list[k].speed for k in range(len(event_list))]
event_time = [event_list[k].timestamp for k in range(len(event_list))]
event_axle_count = [x.index for x in event_list]

df_time = pd.DataFrame(event_time, columns=['timestamp'])
df_time['timestamp'] = df_time['timestamp'].astype("datetime64")+datetime.timedelta(seconds=0)
time_temp = df_time['timestamp'].to_list()

plt.figure(9)
plt.clf()
plt.stem(time_temp, event_speed, use_line_collection=True)
plt.stem(ref_time_1[:], ref_speed_1[:], markerfmt='C1o', linefmt='C1-', use_line_collection=True)

plt.xticks(rotation=20)
plt.ylabel('Speed (KPH)')

a, b, c, d, e, f, g = compare_viper(df_lan2, event_list)

plt.figure(1)
plt.clf()
plt.stem(time_temp, event_speed, use_line_collection=True)
plt.stem(ref_time_1[1:], ref_speed_1[1:], markerfmt='C1o', linefmt='C1-', use_line_collection=True)
plt.stem(np.asarray(ref_time_1)[c][1:], ref_speed_1[c][1:], markerfmt='C4o', linefmt='C4-', use_line_collection=True)
plt.stem(np.asarray(ref_time_1)[e], ref_speed_1[e], markerfmt='C3o', linefmt='C3-', use_line_collection=True)

plt.xticks(rotation=20)
plt.ylabel('Speed (KPH)')
plt.title('Speed validation')

plt.figure(2)
plt.clf()
plt.stem(time_temp, event_axle_count, use_line_collection=True)
plt.stem(ref_time_1[:], ref_index[:], markerfmt='C1o', linefmt='C1-', use_line_collection=True)
plt.stem(np.asarray(ref_time_1)[c], ref_index[c], markerfmt='C4o', linefmt='C4-', use_line_collection=True)
plt.stem(np.asarray(ref_time_1)[g], ref_index[g], markerfmt='C3o', linefmt='C3-', use_line_collection=True)

plt.xticks(rotation=20)
plt.ylabel('Speed (KPH)')
plt.title('Axle count validation')
#
#
# ## statistics of classification
# axle_count = [x.index for x in event_list]
# a_count = 0
# b_count = 0
# c_count = 0
# for j in range(len(axle_count)):
#     if axle_count[j] <= 2:
#         a_count = a_count + 1
#     else:
#         if axle_count[j] < 4:
#             b_count = b_count + 1
#         else:
#             c_count = c_count + 1
#

# ref_index = df_lan2['Axles'].astype(int)
# plt.figure(4)
# plt.clf()
# plt.subplot(1,2,1)
# plt.hist(event_axle_count, bins=np.linspace(0,12,13))
# plt.xlabel('Axle count')
# plt.ylabel('Counts')
# plt.subplot(1,2,2)
# plt.hist(ref_index, bins=np.linspace(0,12,13), color='C1')
# plt.xlabel('Axle count')
#
#
# plt.figure(13)
# plt.clf()
# plt.plot()
# plt.stem(ref_time_1[:], ref_index, markerfmt='C1o', linefmt='C1-', use_line_collection=True)
# plt.stem(time_temp, axle_count, use_line_collection=True)

# lcw = importlib.import_module('load_clean_wav')
#
# filename_1 = r'C:\Users\qchen\PARC\Fibridge-PARC - Drive Easy\AustraliaDeploy\Francis\raw_data\Melbourne_time_20201120_8-10AM\wav\wav_20201119_205844_F01_UTC.npz'
#
# df_1 = lcw.read_npz_file(filename_1)
#
# df_1 = lcw.clean_wav(df_1)

# outliers_1 = lcw.find_outliers(df_1, percent_low=0.001, percent_high=0.9999)
#
# df_1, df_2 = lcw.remove_outliers_from_paired_fibers(outliers_1, outliers_1, df_1, df_1)
#
#
# def define_baseline_alg1(data_seg, threshold):
#     if np.max(np.abs(data_seg - np.median(data_seg))) > threshold:
#         return 1
#     else:
#         return 0
#
#
# def moving_average(data_set, periods=3):
#     weights = np.ones(periods) / periods
#     return np.convolve(data_set, weights, mode='valid')
#
#
# def de_trend(data, n_ave=200):
#     trend = moving_average(data, periods=n_ave)
#     return data[:-(n_ave - 1)] - trend
#
# import scipy.signal as signal
#
# trace_t = np.asarray(df_1.iloc[:, 1:26])
# BIN_SIZE = 50
#
# lane_sensor_1 = list(range(0, 11))
# lane_sensor_2 = list(range(11, 21))
# for i in range(25):
#     trace_t[:len(trace_t)-BIN_SIZE+1,i]=de_trend(trace_t[:,i], n_ave=BIN_SIZE)
#
# trace_t_1 = np.sum(np.abs(trace_t[:len(trace_t)-BIN_SIZE+1, lane_sensor_1]), axis=1)
#
# plt.figure(5)
# plt.clf()
# plt.stem(time_temp, event_axle_count, use_line_collection=True)
# plt.stem(ref_time_1[:], ref_index[:], markerfmt='C1o', linefmt='C1-', use_line_collection=True)
# plt.stem(np.asarray(ref_time_1)[c], ref_index[c], markerfmt='C4o', linefmt='C4-', use_line_collection=True)
# plt.stem(np.asarray(ref_time_1)[g], ref_index[g], markerfmt='C3o', linefmt='C3-', use_line_collection=True)
# plt.plot((df_1['timestamp'].astype('datetime64')+datetime.timedelta(hours=11))[10:len(trace_t_1)], (trace_t_1*-100)[10:])
# plt.xticks(rotation=20)
# plt.ylabel('Axle count')
# plt.title('Axle count validation')

# for j in range(len(peaks[0])):
#     plt.vlines(x=peaks[1]['left_bases'][j]*BIN_SIZE, ymin=0, ymax=1,colors='green', lw=2)
#     plt.vlines(x=peaks[1]['right_bases'][j]*BIN_SIZE, ymin=0, ymax=1,colors='red',lw=2)
# print('Total peak detected: {}'.format(len(peaks[0])))
#
# def cal_occupancy(event_vis, bin_size=BIN_SIZE, agg_size=20):
#     oc = np.zeros(int(len(event_vis)/agg_size))
#     for i in range(len(oc)):
#         oc[i] = np.sum(event_vis[i*agg_size:(i+1)*agg_size])/agg_size
#     return oc
#
# AGG_SIZE = 80
# oc = cal_occupancy(event_vis_1, agg_size=AGG_SIZE)
# timeseg2 = np.linspace(0, len(oc)*AGG_SIZE*BIN_SIZE, len(oc))
# timeseg3 = np.linspace(0, len(event_vis_1)*BIN_SIZE, len(event_vis_1))
# start_time = df_1.iloc[0,0]
# oc_time = [start_time + datetime.timedelta(hours=11, seconds=timeseg2[i]/200) for i in range(len(timeseg2))]
# vis_time = [start_time + datetime.timedelta(hours=11, seconds=timeseg3[i]/200) for i in range(len(timeseg3))]
#
# plt.figure(13)
# plt.clf()
# plt.plot(oc_time, oc)
# # plt.plot(vis_time, -0.5*np.asarray(event_vis_1))

# def volume_compare(df_tirt, event_list, timestamp_start, timestamp_end):
#     delta_t = (timestamp_end - timestamp_start).seconds
#     tirt_volume = np.zeros(int(delta_t/20))
#     j_tirt = 0
#     fibridge_volume = np.zeros(int(delta_t/20))
#     j_fibridge = 0
#     for i in range(len(tirt_volume)):
#         for j in range(j_tirt, len(df_tirt)):
#             if pd.Timestamp(df_tirt.iloc[i, 2], unit='s') + datetime.timedelta(hours=11, seconds=0) > timestamp_start + datetime.timedelta(seconds=) and
#

