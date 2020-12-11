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

# read the VIPER data in given time frame
def addtodf(file, timerange=['2020-11-09 13:45:29', '2020-11-09 15:45:29']):
    filename = file
    time_start = datetime.datetime.strptime(timerange[0], "%Y-%m-%d %H:%M:%S")
    time_end = datetime.datetime.strptime(timerange[1], "%Y-%m-%d %H:%M:%S")
    with open(filename, 'r') as f:
        count = 0

        while True:
            b = f.readline()
            b = re.split(r'\t+', b)
            # print(b)

            if count == 0:
                colums = b[:20]
                count = count + 1
                b = f.readline()
                b = re.split(r'\t+', b)
                bseries = b[:20]
                # print(type(bseries))
                df = pd.DataFrame([bseries], columns=colums)
                continue
            current_time = datetime.datetime.strptime(b[0], "%Y/%m/%d %H:%M:%S:%f")
            if current_time < time_start:
                continue
            dftemp = pd.DataFrame([b[:20]], columns=colums)
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
    mis_matched_ind = []
    class_error_counter = 0
    event_id = 0
    list_temp = 0
    for i in range(len(df_viper)):
        viper_time = datetime.datetime.strptime(df_viper.iloc[i, 0], "%Y/%m/%d %H:%M:%S:%f")
        viper_time = viper_time + datetime.timedelta(hours=1)
        for j in list(range(list_temp, len(event_list))):
            if (event_list[j].timestamp - viper_time) < datetime.timedelta(seconds=1):
                if (event_list[j].timestamp - viper_time) > datetime.timedelta(seconds=-1):
                    viper_matched_counter = viper_matched_counter + 1
                    list_temp = j + 1
                    if np.abs(int(df_viper['Speed KPH'][i])-event_list[j].speed > 5):
                        speed_error_counter = speed_error_counter + 1
                    break
            if (event_list[j].timestamp - viper_time) >= datetime.timedelta(seconds=1):
                viper_mismatched_counter = viper_mismatched_counter + 1
                mis_matched_ind.append(i)
                break
    return viper_matched_counter, viper_mismatched_counter, speed_error_counter, mis_matched_ind


filename = r'C:\Users\qchen\PARC\Fibridge-PARC - Drive Easy\AustraliaDeploy\Francis\VIPER VIM validation\1120-1122\43_20201122[1].txt'

df_vim = addtodf(filename, timerange=['2020-11-20 06:58:42', '2020-11-20 9:00:00'])

df_lan2 = df_vim[df_vim['Lane'].isin(['2'])].reset_index(drop=True)
# df_lan3 = df_vim[df_vim['Lane'].isin(['4'])].reset_index(drop=True)

event_dir = Path(
    r'C:\driveeasy2020\driveeasy-analytics\driveeasy-analytics\explore\1_qc\data')
event_files = list(event_dir.glob('*.npz'))

event_list = []
for file in event_files:
    data = np.load(file, allow_pickle=True)
    event_list.append(data['events'].tolist())

event_list = list(itertools.chain.from_iterable(x for x in event_list))

for i in range(len(event_list)):
    event_list[i].timestamp = event_list[i].timestamp + datetime.timedelta(hours=11)

ref_time_1 = [datetime.datetime.strptime(df_lan2.iloc[i, 0], "%Y/%m/%d %H:%M:%S:%f") for i in range(len(df_lan2))]
ref_time_1 = np.asarray(ref_time_1) + datetime.timedelta(hours=1)
ref_speed_1 = np.asarray(df_lan2.iloc[:, 5]).astype('int')

# Count the number of errors
speed_err = 0
for i in range(len(event_list)):
    if event_list[i].speed < 0 or event_list[i].speed > 100:
        speed_err = speed_err + 1
        event_list[i].speed = 0

event_speed = [event_list[k].speed for k in range(len(event_list))]
event_time = [event_list[k].timestamp for k in range(len(event_list))]

df_time = pd.DataFrame(event_time, columns=['timestamp'])
df_time['timestamp'] = df_time['timestamp'].astype("datetime64")
time_temp = df_time['timestamp'].to_list()

plt.figure(9)
plt.clf()
plt.stem(time_temp, event_speed, use_line_collection=True)
plt.stem(ref_time_1[:], ref_speed_1[:], markerfmt='C1o', linefmt='C1-', use_line_collection=True)

plt.xticks(rotation=20)
plt.ylabel('Speed (KPH)')

plt.figure()
plt.hist(event_speed, bins=np.linspace(0, 100, 50))

plt.figure()
plt.hist(ref_speed_1, bins=np.linspace(0, 100, 50))

a,b,c,d = compare_viper(df_lan2, event_list)

plt.figure(10)
plt.clf()
plt.stem(time_temp, event_speed, use_line_collection=True)
plt.stem(ref_time_1[d], ref_speed_1[d], markerfmt='C4o', linefmt='C4-', use_line_collection=True)

plt.xticks(rotation=20)
plt.ylabel('Speed (KPH)')


## statistics of classification
axle_count = [x.index for x in event_list]
a_count = 0
b_count = 0
c_count = 0
for j in range(len(axle_count)):
    if axle_count[j] <= 2:
        a_count = a_count + 1
    else:
        if axle_count[j] < 4:
            b_count = b_count + 1
        else:
            c_count = c_count + 1

ref_index = [int(df_lan2.iloc[i, 9]) for i in range(len(df_lan2))]
plt.figure()
plt.clf()
plt.subplot(1,2,1)
plt.hist(axle_count, bins=np.linspace(0,12,13))
plt.xlabel('Number of Axle')
plt.ylabel('Counts')
plt.subplot(1,2,2)
plt.hist(ref_index, bins=np.linspace(0,12,13))
plt.xlabel('Class ID')
