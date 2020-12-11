from pydea.ETL import tirtle, wav
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot
from datetime import datetime, timedelta
from scipy import signal
import matplotlib.dates as md
import pytz
from sklearn.preprocessing import minmax_scale
from pydea.event.event_extraction import EventExtractionHY
from pydea.feature.featureextraction import FeatureExtraction

# from pydriveeasy.feature import FeatureExtraction
#%%
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


# @st.cache
# def extract_peaks(arr, peaks, half_window_size=300 * 2):
#     index_list = []
#     for pk in peaks:
#         left = max(0, pk - half_window_size)
#         right = min(len(arr), pk + half_window_size)
#         index_list.append((left, right))
#     return index_list


def create_events_dataset(f1_data, f2_data, index_list, lane_id='lane45', out_dir=None):
    time_index = f1_data.index
    t0 = time_index[0]
    time_str = datetime.strftime(t0, format='%H_%M_%S')
    if out_dir is None:
        out_dir = Path('.')
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        
    for i,(start, end) in enumerate(index_list):
        
        event_line1 = f1_data.iloc[start:end,:]
        event_line2 = f2_data.iloc[start:end,:]
        event_line1.to_csv(out_dir/f'{lane_id}_event{i+1}_line1.csv')
        event_line2.to_csv(out_dir/f'{lane_id}_event{i+1}_line2.csv')
        
        
def two_stage_event_extraction(wls, large_threshold=20, small_threshold=2.5):
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
                                   event_distance=timedelta(seconds=0.45),
                                   event_length=timedelta(seconds=0.8), height=small_threshold)
    
    return large_peaks,large_index_range,small_peaks, small_index_range

def plot_events(peaks, min_agg):
    plt.figure()
    time_index = min_agg.index
    plt.vlines(time_index[peaks],-5,5,colors='r')
    min_agg.plot()
    plt.show()
    
def plot_events_two_stage(peaks,peaks1, min_agg):
    plt.figure()
    time_index = min_agg.index
    plt.vlines(time_index[peaks],-5,2,colors='g')
    plt.vlines(time_index[peaks1],-5,4,colors='r')
    min_agg.plot()
    plt.show()
#%%
def remove_outliers(df, outlier_threshold=1000):
    df[df > outlier_threshold] = 0
    df[df < -outlier_threshold] = 0
    # print(df)
    return df

def load_tirtl_data(data_files, 
                    timezone=pytz.timezone('Australia/Melbourne'),
                    ): #time_shift_from_timestamp_to_melbourne=timedelta(hours=19)
    df = pd.read_csv(data_files, sep=",", header=0, low_memory=False)
    tirtle_ts = df.timestamp
    # tirtle_ts = df.iloc[:,1]
    # tirtle_dt = [datetime.fromtimestamp(ts, timezone)+time_shift_from_timestamp_to_melbourne for ts in tirtle_ts]

    tirtle_dt = [datetime.fromtimestamp(ts, timezone) for ts in tirtle_ts]
    df.index = pd.DatetimeIndex(tirtle_dt)
    return df

def get_unique_events(data_dir):
    files = list(data_dir.glob('*.csv'))
    # unique_events = sorted(list(set([file.stem[6:-3] for file in files])))
    unique_events = list(set([file.stem[0:-6] for file in files]))
    st.write(unique_events)
    # unique_events = sorted(list(set([file.stem[0:-6] for file in files])), key=lambda x:int(x[12:]))
    return unique_events

# def estimate_speed_2sensors(s1, s2):
#     fe = FeatureExtraction(s1, s2, frequency=SAMPLE_FREQUENCY)
#     delta_t = fe.estimate_delta_t(s1, s2)
#     speed = FIBER_DISTANCE / delta_t
#     return speed


# def estimate_speed(fiber1, fiber2):
#     data1 = fiber1.min(axis=1)
#     data2 = fiber2.min(axis=1)
#     # data1 = fiber1
#     # data2 = fiber2
#     fe = FeatureExtraction(data1, data2, frequency=SAMPLE_FREQUENCY)
#     delta_t = fe.estimate_delta_t(data1, data2)
#     speed = FIBER_DISTANCE / delta_t
#     return speed
# # def extract_features2(line1_data, line2_data):
# #     # line1_data = lane_data.line1_data
# #     # line2_data = lane_data.line2_data
# #     # step 1, agg to min sensor history
# #     line1_data = line1_data.values
# #     line2_data = line2_data.values
# #     line1_history = line1_data.min(axis=1)
# #     line2_history = line2_data.min(axis=1)
# #     # st.subheader('Step 1: min agg to get sensing line history')
# #     # fig = plt.figure()
# #     # plt.plot(line1_history, label = 'Line 1')
# #     # plt.plot(line2_history, label='Line 2')
# #     # plt.legend()
# #     # plt.tight_layout()
# #     # st.write(fig)
# #
# #     # step 2, find peaks from sensor history (timestamp wheel on sensor (WOS))
# #     time_peaks1, peak_prop1 = signal.find_peaks(abs(line1_history), distance=50, height=3)
# #     time_peaks2, peak_prop2 = signal.find_peaks(abs(line2_history), distance=50, height=3)
# #     num_axles = len(time_peaks1)
# #
# #     # if len(time_peaks1) == len(time_peaks2):
# #     #     delta_idx = np.array(time_peaks2) - np.array(time_peaks1)
# #     #     avg_delta_idx = np.mean(delta_idx)
# #     #     delta_t = avg_delta_idx/200
# #     #     speed = 2.0/(0.000001+delta_t)
# #
# #     # st.write(f'rough speed estimation (negative sign means travelling uphill): {speed} m/s, or {speed*ms_TO_MPH} MPH')
# #
# #     # step 3. estimate wheel location using sensor profile at ts_wos
# #     # line1_profile_wos = np.array([line1_data[time_wos,:] for time_wos in time_peaks1])
# #     if len(time_peaks1) < 1:
# #         return None
# #
# #     time_wos = time_peaks1[0]
# #     profile_at_wos = np.median(line1_data[time_wos - 1:time_wos + 1, :], axis=0)
# #
# #     # profile_at_wos = np.median(line1_data.iloc[time_wos - 1:time_wos + 1, :].values, axis=0)
# #     minmax_sensors = minmax_scale(abs(profile_at_wos))
# #     minmax_sensors[minmax_sensors < 0.1] = 0
# #     # loc_peaks, peak_props = signal.find_peaks(minmax_sensors, height=0.1, distance=5)
# #     wheel_locations = find_topk_peaks(minmax_sensors, topK=2, height=0.1, distance=5)
# #     vehicle_location = np.mean(wheel_locations)
# #
# #     # step 4. estimate speed using sensor history at wheel locations (2 sensors for example)
# #     speed_estimations = []
# #     # fig = plt.figure()
# #     fig, axes = plt.subplots(2, 1)
# #     for ax_id, wheel_location_idx in enumerate(wheel_locations):
# #         channel_id, sensor_id = lane_sensors['lane4'].line1[wheel_location_idx]
# #         line1_sensor_history = line1_data[:, wheel_location_idx]
# #         line2_sensor_history = line2_data[:, wheel_location_idx]
# #         # st.write(line1_sensor_history)
# #         # st.write(line2_sensor_history)
# #         axes[ax_id].plot(line1_sensor_history, label=f'Line1 at wheel_location:{wheel_location_idx}')
# #         axes[ax_id].plot(line2_sensor_history, label=f'Line2 at wheel_location:{wheel_location_idx}')
# #
# #         speed = estimate_speed_2sensors(abs(line1_sensor_history), abs(line2_sensor_history))
# #         speed_estimations.append(speed)
# #     speed = np.median(speed_estimations)
# #     # plt.legend()
# #     # st.write(fig)
# #     return vehicle_location, speed, num_axles

def read_ch_wls_file(file):
    data = pd.read_csv(file,header=0, parse_dates=[0], index_col=0)
    data.columns = [f'sensor_{i + 1}' for i in range(len(data.columns))]
    time_index = data.index
    # time_duration = data.index[-1] - data.index[0]
    # dt = time_duration.total_seconds() / len(time_index)
    # time_array = time_index[0] + np.arange(len(time_index)) * timedelta(seconds=dt)
    # data.index = time_array
    # wls = to_wls(data)
    clean_wls = data
    # clean_wls = remove_outlier(wls, outlier_threshold=550)
    event_time = time_index[0]
    return clean_wls, event_time

# def read_ch_file(file):
#     data = pd.read_csv(file,header=0, parse_dates=[0], index_col=0)
#     data.columns = [f'sensor_{i + 1}' for i in range(len(data.columns))]
#     time_index = data.index
#     time_duration = data.index[-1] - data.index[0]
#     dt = time_duration.total_seconds() / len(time_index)
#     time_array = time_index[0] + np.arange(len(time_index)) * timedelta(seconds=dt)
#     data.index = time_array
#     clean_wls = to_wls(data)
#     # clean_wls = remove_outlier(wls, outlier_threshold=550)
#     event_time = time_index[0]
#     return clean_wls, event_time
#%%
def load_event_data(data_dir, event, wls_format=True):
    # st.write(event)
    data = {}
    # event_files = list(data_dir.glob(f'{event}ch*.csv'))
    event_files = list(data_dir.glob(f'{event}*.csv'))
    # event_files = list(data_dir.glob(f'event_{event}ch*.csv'))
    num_sensors = {}
    for event_file in event_files:
        ch_id = event_file.stem[-1]
        # line_id = event_file.stem[-1]
        # data[f'ch{ch_id}'] = read_ch(event=event, ch=ch_id)
        if wls_format:
            print(f'event file:{event_file}')
            wls, event_time = read_ch_wls_file(event_file)
        # else:
        #     wls, event_time  = read_ch_file(event_file)

        data[f'ch{ch_id}'] = wls
        num_sensors[f'ch{ch_id}'] = data[f'ch{ch_id}'].shape[1]
    return data, num_sensors, event_time
#%%
def calc_events_stat(event_dir):
    # data_dir = Path(r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\Francis\raw_data\10_mins_batches\from_batch64\extracted_events')
    # data_dir = Path(r'C:\Users\hyu\gitlab_repos\driveeasy\explore\M80')
    unique_events = get_unique_events(event_dir)
    vehicle_location_list = []
    speed_list = []
    num_axles_list = []
    ts = []
    for event in unique_events:
        data, channel_sensors, event_time = load_event_data(data_dir, event)
        # batch_id = event[12]
        # print(f'bid:{batch_id}')


        # f1_id = st.sidebar.selectbox('Select 1st fiber',list(channel_sensors.keys()))
        # f2_id = st.sidebar.selectbox('Select 2st fiber',list(channel_sensors.keys()))
        event_str = str(event)
        print(event_str)
        f1_id = 'ch1'
        f2_id = 'ch2'

        try:
            res = extract_features2(data[f1_id], data[f2_id])
        except:
            continue

        if res:
            vehicle_location, speed, num_axles = res
            # st.write(f'speed:{abs(speed):0.3f} m/s, or {abs(speed) * ms_TO_MPH:0.3f} MPH')
            # st.write(f'location:{vehicle_location * 0.25} m')
            ts.append(event_time)
            speed_list.append(abs(speed))
            vehicle_location_list.append(vehicle_location)
            # plot_heatmap(data[f1_id], figure_filename=f'./{event}')
        out_df = pd.DataFrame({'timestamp':ts, 'speed':speed_list,'location':vehicle_location_list})
        out_df.to_csv(f'./M80_1124_out.csv')
        print(out_df.head())

        print('out df saved.')

    # df.index = pd.to_datetime(df['datetime'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    # df.drop(['datetime'],axis=1,inplace=True)
    # if timestamp_start is not None:
    #     df = df[(df.index >= timestamp_start) & (df.index <= timestamp_end)]
# def remove_outliers(df, threshold=1000):
    #%%
#%%
def load_event_data(event_id=2, type='large', event_dir=None):
    data_dir = event_dir
    # data_dir = Path(
    #     r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\TIRTLE validation\20201124\extracted_events')
    # # unique_events = get_unique_events(data_dir)
    file1 = f'lane5_{type}_event{event_id}_line1.csv'
    file2 = f'lane5_{type}_event{event_id}_line2.csv'
    line1 = pd.read_csv(data_dir/file1, parse_dates=[0], index_col=0)
    line2 = pd.read_csv(data_dir/file2,parse_dates=[0], index_col=0)
    lane_sensors = list(range(12))
    return line1.iloc[:,lane_sensors], line2.iloc[:,lane_sensors]
#%%
event_dir = Path(r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\TIRTLE validation\20201124\extracted_events')
calc_events_stat(event_dir)
#%% create evnets



#%% extract events
# td = load_tritl_data(tirtle_file)

# tirtle_file = r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\TIRTLE validation\20201124\TIRTLE\T0490_vehicle_20201125_070000_+1100_1h.csv'
tirtle_file = Path(r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\Calibration Test 20201201\M80\M80_TIRTL\data-1606945900265 - Copy.csv')
tirtle_data = load_tirtl_data(tirtle_file)

tirtle_ts = tirtle_data.timestamp
tirtle_dt = [datetime.fromtimestamp(ts)+timedelta(hours=19) for ts in tirtle_ts]
tirtle_data.index = pd.DatetimeIndex(tirtle_dt)
# data_files = list(data_dir.glob('*.npz'))
#%%
# data_dir = Path(r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\TIRTLE validation\20201124\driveeasy_wav')
# ch1_file = 'wav_20201124_195912_F01_UTC.npz'
# ch2_file = 'wav_20201124_195912_F02_UTC.npz'
# ch3_file = 'wav_20201124_195912_F03_UTC.npz'
# ch4_file = 'wav_20201124_195912_F04_UTC.npz'

data_dir = Path(r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\Calibration Test 20201201\M80\M80_DriveEasy\wav')

# ch1_file = 'wav_20201201_235115_F01_UTC.npz'
# ch2_file = 'wav_20201201_235115_F02_UTC.npz'
# ch3_file = 'wav_20201201_235115_F03_UTC.npz'
# ch4_file = 'wav_20201201_235115_F04_UTC.npz'

ch1_file = 'wav_20201202_024632_F01_UTC.npz'
ch2_file = 'wav_20201202_024632_F02_UTC.npz'
ch3_file = 'wav_20201202_024632_F03_UTC.npz'
ch4_file = 'wav_20201202_024632_F04_UTC.npz'
#%%
df3 = wav.load_wav_into_dataframe(data_dir/ch3_file)
# df4 = wav.load_wav_into_dataframe(data_dir/ch4_file)
#%%
df3.sort_index(inplace=True)
# df4.sort_index(inplace=True)
#%%
from pydea.preprocessing.detrend import detrend_df
wls3 = detrend_df(df3)*1000
wls4 = detrend_df(df4)*1000
#%%
wls3.sort_index(inplace=True)
wls4.sort_index(inplace=True)
#%%
wls3 = remove_outliers(wls3,outlier_threshold=600)
wls4 = remove_outliers(wls4,outlier_threshold=600)
l5f1 = wls3.iloc[:,list(range(12))]
l5f2 = wls4.iloc[:,list(range(12))]
#%% Extract events from lane 5 (slowest lane)
# lane5_sensors = list(range(1,12))
# lane4_sensors = list(range(12,22))
min_agg = l5f1.min(axis=1)
#%%
plt.figure()
plt.plot(min_agg)
#%%
large_peaks,large_index_range,small_peaks, small_index_range = EventExtractionHY.two_stage_event_extraction(min_agg, large_threshold=15, small_threshold=2.0)
plot_events_two_stage(large_peaks, small_peaks, min_agg)
#%%
out_dir = Path(r'.\m80_1202_0157PM_events')
create_events_dataset(wls3, wls4, large_index_range,lane_id='lane5_large', out_dir= out_dir)
create_events_dataset(wls3, wls4, small_index_range,lane_id='lane5_small', out_dir = out_dir)
#%% calc stats
event_dir = Path(r'C:\Users\hyu\gitlab_repos\driveeasy-analytics\explore\3_hy\M80_compare_tirtle\m80_1202_0157PM_events')
out_time = []
out_loc = []
out_speed = []
out_num_axles = []
out_ids = []
events_dict = {'large':159}
for event_id in range(1,160): # max 424
    line1, line2 = load_event_data(event_id,type='large', event_dir=event_dir)
    FE = FeatureExtraction()
    event_time, vehicle_location, speed, num_axles = FE.extract_features(line1, line2)
    out_ids.append(event_id)
    out_time.append(event_time)
    out_loc.append(vehicle_location)
    out_speed.append(speed)
    out_num_axles.append(num_axles)
out_df = pd.DataFrame({'event_id':out_ids,
                       'timestamp':out_time,
                       'location':out_loc,
                       'speed':out_speed,
                       'num_axles':num_axles})
out_df.to_csv(Path(r'.')/'m80_1202_0157PM_res_large_v1.csv')
#%% small
out_time = []
out_loc = []
out_speed = []
out_num_axles = []
out_ids = []
events_dict = {'large':159,'small':1005}
event_type = 'large'
for event_id in range(1,events_dict[event_type]+1): # max 424
    line1, line2 = load_event_data(event_id,type=event_type, event_dir=event_dir)
    FE = FeatureExtraction()
    res = FE.extract_features(line1, line2, promin_1=8, promin_2=8, dist_1=8, dist_2=8)
    if res:
        event_time, vehicle_location, speed, num_axles = res
        out_ids.append(event_id)
        out_time.append(event_time)
        out_loc.append(vehicle_location)
        out_speed.append(speed)
        out_num_axles.append(num_axles)
out_df = pd.DataFrame({'event_id':out_ids,
                       'timestamp':out_time,
                       'location':out_loc,
                       'speed':out_speed,
                       'num_axles':num_axles})
out_df.to_csv(Path(r'.')/f'm80_1202_0157PM_res_{event_type}_v2.csv')
#%%
wls3.to_pickle(data_dir/'wls_20201124_195912_F03_UTC.pickle')
wls4.to_pickle(data_dir/'wls_20201124_195912_F04_UTC.pickle')
#%%


#%% extract events

# peaks1,index_list1,peaks,index_list = two_stage_event_extraction(min_agg, large_threshold=15, small_threshold=5)
#%%a

#%%
all_index = index_list1.extend(index_list)
create_events_dataset(wls3, wls4,all_index,lane_id='lane5')


#%%
all_events = np.concatenate((peaks1, peaks))
#%%
sorted_events = sorted(all_events)

#%%
# tirtle_data.loc[:,'lane'] = 
# lane_data = tirtle_data[tirtle_data.lane=='5']
lane_data = tirtle_data[tirtle_data.lane==5]
fbgs_timestamp = df3.index

timestamp_start = fbgs_timestamp[0] + timedelta(hours=11)
timestamp_end = fbgs_timestamp[0] + timedelta(hours=11, minutes=120)
lane_window = lane_data[timestamp_start:timestamp_end]
#%%
# timestamp_start=str(timestamp_start)
# timestamp_end=str(timestamp_end)
# tirtle_data.set_index('time',inplace=True)
#%% compare
m80_file = Path(r'C:\Users\hyu\gitlab_repos\driveeasy\apps\M80_1124_out.csv')
m80_res = pd.read_csv(m80_file,header=0)
#%%
# title_name = 'lane 5'
# fig, ax = plt.subplots()
# plt.scatter(lane_window.index, lane_window["speed(kph)"])
# plt.vlines(lane_window.index, 0, lane_window["speed(kph)"])
# plt.ylabel('Speed (KPH)')
# plt.title(title_name)
# # if timestamp_start is not None:
# #     plt.xlim([timestamp_start, timestamp_end])
# xfmt = md.DateFormatter('%H:%M:%S')
# ax.xaxis.set_major_formatter(xfmt)
# # add driveeasy data
# time_index0 = min_agg.index
# time_index = time_index0 + timedelta(hours=11,seconds=3.2)

# load calc stats
# large_res = Path(r'C:\Users\hyu\gitlab_repos\driveeasy-analytics\m80_1124_res_large.csv')
# small_res = Path(r'C:\Users\hyu\gitlab_repos\driveeasy-analytics\m80_1124_res_small.csv')
#%%
large_res = Path(r'.\m80_1202_0157PM_res_large_v3.csv')
small_res = Path(r'.\m80_1202_0157PM_res_small_v3.csv')
large_df = pd.read_csv(large_res,header=0,index_col='timestamp', parse_dates=['timestamp'])#,parse_dates=[0],))
small_df = pd.read_csv(small_res,header=0,index_col='timestamp', parse_dates=['timestamp'])#,parse_dates=[0],index_col=0)
#%%
small_df = small_df.speed
large_df = large_df.speed
#%%
print(small_df.head())
def remove_outliers(df,low=50/3.6, high=200/3.6):
    return df[(df>low) &(df<high)]
small_df= remove_outliers(small_df)
large_df= remove_outliers(large_df)
#%% Plot
#################
small_df = small_df.speed
large_df = large_df.speed

#%% plot speed

title_name = 'lane 5'
fig, ax = plt.subplots()

plt.scatter(lane_window.index, lane_window["speed(kph)"])
plt.vlines(lane_window.index, 0, lane_window["speed(kph)"])
plt.ylabel('Speed (KPH)')
plt.title(title_name)
# if timestamp_start is not None:
#     plt.xlim([timestamp_start, timestamp_end])
xfmt = md.DateFormatter('%H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
# add driveeasy data

time_index0 = small_df.index
time_index = time_index0 + timedelta(hours=11,seconds=1.2) #3.2
plt.scatter(time_index, small_df*3.6, color='r')
plt.vlines(time_index, 0, small_df*3.6,color='r')
#% plot large
time_index0 = large_df.index
time_index = time_index0 + timedelta(hours=11,seconds=1.2)
plt.scatter(time_index, large_df*3.6*1, color='g')
plt.vlines(time_index, 0, large_df*3.6*1,color='g')

# ax2 = ax.twinx()
time_index0 = min_agg.index
time_index = time_index0 + timedelta(hours=11,seconds=1.2)
ax.plot(time_index, min_agg.values,color='m')
# ax2.set_ylim([-40,1])
# add raw data

plt.show()

#%% plot num axles
large_df = large_df.num_axles
small_df = small_df.num_axles
#%%
title_name = 'lane 5'
fig, ax = plt.subplots()
plt.scatter(lane_window.index, lane_window["axle_count"])
plt.vlines(lane_window.index, 0, lane_window["axle_count"])
# plt.scatter(lane_window.index, lane_window["speed(kph)"])
# plt.vlines(lane_window.index, 0, lane_window["speed(kph)"])
plt.ylabel('axle count')

time_index0 = large_df.index
time_index = time_index0 + timedelta(hours=11,seconds=1.2)
plt.scatter(time_index, large_df, color='g')
plt.vlines(time_index, 0, large_df,color='g')

time_index0 = small_df.index
time_index = time_index0 + timedelta(hours=11,seconds=1.2) #3.2
plt.scatter(time_index, small_df, color='r')
plt.vlines(time_index, 0, small_df,color='r')
plt.title(title_name)
plt.show()
##################
#%% all res
all_df = pd.concat([large_df, small_df])
#%%
plt.figure()
all_df.hist()

plt.figure()
lane_window["axle_count"].hist()
plt.show()
#%%
plt.figure()
(all_df*3.6*0.8).hist()
#%%
all_speed = all_df*3.6*0.8
all_speed.sort_index(inplace=True)
rolling_speed = all_speed.rolling('20S').count()
#%%
plt.figure()
time_index0 = rolling_speed.index
time_index = time_index0 + timedelta(hours=11,seconds=1.2) #3.2
t_start = time_index[0]
t_end = time_index[-1]

plt.plot(time_index,rolling_speed.values,label='Fibridge DriveEasy')
plt.xlabel('Melourne time')
plt.ylabel('Volumne over 20s')


rolling_tirtl = lane_window["speed(kph)"].rolling('20s').count()
tirtl_window = rolling_tirtl[t_start:t_end]
plt.plot(tirtl_window.index, tirtl_window,label='TIRTL')
plt.legend()
plt.show()
# rolling_speed.plot()
#%%
t_start = all_df
#%%

plt.vlines(time_index[peaks],-5,2,colors='r')
plt.vlines(time_index[peaks1],-5,4,colors='r')
plt.plot(time_index, min_agg.values,color='g')

# min_agg.plot()

plt.show()

#%%

fig, ax = plt.subplots()
plt.scatter(lane_data.time, lane_data["Speed KPH"])
plt.vlines(lane_data.time, 0, lane_data["Speed KPH"])
plt.ylabel('Speed (KPH)')
plt.title(title_name)
plt.xlim([timestamp_start, timestamp_end])
ax.set_xticks([])
plt.legend(['VIPER VIM'])
plt.xlabel('Timestamps: ' + str(timestamp_start) + ' ~ ' + str(timestamp_end))
# my_xticks = ax.get_xticks()
# plt.xticks([my_xticks[0], my_xticks[-1]], visible=True, rotation=00)
plt.show()

#%%
utc_offset = datetime.fromtimestamp(ts) - datetime.utcfromtimestamp(ts)













#%%
#%%
plt.figure()
wls3.sensor1.plot()
plt.show()
#%%
fig = px.line(l5f1)
plot(fig)
#%%

#%%
df30 = df3.sort_index()
#%%
ts = df1.index
from datetime import datetime, timedelta, timezone
ts_melbourne = ts + timedelta(hours=11)
#%%
import matplotlib.pyplot as plt
plt.figure()
df1.sensor19.iloc[0:-90000].plot()
plt.show()

#%% plot overlay 
lane_data = tirtle_data[0:100]
turtle_ts = lane_data.time
# turtle_ts
# turtle_ts_rel = turtle_ts - turtle_ts[0]
#%%
#
# plt.figure()
fig, ax = plt.subplots()
plt.scatter(lane_data.time, lane_data["speed(kph)"])
plt.vlines(lane_data.time, 0, lane_data["speed(kph)"])
# plt.plot(ts_melbourne,df1.sensor19.values)
# plt.title(tirtle_name)
# if timestamp_start is not None:
#     plt.xlim([timestamp_start, timestamp_end])
my_xticks = ax.get_xticks()
plt.xticks([my_xticks[0], my_xticks[-1]], visible=True, rotation=00)
plt.show()

#%%
plt.figure()
plt.plot(df30.index,df30.sensor7.values)
plt.show()

#%%
ts = lane_data.timestamp
dt1 = datetime.fromtimestamp(ts[0])

#%%
d1 = np.load(data_dir/ch1_file, allow_pickle=True)
#%%
plt.figure()
plt.plot(d1['timestamp'],d1['wav'][:,9])
plt.show()
#%%
ts =  d1['timestamp']
ts_diff = np.diff(ts)
is_mono = all([dt.astype('float')>0 for dt in ts_diff])
#%%
bad_ind = []
for i, dt in enumerate(ts_diff):
    if dt.astype('float') < 0:
        bad_ind.append(i)
        print(ts[i-1:i+2])
        print(d1['wav'][i])

#%%

