from pydea.ETL import tirtl, wav
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
# %%
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

    for i, (start, end) in enumerate(index_list):
        event_line1 = f1_data.iloc[start:end, :]
        event_line2 = f2_data.iloc[start:end, :]
        event_line1.to_csv(out_dir / f'{lane_id}_event{i + 1}_line1.csv')
        event_line2.to_csv(out_dir / f'{lane_id}_event{i + 1}_line2.csv')


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

    return large_peaks, large_index_range, small_peaks, small_index_range


def plot_events(peaks, min_agg):
    plt.figure()
    time_index = min_agg.index
    plt.vlines(time_index[peaks], -5, 5, colors='r')
    min_agg.plot()
    plt.show()


def plot_events_two_stage(peaks, peaks1, min_agg):
    plt.figure()
    time_index = min_agg.index
    plt.vlines(time_index[peaks], -5, 2, colors='g')
    plt.vlines(time_index[peaks1], -5, 4, colors='r')
    min_agg.plot()
    plt.show()


# %%
def remove_outliers(df, outlier_threshold=1000):
    df[df > outlier_threshold] = 0
    df[df < -outlier_threshold] = 0
    # print(df)
    return df


def load_tirtl_data(data_files,
                    timezone=pytz.timezone('Australia/Melbourne'),
                    ):  # time_shift_from_timestamp_to_melbourne=timedelta(hours=19)
    df = pd.read_csv(data_files, sep=",", header=0, low_memory=False)
    tirtle_ts = df.timestamp
    # tirtle_ts = df.iloc[:,1]
    # tirtle_dt = [datetime.fromtimestamp(ts, timezone)+time_shift_from_timestamp_to_melbourne for ts in tirtle_ts]

    tirtle_dt = [datetime.fromtimestamp(ts, timezone) for ts in tirtle_ts]
    df.index = pd.DatetimeIndex(tirtle_dt)
    return df

def load_event_data(event_id=2, type='large', event_dir=None):
    data_dir = event_dir
    # data_dir = Path(
    #     r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\TIRTLE validation\20201124\extracted_events')
    # # unique_events = get_unique_events(data_dir)
    file1 = f'lane5_{type}_event{event_id}_line1.csv'
    file2 = f'lane5_{type}_event{event_id}_line2.csv'
    line1 = pd.read_csv(data_dir / file1, parse_dates=[0], index_col=0)
    line2 = pd.read_csv(data_dir / file2, parse_dates=[0], index_col=0)
    lane_sensors = list(range(12))
    return line1.iloc[:, lane_sensors], line2.iloc[:, lane_sensors]


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


# %% calc stats
if __name__ == "__main__":
    event_dir = Path(
        r'C:\Users\hyu\gitlab_repos\driveeasy-analytics\explore\3_hy\M80_compare_tirtle\m80_1202_0157PM_events')
    #%%
    out_time = []
    out_loc = []
    out_speed = []
    out_num_axles = []
    out_ids = []
    events_dict = {'large': 159, 'small': 1005}
    event_type = 'large'
    for event_id in range(1, events_dict[event_type] + 1):  # max 424
        line1, line2 = load_event_data(event_id, type=event_type, event_dir=event_dir)
        FE = FeatureExtraction()
        res = FE.extract_features(line1, line2, promin_1=2, promin_2=2, dist_1=6, dist_2=6)
        if res:
            event_time, vehicle_location, speed, num_axles = res
            num_axles = int(np.ceil(num_axles))
            out_ids.append(event_id)
            out_time.append(event_time)
            out_loc.append(vehicle_location)
            out_speed.append(speed)
            out_num_axles.append(num_axles)
    out_df = pd.DataFrame({'event_id': out_ids,
                           'timestamp': out_time,
                           'location': out_loc,
                           'speed': out_speed,
                           'num_axles': out_num_axles})
    print(f'num data:{out_df.shape[0]}')
    out_df.to_csv(Path(r'.') / f'm80_1202_0157PM_res_{event_type}_v5.csv')

