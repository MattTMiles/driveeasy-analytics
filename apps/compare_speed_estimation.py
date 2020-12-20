import streamlit as st
import pandas as pd
import numpy as np
import scipy.signal as signal
import datetime
from pathlib import Path
from sklearn.preprocessing import minmax_scale

import plotly.express as px
from pydea.viz import plot_heatmap
from pydea.feature.speed import SpeedEstimationHY, SpeedEstimationQC, SpeedEstimationJY

@st.cache
def load_event_data(event_id=2, type='large'):
    data_dir = Path(
        r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\TIRTLE validation\20201124\extracted_events')
    # unique_events = get_unique_events(data_dir)
    file1 = f'lane5_{type}_event{event_id}_line1.csv'
    file2 = f'lane5_{type}_event{event_id}_line2.csv'
    line1 = pd.read_csv(data_dir / file1, parse_dates=[0], index_col=0)
    line2 = pd.read_csv(data_dir / file2, parse_dates=[0], index_col=0)
    lane_sensors = list(range(12))
    return line1.iloc[:, lane_sensors], line2.iloc[:, lane_sensors]

def create_sidebar():
    st.sidebar.title('Compare Speed Estations Algorithms')
    event_type = st.sidebar.selectbox('Choose event type {large truck, small vehicle}',['large','small'])
    event_id = st.sidebar.slider('Choose event id (max 60 for large truck,max 424 for small vehicles',1,424,1)
    return event_type, event_id

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

def calc_num_axles(line1, line2):
    wav1 = line1.min(axis=1)
    wav2 = line2.min(axis=1)
    ax_count, ax_list = find_axle_location(wav1, wav2, promin_1=5, promin_2=5)
    return ax_count

def line_plot(min_agg):
    fig = px.line(min_agg)
    return fig

from pydea.feature.featureextraction import FeatureExtraction

def extract_features(line1, line2):
    FE = FeatureExtraction()
    res = FE.extract_features(line1, line2)
    return res
    # if res:
    #     event_time, vehicle_location, speed, num_axles = res
    #     return

def extract_features2(line1_data, line2_data):
    # line1_data = lane_data.line1_data
    # line2_data = lane_data.line2_data
    # step 1, agg to min sensor history
    line1_data = line1_data.values
    line2_data = line2_data.values
    line1_history = line1_data.min(axis=1)
    line2_history = line2_data.min(axis=1)
    # st.subheader('Step 1: min agg to get sensing line history')
    # fig = plt.figure()
    # plt.plot(line1_history, label = 'Line 1')
    # plt.plot(line2_history, label='Line 2')
    # plt.legend()
    # plt.tight_layout()
    # st.write(fig)

    # step 2, find peaks from sensor history (timestamp wheel on sensor (WOS))
    time_peaks1, peak_prop1 = signal.find_peaks(abs(line1_history), distance=50, height=3)
    time_peaks2, peak_prop2 = signal.find_peaks(abs(line2_history), distance=50, height=3)
    num_axles = len(time_peaks1)

    # if len(time_peaks1) == len(time_peaks2):
    #     delta_idx = np.array(time_peaks2) - np.array(time_peaks1)
    #     avg_delta_idx = np.mean(delta_idx)
    #     delta_t = avg_delta_idx/200
    #     speed = 2.0/(0.000001+delta_t)

    # st.write(f'rough speed estimation (negative sign means travelling uphill): {speed} m/s, or {speed*ms_TO_MPH} MPH')

    # step 3. estimate wheel location using sensor profile at ts_wos
    # line1_profile_wos = np.array([line1_data[time_wos,:] for time_wos in time_peaks1])
    if len(time_peaks1) < 1:
        return None

    time_wos = time_peaks1[0]
    profile_at_wos = np.median(line1_data[time_wos - 1:time_wos + 1, :], axis=0)

    # profile_at_wos = np.median(line1_data.iloc[time_wos - 1:time_wos + 1, :].values, axis=0)
    minmax_sensors = minmax_scale(abs(profile_at_wos))
    minmax_sensors[minmax_sensors < 0.1] = 0
    # loc_peaks, peak_props = signal.find_peaks(minmax_sensors, height=0.1, distance=5)
    wheel_locations = find_topk_peaks(minmax_sensors, topK=2, height=0.1, distance=5)
    vehicle_location = np.mean(wheel_locations)

    # step 4. estimate speed using sensor history at wheel locations (2 sensors for example)
    speed_estimations = []
    # fig = plt.figure()
    fig, axes = plt.subplots(2, 1)
    for ax_id, wheel_location_idx in enumerate(wheel_locations):
        channel_id, sensor_id = lane_sensors['lane5'].line1[wheel_location_idx]
        line1_sensor_history = line1_data[:, wheel_location_idx]
        line2_sensor_history = line2_data[:, wheel_location_idx]
        # st.write(line1_sensor_history)
        # st.write(line2_sensor_history)
        axes[ax_id].plot(line1_sensor_history, label=f'Line1 at wheel_location:{wheel_location_idx}')
        axes[ax_id].plot(line2_sensor_history, label=f'Line2 at wheel_location:{wheel_location_idx}')

        speed = estimate_speed_2sensors(abs(line1_sensor_history), abs(line2_sensor_history))
        speed_estimations.append(speed)
    speed = np.median(speed_estimations)
    # plt.legend()
    # st.write(fig)
    return vehicle_location, speed, num_axles
event_type, event_id = create_sidebar()
line1, line2 = load_event_data(event_id=event_id, type=event_type)
hm = plot_heatmap(line1, figsize=[500,500])
st.write(hm)
SE = SpeedEstimationHY(line1,line2)
speed = SE.estimate_speed_2sensors(line1.min(axis=1), line2.min(axis=1))
st.write(f'speed:{speed}')
st.write(line_plot(line1.min(axis=1)))
st.write(f'num axles: {calc_num_axles(line1, line2)}')

res = extract_features(line1, line2)
if res:
    event_time, vehicle_location, speed, num_axles = res
    st.write(f'num axles:{num_axles}')
