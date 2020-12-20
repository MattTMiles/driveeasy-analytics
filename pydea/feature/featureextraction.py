from sklearn.preprocessing import minmax_scale
from pydea.utils.peak_utils import find_topk_peaks
from scipy import signal
import numpy as np
from pydea.datamodels import LaneSensors
from pydea.feature.speed import SpeedEstimationHY
from pydea.utils.peak_utils import get_top_peaks
class FeatureExtraction:
    def __init__(self):
        self.lane_id = 'lane5'


    @staticmethod
    def create_lane_sensors():
        lane1_line1 = [(1, sensor) for sensor in range(1, 15)]
        lane1_line2 = [(2, sensor) for sensor in range(1, 15)]

        lane2_line1 = [(1, sensor) for sensor in range(12, 25)]
        lane2_line2 = [(2, sensor) for sensor in range(12, 25)]

        lane4_line1 = [(3, sensor) for sensor in range(12, 25)]
        lane4_line2 = [(4, sensor) for sensor in range(12, 25)]

        lane5_line1 = [(3, sensor) for sensor in range(1, 12)]
        lane5_line2 = [(4, sensor) for sensor in range(1, 12)]

        lane1_line1 = [(1, sensor) for sensor in range(1, 15)]
        lane1_line2 = [(2, sensor) for sensor in range(1, 15)]

        lane1 = LaneSensors(line1=lane1_line1, line2=lane1_line2)
        lane2 = LaneSensors(line1=lane2_line1, line2=lane2_line2)

        lane4 = LaneSensors(line1=lane4_line1, line2=lane4_line2)
        lane5 = LaneSensors(line1=lane5_line1, line2=lane5_line2)
        lane_sensors = {'lane4': lane4, 'lane5': lane5}

        return lane_sensors

    # lane_sensors = create_lane_sensors()

    def find_axle_location(self, wav1, wav2, promin_1=0.001, promin_2=0.001, distance=5):
        peaks_1 = signal.find_peaks(wav1, prominence=promin_1,distance=distance)
        peaks_2 = signal.find_peaks(wav2, prominence=promin_2,distance=distance)
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
                        axle_list[i] = (axle_fiber1[i] + axle_fiber2[i]) / 2
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

    def extract_features(self, line1, line2, promin_1=8, promin_2=8, dist_1=8, dist_2=8):
        num_axles_res = []
        lane_sensors = FeatureExtraction.create_lane_sensors()
        # line1_data = lane_data.line1_data
        # line2_data = lane_data.line2_data
        # step 1, agg to min sensor history
        line1_data = line1.values
        line2_data = line2.values
        line1_history = line1_data.min(axis=1)
        line2_history = line2_data.min(axis=1)
        line1_profile = line2_data.min(axis=0)
        # peaks1, prop1 = signal.find_peaks(abs(line1_profile), distance=2, height=1.5)
        # wheel_loc = get_top_peaks(peaks1, prop1['peak_heights'], topK=1)[0]
        # wos_history =

        # st.subheader('Step 1: min agg to get sensing line history')
        # fig = plt.figure()
        # plt.plot(line1_history, label = 'Line 1')
        # plt.plot(line2_history, label='Line 2')
        # plt.legend()
        # plt.tight_layout()
        # st.write(fig)

        # step 2, find peaks from sensor history (timestamp wheel on sensor (WOS))
        time_peaks1, peak_prop1 = signal.find_peaks(abs(line1_history), distance=dist_1, prominence=promin_1)
        time_peaks2, peak_prop2 = signal.find_peaks(abs(line2_history), distance=dist_2, prominence=promin_2)
        if len(time_peaks1) < 1:
            return None

        num_axles1 = len(time_peaks1)
        num_axles2 = len(time_peaks2)
        num_axles_res.append(num_axles1)
        num_axles_res.append(num_axles2)
        event_time_index = time_peaks1[0]
        event_time = line1.index[event_time_index]

        # if len(time_peaks1) == len(time_peaks2):
        #     delta_idx = np.array(time_peaks2) - np.array(time_peaks1)
        #     avg_delta_idx = np.mean(delta_idx)
        #     delta_t = avg_delta_idx/200
        #     speed = 2.0/(0.000001+delta_t)

        # st.write(f'rough speed estimation (negative sign means travelling uphill): {speed} m/s, or {speed*ms_TO_MPH} MPH')

        # step 3. estimate wheel location using sensor profile at ts_wos
        # line1_profile_wos = np.array([line1_data[time_wos,:] for time_wos in time_peaks1])
        # if len(time_peaks1) < 1:
        #     return None

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
        # fig, axes = plt.subplots(2, 1)
        SE = SpeedEstimationHY(line1_data,line2_data)
        agg_speed = SE.estimate_speed_2sensors(line1_history, line2_history)
        speed_estimations.append(agg_speed)
        for ax_id, wheel_location_idx in enumerate(wheel_locations):
            print(lane_sensors[self.lane_id].line1)
            print(wheel_location_idx)
            # channel_id, sensor_id = lane_sensors[self.lane_id].line1[wheel_location_idx]
            line1_sensor_history = line1_data[:, wheel_location_idx]
            line2_sensor_history = line2_data[:, wheel_location_idx]
            # st.write(line1_sensor_history)
            # st.write(line2_sensor_history)
            # axes[ax_id].plot(line1_sensor_history, label=f'Line1 at wheel_location:{wheel_location_idx}')
            # axes[ax_id].plot(line2_sensor_history, label=f'Line2 at wheel_location:{wheel_location_idx}')
            #
            # line1_axels =
            axle_count, axle_list = self.find_axle_location(abs(line1_sensor_history), abs(line2_sensor_history), promin_1=promin_1, promin_2=promin_2, distance=dist_1)
            print(f'num axles qc: {axle_count}')

            speed = SE.estimate_speed_2sensors(abs(line1_sensor_history), abs(line2_sensor_history))
            num_axles_res.append(axle_count)

            # speed = estimate_speed_2sensors(abs(line1_sensor_history), abs(line2_sensor_history))
            speed_estimations.append(speed)

        speed = np.median(speed_estimations)
        print(f'num axles: {num_axles_res}')
        num_axles_out = np.max(num_axles_res)
        num_axles_out = min(11,num_axles_out)
        num_axles_out = max(2, num_axles_out)
        # plt.legend()
        # st.write(fig)
        return event_time, vehicle_location, speed, num_axles_out

from pathlib import Path
import pandas as pd

def create_test_event_data(event_id=2, type='large'):
    data_dir = Path(
        r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\TIRTLE validation\20201124\extracted_events')
    # unique_events = get_unique_events(data_dir)
    file1 = f'lane5_{type}_event{event_id}_line1.csv'
    file2 = f'lane5_{type}_event{event_id}_line2.csv'
    line1 = pd.read_csv(data_dir/file1, parse_dates=[0], index_col=0)
    line2 = pd.read_csv(data_dir/file2,parse_dates=[0], index_col=0)
    lane_sensors = list(range(12))
    return line1.iloc[:,lane_sensors], line2.iloc[:,lane_sensors]

def test_speed_estimation():
    line1, line2 = create_test_event_data()
    SE = SpeedEstimationHY(line1, line2)
    s = SE.estimate_speed_2sensors(line1.min(axis=1), line2.min(axis=1))
    print(f'speed: {s} m/s or {s*3.6:.3f} kmh')

def test_feature_extraction():
    line1, line2 = create_test_event_data(4)
    FE = FeatureExtraction()
    event_time, vehicle_location, speed, num_axles = FE.extract_features(line1, line2)
    print(event_time,vehicle_location, speed, num_axles)

if __name__ == '__main__':


    out_time = []
    out_loc = []
    out_speed = []
    out_num_axles = []
    out_ids = []
    for event_id in range(1,424): # max 424
        line1, line2 = create_test_event_data(event_id,type='small')
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
    out_df.to_csv(Path(r'.')/'m80_1124_res_small.csv')