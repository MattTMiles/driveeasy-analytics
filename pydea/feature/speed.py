from scipy import signal
import numpy as np


FIBER_DISTANCE = 2.5
SAMPLING_RATE = 200.0
class SpeedEstimationJY:
    # def __init__(self, fiber_distance=FIBER_DISTANCE, sampling_rate=SAMPLING_RATE):
    #     self.fiber_distance = fiber_distance
    #     self.sampling_rate = sampling_rate
    @staticmethod
    def speed_from_all_sensor_pairs(fiber1, fiber2, low_threshold, high_threshold):
        """Adapted from Jin's implementation.

        :param fiber1:
        :param fiber2:
        :return:
        """
        time_diff = fiber1.idxmax().values - fiber2.idxmax().values
        time_diff = time_diff / np.timedelta64(1, 's')

        # distance (2m) divide by the time difference, 2.23694 is tranferring m/s to mph
        velocity = FIBER_DISTANCE  / time_diff * 3.6

        # threshold the abnormal data beyond vehicle speed limit.
        velocity[(-low_threshold < velocity) & (velocity < low_threshold)] = np.nan
        velocity[(velocity > high_threshold) | (velocity < -high_threshold)] = np.nan
        velocity[velocity > 0] = np.nan
        # mode = histogram_mode(velocity, visualization=False)
        speed_kph  = np.nanmedian(velocity)
        # speed_kph = speed_mps * 3.6
        return speed_kph


class SpeedEstimationQC:
    @staticmethod
    def speed_from_single_sensor_pair(event, sensor_id = 19):
        """Speed estimation using only on pair of sensors from leading and trailing fibers. Adapted from Qiushu's implementation.

        :param event:
        :param sensor_id: sensor to be used for speed estimation
        :return: speed in km/h
        """
        fiber1 = event.fiber1
        fiber2 = event.fiber2
        fiber1_remove_base = fiber1[:,sensor_id-1] -  fiber1[0,sensor_id-1]
        fiber2_remove_base = fiber2[:, sensor_id - 1] - fiber2[0, sensor_id - 1]
        corr = signal.correlate(fiber1_remove_base, fiber2_remove_base)
        speed_mps = (-1 * FIBER_DISTANCE)/(corr.argmax() - len(fiber1))*SAMPLING_RATE
        speed_kph = speed_mps*3.6
        return speed_kph


class SpeedEstimationHY:
    pass




# speed_valid = signal.correlate(event_list[j].data['leading'].iloc[:,ll]-event_list[j].data['leading'].iloc[0,ll], event_list[j].data['trailing'].iloc[:,ll]-event_list[j].data['trailing'].iloc[0,ll])
# speed = -1*2.5/(speed_valid.argmax()-len(event_list[j].data['leading']))*SAMPLING_RATE*3.6
# print(speed)
# plt.figure()
# plt.plot(speed_valid)
