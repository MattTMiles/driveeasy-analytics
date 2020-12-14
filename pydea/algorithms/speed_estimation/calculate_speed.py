from typing import Union, List

import numpy as np
import scipy.signal as signal

from pydea.datamodels.datamodel import Event


def calculate_speed_alg1(event: Event,
                         lane_sensor: List,
                         fiber_distance: Union[float, int] = 2.5,
                         sampling_rate: Union[float, int] = 200.0):
    """
    Calcualte speed using correlation of aggregated sensor history.
    Steps:
    1. Pick sensor data that belongs to one lane using `lane_sensor`
    2. Aggregate the chosen wavelength data in sensor location dimension. Resulting data index is timestamp.
    3. Calculate speed from the two sensor data history using correlation
    Args:
        event: Event type.
        lane_sensor: list of sensors within a lane. Example: [0,1,2,3,4,5,6,7,8,9]
        fiber_distance: distance between leading and trailing sensors. Default is 2.5m.
        sampling_rate: data collection rate, default is 200Hz.

    Returns:

    """
    trace_temp1 = np.min(event.wav1[:, lane_sensor], axis=1)
    trace_temp2 = np.min(event.wav2[:, lane_sensor], axis=1)
    speed_valid = signal.correlate(trace_temp1, trace_temp2)
    try:
        speed_corr = -1 * fiber_distance / (speed_valid.argmax() - len(event.wav1)) * sampling_rate * 3.6
    except:
        speed_corr = 0
    return speed_corr


def calculate_speed_alg2(event: Event,
                         lane_sensor: List,
                         fiber_distance: Union[float, int] = 2.5,
                         sampling_rate: Union[float, int] = 200.0):
    """
    Calcualte speed using 1) correlation and 2)distance between the 1st peaks from the leading and trailing sensors.
    Steps:
    1. Pick sensor data that belongs to one lane using `lane_sensor`
    2. Aggregate the chosen wavelength data in time dimension. Resulting data index is sensor id.
    3. Find the sensor location that sees max compression (mininum signal value). This corresponds to the wheel location (Wheel on Sensor --WOS)
    4. Pick sensor history at the WOS location from leading and trailing fibers (sensor data history)
    5. Calculate speed from the two sensor data history using correlation
    6. Calculate speed from the two sensor data history using the delta_time between the 1st peaks from the leading and trailing sensors.

    Args:
        event: Event type.
        lane_sensor: list of sensors within a lane. Example: [0,1,2,3,4,5,6,7,8,9]
        fiber_distance: distance between leading and trailing sensors. Default is 2.5m.
        sampling_rate: data collection rate, default is 200Hz.

    Returns:

    """
    kk = np.min(event.wav1[:, lane_sensor], axis=0).argmin()
    # trace_temp1 = np.max(np.abs(event_list[j].wav1[:,lane_sensor_1]), axis=1)
    trace_temp1 = np.abs(event.wav1[:, kk])
    trace_temp2 = np.abs(event.wav2[:, kk])

    #     trace_temp2 = np.max(np.abs(event.wav2[:, lane_sensor]), axis=1)
    #     trace_temp1 = np.max(np.abs(event.wav1[:, lane_sensor]), axis=1)
    speed_valid = signal.correlate(trace_temp1, trace_temp2)
    try:
        speed_corr = -1 * fiber_distance / (speed_valid.argmax() - len(event.wav1)) * sampling_rate * 3.6
    except:
        speed_corr = 0

    peaks_temp1, peaks_prop1 = signal.find_peaks(trace_temp1, prominence=0.002)
    peaks_temp2, peaks_prop2 = signal.find_peaks(trace_temp2, prominence=0.002)

    if len(peaks_temp1) == 0 or len(peaks_temp2) == 0:
        return speed_corr, 0, 0
    else:
        speed_agg_peak = -1 * fiber_distance / (peaks_temp1[0] - peaks_temp2[0]) * sampling_rate * 3.6
        return speed_corr, speed_agg_peak, np.max([len(peaks_temp1), len(peaks_temp1)])


def calculate_speed_ensemble(event: Event,
                             lane_sensor: List,
                             fiber_distance: Union[float, int] = 2.5,
                             sampling_rate: Union[float, int] = 200.0):
    """
    Calculate speed using multiple approaches:
    1) correlation from aggregated sensor data (using all sensor within a lane),
    2) correlation from WOS sensor data (only using single WOS sensor),
    3) delta_time from first peaks from leading and trailing fibers (only using single WOS sensor).
    Args:
        event:
        lane_sensor:
        fiber_distance:
        sampling_rate:

    Returns:

    """
    speed_temp1 = calculate_speed_alg1(event=event,
                                       lane_sensor=lane_sensor,
                                       fiber_distance=fiber_distance,
                                       sampling_rate=sampling_rate)
    speed_temp2, speed_temp3, axle_index = calculate_speed_alg2(event=event,
                                                                lane_sensor=lane_sensor,
                                                                fiber_distance=fiber_distance,
                                                                sampling_rate=sampling_rate)
    speed_list = [speed_temp1, speed_temp2, speed_temp3]
    return speed_list

