from pathlib import Path

import numpy as np

from pydea.algorithms.speed_estimation.calculate_speed import calculate_speed_alg1, calculate_speed_alg2, \
    calculate_speed_ensemble
from pydea.datamodels.datamodel import Event
from ..data_utils import prepare_event_data

# def prepare_event_data():
#     event_dir = Path(r'..\test_data\events')
#     files = list(event_dir.glob('evt*.npz'))
#     events = []
#     for file in files:
#         events.append(Event(file))
#     return events


def load_event(event_file=Path(r'..\test_data\events\evt_20201130_223320_UTC_Francis.npz')):
    event = Event(event_file)
    print(event.timestamp)
    return event


def test_load_event():
    event = load_event()
    print(event.fiber1_sensors)


def test_prepare_event_data():
    events = prepare_event_data()
    assert len(events) == 2


def test_calculate_speed_alg1():
    events = prepare_event_data()
    print(f'check condition: speed > 0 and speed <= 160 ')
    for event in events:
        speed = calculate_speed_alg1(event=event,
                                     lane_sensor=list(range(0, 11)),
                                     fiber_distance=2.5,
                                     sampling_rate=200)
        print(f'estimated speed: {speed}')
        assert (speed > 0 and speed <= 160)


def test_calculate_speed_alg2():
    events = prepare_event_data()
    print(f'check condition: speed > 0 and speed <= 160 ')
    for event in events:
        speed_corr, speed_delta_t, _ = calculate_speed_alg2(event=event,
                                                            lane_sensor=list(range(0, 11)),
                                                            fiber_distance=2.5,
                                                            sampling_rate=200)
        print(f'speed_corr: {speed_corr}')
        assert (speed_corr > 0 and speed_corr <= 160)
        print(f'speed_delta_t: {speed_delta_t}')
        assert (speed_delta_t > 0 and speed_delta_t <= 160)


def test_calculate_speed_ensemble():
    events1 = prepare_event_data()
    print(f'check condition: speed > 0 and speed <= 160 ')
    for event in events1:
        speed_list = calculate_speed_ensemble(event=event,
                                              lane_sensor=list(range(0, 11)),
                                              fiber_distance=2.5,
                                              sampling_rate=200)
        speed = np.median(speed_list)
        speed_std = np.std(speed_list)
        print(f'speed: {speed}')
        assert (speed > 0 and speed <= 160)
        print(f'speed std: {speed_std}')
        assert (speed_std <= 1.0)
