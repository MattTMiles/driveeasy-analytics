from pydea.datamodels.datamodel import Event
# from ..data_utils import prepare_event_data
from pathlib import Path
import numpy as np
from pydea.algorithms.count_axles.count_axles_qc import calculate_axle_number_distance

def prepare_event_data():
    event_dir = Path(r'..\test_data\events')
    files = list(event_dir.glob('evt*.npz'))
    events = []
    for file in files:
        events.append(Event(file))
    return events
#
# # events = prepare_event_data()

def test_calculate_axle_number_distance():
    events = prepare_event_data()
    print(len(events))
    event0 = events[0]
    axle_count, axle_distances, num_groups = calculate_axle_number_distance(event=event0, lane_sensor=list(range(2, 15)))
    print(axle_count)
    print(axle_distances)
    print(f'check condtion 1: axle_count>=2 and axle_count<=13 ')
    print(f'check condiction 2: np.min(axle_distances) >=0.2 and np.max(axle_distances)<=12')
    assert (axle_count>=2 and axle_count<=13)
    assert (np.min(axle_distances) >=0.2 and np.max(axle_distances)<=12)

