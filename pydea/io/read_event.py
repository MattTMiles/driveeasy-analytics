import numpy as np
from ..datamodels.event import EventData

def read_event(filename):
    if not str(filename).endswith('.npz'):
        raise TypeError('Event filename not ended with .npz')
    data = np.load(filename, allow_pickle=True)
    event = EventData()
    event.from_dict(data)
    return EventData(filename)