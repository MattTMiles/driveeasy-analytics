from pydea.datamodels.datamodel import Event
from pathlib import Path

import numpy as np


def prepare_event_data(event_dir=Path(r'.\test_data\events')):
    files = list(event_dir.glob('evt*.npz'))
    events = []
    for file in files:
        events.append(Event(file))
    return events