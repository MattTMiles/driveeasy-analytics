#TODO
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import os.path as op
import re
from copy import deepcopy
from itertools import takewhile
from collections import Counter
from collections.abc import Iterable
import warnings
from textwrap import shorten

from datetime import datetime, timezone, timedelta

# pydea modules
from pydea.utils.check import _check_dt
from pydea.utils.time_utils import _stamp_to_dt
from pydea.utils.misc import _plural


class Annotations:
    def __init__(self, onset, duration, description, orig_time=None):
        self._orig_time = _handle_meas_date(orig_time)
        self.onset, self.duration, self.description = _check_annotation_init(onset,duration,description)
        self._sort()

    def copy(self):
        """The time base of the Annotations."""
        return deepcopy(self)

    def __eq__(self, other):
        """Compare to another Annotations instance."""
        if not isinstance(other, Annotations):
            return False
        return (np.array_equal(self.onset, other.onset) and
                np.array_equal(self.duration, other.duration) and
                np.array_equal(self.description, other.description) and
                self.orig_time == other.orig_time)

    def __repr__(self):
        """Show the representation."""
        counter = Counter(self.description)
        kinds = ', '.join(['%s (%s)' % k for k in sorted(counter.items())])
        kinds = (': ' if len(kinds) > 0 else '') + kinds
        s = ('Annotations | %s segment%s%s' %
             (len(self.onset), _plural(len(self.onset)), kinds))
        return '<' + shorten(s, width=77, placeholder=' ...') + '>'

    def __len__(self):
        """Return the number of annotations."""
        return len(self.duration)

    def _sort(self):
        """Sort in place."""
        # instead of argsort here we use sorted so that it gives us
        # the onset-then-duration hierarchy
        vals = sorted(zip(self.onset, self.duration, range(len(self))))
        order = list(list(zip(*vals))[-1]) if len(vals) else []
        self.onset = self.onset[order]
        self.duration = self.duration[order]
        self.description = self.description[order]






def read_annotations():
    pass

def annotations_from_events():
    pass

def events_from_annotations():
    pass

def _sync_onset(raw, onset, inverse=False):
    """Adjust onsets in relation to raw data."""
    offset = (-1 if inverse else 1) * raw._first_time
    assert raw.info['meas_date'] == raw.annotations.orig_time
    annot_start = onset - offset
    return annot_start

def _handle_meas_date(meas_date):
    """Convert meas_date to datetime or None.

    If `meas_date` is a string, it should conform to the ISO8601 format.
    More precisely to this '%Y-%m-%d %H:%M:%S.%f' particular case of the
    ISO8601 format where the delimiter between date and time is ' '.
    Note that ISO8601 allows for ' ' or 'T' as delimiters between date and
    time.
    """
    if isinstance(meas_date, str):
        ACCEPTED_ISO8601 = '%Y-%m-%d %H:%M:%S.%f'
        try:
            meas_date = datetime.strptime(meas_date, ACCEPTED_ISO8601)
        except ValueError:
            meas_date = None
        else:
            meas_date = meas_date.replace(tzinfo=timezone.utc)
    elif isinstance(meas_date, tuple):
        # old way
        meas_date = _stamp_to_dt(meas_date)

    if meas_date is not None:
        if np.isscalar(meas_date):
            # It would be nice just to do:
            #
            #     meas_date = datetime.fromtimestamp(meas_date, timezone.utc)
            #
            # But Windows does not like timestamps < 0. So we'll use
            # our specialized wrapper instead:
            meas_date = np.array(np.modf(meas_date)[::-1])
            meas_date *= [1, 1e6]
            meas_date = _stamp_to_dt(np.round(meas_date))
        _check_dt(meas_date)  # run checks
    return meas_date

def _check_annotation_init(onset, duration, description):
    onset = np.atleast_1d(np.array(onset, dtype=float))
    if onset.ndim != 1:
        raise ValueError('Onset must be a one dimensional array, got %s '
                         '(shape %s).'
                         % (onset.ndim, onset.shape))
    duration = np.array(duration, dtype=float)
    if duration.ndim == 0 or duration.shape == (1,):
        duration = np.repeat(duration, len(onset))
    if duration.ndim != 1:
        raise ValueError('Duration must be a one dimensional array, '
                         'got %d.' % (duration.ndim,))

    description = np.array(description, dtype=str)
    if description.ndim == 0 or description.shape == (1,):
        description = np.repeat(description, len(onset))
    if description.ndim != 1:
        raise ValueError('Description must be a one dimensional array, '
                         'got %d.' % (description.ndim,))
    if any(['{COLON}' in desc for desc in description]):
        raise ValueError('The substring "{COLON}" '
                         'in descriptions not supported.')

    if not (len(onset) == len(duration) == len(description)):
        raise ValueError('Onset, duration and description must be '
                         'equal in sizes, got %s, %s, and %s.'
                         % (len(onset), len(duration), len(description)))
    return onset, duration, description
