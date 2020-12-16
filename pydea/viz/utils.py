from collections import defaultdict
from contextlib import contextmanager
from functools import partial
import math
import numpy as np
from copy import deepcopy
from distutils.version import LooseVersion
import warnings
from loguru import logger

@contextmanager
def _events_off(obj):
    obj.eventson = False
    try:
        yield
    finally:
        obj.eventson = True

def plt_show(show=True, fig=None, **kwargs):
    """Show a figure while suppressing warnings.

    Parameters
    ----------
    show : bool
        Show the figure.
    fig : instance of Figure | None
        If non-None, use fig.show().
    **kwargs : dict
        Extra arguments for :func:`matplotlib.pyplot.show`.
    """
    from matplotlib import get_backend
    import matplotlib.pyplot as plt
    if show and get_backend() != 'agg':
        (fig or plt).show(**kwargs)

def _merge_annotations(start, stop, description, annotations, current=()):
    """Handle drawn annotations."""
    ends = annotations.onset + annotations.duration
    idx = np.intersect1d(np.where(ends >= start)[0],
                         np.where(annotations.onset <= stop)[0])
    idx = np.intersect1d(idx,
                         np.where(annotations.description == description)[0])
    new_idx = np.setdiff1d(idx, current)  # don't include modified annotation
    end = max(np.append((annotations.onset[new_idx] +
                         annotations.duration[new_idx]), stop))
    onset = min(np.append(annotations.onset[new_idx], start))
    duration = end - onset
    annotations.delete(idx)
    annotations.append(onset, duration, description)


def tight_layout(pad=1.2, h_pad=None, w_pad=None, fig=None):
    """Adjust subplot parameters to give specified padding.

    .. note:: For plotting please use this function instead of
              ``plt.tight_layout``.

    Parameters
    ----------
    pad : float
        Padding between the figure edge and the edges of subplots, as a
        fraction of the font-size.
    h_pad : float
        Padding height between edges of adjacent subplots.
        Defaults to ``pad_inches``.
    w_pad : float
        Padding width between edges of adjacent subplots.
        Defaults to ``pad_inches``.
    fig : instance of Figure
        Figure to apply changes to.

    Notes
    -----
    This will not force constrained_layout=False if the figure was created
    with that method.
    """
    import matplotlib.pyplot as plt
    fig = plt.gcf() if fig is None else fig

    fig.canvas.draw()
    constrained = fig.get_constrained_layout()
    if constrained:
        return  # no-op
    try:  # see https://github.com/matplotlib/matplotlib/issues/2654
        with warnings.catch_warnings(record=True) as ws:
            fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    except Exception:
        try:
            with warnings.catch_warnings(record=True) as ws:
                fig.set_tight_layout(dict(pad=pad, h_pad=h_pad, w_pad=w_pad))
        except Exception:
            logger.warn('Matplotlib function "tight_layout" is not supported.'
                 ' Skipping subplot adjustment.')
            return
    for w in ws:
        w_msg = str(w.message) if hasattr(w, 'message') else w.get_message()
        if not w_msg.startswith('This figure includes Axes'):
            logger.warning(w_msg, w.category, 'matplotlib')


def _compute_scalings(scalings, inst, remove_dc=False, duration=10):
    """Compute scalings for each channel type automatically.

    Parameters
    ----------
    scalings : dict
        The scalings for each channel type. If any values are
        'auto', this will automatically compute a reasonable
        scaling for that channel type. Any values that aren't
        'auto' will not be changed.
    inst : instance of Raw or Epochs
        The data for which you want to compute scalings. If data
        is not preloaded, this will read a subset of times / epochs
        up to 100mb in size in order to compute scalings.
    remove_dc : bool
        Whether to remove the mean (DC) before calculating the scalings. If
        True, the mean will be computed and subtracted for short epochs in
        order to compensate not only for global mean offset, but also for slow
        drifts in the signals.
    duration : float
        If remove_dc is True, the mean will be computed and subtracted on
        segments of length ``duration`` seconds.

    Returns
    -------
    scalings : dict
        A scalings dictionary with updated values
    """
    from ..io.base import BaseRaw
    from ..epochs import BaseEpochs
    scalings = _handle_default('scalings_plot_raw', scalings)
    if not isinstance(inst, (BaseRaw, BaseEpochs)):
        raise ValueError('Must supply either Raw or Epochs')

    ch_types = channel_indices_by_type(inst.info)
    ch_types = {i_type: i_ixs
                for i_type, i_ixs in ch_types.items() if len(i_ixs) != 0}
    scalings = deepcopy(scalings)

    if inst.preload is False:
        if isinstance(inst, BaseRaw):
            # Load a window of data from the center up to 100mb in size
            n_times = 1e8 // (len(inst.ch_names) * 8)
            n_times = np.clip(n_times, 1, inst.n_times)
            n_secs = n_times / float(inst.info['sfreq'])
            time_middle = np.mean(inst.times)
            tmin = np.clip(time_middle - n_secs / 2., inst.times.min(), None)
            tmax = np.clip(time_middle + n_secs / 2., None, inst.times.max())
            data = inst._read_segment(tmin, tmax)
        elif isinstance(inst, BaseEpochs):
            # Load a random subset of epochs up to 100mb in size
            n_epochs = 1e8 // (len(inst.ch_names) * len(inst.times) * 8)
            n_epochs = int(np.clip(n_epochs, 1, len(inst)))
            ixs_epochs = np.random.choice(range(len(inst)), n_epochs, False)
            inst = inst.copy()[ixs_epochs].load_data()
    else:
        data = inst._data
    if isinstance(inst, BaseEpochs):
        data = inst._data.swapaxes(0, 1).reshape([len(inst.ch_names), -1])
    # Iterate through ch types and update scaling if ' auto'
    for key, value in scalings.items():
        if key not in ch_types:
            continue
        if not (isinstance(value, str) and value == 'auto'):
            try:
                scalings[key] = float(value)
            except Exception:
                raise ValueError(
                    f'scalings must be "auto" or float, got scalings[{key!r}]='
                    f'{value!r} which could not be converted to float')
            continue
        this_data = data[ch_types[key]]
        if remove_dc and (this_data.shape[1] / inst.info["sfreq"] >= duration):
            length = int(duration * inst.info["sfreq"])  # segment length
            # truncate data so that we can divide into segments of equal length
            this_data = this_data[:, :this_data.shape[1] // length * length]
            shape = this_data.shape  # original shape
            this_data = this_data.T.reshape(-1, length, shape[0])  # segment
            this_data -= np.nanmean(this_data, 0)  # subtract segment means
            this_data = this_data.T.reshape(shape)  # reshape into original
        this_data = this_data.ravel()
        this_data = this_data[np.isfinite(this_data)]
        if this_data.size:
            iqr = np.diff(np.percentile(this_data, [25, 75]))[0]
        else:
            iqr = 1.
        scalings[key] = iqr
    return scalings


def _get_channel_plotting_order(order, ch_types, picks=None):
    """Determine channel plotting order for browse-style Raw/Epochs plots."""
    # TODO
    if order is None:
        # for backward compat, we swap the first two to keep grad before mag
        ch_type_order = list(('lane_1','lane_2','lane_3','lane_4','lane_5'))
        # ch_type_order = tuple(['grad', 'mag'] + ch_type_order[2:])
        order = [pick_idx for order_type in ch_type_order
                 for pick_idx, pick_type in enumerate(ch_types)
                 if order_type == pick_type]
    elif not isinstance(order, (np.ndarray, list, tuple)):
        raise ValueError('order should be array-like; got '
                         f'"{order}" ({type(order)}).')
    if picks is not None:
        order = [ch for ch in order if ch in picks]
    return np.asarray(order)