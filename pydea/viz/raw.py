from ..defaults import _handle_default

def plot_raw(raw, events=None, sensors_to_plot=None, color=None, show_scrollbars=True, show_scalebars=True,
             duration=10):

    from .figureX import create_browse_figure

    info = raw.info.copy()
    sampling_rate = info['sampling_rate']
    color = _handle_default('color', color)
    scalings = 1 # TODO

    duration = min(raw.times[-1], float(duration))
    ch_names =
    ch_types = list(('lane_1','lane_2','lane_3','lane_4','lane_5'))
    ch_order = list(('lane_1','lane_2','lane_3','lane_4','lane_5'))

    ch_names = raw.sensor_names



    # duration = min(raw.timestamp[-1], float(duration))
    title = 'DriveEasy Plot'

    params = dict(inst=raw,
                  info=info,
                  ch_names=ch_names,
                  ch_types=ch_types,
                  ch_order=ch_order,
                  picks=order[:n_channels],
                  n_channels=n_channels,
                  picks_data=picks_data,
                  group_by=group_by,
                  ch_selections=selections,
                  t_start=start,
                  duration=duration,
                  n_times=raw.n_times,
                  first_time=first_time,
                  decim=decim,
                  # events
                  event_color_dict=event_color_dict,
                  event_times=event_times,
                  event_nums=event_nums,
                  event_id_rev=event_id_rev,
                  # preprocessing
                  projs=projs,
                  projs_on=projs_on,
                  apply_proj=proj,
                  remove_dc=remove_dc,
                  filter_coefs=ba,
                  filter_bounds=filt_bounds,
                  noise_cov=noise_cov,
                  # scalings
                  scalings=scalings,
                  units=units,
                  unit_scalings=unit_scalings,
                  # colors
                  ch_color_bad=bad_color,
                  ch_color_dict=color,
                  # display
                  butterfly=butterfly,
                  clipping=clipping,
                  scrollbars_visible=show_scrollbars,
                  scalebars_visible=show_scalebars,
                  window_title=title)



    fig = create_browse_figure(**params)
    fig._update_picks()

    return fig
