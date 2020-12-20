from contextlib import contextmanager

from matplotlib.figure import Figure
from .utils import _events_off, plt_show, _merge_annotations
from pydea.datamodels.datamodel import  Raw
import numpy as np
from ..annotations import _sync_onset
from ..utils import  Bunch

class FigureParams:
    def __init__(self, **kwargs):
        self.close_key = 'escape'
        vars(self).update(**kwargs)

class DriveEasyFigure(Figure):
    """Base class for 2D figures & dialogs; wraps matplotlib.figure.Figure."""

    def __init__(self, **kwargs):
        super().__init__()
        self.fig_params = FigureParams(**kwargs)

    def _close(self, event):
        pass

    def _keypress(self, event):
        """Handle keypress events."""
        if event.key == self.fig_params.close_key:
            from matplotlib.pyplot import close
            close(self)
        elif event.key == 'f11':  # full screen
            self.canvas.manager.full_screen_toggle()

    def _buttonpress(self, event):
        """Handle buttonpress events."""
        pass

    def _pick(self, event):
        """Handle matplotlib pick events."""
        pass

    def _resize(self, event):
        """Handle window resize events."""
        pass

    def _add_default_callbacks(self, **kwargs):
        """Remove some matplotlib default callbacks and add Python-DriveEasy ones."""
        # Remove matplotlib default keypress catchers
        default_callbacks = list(
            self.canvas.callbacks.callbacks.get('key_press_event', {}))
        for callback in default_callbacks:
            self.canvas.callbacks.disconnect(callback)
        # add our event callbacks
        callbacks = dict(resize_event=self._resize,
                         key_press_event=self._keypress,
                         button_press_event=self._buttonpress,
                         close_event=self._close,
                         pick_event=self._pick)
        callbacks.update(kwargs)
        callback_ids = dict()
        for event, callback in callbacks.items():
            callback_ids[event] = self.canvas.mpl_connect(event, callback)
        # store callback references so they aren't garbage-collected
        self.fig_params._callback_ids = callback_ids

    def _get_dpi_ratio(self):
        """Get DPI ratio (to handle hi-DPI screens)."""
        dpi_ratio = 1.
        for key in ('_dpi_ratio', '_device_scale'):
            dpi_ratio = getattr(self.canvas, key, dpi_ratio)
        return dpi_ratio

    def _get_size_px(self):
        """Get figure size in pixels."""
        dpi_ratio = self._get_dpi_ratio()
        return self.get_size_inches() * self.dpi / dpi_ratio

    def _inch_to_rel(self, dim_inches, horiz=True):
        """Convert inches to figure-relative distances."""
        fig_w, fig_h = self.get_size_inches()
        w_or_h = fig_w if horiz else fig_h
        return dim_inches / w_or_h

class DriveEasyAnnotationFigure(DriveEasyFigure):

    def _close(self, event):
        """Handle close events (via keypress or window [x])."""
        parent = self.fig_params.parent_fig
        # disable span selector
        parent.fig_params.ax_main.selector.active = False
        # clear hover line
        parent._remove_annotation_hover_line()
        # disconnect hover callback
        callback_id = parent.fig_params._callback_ids['motion_notify_event']
        parent.canvas.callbacks.disconnect(callback_id)
        # do all the other cleanup activities
        super()._close(event)

    def _keypress(self, event):
        text = self.label.get_text()
        key = event.key
        if key == self.fig_params.close_key:
            from matplotlib.pyplot import close
            close(self)
        elif key == 'backspace':
            text = text[:-1]
        elif key == 'enter':
            self.fig_params._add_annotation_label(event)
            return

        elif len(key) > 1 or key == ';': # ignore
            return
        else:
            text = text + key
        self.label.set_text(text)
        self.cavas.draw()

    def _radiopress(self, event):
        buttons = self.fig_params.radio_ax.buttons
        labels = [label.get_text() for label in buttons.labels]
        idx = labels.index(buttons.value_selected)
        self._set_active_button(idx)

        color = buttons.circles[idx].get_edgecolor()
        selector = self.fig_params.parent_fig.fig_params.ax_main.selector
        selector.rect.set_color(color)
        selector.rectprops.update(dict(facecolor=color))

    def _click_override(self, event):
        ax = self.fig_params.radio_ax
        buttons = ax.buttons
        if (buttons.ignore(event) or event.button != 1 or event.inaxes != ax):
            return
        pclicked = ax.transData.inverted().transform((event.x, event.y))
        distances = {}
        for i, (p,t) in enumerate(zip(buttons.circles, buttons.labels)):
            if (t.get_window_extent().contains(event.x, event.y) or np.linalg.norm(pclicked - p.center) < p.radius):
                distances[i] = np.linalg.norm(pclicked - p.center)
        if len(distances) > 0:
            closest = min(distances, key=distances.get)
            buttons.set_active(closest)

    def _set_active_button(self, idx):
        buttons = self.fig_params.radio_ax.buttons
        with _events_off(buttons):
            buttons.set_active(idx)
        for circle in buttons.circles:
            circle.set_facecolor(self.fig_params.parent_fig.fig_params.bgcolor)
        # active circle gets filled in, partially transparent
        color = list(buttons.circles[idx].get_edgecolor())
        color[-1] = 0.5
        buttons.circles[idx].set_facecolor(color)
        self.canvas.draw()

class DriveEasyBrowseFigure(DriveEasyFigure):

    def __init__(self, inst, figsize, xlabel='Time (s)', **kwargs):
        from matplotlib import rcParams
        from matplotlib.colors import to_rgba_array
        from matplotlib.ticker import (FixedLocator, FixedFormatter, FuncFormatter, NullFormatter)
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import Button
        from matplotlib.transforms import blended_transform_factory
        from mpl_toolkits.axes_grid1.axes_size import Fixed
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

        fgcolor = rcParams['axes.edgecolor']
        bgcolor = rcParams['axes.facecolor']

        super().__init__(figsize=figsize, inst=inst, **kwargs)

        if isinstance(inst, Raw):
            self.fig_params.instance_type = 'raw'
        else:
            raise TypeError(f'Expected an instance of Raw, got {type(inst)}.')

        # main ax
        l_margin = 1.
        r_margin = 0.1
        b_margin = 0.45
        t_margin = 0.25
        scroll_width = 0.25
        hscroll_dist = 0.25
        vscroll_dist = 0.1
        help_width = scroll_width * 2
        # MAIN AXES: default margins (figure-relative coordinates)
        left = self._inch_to_rel(l_margin - vscroll_dist - help_width)
        right = 1 - self._inch_to_rel(r_margin)
        bottom = self._inch_to_rel(b_margin, horiz=False)
        top = 1 - self._inch_to_rel(t_margin, horiz=False)
        width = right - left
        height = top - bottom
        position = [left, bottom, width, height]

        ax_main = self.add_subplot(1,1,1,position=position)
        self.subplotpars.update(left=left, bottom=bottom, top=top, right=right)
        div = make_axes_locatable(ax_main)

        # scroll bars
        ax_hscroll = div.append_axes(position='bottom',
                                     size=Fixed(scroll_width),
                                     pad=Fixed(hscroll_dist))
        ax_vscroll = div.append_axes(position='right',
                                     size=Fixed(scroll_width),
                                     pad=Fixed(vscroll_dist))
        ax_hscroll.get_yaxis().set_visible(False)
        ax_hscroll.set_xlabel(xlabel)
        ax_vscroll.set_axis_off()

        # VERTICAL SCROLLBAR PATCHES (COLORED BY LANE ID)
        ch_order = self.fig_params.ch_order
        for ix, pick in enumerate(ch_order):
            this_color = (self.fig_params.ch_color_bad
                          if self.fig_params.ch_names[pick] in self.fig_params.info['bads']
                          else self.fig_params.ch_color_dict)
            if isinstance(this_color, dict):
                this_color = this_color[self.fig_params.ch_types[pick]]
            ax_vscroll.add_patch(Rectangle((0,ix), 1, 1, color=this_color, zorder=self.fig_params.zorder['patch']))

        ax_vscroll.set_ylim(len(ch_order), 0)
        ax_vscroll.set_visible(not self.fig_params.butterfly) #TODO
        # SCROLLBAR VISIBLE SELECTION PATCHES
        sel_kwargs = dict(alpha=0.3, linewidth=4, clip_on=False, edgecolor=fgcolor)
        vsel_patch = Rectangle((0,0),1, self.fig_params.n_channels, facecolor=bgcolor, **sel_kwargs)
        ax_vscroll.add_patch(vsel_patch)
        hsel_facecolor = np.average(np.vstack((to_rgba_array(fgcolor),to_rgba_array(bgcolor))),axis=0, weights=(3,1)) # 75% foreground, 25% background
        hsel_patch = Rectangle((self.fig_params.t_start, 0), self.fig_params.duration, 1, facecolor=hsel_facecolor, **sel_kwargs)
        ax_hscroll.add_patch(hsel_patch)
        ax_hscroll.set_xlim(self.fig_params.first_time, self.fig_params.first_time + self.fig_params.n_times/self.fig_params.info['sampling_rate'])

        # VLINE
        vline_color = (0., 0.75, 0.)
        vline_kwargs = dict(visible=False, animated=True, zorder=self.fig_params.zorder['vline'])

        if self.fig_params.is_epochs:
            pass
        else:
            vline = ax_main.axvline(0, color=vline_color, **vline_kwargs)
            vline_hscroll = ax_hscroll.axvline(0, color=vline_color, **vline_kwargs)

        vline_text = ax_hscroll.text(self.fig_params.first_time, 1.2, '', fontsize=10, ha='right', va='bottom',
                                     color=vline_color, **vline_kwargs)

        # HELP button
        ax_help = div.append_axes(position='left', size=Fixed(help_width), pad=Fixed(vscroll_dist))

        # HELP BUTTON: ...move it down by changing its locator
        loc = div.new_locator(nx=0, ny=0)
        ax_help.set_axes_locator(loc)

        with _patched_canvas(ax_help.figure):
            self.fig_params.button_help = Button(ax_help, 'Help')

        # Proj
        # TODO

        # INIT TRACES
        self.fig_params.trace_kwargs = dict(antialiased=True, linewidth=0.5)
        self.fig_params.traces = ax_main.plot(np.full((1, self.fig_params.n_channels), np.nan), **self.fig_params.trace_kwargs)

        # SAVE UI ELEM HANDLES

        vars(self.fig_params).update(
            ax_main=ax_main, ax_help=ax_help, ax_hscroll=ax_hscroll,ax_vscroll=ax_vscroll,
            vsel_patch=vsel_patch, hsel_patch=hsel_patch, vline=vline, vline_hscroll=vline_hscroll,
            vline_text=vline_text, fgcolor=fgcolor, bgcolor=bgcolor)

    def _close(self, event):
        from matplotlib.pyplot import close
        if self.fig_params.instance_type in ['Raw','Event']:
            self.fig_params.inst.info['bads'] = self.fig_params.info['bads']

        size = ','.join(self.get_size_inches().astype(str))
        # TODO
        # set_config('BROWSE_RAW_SIZE', size, set_env=False)
        while len(self.fig_params.child_figs):
            fig = self.fig_params.child_figs[-1]
            close(fig)

    def _hover(self,event):
        # TODO
        pass

    def _keypress(self, event):
        key = event.key
        n_channels = self.fig_params.n_channels
        if self.fig_params.is_epochs:
            # TODO
            pass

        else:
            last_time = self.fig_params.times[-1]

        # scroll up down
        # TODO
        if key == 'a': # annotation mode
            self._toggle_annotation_fig()
        else:
            super._keypress(event)

    def _new_child_figure(self, fig_name, **kwargs):
        """Instantiate a new MNE dialog figure (with event listeners)."""
        fig = create_figure(toolbar=False, parent_fig=self, fig_name=fig_name,
                      **kwargs)
        fig._add_default_callbacks()
        self.fig_params.child_figs.append(fig)
        if isinstance(fig_name, str):
            setattr(self.mne, fig_name, fig)
        return fig

    def _toggle_annotation_fig(self):
        if self.fig_params.fig_annotation is None:
            self._create_annotation_fig()
        else:
            from matplotlib.pyplot import close
            close(self.fig_params.fig_annotation)

    def _create_annotation_fig(self):
        """Create the annotation dialog window."""
        from matplotlib.widgets import  Button, SpanSelector, CheckButtons
        from mpl_toolkits.axes_grid1.axes_size import Fixed
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

        # make figure
        labels = np.array(sorted(set(self.fig_params.inst.annotations.description)))
        width, var_height, fixed_height, pad = self._compute_annotation_figsize(len(labels))
        figsize = (width, var_height + fixed_height)
        fig = self._new_child_figure(figsize=figsize,
                                     FigureClass=DriveEasyAnnotationFigure,
                                     fig_name='fig_annotation',
                                     window_title='Fibridge DriveEasy Annotations')

        # make main ax
        left = fig._inch_to_rel(pad)
        bottom = fig._inch_to_rel(pad, horiz=False)
        width = 1 - 2*left
        height = 1 - 2*bottom
        fig.fig_params.radio_ax = fig.add_axes((left,bottom, width, height), frame_on=False, aspect='equal')
        div = make_axes_locatable(fig.fig_params.radio_ax)
        self._update_annotation_fig() # create butions and labels
        # append instructions
        instructions_ax = div.append_axes(position='top', size=Fixed(1), pad=Fixed(5*pad))
        # instructions = '\n'.join(
        #     ['Left click & drag on plot: create/modify annotation',
        #      'Right click on plot annotation: delete annotation',
        #      'Type in annotation window: modify new label name',
        #      'Enter (or click button): add new label to list',
        #      'Esc: exit annotation mode & close window'])

        # only works for matplotlib version >= 3.x
        instructions = '\n'.join(
            [r'$\mathbf{Left‐click~&~drag~on~plot:}$ create/modify annotation',  # noqa E501
             r'$\mathbf{Right‐click~on~plot~annotation:}$ delete annotation',
             r'$\mathbf{Type~in~annotation~window:}$ modify new label name',
             r'$\mathbf{Enter~(or~click~button):}$ add new label to list',
             r'$\mathbf{Esc:}$ exit annotation mode & close window'])
        instructions_ax.test(0, 1, instructions, va='top', ha='left', usetex=False)
        instructions_ax.set_axis_off()
        #append text entry axis at bottom
        text_entry_ax = div.append_axes(position='bottom', size=Fixed(3 *pad), pad= Fixed(pad))
        text_entry_ax.text(0.4, 0.5, 'New label:', va='center', ha='right', weight='bold')
        fig.label = text_entry_ax.text(0.5, 0.5, 'Truck_AX9', va='center', ha='left')
        text_entry_ax.set_axis_off()

        #append button at bottom
        button_ax = div.append_axes(position='bottom', size=Fixed(3*pad), pad=Fixed(pad))
        fig.button = Button(button_ax, 'Add new labels')
        fig.button.on_clicked(self._add_annotation_labels)
        plt_show(fig=fig)

        # add "draggable" checkbox
        drag_ax_height = 3 *pad
        drag_ax = div.append_axes(position='bottom', size=Fixed(drag_ax_height), pad=Fixed(pad), aspect='equal')
        checkbox = CheckButtons(drag_ax, labels=('Draggable edges?',), actives=(self.fig_params.draggable_annotations,))
        checkbox.on_clicked(self._toggle_draggable_annotations)
        fig.fig_params.drag_checkbox = checkbox
        # reposition & resize axes
        # TODO

        # setup intractivity in plot
        col = ('#ff0000' if len(fig.fig_params.radio_ax.buttons.circles) < 1 else fig.fig_params.radio_ax.buttons.circles[0].get_edgecolor())
        selector = SpanSelector(self.fig_params.ax_main, self._select_annotation_span, direction='horizontal', minspan=0.1,
                                useblit=False, rectprops=dict(alpha=0.5, facecolor=col))
        self.fig_params.ax_main.selector = selector
        self.fig_params._callback_ids['motion_notify_event'] = self.canvas.mpl_connect('motion_notify_event', self._hover)

    def _toggle_draggable_annotations(self, event):
        """Enable/disable draggable annotation edges."""
        self.fig_params.draggable_annotations = not self.fig_params.draggable_annotations

    def _update_annotation_fig(self):
        """Draw or redraw the radio buttons and annotation labels."""
        from matplotlib.widgets import RadioButtons
        fig = self.fig_params.fig_annotation
        ax = fig.fig_params.radio_ax
        labels = list(set(self.fig_params.inst.annotations.description))
        labels = np.union1d(labels, self.fig_params.new_annotation_labels)
        # compute new figsize
        width, var_height, fixed_height, pad = \
            self._compute_annotation_figsize(len(labels))
        fig.set_size_inches(width, var_height + fixed_height, forward=True)
        ax.clear()
        title = 'Existing labels:' if len(labels) else 'No existing labels'
        ax.set_title(title, size=None, loc='left')
        ax.buttons = RadioButtons(ax, labels)
        aspect = (width - 2*pad) / var_height
        ax.set_xlim((0, aspect))
        # style the buttons & adjust spacing
        radius = 0.15
        circles = ax.buttons.circles
        for circle, label in zip(circles, ax.buttons.labels):
            circle.set_transform(ax.transData)
            center = ax.transData.inverted().transform(
                ax.transAxes.transform((0.1, 0)))
            # XXX older MPL doesn't have circle.set_center
            circle.center = (center[0], circle.center[1])
            circle.set_edgecolor(
                self.mne.annotation_segment_colors[label.get_text()])
            circle.set_linewidth(4)
            circle.set_radius(radius / len(labels))
        # style the selected button
        if len(labels):
            fig._set_active_button(0)
        # add event listeners
        ax.buttons.disconnect_events()  # clear MPL default listeners
        ax.buttons.on_clicked(fig._radiopress)
        ax.buttons.connect_event('button_press_event', fig._click_override)

    def _compute_annotation_figsize(self, n_labels):
        """Adapt size of Annotation UI to accommodate the number of buttons.

        self._create_annotation_fig() implements the following:

        Fixed part of height:
        0.1  top margin
        1.0  instructions
        0.5  padding below instructions
        ---  (variable-height axis for label list)
        0.1  padding above text entry
        0.3  text entry
        0.1  padding above button
        0.3  button
        0.1  padding above checkbox
        0.3  checkbox
        0.1  bottom margin
        ------------------------------------------
        2.9  total fixed height
        """
        pad = 0.1
        width = 4.5
        var_height = max(pad, 0.7 * n_labels)
        fixed_height = 2.9
        return (width, var_height, fixed_height, pad)

    def _select_annotation_span(self, vmin, vmax):
        """Handle annotation span selector."""
        onset = _sync_onset(self.fig_params.inst, vmin, True) - self.fig_params.first_time
        duration = vmax - vmin
        buttons = self.mne.fig_annotation.mne.radio_ax.buttons
        labels = [label.get_text() for label in buttons.labels]
        active_idx = labels.index(buttons.value_selected)
        _merge_annotations(onset, onset + duration, labels[active_idx],
                           self.mne.inst.annotations)
        self._draw_annotations()
        self.canvas.draw_idle()


def create_figure(toolbar=True, FigureClass = DriveEasyFigure, **kwargs):
    """Instantiate a new figure."""
    from matplotlib import rc_context
    from matplotlib.pyplot import figure
    rc = dict() if toolbar else dict(toolbar='none')
    with rc_context(rc=rc):
        fig = figure(FigureClass=FigureClass, **kwargs)
    # TODO:set window title
    return fig


def create_browse_figure(inst, **kwargs):
    fig = create_figure(inst=inst, toolbar=False, FigureClass=DriveEasyBrowseFigure, **kwargs)
    fig.canvas.draw()
    fig.fig_params.fig_size_px = fig._get_size_px()
    fig.fig_params.zen_w = (fig.fig_params.ax_vscroll.get_position().xmax - fig.fig_params.ax_main.get_position().xmax)
    fig.fig_params.zen_h = (fig.fig_params.ax_main.get_position().ymin - fig.fig_params.ax_hscroll.get_position().ymin)
    # if scrollbars are supposed to start hidden, set to True and then toggle
    if not fig.fig_params.scrollbars_visible:
        fig.fig_params.scrollbars_visible = True
        fig._toggle_scrollbars()
    # add event callbacks
    fig._add_default_callbacks()
    return fig


@contextmanager
def _patched_canvas(fig):
    old_canvas = fig.canvas
    if fig.canvas is None:  # XXX old MPL (at least 3.0.3) does this for Agg
        fig.canvas = Bunch(mpl_connect=lambda event, callback: None)
    try:
        yield
    finally:
        fig.canvas = old_canvas

if __name__ == '__main__':
    from pydea.datamodels.datamodel import Raw
    from pathlib import Path
    raw_file = Path(r'C:\Users\hyu\gitlab_repos\driveeasy-analytics\data\M80\333Hz\raw_2020-12-12_19-00-24.gzip')
    raw = Raw(raw_file)
    params = dict(inst=raw)
    fig = create_browse_figure(**params)
    