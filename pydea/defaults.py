# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy
from datetime import timedelta

M80_TIME_SHIFT_TO_UTC = timedelta(hours=3,minutes=53,seconds=9) #M80
Francis_TIME_SHIFT_TO_UTC = timedelta(hours=0,minutes=57,seconds=52) #Francis st.


DEFAULTS = dict(
    color=dict(raw='darkblue', wls='b', wav='k', eog='k', ecg='m', emg='k',
               ref_meg='steelblue', misc='k', stim='k', resp='k', chpi='k',
               exci='k', ias='k', syst='k', seeg='saddlebrown', dipole='k',
               gof='k', bio='k', ecog='k', hbo='#AA3377', hbr='b',
               fnirs_cw_amplitude='k', fnirs_od='k', csd='k'),
    units=dict(wls='pm', wav='nm'),
    scalings=dict(wls=1, wav=1), # scalings for the units

    scalings_plot_raw=dict(wls=100, wav=1000), # rough guess for a good plot
    scalings_cov_rank=dict(wls=1,wav=1),
    ylim=dict(raw=(1500,1600), wls=(-100,50), wav=(1500,1600)),

    titles=dict(raw='Raw Data', wls='Wavelength shift', wav='Wavelength'),
    mask_params=dict(marker='o',
                     markerfacecolor='w',
                     markeredgecolor='k',
                     linewidth=0,
                     markeredgewidth=1,
                     markersize=4),
    coreg=dict(
        mri_fid_opacity=1.0,
        dig_fid_opacity=0.3,

        mri_fid_scale=1e-2,
        dig_fid_scale=3e-2,
        extra_scale=4e-3,
        eeg_scale=4e-3, eegp_scale=20e-3, eegp_height=0.1,
        ecog_scale=5e-3,
        seeg_scale=5e-3,
        fnirs_scale=5e-3,
        source_scale=5e-3,
        detector_scale=5e-3,
        hpi_scale=15e-3,

        head_color=(0.988, 0.89, 0.74),
        hpi_color=(1., 0., 1.),
        extra_color=(1., 1., 1.),
        eeg_color=(1., 0.596, 0.588), eegp_color=(0.839, 0.15, 0.16),
        ecog_color=(1., 1., 1.),
        seeg_color=(1., 1., .3),
        fnirs_color=(1., .647, 0.),
        source_color=(1., .05, 0.),
        detector_color=(.3, .15, .15),
        lpa_color=(1., 0., 0.),
        nasion_color=(0., 1., 0.),
        rpa_color=(0., 0., 1.),
    ),
    noise_std=dict(grad=5e-13, mag=20e-15, eeg=0.2e-6),
    eloreta_options=dict(eps=1e-6, max_iter=20, force_equal=False),
    depth_mne=dict(exp=0.8, limit=10., limit_depth_chs=True,
                   combine_xyz='spectral', allow_fixed_depth=False),
    depth_sparse=dict(exp=0.8, limit=None, limit_depth_chs='whiten',
                      combine_xyz='fro', allow_fixed_depth=True),
    interpolation_method=dict(eeg='spline', meg='MNE', fnirs='nearest'),
    volume_options=dict(
        alpha=None, resolution=1., surface_alpha=None, blending='mip',
        silhouette_alpha=None, silhouette_linewidth=2.),
)


def _handle_default(k, v=None):
    """Avoid dicts as default keyword arguments.

    Use this function instead to resolve default dict values. Example usage::

        scalings = _handle_default('scalings', scalings)

    """
    this_mapping = deepcopy(DEFAULTS[k])
    if v is not None:
        if isinstance(v, dict):
            this_mapping.update(v)
        else:
            for key in this_mapping:
                this_mapping[key] = v
    return this_mapping


HEAD_SIZE_DEFAULT = 0.095  # in [m]
_BORDER_DEFAULT = 'mean'
_EXTRAPOLATE_DEFAULT = 'auto'
