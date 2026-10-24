from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.special import erf
from astropy import units as u


def find_signal_in_cube(
        cube, noisecube, mask=None,
        nchan_hi=3, snr_hi=3.5, nchan_lo=2, snr_lo=2,
        prune_by_npix=None, prune_by_fracbeam=1.,
        expand_by_npix=None, expand_by_fracbeam=0.,
        expand_by_nchan=2, verbose=False):
    """
    Identify (positive) signal in a cube based on S/N ratio.

    This is a revised version of the signal identification scheme
    described in Sun et al. (2018, ApJ, 860, 172).

    Here is a description of the scheme:
    1. Generate a 'core mask' by identifying detections with
       S/N >= `snr_hi` over at least `nchan_hi` consecutive channels.
       (if a user-specified mask exists, do this step inside the mask)
    2. Generate a 'wing mask' by identifying detections with
       S/N >= `snr_lo` over at least `nchan_lo` consecutive channels.
       (if a user-specified mask exists, do this step inside the mask)
    3. Dilate the 'core mask' inside the 'wing mask' to get
       a 'signal mask' that defines 'detections'.
    4. Label 'detections' by connectivity, and prune 'detections'
       in the 'signal mask' if projected area on sky smaller than
       a given number of pixels or a fraction of the beam area.
       (fraction specified by `prune_by_npix` or `prune_by_fracbeam`)
    5. Expand the 'signal mask' along the spatial dimensions by
       a given number of pixels or a fraction of the beam FWHM.
       (fraction specified by `expand_by_npix` or `expand_by_fracbeam`)
    6. Expand the 'signal mask' along the spectral dimension by
       a given number of channels.
       (# of channels specified by `expand_by_nchan`)

    Parameters
    ----------
    cube : SpectralCube object
        Input spectral cube
    noisecube : SpectralCube object
        Estimated rms noise cube
    mask : `np.ndarray` object, optional
        User-specified mask, within which all steps are performed
    nchan_hi : int, optional
        # of consecutive channels specified for the 'core mask'
        (Default: 3)
    snr_hi : float, optional
        S/N threshold specified for the 'core mask'
        (Default: 3.5)
    nchan_lo : int, optional
        # of consecutive channels specified for the 'wing mask'
        (Default: 2)
    snr_lo : float, optional
        S/N threshold specified for the 'wing mask'
        (Default: 2.0)
    prune_by_npix : int, optional
        Threshold for pruning. All detections with projected area
        smaller than this number of pixels will be pruned.
        If not None, `prune_by_fracbeam` will be ignored.
        (Default: None)
    prune_by_fracbeam : float, optional
        Threshold for pruning. All detections with projected area
        smaller than this threshold times the beam area will be pruned.
        (Default: 1.0)
    expand_by_npix : int, optional
        Expand the signal mask along the spatial dimensions by this
        number of pixels. If not None, `expand_by_fracbeam` will be
        ignored. (Default: None)
    expand_by_fracbeam : float, optional
        Expand the signal mask along the spatial dimensions by this
        fraction times the beam HWHM.
        (Default: 0.0)
    expand_by_nchan : int, optional
        Expand the signal mask along the spectral dimensions by this
        number of channels.
        (Default: 2)
    verbose : bool, optional
        Whether to print the detailed processing information
        Default is to not print.

    Returns
    -------
    outcube : SpectralCube object
        Input cube masked by the generated 'signal mask'.
    """

    if not cube.unit.is_equivalent(noisecube.unit):
        raise ValueError(
            "Incompatable units between 'cube' and 'noisecube'!")

    snr = cube.filled_data[:] / noisecube.filled_data[:]
    snr = snr.to(u.dimensionless_unscaled).value
    if mask is None:
        mask = np.ones_like(snr).astype('?')
    
    # generate core mask
    if verbose:
        print(
            "Generating core mask (S/N >= {} over {} consecutive "
            "channels)...".format(snr_hi, nchan_hi))
    mask_core = (snr > snr_hi)
    for iiter in range(nchan_hi-1):
        mask_core &= np.roll(mask_core, 1, 0)
    mask_core[:nchan_hi-1, :] = False
    for iiter in range(nchan_hi-1):
        mask_core |= np.roll(mask_core, -1, 0)
    mask_core &= mask

    # generate wing mask
    if verbose:
        print(
            "Generating wing mask (S/N >= {} over {} consecutive "
            "channels)...".format(snr_lo, nchan_lo))
    mask_wing = snr > snr_lo
    for iiter in range(nchan_lo-1):
        mask_wing &= np.roll(mask_wing, 1, 0)
    mask_wing[:nchan_lo-1, :] = False
    for iiter in range(nchan_lo-1):
        mask_wing |= np.roll(mask_wing, -1, 0)
    mask_wing &= mask
    
    # dilate core mask inside wing mask
    if verbose:
        print("Dilating core mask inside wing mask...")
    mask_signal = binary_dilation(
        mask_core, iterations=0, mask=mask_wing)

    # prune detections with small projected area on the sky
    if (prune_by_fracbeam > 0) or (prune_by_npix is not None):
        if verbose:
            print("Pruning detections with small projected area")
        if prune_by_npix is None:
            beamarea_pix = np.abs(
                cube.beam.sr.to(u.deg**2).value /
                cube.wcs.celestial.wcs.cdelt.prod())
            prune_by_npix = beamarea_pix * prune_by_fracbeam
        labels, count = label(mask_signal)
        for ind in np.arange(count)+1:
            if (labels == ind).any(axis=0).sum() < prune_by_npix:
                mask_signal[labels == ind] = False

    # expand along spatial dimensions by a fraction of beam FWHM
    if (expand_by_fracbeam > 0) or (expand_by_npix is not None):
        if verbose:
            print("Expanding signal mask along spatial dimensions")
        if expand_by_npix is None:
            beamHWHM_pix = np.ceil(
                cube.beam.major.to(u.deg).value / 2 /
                np.abs(cube.wcs.celestial.wcs.cdelt.min()))
            expand_by_npix = int(beamHWHM_pix * expand_by_fracbeam)
        structure = np.zeros(
            [3, expand_by_npix*2+1, expand_by_npix*2+1])
        Y, X = np.ogrid[:expand_by_npix*2+1, :expand_by_npix*2+1]
        R = np.sqrt((X - expand_by_npix)**2 + (Y-expand_by_npix)**2)
        structure[1, :] = (R <= expand_by_npix)
        mask_signal = binary_dilation(
            mask_signal, iterations=1, structure=structure, mask=mask)

    # expand along spectral dimension by a number of channels
    if expand_by_nchan > 0:
        if verbose:
            print("Expanding along spectral dimension by {} channels"
                  "".format(expand_by_nchan))
        for iiter in range(expand_by_nchan):
            tempmask = np.roll(mask_signal, 1, axis=0)
            tempmask[0, :] = False
            mask_signal |= tempmask
            tempmask = np.roll(mask_signal, -1, axis=0)
            tempmask[-1, :] = False
            mask_signal |= tempmask
        mask_signal &= mask
    
    return cube.with_mask(mask_signal)


def censoring_function(
        line_rms_width, channel_width, T_noise,
        spec_resp_kernel=[0, 1, 0],
        nchan_crit=2, snr_crit=2, scenario='optimistic'):
    """
    Calculate the shape of the censoring function.

    Parameters
    ----------
    line_rms_width: float or array-like
        The rms width of the Gaussian line profile
    channel_width: float or array-like
        The width of spectral channels, in the same unit as
        `line_rms_width`
    T_noise: float or array-like
        Noise temperature (1-sigma level)
    spec_resp_kernel: array-like with odd number of elements, optional
        A spectral response kernel designed to reproduce the
        appropriate channel-to-channel correlation
        (see Appendix A in Leroy et al. 2016, ApJ, 831, 16)
    nchan_crit: int, optional (default=2)
        # of channels used in determining significant detections
        (e.g., if the criterion is 2 consecutive channels above S/N=5,
        then `nchan_crit` should be set to 2)
    snr_crit: float, optional (default=2)
        S/R ratio used in determining significant detections
        (e.g., if the criterion is 2 consecutive channels above S/N=5,
        then `snr_crit` should be set to 5)
    scenario: {'optimistic', 'pessimistic'}, optional
        If 'optimistic' (default), this function will return the
        threshold below which the completeness is 0%.
        If 'pessimistic', this function will return the threshold
        above which the completeness is 100%.

    Returns
    -------
    line_intensity: float or array-like
        Completeness threshold in terms of integrated line intensity,
        carrying a unit of [T_noise] * [line_rms_width]
    """
    x = (np.asfarray(channel_width) /
         (2**0.5 * np.asfarray(line_rms_width)))

    if scenario == 'optimistic':
        # line center aligns perfectly with
        # the center of the `nchan_crit` consecutive channels
        uplim = nchan_crit / 2
        lolim = nchan_crit / 2 - 1
    elif scenario == 'pessimistic':
        # line center is off by half a channel width from
        # the center of the `nchan_crit` consecutive channels
        uplim = nchan_crit / 2 + 0.5
        lolim = nchan_crit / 2 - 0.5
    else:
        raise ValueError("Invalid `scenario`")

    kernel = np.asfarray(spec_resp_kernel).flatten()
    if (kernel.size % 2 == 0) or (kernel.sum() != 1):
        raise ValueError("Invalid spectral response kernel")

    ind = np.arange(-(kernel.size-1)/2, (kernel.size+1)/2)
    uplim = (uplim + ind) * x[..., np.newaxis]
    lolim = (lolim + ind) * x[..., np.newaxis]
    fraction = ((erf(uplim) - erf(lolim)) / 2 * kernel).sum(axis=-1)

    return np.asfarray(T_noise) * channel_width * snr_crit / fraction
