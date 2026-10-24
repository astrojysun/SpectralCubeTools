from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
from scipy.ndimage import generic_filter
from scipy.stats import pearsonr
from astropy import units as u
from astropy.stats import mad_std
from spectral_cube import SpectralCube


def calc_noise_in_cube(
        cube, masking_scheme='simple', mask=None,
        spatial_average_npix=None, spatial_average_nbeam=5.0,
        spectral_average_nchan=5, verbose=False):
    """
    Estimate rms noise in a (continuum-subtracted) spectral cube.

    Parameters
    ----------
    cube : SpectralCube object
        Input spectral cube (needs to be continuum-subtracted)
    masking_scheme : {'simple', 'user'}, optional
        Scheme for flagging signal in the cube. 'simple' means to flag
        all values above 3*rms or below -3*rms (default scheme);
        'user' means to use the user-specified mask (i.e., `mask`).
    mask : `np.ndarray` object, optional
        User-specified signal mask (this parameter is ignored if
        `masking_scheme` is not 'user')
    spatial_average_npix : int, optional
        Size of the spatial averaging box, in terms of pixel number
        If not None, `spatial_average_nbeam` will be ingored.
        (Default: None)
    spatial_average_nbeam : float, optional
        Size of the spatial averaging box, in the unit of beam FWHM
        (Default: 5.0)
    spectral_average_nchan : int, optional
        Size of the spectral averaging box, in terms of channel number
        (Default: 5)
    verbose : bool, optional
        Whether to print the detailed processing information in terminal
        Default is to not print.
    
    Returns
    -------
    rmscube : SpectralCube object
        Spectral cube containing the rms noise at each ppv location
    """

    if masking_scheme not in ['simple', 'user']:
        raise ValueError(
            "'masking_scheme' should be specified as "
            "either 'simple' or 'user'")
    elif masking_scheme == 'user' and mask is None:
        raise ValueError(
            "'masking_scheme' set to 'user', yet "
            "no user-specified mask found")

    # extract negative values (only needed if masking_scheme='simple')
    if masking_scheme == 'simple':
        if verbose:
            print("Extracting negative values...")
        negmask = cube < (0 * cube.unit)
        negdata = cube.with_mask(negmask).filled_data[:].value
        negdata = np.stack([negdata, -1 * negdata], axis=-1)
    else:
        negdata = None

    # find rms noise as a function of channel
    if verbose:
        print("Estimating rms noise as a function of channel...")
    if masking_scheme == 'user':
        mask_v = mask
    else:
        rms_v = mad_std(negdata, axis=(1, 2, 3), ignore_nan=True)
        uplim_v = (3 * rms_v * cube.unit).reshape(-1, 1, 1)
        lolim_v = (-3 * rms_v * cube.unit).reshape(-1, 1, 1)
        mask_v = (
            ((cube - uplim_v) < (0 * cube.unit)) &
            ((cube - lolim_v) > (0 * cube.unit)))
    rms_v = cube.with_mask(mask_v).mad_std(axis=(1, 2)).quantity.value
    rms_v = generic_filter(
        rms_v, np.nanmedian, mode='constant', cval=np.nan,
        size=spectral_average_nchan)
    
    # find rms noise as a function of sightline
    if verbose:
        print("Estimating rms noise as a function of sightline...")
    if masking_scheme == 'user':
        mask_s = mask
    else:
        rms_s = mad_std(negdata, axis=(0, 3), ignore_nan=True)
        uplim_s = 3 * rms_s * cube.unit
        lolim_s = -3 * rms_s * cube.unit
        mask_s = (
            ((cube - uplim_s) < (0 * cube.unit)) &
            ((cube - lolim_s) > (0 * cube.unit)))
    rms_s = cube.with_mask(mask_s).mad_std(axis=0).quantity.value
    if spatial_average_npix is None:
        beamFWHM_pix = (
            cube.beam.major.to(u.deg).value /
            np.abs(cube.wcs.celestial.wcs.cdelt.min()))
        beamFWHM_pix = np.max([beamFWHM_pix, 3.])
        spatial_average_npix = int(
            spatial_average_nbeam * beamFWHM_pix)
    rms_s = generic_filter(
        rms_s, np.nanmedian, mode='constant', cval=np.nan,
        size=spatial_average_npix)

    # create rms noise cube from the tensor product of rms_v and rms_s
    if verbose:
        print("Creating rms noise cube (direct tensor product)...")
    rmscube = SpectralCube(
        np.einsum('i,jk', rms_v, rms_s), wcs=cube.wcs,
        header=cube.header.copy()).with_mask(cube.mask.include())
    rmscube.allow_huge_operations = cube.allow_huge_operations
    # correct the normalization of the rms cube
    if masking_scheme == 'user':
        mask_n = mask
    else:
        rms_n = mad_std(negdata, ignore_nan=True)
        uplim_n = 3 * rms_n * cube.unit
        lolim_n = -3 * rms_n * cube.unit
        mask_n = (
            ((cube - uplim_n) < (0 * cube.unit)) &
            ((cube - lolim_n) > (0 * cube.unit)))
    rms_n = cube.with_mask(mask_n).mad_std().value
    rmscube /= rms_n

    # check unit
    if rmscube.unit != cube.unit:
        rmscube = rmscube * (cube.unit / rmscube.unit)

    return rmscube
    

def calc_channel_corr(cube, mask=None, channel_lag=1):
    """
    Calculate the channel-to-channel correlation coefficient (Pearson's r)

    Parameters
    ----------
    cube : SpectralCube object
        Input spectral cube
    mask : `np.ndarray` object, optional
        User-specified mask, within which to calculate r
    channel_lag : int
        Number of channel lag at which the correlation is calculated.
        Default is 1, which means to estimate correlation at one
        channel lag (i.e., between immediately adjacent channels)

    Returns
    -------
    r : float
        Pearson's correlation coefficient
    p-value : float
        Two-tailed p-value
    """
    if mask is None:
        mask = cube.mask.include()
    mask[-1, :] = False
    for i in np.arange(channel_lag):
        mask &= np.roll(mask, -1, axis=0)
    return pearsonr(
        cube.filled_data[mask],
        cube.filled_data[np.roll(mask, channel_lag, axis=0)])
