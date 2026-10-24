from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
from scipy.ndimage import generic_filter
from scipy.stats import pearsonr
from astropy import units as u
from astropy.stats import mad_std
from radio_beam import Beam
from spectral_cube import SpectralCube
from .convolve import convolve_cube


def measure_noise_in_cube(
        cube, masking_scheme='simple', mask=None,
        spatial_average_npix=None, spatial_average_nbeam=5.0,
        spectral_average_nchan=5, verbose=False):
    """
    Measure rms noise in a (continuum-subtracted) spectral cube.

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
    

def measure_channel_corr(cube, mask=None, channel_lag=1):
    """
    Measure the channel-to-channel correlation coefficient (Pearson's r)

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


def _gen_correlated_unit_noise(
        cube, beam=None, chan_corr=None, verbose=True, seed=None):
    """
    Generate unit-rms, spatially and spectrally correlated Gaussian noise cube
    """

    # generate zero-mean, unit-rms white noise (2 extra channels at the edges)
    rng = np.random.default_rng(seed)
    white_noise = rng.normal(
        0.0, 1.0, size=cube.shape + np.array([2, 0, 0])).astype(np.float32)

    if chan_corr is not None:
        # impose channel correlation to get spectrally correlated noise
        if verbose:
            print(f"Imposing channel correlation of {chan_corr:.2f}")
        r = np.float32(chan_corr)
        if not 0 <= r <= 0.65:
            raise ValueError("chan_corr must be between 0 and 0.65")
        k = 0.47 * r - 0.23 * r**2 - 0.16 * r**3 + 0.43 * r**4  # Leroy+16
        kernel = np.array([k, 1 - 2 * k, k])
        noise = (
            kernel[0] * white_noise[:-2, :, :] +
            kernel[1] * white_noise[1:-1, :, :] +
            kernel[2] * white_noise[2:, :, :]).astype(np.float32)
    else:
        noise = white_noise[1:-1, :, :]
    
    if beam is not None:
        # impose beam to get spatially correlated noise
        if verbose:
            print(
                f"Imposing beam of {beam.major.to(u.arcsec):.2f} x "
                f"{beam.minor.to(u.arcsec):.2f} at {beam.pa.to(u.deg):.1f}")
        noise_cube = SpectralCube(
            noise * cube.unit, wcs=cube.wcs, mask=cube.mask,
            beam=Beam(0.0 * u.arcsec), allow_huge_operations=True)
        noise_cube = convolve_cube(
            noise_cube, beam, min_coverage=None, verbose=verbose)
        noise = noise_cube.unmasked_data[:]
    else:
        noise = noise * cube.unit

    # measure the new noise rms and rescale to unit rms
    rms = np.nanstd(noise.value)
    return (noise / rms).to(cube.unit)


def shape_noise_in_cube(
        cube, current_rms, target_rms,
        impose_beam=True, chan_corr=None, verbose=True, seed=None):
    """
    Add correlated noise to a cube to reach a target noise level.

    Parameters
    ----------
    cube : `spectral_cube.SpectralCube`
        The input data cube (needs to carry a beam if `impose_beam` is True).
    current_rms : float or np.ndarray
        Current rms noise (in cube units), broadcastable to the cube shape.
    target_rms : float or np.ndarray
        Desired rms noise (in cube units), broadcastable to the cube shape.
    impose_beam : bool, optional
        If True, the cube's beam is imposed on the added noise.
    chan_corr : float, optional
        Correlation coefficient between adjacent channels. If None, no spectral
        correlation is imposed on the added noise. Must be between 0 and 0.65.
    verbose : bool, optional
        If True, print progress messages.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    new_cube : `spectral_cube.SpectralCube`
        Cube with added noise, same unit and WCS as the input.
    """
    
    current_rms = np.broadcast_to(np.asarray(
        current_rms.to(cube.unit).value, dtype=np.float32), cube.shape)
    target_rms = np.broadcast_to(np.asarray(
        target_rms.to(cube.unit).value, dtype=np.float32), cube.shape)

    # rms of the noise to add; never remove noise
    diff_var = target_rms**2 - current_rms**2
    if np.nanmin(diff_var) < 0:
        print(
            "Warning: target_rms < current_rms in some pixels; "
            "no noise added there")
    diff_rms = np.sqrt(np.clip(diff_var, 0, None))

    # generate the noise
    beam = cube.beam if impose_beam else None
    noise = _gen_correlated_unit_noise(
        cube, beam=beam, chan_corr=chan_corr, verbose=verbose, seed=seed)
    noise = (noise * diff_rms).astype(np.float32)

    return cube + noise
