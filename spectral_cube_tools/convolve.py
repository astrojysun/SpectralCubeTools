from __future__ import (
    division, print_function, absolute_import, unicode_literals)

from functools import partial
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve_fft
from radio_beam import Beam
from spectral_cube import SpectralCube, Projection


def convolve_cube(
        incube, newbeam, mode='datacube',
        res_tol=0.0, min_coverage=0.8,
        append_raw=False, verbose=False, suppress_error=False):
    """
    Convolve a spectral cube or an rms noise cube to a specified beam.

    'datacube' mode: This is basically a wrapper around
    `~spectral_cube.SpectralCube.convolve_to()`,
    but it treats NaN values / edge effect in a more careful way
    (see the description of keyword 'min_coverage' below).

    'noisecube' mode: It handles rms noise cubes in a way that it
    correctly predicts the rms noise for the corresponding data cube
    convolved to the same specified beam.

    Parameters
    ----------
    incube : FITS HDU object or SpectralCube object
        Input spectral cube
    newbeam : radio_beam.Beam object
        Target beam to convolve to
    mode : {'datacube', 'noisecube'}, optional
        Whether the input cube is a data cube or an rms noise cube.
        In the former case, a direct convolution is performed.
        In the latter case, the convolution is done in a way that
        it returns the appropriate noise cube corresponding to
        the convolved data cube. (Default: 'datacube')
    res_tol : float, optional
        Tolerance on the difference between input/output resolution
        By default, a convolution is performed on the input cube
        when its native resolution is different from (sharper than)
        the target resolution. Use this keyword to specify a tolerance
        on resolution, within which no convolution will be performed.
        For example, res_tol=0.1 will allow a 10% tolerance.
    min_coverage : float or None, optional
        When the convolution meets NaN values or edges, the output is
        calculated by beam-weighted average. This keyword specifies
        a minimum beam covering fraction of valid (np.finite) pixels.
        Pixels with less beam covering fraction will be assigned NaNs.
        Default is an 80% beam covering fraction (min_coverage=0.8).
        If the user would rather use the interpolation strategy in
        `astropy.convolution.convolve_fft`, set this keyword to None.
        Note that the NaN pixels will be kept as NaN.
    append_raw : bool, optional
        Whether to append the raw convolved cube and weight cube.
        Default is not to append.
    verbose : bool, optional
        Whether to print the detailed processing log in terminal.
        Default is to not print.
    suppress_error : bool, optional
        Whether to suppress the error message when convolution is
        unsuccessful. Default is to not suppress.

    Returns
    -------
    outcube : FITS HDU objects or SpectralCube objects
        Convolved spectral cubes (when append_raw=False), or a 3-tuple
        including a masked verson, an unmaked version, and a coverage
        fraction cube (when append_raw=True).
        The output will be the same type of objects as the input.
    """

    if isinstance(incube, SpectralCube):
        cube = incube
    elif isinstance(incube, (fits.PrimaryHDU, fits.ImageHDU)):
        if 'BUNIT' in incube.header:
            unit = u.Unit(incube.header['BUNIT'], parse_strict='warn')
            if isinstance(unit, u.UnrecognizedUnit):
                unit = u.dimensionless_unscaled
        else:
            unit = u.dimensionless_unscaled
        cube = SpectralCube(
            data=incube.data*unit,
            wcs=WCS(incube.header), header=incube.header,
            allow_huge_operations=True).with_mask(
                np.isfinite(incube.data))
    else:
        raise ValueError(
            "`incube` needs to be either a SpectralCube object "
            "or a FITS HDU object")

    if mode not in ('datacube', 'noisecube'):
        raise ValueError("Invalid `mode` value: {}".format(mode))

    if (res_tol > 0) and (newbeam.major != newbeam.minor):
        raise ValueError(
            "Cannot handle a non-zero resolution torelance "
            "when the target beam is not round")

    if min_coverage is None:
        # Skip coverage check and preserve NaN values.
        # This uses the default interpolation strategy
        # implemented in 'astropy.convolution.convolve_fft'
        convolve_func = partial(
            convolve_fft, preserve_nan=True, allow_huge=True)
    else:
        # Do coverage check to determine the mask on the output
        convolve_func = partial(
            convolve_fft, nan_treatment='fill',
            boundary='fill', fill_value=0., allow_huge=True)

    tol = newbeam.major * np.array([1-res_tol, 1+res_tol])
    if ((tol[0] < cube.beam.major < tol[1]) and
            (tol[0] < cube.beam.minor < tol[1])):
        if verbose:
            print(
                "Native resolution within tolerance - "
                "copying original cube...")
        my_append_raw = False
        convcube = wtcube = None
        newcube = cube.unmasked_copy().with_mask(cube.mask.include())
    else:
        if verbose:
            print("Deconvolving beam...")
        try:
            beamdiff = newbeam.deconvolve(cube.beam)
        except ValueError as err:
            if suppress_error:
                if verbose:
                    print(
                        "{}\n"
                        "Old:  {:.3g} x {:.3g}  PA = {:.1f}\n"
                        "New:  {:.3g} x {:.3g}  PA = {:.1f}".format(
                            err,
                            cube.beam.major.to('arcsec'),
                            cube.beam.minor.to('arcsec'),
                            cube.beam.pa.to('deg'),
                            newbeam.major.to('arcsec'),
                            newbeam.minor.to('arcsec'),
                            newbeam.pa.to('deg')))
                    print("Exiting...")
                return
            else:
                raise ValueError(
                    "{}\n"
                    "Old:  {:.3g} x {:.3g}  PA = {:.1f}\n"
                    "New:  {:.3g} x {:.3g}  PA = {:.1f}".format(
                        err,
                        cube.beam.major.to('arcsec'),
                        cube.beam.minor.to('arcsec'),
                        cube.beam.pa.to('deg'),
                        newbeam.major.to('arcsec'),
                        newbeam.minor.to('arcsec'),
                        newbeam.pa.to('deg')))
        if verbose:
            print("Convolving cube...")
        if mode == 'datacube':
            # do convolution
            convcube = cube.convolve_to(
                newbeam, convolve=convolve_func)
            if min_coverage is not None:
                # divide the raw convolved image by the weight image
                # to correct for filling fraction
                my_append_raw = True
                wtcube = SpectralCube(
                    cube.mask.include().astype('float'),
                    cube.wcs, beam=cube.beam).with_mask(
                        np.ones(cube.shape).astype('?'))
                wtcube.allow_huge_operations = (
                    cube.allow_huge_operations)
                wtcube = wtcube.convolve_to(
                    newbeam, convolve=convolve_func)
                newcube = convcube / wtcube.unmasked_data[:]
                # mask all pixels w/ weight smaller than min_coverage
                threshold = min_coverage * u.dimensionless_unscaled
                newcube = newcube.with_mask(wtcube >= threshold)
            else:
                my_append_raw = False
                newcube = convcube
                wtcube = None
        else:  # mode == noisecube
            # Analytically derive the rms noise cube for the low
            # resolution data cube. Steps are described as follows:
            # Step 1: square the high resolution noise cube
            cubesq = cube**2
            # Step 2: convolve the squared noise cube with a kernel
            #         that is sqrt(2) times narrower than the one
            #         used for data cube convolution
            beamdiff_small = Beam(
                major=beamdiff.major/np.sqrt(2),
                minor=beamdiff.minor/np.sqrt(2), pa=beamdiff.pa)
            newbeam_small = cube.beam.convolve(beamdiff_small)
            convcubesq = cubesq.convolve_to(
                newbeam_small, convolve=convolve_func)
            if min_coverage is not None:
                # divide the raw convolved image by the weight image
                # to correct for filling fraction
                my_append_raw = True
                wtcube_o = SpectralCube(
                    cube.mask.include().astype('float'),
                    cube.wcs, beam=cube.beam).with_mask(
                        np.ones(cube.shape).astype('?'))
                wtcube_o.allow_huge_operations = (
                    cube.allow_huge_operations)
                wtcube = wtcube_o.convolve_to(
                    newbeam_small, convolve=convolve_func)
                newcubesq = convcubesq / wtcube.unmasked_data[:]
                # mask all pixels w/ weight smaller than min_coverage
                # (here I force the masking of the noise cube to be
                #  consistent with that of the data cube)
                threshold = min_coverage * u.dimensionless_unscaled
                wtcube_d = wtcube_o.convolve_to(
                    newbeam, convolve=convolve_func)
                newcubesq = newcubesq.with_mask(wtcube_d >= threshold)
            else:
                my_append_raw = False
                newcubesq = convcubesq
                wtcube = None
            # Step 3: find the square root of the convolved noise cube
            convcube = np.sqrt(convcubesq)
            newcube = np.sqrt(newcubesq)
            # Step 4: apply a multiplicative factor, which accounts
            #         for the decrease in rms noise due to averaging
            convcube *= np.sqrt(cube.beam.sr/newbeam.sr).to('').value
            newcube *= np.sqrt(cube.beam.sr/newbeam.sr).to('').value

    if isinstance(incube, SpectralCube):
        if append_raw and my_append_raw:
            return newcube, convcube, wtcube
        else:
            return newcube
    elif isinstance(incube, (fits.PrimaryHDU, fits.ImageHDU)):
        if append_raw and my_append_raw:
            return newcube.hdu, convcube.hdu, wtcube.hdu
        else:
            return newcube.hdu


def convolve_image(
        inimage, newbeam, res_tol=0.0, min_coverage=0.8,
        append_raw=False, verbose=False, suppress_error=False):
    """
    Convolve a 2D image to a specified beam.

    Very similar to `convolve_cube()`, but this function deals with
    2D images (i.e., projections) rather than 3D cubes.

    Parameters
    ----------
    inimage : FITS HDU object or ~spectral_cube.Projection object
        Input 2D image
    newbeam : radio_beam.Beam object
        Target beam to convolve to
    res_tol : float, optional
        Tolerance on the difference between input/output resolution
        By default, a convolution is performed on the input image
        when its native resolution is different from (sharper than)
        the target resolution. Use this keyword to specify a tolerance
        on resolution, within which no convolution will be performed.
        For example, res_tol=0.1 will allow a 10% tolerance.
    min_coverage : float or None, optional
        When the convolution meets NaN values or edges, the output is
        calculated by beam-weighted average. This keyword specifies
        a minimum beam covering fraction of valid (np.finite) values.
        Pixels with less beam covering fraction will be assigned NaNs.
        Default is an 80% beam covering fraction (min_coverage=0.8).
        If the user would rather use the interpolation strategy in
        `astropy.convolution.convolve_fft`, set this keyword to None.
        Note that the NaN pixels will be kept as NaN.
    append_raw : bool, optional
        Whether to append the raw convolved image and weight image.
        Default is not to append.
    verbose : bool, optional
        Whether to print the detailed processing log in terminal.
        Default is to not print.
    suppress_error : bool, optional
        Whether to suppress the error message when convolution is
        unsuccessful. Default is to not suppress.

    Returns
    -------
    outimage : FITS HDU objects or Projection objects
        Convolved 2D images (when append_raw=False), or a 3-tuple
        including a masked verson, an unmaked version, and a coverage
        fraction map (when append_raw=True).
        The output will be the same type of objects as the input.
    """

    if isinstance(inimage, Projection):
        proj = inimage
    elif isinstance(inimage, (fits.PrimaryHDU, fits.ImageHDU)):
        proj = Projection.from_hdu(inimage)
    else:
        raise ValueError(
            "`inimage` needs to be either a FITS HDU object "
            "or a spectral_cube.Projection object")

    if (res_tol > 0) and (newbeam.major != newbeam.minor):
        raise ValueError(
            "Cannot handle a non-zero resolution torelance "
            "when the target beam is not round")

    if min_coverage is None:
        # Skip coverage check and preserve NaN values.
        # This uses the default interpolation strategy
        # implemented in 'astropy.convolution.convolve_fft'
        convolve_func = partial(
            convolve_fft, preserve_nan=True, allow_huge=True)
    else:
        # Do coverage check to determine the mask on the output
        convolve_func = partial(
            convolve_fft, nan_treatment='fill',
            boundary='fill', fill_value=0., allow_huge=True)

    tol = newbeam.major * np.array([1-res_tol, 1+res_tol])
    if ((tol[0] < proj.beam.major < tol[1]) and
            (tol[0] < proj.beam.minor < tol[1])):
        if verbose:
            print(
                "Native resolution within tolerance - "
                "Copying original image...")
        my_append_raw = False
        newproj = proj.copy()
        convproj = wtproj = None
    else:
        if verbose:
            print("Convolving image...")
        try:
            convproj = proj.convolve_to(
                newbeam, convolve=convolve_func)
            if min_coverage is not None:
                # divide the raw convolved image by the weight image
                my_append_raw = True
                wtproj = Projection(
                    np.isfinite(proj.data).astype('float'),
                    wcs=proj.wcs, beam=proj.beam)
                wtproj = wtproj.convolve_to(
                    newbeam, convolve=convolve_func)
                newproj = convproj / wtproj.hdu.data
                # mask all pixels w/ weight smaller than min_coverage
                threshold = min_coverage * u.dimensionless_unscaled
                newproj[wtproj < threshold] = np.nan
            else:
                my_append_raw = False
                newproj = convproj
                wtproj = None
        except ValueError as err:
            if suppress_error:
                return
            else:
                raise ValueError(
                    "Unsuccessful convolution: {}\nOld: {}\nNew: {}"
                    "".format(err, proj.beam, newbeam))

    if isinstance(inimage, Projection):
        if append_raw and my_append_raw:
            return newproj, convproj, wtproj
        else:
            return newproj
    elif isinstance(inimage, (fits.PrimaryHDU, fits.ImageHDU)):
        if append_raw and my_append_raw:
            return newproj.hdu, convproj.hdu, wtproj.hdu
        else:
            return newproj.hdu
