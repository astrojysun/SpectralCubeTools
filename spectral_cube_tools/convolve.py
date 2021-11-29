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
        nan_treatment='fill', boundary='fill', fill_value=0.,
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
        In the former case, a direct convolution is performed;
        in the latter case, the convolution attempts to mimic the
        error propagation process to the specified lower resolution.
        (Default: 'datacube')
    res_tol : float, optional
        Tolerance on the difference between input/output resolution
        By default, a convolution is performed on the input cube
        when its native resolution is different from (sharper than)
        the target resolution. Use this keyword to specify a tolerance
        on resolution, within which no convolution will be performed.
        For example, ``res_tol=0.1`` will allow a 10% tolerance.
    min_coverage : float or None, optional
        This keyword specifies a minimum beam covering fraction of
        valid pixels for convolution (Default: 0.8).
        Locations with a beam covering fraction less than this value
        will be overwritten to "NaN" in the convolved cube.
        If the user would rather use the ``preserve_nan`` mode in
        `astropy.convolution.convolve_fft`, set this keyword to None.
    nan_treatment: {'interpolate', 'fill'}, optional
        To be passed to `astropy.convolution.convolve_fft`.
        (Default: 'fill')
    boundary: {'fill', 'wrap'}, optional
        To be passed to `astropy.convolution.convolve_fft`.
        (Default: 'fill')
    fill_value : float, optional
        To be passed to `astropy.convolution.convolve_fft`.
        (Default: 0)
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

    if (res_tol > 0) and (newbeam.major != newbeam.minor):
        raise ValueError(
            "Cannot handle a non-zero resolution torelance "
            "when the target beam is not round")

    if min_coverage is None:
        # Skip coverage check and preserve NaN values.
        # This uses the default 'preserve_nan' scheme
        # implemented in 'astropy.convolution.convolve_fft'
        convolve_func = partial(
            convolve_fft, fill_value=fill_value,
            nan_treatment=nan_treatment, boundary=boundary,
            preserve_nan=True, allow_huge=True)
    else:
        # Do coverage check to determine the mask on the output
        convolve_func = partial(
            convolve_fft, fill_value=fill_value,
            nan_treatment=nan_treatment, boundary=boundary,
            allow_huge=True)
    convolve_func_w = partial(
        convolve_fft, fill_value=0., boundary='fill',
        allow_huge=True)

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
                        "Unsuccessful beam deconvolution: "
                        "{}\nOld: {}\nNew: {}"
                        "".format(err, cube.beam, newbeam))
                    print("Exiting...")
                return
            else:
                raise ValueError(
                    "Unsuccessful beam deconvolution: "
                    "{}\nOld: {}\nNew: {}"
                    "".format(err, cube.beam, newbeam))
        if verbose:
            print("Convolving cube...")
        if mode == 'datacube':
            # do convolution
            convcube = cube.convolve_to(
                newbeam, convolve=convolve_func)
            if min_coverage is not None:
                my_append_raw = True
                wtcube = SpectralCube(
                    cube.mask.include().astype('float'),
                    cube.wcs, beam=cube.beam).with_mask(
                        np.ones(cube.shape).astype('?'))
                wtcube.allow_huge_operations = (
                    cube.allow_huge_operations)
                wtcube = wtcube.convolve_to(
                    newbeam, convolve=convolve_func_w)
                # divide the raw convolved image by the weight image
                # to correct for filling fraction
                newcube = convcube / wtcube.unmasked_data[:]
                # mask all pixels w/ weight smaller than min_coverage
                threshold = min_coverage * u.dimensionless_unscaled
                newcube = newcube.with_mask(wtcube >= threshold)
            else:
                my_append_raw = False
                newcube = convcube
                wtcube = None
        elif mode == 'noisecube':
            # Empirically derive a noise cube at the lower resolution
            # Step 1: square the high resolution noise cube
            cubesq = cube**2
            # Step 2: convolve the squared noise cube with a kernel
            #         that is sqrt(2) times narrower than the one
            #         used for data cube convolution (this is because
            #         the Gaussian weight needs to be squared in
            #         error propagation)
            beamdiff_small = Beam(
                major=beamdiff.major/np.sqrt(2),
                minor=beamdiff.minor/np.sqrt(2), pa=beamdiff.pa)
            newbeam_small = cube.beam.convolve(beamdiff_small)
            convcubesq = cubesq.convolve_to(
                newbeam_small, convolve=convolve_func)
            del cubesq  # release memory
            if min_coverage is not None:
                my_append_raw = True
                wtcube = SpectralCube(
                    cube.mask.include().astype('float'),
                    cube.wcs, beam=cube.beam).with_mask(
                        np.ones(cube.shape).astype('?'))
                wtcube.allow_huge_operations = (
                    cube.allow_huge_operations)
                # divide the raw convolved cube by the weight cube
                # to correct for filling fraction
                wtcube_d = wtcube.convolve_to(
                    newbeam_small, convolve=convolve_func_w)
                newcubesq = convcubesq / wtcube_d.unmasked_data[:]
                del convcubesq, wtcube_d  # release memory
                # mask all pixels w/ weight smaller than min_coverage
                # (here I force the masking of the noise cube to be
                #  consistent with that of the data cube)
                wtcube = wtcube.convolve_to(
                    newbeam, convolve=convolve_func_w)
                threshold = min_coverage * u.dimensionless_unscaled
                newcubesq = newcubesq.with_mask(wtcube >= threshold)
            else:
                my_append_raw = False
                newcubesq = convcubesq
                wtcube = None
            # Step 3: find the sqrt of the convolved noise cube
            convcube = np.sqrt(convcubesq)
            del convcubesq  # release memory
            newcube = np.sqrt(newcubesq)
            del newcubesq  # release memory
            # Step 4: apply a multiplicative factor, which accounts
            #         for the decrease in rms noise due to averaging
            convcube *= np.sqrt(cube.beam.sr/newbeam.sr).to('').value
            newcube *= np.sqrt(cube.beam.sr/newbeam.sr).to('').value
        else:
            raise ValueError("Invalid `mode` value: {}".format(mode))

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
        inimage, newbeam, mode='dataimage',
        res_tol=0.0, min_coverage=0.8,
        nan_treatment='fill', boundary='fill', fill_value=0.,
        append_raw=False, verbose=False, suppress_error=False):
    """
    Convolve a 2D image or an rms noise image to a specified beam.

    This function is similar to `convolve_cube()`, but it deals with
    2D images (i.e., projections) rather than 3D cubes.

    Parameters
    ----------
    inimage : FITS HDU object or ~spectral_cube.Projection object
        Input 2D image
    newbeam : radio_beam.Beam object
        Target beam to convolve to
    mode : {'dataimage', 'noiseimage'}, optional
        Whether the input image is a data image or an rms noise image.
        In the former case, a direct convolution is performed;
        in the latter case, the convolution attempts to mimic the
        error propagation process to the specified lower resolution.
        (Default: 'dataimage')
    res_tol : float, optional
        Tolerance on the difference between input/output resolution
        By default, a convolution is performed on the input image
        when its native resolution is different from (sharper than)
        the target resolution. Use this keyword to specify a tolerance
        on resolution, within which no convolution will be performed.
        For example, res_tol=0.1 will allow a 10% tolerance.
    min_coverage : float or None, optional
        This keyword specifies a minimum beam covering fraction of
        valid pixels for convolution (Default: 0.8).
        Locations with a beam covering fraction less than this value
        will be overwritten to "NaN" in the convolved cube.
        If the user would rather use the ``preserve_nan`` mode in
        `astropy.convolution.convolve_fft`, set this keyword to None.
    nan_treatment: {'interpolate', 'fill'}, optional
        To be passed to `astropy.convolution.convolve_fft`.
        (Default: 'fill')
    boundary: {'fill', 'wrap'}, optional
        To be passed to `astropy.convolution.convolve_fft`.
        (Default: 'fill')
    fill_value : float, optional
        To be passed to `astropy.convolution.convolve_fft`.
        (Default: 0)
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
        # This uses the default 'preserve_nan' scheme
        # implemented in 'astropy.convolution.convolve_fft'
        convolve_func = partial(
            convolve_fft, fill_value=fill_value,
            nan_treatment=nan_treatment, boundary=boundary,
            preserve_nan=True, allow_huge=True)
    else:
        # Do coverage check to determine the mask on the output
        convolve_func = partial(
            convolve_fft, fill_value=fill_value,
            nan_treatment=nan_treatment, boundary=boundary,
            allow_huge=True)
    convolve_func_w = partial(
        convolve_fft, fill_value=0., boundary='fill',
        allow_huge=True)

    tol = newbeam.major * np.array([1-res_tol, 1+res_tol])
    if ((tol[0] < proj.beam.major < tol[1]) and
            (tol[0] < proj.beam.minor < tol[1])):
        if verbose:
            print(
                "Native resolution within tolerance - "
                "Copying original image...")
        my_append_raw = False
        convproj = wtproj = None
        newproj = proj.copy()
    else:
        if verbose:
            print("Deconvolving beam...")
        try:
            beamdiff = newbeam.deconvolve(proj.beam)
        except ValueError as err:
            if suppress_error:
                if verbose:
                    print(
                        "Unsuccessful beam deconvolution: "
                        "{}\nOld: {}\nNew: {}"
                        "".format(err, proj.beam, newbeam))
                    print("Exiting...")
                return
            else:
                raise ValueError(
                    "Unsuccessful beam deconvolution: "
                    "{}\nOld: {}\nNew: {}"
                    "".format(err, proj.beam, newbeam))
        if verbose:
            print("Convolving image...")
        if mode == 'dataimage':
            # do convolution
            convproj = proj.convolve_to(
                newbeam, convolve=convolve_func)
            if min_coverage is not None:
                my_append_raw = True
                wtproj = Projection(
                    np.isfinite(proj.data).astype('float'),
                    wcs=proj.wcs, beam=proj.beam)
                wtproj = wtproj.convolve_to(
                    newbeam, convolve=convolve_func_w)
                # divide the raw convolved image by the weight image
                # to correct for filling fraction
                newproj = convproj / wtproj.hdu.data
                # mask all pixels w/ weight smaller than min_coverage
                threshold = min_coverage * u.dimensionless_unscaled
                newproj[wtproj < threshold] = np.nan
            else:
                my_append_raw = False
                newproj = convproj
                wtproj = None
        elif mode == 'noiseimage':
            # Empirically derive a noise image at the lower resolution
            # Step 1: square the high resolution noise image
            projsq = proj**2
            # Step 2: convolve the squared noise image with a kernel
            #         that is sqrt(2) times narrower than the one
            #         used for data image convolution (this is because
            #         the Gaussian weight needs to be squared in
            #         error propagation)
            beamdiff_small = Beam(
                major=beamdiff.major/np.sqrt(2),
                minor=beamdiff.minor/np.sqrt(2), pa=beamdiff.pa)
            newbeam_small = proj.beam.convolve(beamdiff_small)
            convprojsq = projsq.convolve_to(
                newbeam_small, convolve=convolve_func)
            if min_coverage is not None:
                my_append_raw = True
                wtproj = Projection(
                    np.isfinite(proj.data).astype('float'),
                    wcs=proj.wcs, beam=proj.beam)
                # divide the raw convolved image by the weight image
                # to correct for filling fraction
                wtproj_d = wtproj.convolve_to(
                    newbeam_small, convolve=convolve_func_w)
                newprojsq = convprojsq / wtproj_d.hdu.data
                # mask all pixels w/ weight smaller than min_coverage
                # (here I force the masking of the noise image to be
                #  consistent with that of the data image)
                wtproj = wtproj.convolve_to(
                    newbeam, convolve=convolve_func_w)
                threshold = min_coverage * u.dimensionless_unscaled
                newprojsq[wtproj < threshold] = np.nan
            else:
                my_append_raw = False
                newprojsq = convprojsq
                wtproj = None
            # Step 3: find the sqrt of the convolved noise image
            convproj = np.sqrt(convprojsq)
            newproj = np.sqrt(newprojsq)
            # Step 4: apply a multiplicative factor, which accounts
            #         for the decrease in rms noise due to averaging
            convproj = (
                convproj *
                np.sqrt(proj.beam.sr/newbeam.sr).to('').value)
            newproj = (
                newproj *
                np.sqrt(proj.beam.sr/newbeam.sr).to('').value)
        else:
            raise ValueError("Invalid `mode` value: {}".format(mode))

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
