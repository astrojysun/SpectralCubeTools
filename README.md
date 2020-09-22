# SpectralCubeTools

Python tools for manipulating/analyzing radio spectral cubes.

(Migrated from [Sun_Astro_Tools](https://github.com/astrojysun/Sun_Astro_Tools))

## Table of content

+ [`spectral_cube_tools.convolve`](https://github.com/astrojysun/SpectralCubeTools/blob/master/spectral_cube_tools/convolve.py)
  - `convolve_cube`: convolve a spectral cube or an rms noise cube to a specified beam
  - `convolve_image`: convolve a 2D image to a specified beam
+ [`spectral_cube_tools.characterize`](https://github.com/astrojysun/SpectralCubeTools/blob/master/spectral_cube_tools/characterize.py)
  - `calc_noise_in_cube`: estimate rms noise in a (continuum-subtracted) spectral cube
  - `calc_channel_corr`: estimate the channel-to-channel correlation coefficient
+ [`spectral_cube_tools.identify_signal`](https://github.com/astrojysun/SpectralCubeTools/blob/master/spectral_cube_tools/identify_signal.py)
  - `find_signal_in_cube`: identify signal in a spectral cube with S/N-based criteria
  - `censoring_function`: calculate the shape of the corresponding censoring function

## Credits

This package is developed by [Jiayi Sun](https://github.com/astrojysun) in close collaboration with [Adam Leroy](https://github.com/akleroy).

If you make use of this package in a publication, please consider citing the following papers:
+ [Sun, Leroy, Schruba, et al., *"Cloud-scale Molecular Gas Properties in 15 Nearby Galaxies"*, 2018, ApJ, 860, 172](https://ui.adsabs.harvard.edu/abs/2018ApJ...860..172S)
+ [Sun, Leroy, Schinnerer, et al., *"Molecular Gas Properties on Cloud Scales Across the Local Star-forming Galaxy Population"*, 2020, ApJL, 901, L8](https://ui.adsabs.harvard.edu/abs/2020arXiv200901842S)

Please also consider acknowledgements to [`astropy`](https://github.com/astropy/astropy), [`spectral-cube`](https://github.com/radio-astro-tools/spectral-cube), and the other required packages.
