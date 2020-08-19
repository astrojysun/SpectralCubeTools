# SpectralCubeTools

Python tools for manipulating/analyzing radio spectral cubes.

(Originally migrated from the repository [Sun_Astro_Tools](https://github.com/astrojysun/Sun_Astro_Tools))

# Table of content

+ [`spectral_cube_tools.convolve`](https://github.com/astrojysun/SpectralCubeTools/blob/master/spectral_cube_tools/convolve.py)
  - `convolve_cube`: convolve a spectral cube or an rms noise cube to a specified beam
  - `convolve_image`: convolve a 2D image to a specified beam
+ [`spectral_cube_tools.characterize`](https://github.com/astrojysun/SpectralCubeTools/blob/master/spectral_cube_tools/characterize.py)
  - `calc_noise_in_cube`: estimate rms noise in a (continuum-subtracted) spectral cube
  - `calc_channel_corr`: estimate the channel-to-channel correlation coefficient
+ [`spectral_cube_tools.identify_signal`](https://github.com/astrojysun/SpectralCubeTools/blob/master/spectral_cube_tools/identify_signal.py)
  - `find_signal_in_cube`: identify signal in a spectral cube based on S/N ratio
  - `censoring_function`: calculate the shape of the censoring function
