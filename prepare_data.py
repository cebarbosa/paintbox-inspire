"""
Prepare data for fitting with paintbox.
"""

import os

import numpy as np
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
from specutils import Spectrum1D

import context

if __name__ == "__main__":
    sample = "test"
    data_dir = os.path.join(context.data_dir, sample)
    galaxies = os.listdir(data_dir)
    wranges = ["UVB", "VIS", "NIR"]
    R = [3300, 5400, 3500]
    target_res = 100.
    velscale = 50.
    c = const.c.to("km/s").value
    for galaxy in galaxies:
        wdir = os.path.join(data_dir, galaxy)
        for i, wr in enumerate(wranges):
            specname = "{}_{}_FLUX_RSF.fits".format(galaxy, wr)
            errorspecname = specname.replace("FLUX", "ERROR")
            spec1d = Spectrum1D.read(os.path.join(wdir, specname))
            wave = np.power(10, spec1d.spectral_axis.value) * u.angstrom
            target_fwhm = target_res / c * wave * 2.355
            # flux = spec1d.flux
            fwhm = wave/ R[i]
            # sigma = np.full(len(wave), const.c.to("km/s").value / 2.355 / R[i])
            plt.plot(wave,  target_fwhm, c="C{}".format(i))
            plt.plot(wave, fwhm, c="C{}".format(i), ls="--")
        plt.show()
