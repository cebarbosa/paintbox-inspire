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
    data_dir = os.path.join(context.data_dir, "test")
    galaxies = os.listdir(data_dir)
    wranges = ["UVB", "VIS", "NIR"]
    R = [3300, 5400, 3500]
    for galaxy in galaxies:
        wdir = os.path.join(data_dir, galaxy)
        for i, wr in enumerate(wranges):
            specname = "{}_{}_FLUX_RSF.fits".format(galaxy, wr)
            errorspecname = specname.replace("FLUX", "ERROR")
            spec1d = Spectrum1D.read(os.path.join(wdir, specname))
            wave = np.power(10, spec1d.spectral_axis.value) * u.angstrom
            flux = spec1d.flux
            fwhm = wave/ R[i]
            vel = const.c.to("km/s") * fwhm / wave / 2.634
            print(vel)
            plt.plot(wave,  vel)
        plt.show()
