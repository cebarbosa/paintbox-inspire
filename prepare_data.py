"""
Prepare data for fitting with paintbox.
"""

import os

import numpy as np
import astropy.units as u
from astropy.io import fits
import astropy.constants as const
from astropy.table import Table
import matplotlib.pyplot as plt
from specutils import Spectrum1D
from scipy.ndimage.filters import gaussian_filter1d
from ppxf.ppxf_util import log_rebin
from spectres import spectres

import context

if __name__ == "__main__":
    sample = "test"
    data_dir = os.path.join(context.data_dir, sample)
    galaxies = os.listdir(data_dir)
    wranges = ["UVB", "VIS", "NIR"]
    R = [3300, 5400, 3500]
    target_sigma = 300
    outvelscale = int(target_sigma / 3)
    c = const.c.to("km/s").value
    for galaxy in galaxies:
        wdir = os.path.join(data_dir, galaxy)
        output = os.path.join(wdir, "{}_sig{}.fits".format(galaxy,
                                                               target_sigma))
        hdulist = [fits.PrimaryHDU()]
        fig = plt.figure(figsize=(context.txtwidth, 2.5))
        for i, wr in enumerate(wranges):
            specname = "{}_{}_FLUX_RSF.fits".format(galaxy, wr)
            ername = specname.replace("FLUX", "ERROR")
            spec1d = Spectrum1D.read(os.path.join(wdir, specname))
            err1d = Spectrum1D.read(os.path.join(wdir, ername))
            wave = np.power(10, spec1d.spectral_axis.value)
            wave2 = np.power(10, err1d.spectral_axis.value)
            assert np.all(np.equal(wave, wave2)), "different wavelenght " \
                                                  "dispersion"
            obssigma = const.c.to("km/s").value / 2.355 / R[i]
            velscale = np.diff(np.log(wave) * const.c.to("km/s"))[0]
            kernel_sigma = np.sqrt(target_sigma**2 - obssigma**2) / velscale
            spec_target_sigma = gaussian_filter1d(spec1d.flux, kernel_sigma.value,
                                                  mode="constant", cval=0.0)
            err_target_sigma = gaussian_filter1d(err1d.flux,
                                                    kernel_sigma.value,
                                                    mode="constant", cval=0.0)
            logwave = log_rebin([wave[0], wave[-1]],
                                spec_target_sigma, velscale=outvelscale)[1]
            owave = np.exp(logwave)[1:-1]
            ospec, ospecerr = spectres(owave, wave, spec_target_sigma,
                                       spec_errs=err_target_sigma)
            plt.plot(wave, spec1d.flux, ls="--", c="0.9")
            plt.plot(owave, ospec, c="C{}".format(i), label=wranges[i])
            plt.plot(owave, ospec + ospecerr, c="C{}".format(i), ls="--")
            plt.plot(owave, ospec - ospecerr, c="C{}".format(i), ls="--")
            t = Table([owave * u.Angstrom, ospec * context.flam_unit,
                       ospecerr * context.flam_unit],
                      names=["wave", "flam", "flamerr"])
            hdu = fits.BinTableHDU(t)
            hdu.header["EXTNAME"] = (wranges[i], "Xshooter arm")
            hdulist.append(hdu)
        hdulist = fits.HDUList(hdulist)
        hdulist.writeto(output, overwrite=True)
        plt.ylim(-1e-17, 0.8e-16)
        plt.title(galaxy)
        plt.xlabel(r"$\lambda$ (\r{A})")
        funit = context.flam_unit
        plt.ylabel(r"erg s$^{-1}$ cm$^{-2} $\r{A}$^{-1}$")
        plt.subplots_adjust(bottom=0.13, left=0.06, right=0.99, top=0.93)
        plt.legend()
        plt.savefig(os.path.join(wdir, "{}_spectrum.png".format(galaxy)),
                    dpi=250)
        plt.show()