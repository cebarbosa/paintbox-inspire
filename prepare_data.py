"""
Prepare data for fitting with paintbox.
"""

import os
import copy

import numpy as np
import astropy.units as u
from astropy.io import fits
import astropy.constants as const
from astropy.table import Table
import matplotlib.pyplot as plt
from specutils import Spectrum1D
from scipy.ndimage.filters import gaussian_filter1d
from spectres import spectres
import paintbox as pb
from paintbox.utils.disp2vel import disp2vel

import context

def prepare_test_data():
    sample = "test"
    data_dir = os.path.join(context.data_dir, sample)
    galaxies = sorted(os.listdir(data_dir))
    wranges = ["UVB", "VIS", "NIR"]
    R = [3300, 5400, 3500]
    target_sigma = 300
    outvelscale = int(target_sigma / 3)
    c = const.c.to("km/s").value
    for galaxy in galaxies:
        wdir = os.path.join(data_dir, galaxy)
        maskfile = os.path.join(wdir, "mranges.txt")
        mwave, marm = [], []
        if os.path.exists(maskfile):
            mwave = np.atleast_2d(np.loadtxt(maskfile, usecols=(0,1)))
            marm = np.atleast_1d(np.loadtxt(maskfile, usecols=(2,),
                                            dtype=np.str))
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
            owave = disp2vel(wave, velscale)
            # Mask out bad regions
            goodpix = np.full(len(owave), True)
            for j, (w0, w1) in enumerate(mwave):
                if marm[j] != wr:
                    continue
                idx = np.where((owave >= w0) & (owave <= w1))
                goodpix[idx] = False
            ospec, ospecerr = spectres(owave, wave, spec_target_sigma,
                                       spec_errs=err_target_sigma)
            plt.plot(wave, spec1d.flux, ls="--", c="0.9")
            y = copy.copy(ospec)
            y[~goodpix] = np.nan
            plt.plot(owave, y, c="C{}".format(i),
                     label=wranges[i])
            plt.plot(owave, y + ospecerr, c="C{}".format(i), ls="--")
            plt.plot(owave, y - ospecerr, c="C{}".format(i), ls="--")
            mask = np.where(goodpix, 1, 0)
            t = Table([owave * u.Angstrom, ospec * context.flam_unit,
                       ospecerr * context.flam_unit, mask],
                      names=["wave", "flam", "flamerr", "mask"])
            # Crop borders
            w1 = t["wave"].data[t["mask"] == 0][0]
            w2 = t["wave"].data[t["mask"] == 0][-1]
            t = t[t["wave"] >= w1]
            t = t[t["wave"] <= w2]
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

def prepare_dr1_data(target_sigma=300):
    sample = "combined_dr1"
    data_dir = os.path.join(context.data_dir, sample)
    filenames = sorted(os.listdir(data_dir))
    galaxies = list(set([_.split("_")[0] for _ in filenames]))
    # R = np.array([3300, 5400, 3500])
    # obssigma = const.c.to("km/s").value / 2.355 / R
    velscale = int(target_sigma / 3)
    outdir = os.path.join(context.home_dir, f"paintbox/dr1_sig{target_sigma}")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for galaxy in galaxies:
        print(galaxy)
        output = os.path.join(outdir, f"{galaxy}_sig{target_sigma}.fits")
        specname = f"{galaxy}_ALL_REBINNED_NORM.fits"
        specerrname = f"{galaxy}_ALL_ERROR_REBINNED_NORM.fits"
        specname_unnorm = f"{galaxy}_ALL_REBINNED.fits"
        spec1d = Spectrum1D.read(os.path.join(data_dir, specname))
        err1d = Spectrum1D.read(os.path.join(data_dir, specerrname))
        spec1d_un = Spectrum1D.read(os.path.join(data_dir, specname_unnorm ))
        ratio = spec1d_un.flux / spec1d.flux
        norm = np.nanmedian(ratio.value)
        wave = np.power(10, spec1d.spectral_axis.value)
        flux = spec1d.flux
        fluxerr = err1d.flux
        mask = np.where(flux == 0, 1, 0)
        owave = disp2vel(wave, velscale)
        flux_rebin, fluxerr_rebin = spectres(owave, wave, flux,
                                             spec_errs=fluxerr, fill=0,
                                             verbose=False)
        omask = spectres(owave, wave, mask, fill=0, verbose=False)
        theta = np.array([1, 0, target_sigma])
        oflux = pb.LOSVDConv(
            pb.NonParametricModel(owave, np.atleast_2d(flux_rebin)))(theta)
        ofluxerr = pb.LOSVDConv(
            pb.NonParametricModel(owave, np.atleast_2d(fluxerr_rebin)))(theta)
        omask[np.isnan(oflux)] = 1
        ofluxerr[np.isnan(oflux)] = 0
        oflux[np.isnan(oflux)] = 0
        table = Table([owave, oflux, ofluxerr, omask],
                      names=["wave", "flux", "fluxerr", "mask"])
        hdu = fits.BinTableHDU(table)
        hdu.header["NORM"] = (norm, "Flux normalization")
        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
        hdulist.writeto(output, overwrite=True)

if __name__ == "__main__":
    # prepare_test_data()
    prepare_dr1_data()