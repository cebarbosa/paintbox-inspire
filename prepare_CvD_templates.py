"""
Prepare templates for use with paintbox
"""
import os
import itertools

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.table import Table, hstack, vstack
from astropy.io import fits
import ppxf.ppxf_util as util
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from spectres import spectres

import context
import fsps

def make_CvD_templates(owave, output, Zs=None, imf1s=None,
                       imf2s=None, agemin=0.5, agemax=20, age_freq=2,
                       dlam=None):
    """ Use python-FSPS to produce models to be used with paintbox. """
    flam_unit = u.erg / u.cm / u.cm / u.s / u.AA
    fnu_unit = u.erg / u.s / u.cm / u.cm / u.Hz
    dlam = 500 * u.AA if dlam is None else dlam
    bands = fsps.list_filters()
    sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,
                                sfh=0, logzsol=0.0, dust_type=2, dust2=0.,
                                add_agb_dust_model=False,
                                add_dust_emission=False,
                                add_neb_emission=False,
                                nebemlineinspec=False,
                                imf_type=2)
    wave_eff, Mvega, Msun = sp.filter_data()
    Zs = np.array([-2.5, -2.0, -1.75, -1.5, -1.25, -1., -0.75, -0.5, -0.25,
                   0., 0.25, 0.5]) if Zs is None else Zs
    logTs = np.arange(5, 10.31, 0.05)
    Ts = np.power(10, logTs) / 1e9
    idx_T = np.where((Ts >= agemin) & (Ts <= agemax))[0][::age_freq]
    Ts = Ts[idx_T]
    imfs = 0.5 + np.arange(16) / 5
    imf1s = imfs if imf1s is None else imf1s
    imf2s = imfs if imf2s is None else imf2s
    params = list(itertools.product(*[Zs, imf1s, imf2s]))
    flams = []
    otable = []
    for Z, imf1, imf2 in tqdm(params, desc="Processing spectra with FSPS"):
        sp.params['logzsol'] = Z
        sp.params["imf1"] = imf1
        sp.params["imf2"] = imf2
        wave, fnu = sp.get_spectrum()
        fnu = fnu[idx_T,:]
        # Making table with magnitudes of models
        mags = sp.get_mags(bands=bands)[idx_T]
        tabmags = Table(mags, names=bands)
        # Calculating M/L
        L = np.power(10, -0.4 * (mags - Msun[None, :]))
        M2L = 1 / L
        tabM2L = Table(M2L, names=["ML_{}".format(b) for b in bands])
        # Processing spectra
        wave *= u.AA
        fnu = fnu * const.L_sun / u.Hz / (4 * np.pi * const.au**2)
        fnu = fnu.to(fnu_unit)
        flam = fnu * const.c / wave ** 2
        flam = flam.to(flam_unit)
        oflam = np.zeros((len(flam), len(owave)))
        idx = np.where((wave >= owave[0] - dlam) & \
                       (wave <= owave[-1] + dlam))[0]
        ones = np.ones(len(Ts))
        tab = Table([Z * ones, Ts, imf1 * ones, imf2 * ones],
                       names=["logzsol", "age", "imf1", "imf2"])
        otable.append(hstack([tab, tabmags, tabM2L]))
        for i, T in enumerate(Ts):
            f = interp1d(wave[idx], flam[i,idx], bounds_error=False)
            oflam[i] = f(owave)
        flams.append(oflam)
    otable = vstack(otable)
    flams = np.vstack(flams)
    norm = float(int(np.log10(np.nanmedian(flams))))
    bscale = np.power(10., norm)
    flams = flams / bscale
    hdu1 = fits.PrimaryHDU(flams)
    hdu1.header["EXTNAME"] = "SSPS"
    hdu1.header["BSCALE"] = (bscale, "Linear factor in scaling equation")
    hdu1.header["BZERO"] = (0, "Zero point in scaling equation")
    hdu1.header["BUNIT"] = ("{}".format(flam_unit),
                           "Physical units of the array values")
    hdu2 = fits.BinTableHDU(otable)
    hdu2.header["EXTNAME"] = "PARAMS"
    # Making wavelength array
    hdu3 = fits.BinTableHDU(Table([owave], names=["wave"]))
    hdu3.header["EXTNAME"] = "WAVE"
    hdulist = fits.HDUList([hdu1, hdu2, hdu3])
    hdulist.writeto(output, overwrite=True)

def prepare_models_wifis(w1=8600, w2=13200, velscale=50):
    logLam, velscale = util.log_rebin([w1, w2], np.ones(10000),
                                               velscale=velscale)[1:]
    wave = np.exp(logLam) * u.AA
    output = "/home/kadu/Dropbox/SPINS/fsps_wifis.fits"
    make_CvD_templates(wave, output)

if __name__ == "__main__":
    prepare_models_wifis()
