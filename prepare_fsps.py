"""
Prepare templates for use with paintbox. These models have lower resolution
in relation to CvD models, but the python-FSPS package provides a simple way
to calculate the M/L of the models.
"""
import os
import itertools

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.table import Table, hstack
from astropy.io import fits
from tqdm import tqdm

import context
import fsps

def make_FSPS_varydoublex(outdir, redo=False, add_neb_emission=False):
    """ Use python-FSPS to produce models to be used with paintbox. """
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    flam_unit = u.erg / u.cm / u.cm / u.s / u.AA
    fnu_unit = u.erg / u.s / u.cm / u.cm / u.Hz
    bands = fsps.list_filters()
    sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,
                                sfh=0, logzsol=0.0, dust_type=2, dust2=0.,
                                add_agb_dust_model=False,
                                add_dust_emission=False,
                                add_neb_emission=add_neb_emission,
                                nebemlineinspec=False,
                                imf_type=2, smooth_lsf=1, sigma_smooth=100)
    wave_eff, Mvega, Msun = sp.filter_data()
    Zs = np.array([-2.5, -2.0, -1.75, -1.5, -1.25, -1., -0.75, -0.5, -0.25,
                   0., 0.25, 0.5])
    logTs = np.arange(5, 10.31, 0.05)
    Ts = np.power(10, logTs) / 1e9
    imf1s = 0.5 + np.arange(16) / 5
    imf2s = 0.5 + np.arange(16) / 5
    params = list(itertools.product(*[Zs, imf1s, imf2s]))
    for Z, imf1, imf2 in tqdm(params, desc="Processing spectra with FSPS"):
        zname = "{:+.2f}".format(Z).replace("+", "p").replace("-", "m")
        x1name = "{:+.2f}".format(imf1).replace("+", "p").replace("-", "m")
        x2name = "{:+.2f}".format(imf2).replace("+", "p").replace("-", "m")
        fname = "FSPS_varydoublex_Z{}_imf1{}_imf2{}.fits".format(zname, x1name,
                                                           x2name)

        output = os.path.join(outdir, fname)
        if os.path.exists(output) and not redo:
            continue
        sp.params['logzsol'] = Z
        sp.params["imf1"] = imf1
        sp.params["imf2"] = imf2
        wave, fnu = sp.get_spectrum()
        # Making table with magnitudes of models
        mags = sp.get_mags(bands=bands)
        tabmags = Table(mags, names=bands)
        # Calculating M/L
        Mstar = sp.stellar_mass
        mstar_table = Table([Mstar], names=["M*+rem"])
        L = np.power(10, -0.4 * (mags - Msun[None, :]))
        M2L = Mstar[:,None] / L
        names_M2L = ["ML_{}".format(b) for b in bands]
        tabM2L = Table(M2L, names=names_M2L)
        # Processing spectra
        fnu = fnu * const.L_sun / u.Hz / (4 * np.pi * const.au**2)
        fnu = fnu.to(fnu_unit)
        flam = fnu * const.c / (wave * u.AA)**2
        flam = flam.to(flam_unit).value
        ones = np.ones(len(Ts))
        tab = Table([Z * ones, Ts, imf1 * ones, imf2 * ones],
                       names=["logzsol", "age", "imf1", "imf2"])
        otable = hstack([tab, mstar_table, tabmags, tabM2L])
        # Saving results in output
        hdu1 = fits.PrimaryHDU(flam)
        hdu1.header["EXTNAME"] = "SSPS"
        hdu1.header["BUNIT"] = ("{}".format(flam_unit),
                               "Physical units of the array values")
        hdu2 = fits.BinTableHDU(otable)
        hdu2.header["EXTNAME"] = "PARAMS"
        # Making wavelength array
        hdu3 = fits.BinTableHDU(Table([wave], names=["wave"]))
        hdu3.header["EXTNAME"] = "WAVE"
        hdulist = fits.HDUList([hdu1, hdu2, hdu3])
        hdulist.writeto(output, overwrite=True)


if __name__ == "__main__":
    FSPS_dir = "/home/kadu/Dropbox/SSPs/FSPS/varydoublex_with_lines"
    make_FSPS_varydoublex(FSPS_dir, redo=False, add_neb_emission=True)