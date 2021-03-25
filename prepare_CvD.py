""" Calculates the M/L for CvD models. """
import os

import numpy as np
from astropy.table import Table, vstack
from astropy.io import fits
from tqdm import tqdm
import speclite.filters
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const

class CvD():
    def __init__(self, output):
        self.output = output
        self.prepare_VCJ17(ssp_dir)
        self.mass2light()

    def prepare_VCJ17(self, data_dir, overwrite=False):
        """ Prepare templates for SSP models from Villaume et al. (2017).

            Parameters
        ----------
        output: str
            Name of the output file (a multi-extension FITS file)
        overwrite: bool (optional)
            Overwrite the output files if they already exist.

        """
        if os.path.exists(self.output) and not overwrite:
            self.data = fits.getdata(self.output)
            self.params = Table.read(self.output, hdu=1)
            self.wave = Table.read(self.output, hdu=2)["wave"].data
            return
        specs = sorted(os.listdir(data_dir))
        nimf = 16
        imfs = 0.5 + np.arange(nimf) / 5
        x2s, x1s=  np.stack(np.meshgrid(imfs, imfs)).reshape(2, -1)
        ssps, params = [], []
        count = 0
        for spec in tqdm(specs, desc="Processing SSP files"):
            T = float(spec.split("_")[3][1:])
            Z = float(spec.split("_")[4][1:-8].replace("p", "+").replace(
                        "m", "-"))
            data = np.loadtxt(os.path.join(data_dir, spec))
            wave = data[:,0]
            for i, (x1, x2) in enumerate(zip(x1s, x2s)):
                params.append(Table([[Z], [T], [x1], [x2]],
                                    names=["Z", "Age", "imf1", "imf2"]))
                ssp = data[:, i+1]
                ssps.append(ssp)
        self.data = np.array(ssps)
        self.params = vstack(params)
        self.wave = wave
        self.save()

    def save(self):
        # Saving models to file
        hdu1 = fits.PrimaryHDU(self.data)
        hdu1.header["EXTNAME"] = "SSPS"
        params = Table(self.params)
        hdu2 = fits.BinTableHDU(params)
        hdu2.header["EXTNAME"] = "PARAMS"
        # Making wavelength array
        hdu3 = fits.BinTableHDU(Table([self.wave], names=["wave"]))
        hdu3.header["EXTNAME"] = "WAVE"
        hdulist = fits.HDUList([hdu1, hdu2, hdu3])
        hdulist.writeto(self.output, overwrite=True)
        return

    def mass2light(self):
        # Constants
        msto_t0=0.33250847
        msto_t1=-0.29560944
        msto_z0=0.95402521
        msto_z1=0.21944863
        msto_z2=0.070565820
        clight = 2.9979E10
        msun   = 1.989E33
        lsun   = 3.839E33
        pc2cm  = 3.08568E18

        krpa_imf1 = 1.3
        krpa_imf2 = 2.3
        krpa_imf3 = 2.3
        # From table
        logage = np.log10(self.params["Age"].data)
        zh = self.params["Z"].data
        imf1 = self.params["imf1"].data
        imf2 = self.params["imf2"].data
        aspec = self.data * lsun / 1E6 * self.wave** 2 / clight / 1E8 \
                / 4 / np.pi / pc2cm ** 2
        msto = 10 ** (msto_t0 + msto_t1 * logage) * \
                (msto_z0 + msto_z1 * zh + msto_z2 * zh ** 2)
        mass = self.getmass(msto, imf1, imf2)
        sdss = speclite.filters.load_filters('sdss2010-*')
        sun_spec = Table.read(os.path.join(home_dir, "sun_composite.fits"))
        Fsun = sun_spec["FLUX"].data * sun_spec["FLUX"].unit * u.A /u.Angstrom
        wsun = sun_spec["WAVE"].data * sun_spec["WAVE"].unit
        d = const.au.to(u.parsec).value
        fsun = Fsun * (d / 10)**2
        mags_sun = sdss.get_ab_magnitudes(fsun, wsun)
        print(mags_sun)
        input()



    def getmass(self, mto, imf1, imf2):
        # normalize the weights so that 1 Msun formed at t = 0
        m2 = 0.5
        m3 = 1.0
        mlo = 0.08
        imfup = 100.0
        imfhi = 100
        bhlim = 40.0
        nslim = 8.5
        # normalize the weights so that 1 Msun formed at t=0
        imfnorm = (m2**(-imf1 + 2) - mlo**(-imf1 + 2)) / (-imf1 + 2) + \
                   m2**(-imf1 + imf2) * (m3 ** (-imf2 + 2) - m2**(-imf2 + 2))\
                  / (-imf2 + 2) + m2**(-imf1 + imf2) * (imfhi**(-imfup + 2)
                   - m3**(-imfup + 2)) / (imfup + 2)
        # stars still alive
        mass = (m2 ** (-imf1 + 2) - mlo ** (-imf1 + 2)) / (-imf1 + 2)

        mext1=  m2**(-imf1+imf2)*(mto**(-imf2+2)-m2**(-imf2+2))/(-imf2+2)
        mext2 = m2**(-imf1+imf2)*(m3**(-imf2+2)-m2**(-imf2+2))/(-imf2+2) + \
             m2**(-imf1+imf2)*(mto**(-imfup+2)-m3**(-imfup+2))/(-imfup+2)
        mass += np.where(mto < m3, mext1, mext2)
        mass /= imfnorm
        # BH remnants
        # 40<M<imf_up leave behind a 0.5*M BH
        d = m2**(imf2 - imf1)
        mass += 0.5 * m2**(-imf1 + imf2) * (imfhi**(-imfup+2) - bhlim**(
                -imfup+2)) / (-imfup+2) / imfnorm

        # NS remnants
        # 8.5<M<40 leave behind 1.4 Msun NS
        mass += 1.4*m2**(-imf1+imf2)*(bhlim**(-imfup+1)-nslim**(-imfup+1))/(-imfup+1)/imfnorm
        # WD remnants
        # M<8.5 leave behind 0.077*M+0.48 WD
        term1 = 0.48*m2**(-imf1+imf2)*(nslim**(-imfup+1)-m3**(-imfup+1))/(-imfup+1)/imfnorm
        term2 = 0.48*m2**(-imf1+imf2)*(m3**(-imf2+1)-mto**(-imf2+1))/(-imf2+1)/imfnorm
        term3 = 0.077*m2**(-imf1+imf2)*(nslim**(-imfup+2)-m3**(-imfup+2))/(-imfup+2)/imfnorm
        term4 = 0.077*m2**(-imf1+imf2)*(m3**(-imf2+2)-mto**(-imf2+2))/(-imf2+2)/imfnorm
        term5 = 0.48*m2**(-imf1+imf2)*(nslim**(-imfup+1)-mto**(-imfup+1))/(-imfup+1)/imfnorm
        term6 = 0.077*m2**(-imf1+imf2)*(nslim**(-imfup+2)-mto**(-imfup+2))/(-imfup+2)/imfnorm
        mass += np.where(mto < m3, term1 + term2 + term3 + term4, term5 + term6)
        return mass

if __name__ == "__main__":
    home_dir = "/home/kadu/Dropbox/CvD18"
    ssp_dir = os.path.join(home_dir, "VCJ_v8")
    outfile = os.path.join(home_dir, "VCJ_v8.fits")
    cvd = CvD(outfile)