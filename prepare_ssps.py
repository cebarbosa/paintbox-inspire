""" Prepares EMILES templates for the fitting. """
import os

import numpy as np
from spectres import spectres
import astropy.units as u
import astropy.constants as const
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
from tqdm import tqdm
import paintbox as pb
import matplotlib.pyplot as plt

import context

class Emiles_Basti_Interpolated():
    """ Class to handle data from the EMILES SSP models. """
    def __init__(self, wave=None, libpath=None, norm=True, store=None,
                 sigma=300, params=None):
        self.wave = wave
        self.libpath = "" if libpath is None else libpath
        self.norm = norm
        self.store = "" if store is None else store
        self.sigma = sigma
        self.params = {} if params is None else params
        self.read_emiles_resolution()
        self.check_params()
        if os.path.isfile(self.store):
            self.load_templates()
        else:
            self._check_libpath()
            self._ssp_filenames = self.get_ssp_filenames()
            self.set_wave(wave)
            self.process_ssps()
            self.process_respfun()
            if self.store is not None:
                self.save_templates()
        self.build_model()

    def _check_libpath(self):
        """ Test if libpath value is valid. """
        is_valid = os.path.isfile(self.libpath) or os.path.islink(self.libpath)
        msg = "Valid libpath is necessary if models are not stored. "
        if not is_valid:
            raise ValueError(msg)

    def read_emiles_resolution(self):
        """ Get the wavelength dispersion, FWHM and sigma for E-MILES. """
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tables/e-miles_spectral_resolution.dat")
        self.wave_emiles, self.fwhm_emiles, self.sigma_emiles = np.loadtxt(
            filename, unpack=True, skiprows=1)
        return

    def check_params(self):
        """ Get allowed parameters of models. """
        default_params = {}
        default_params["gamma_b"]= np.array([0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8,
                                             2.0, 2.3, 2.5, 2.8, 3.0, 3.3, 3.5])
        default_params["ZH"] = np.array([-0.96, -0.66, -0.35, -0.25, 0.06,
                                          0.15, 0.26, 0.4])
        default_params["age"] = np.linspace(1., 14., 27)
        default_params["alphaFe"] = np.array([0., 0.2, 0.4])
        default_params["NaFe"] = np.array([0., 0.3, 0.6])
        for k in default_params.keys():
            if k not in self.params:
                self.params[k] = default_params[k]
            else:
                test = np.isin(self.params[k], default_params[k]).all()
                if not test:
                    msg = f"Parameters for {k} should be a" \
                          f"subset of {default_params[k]}"
                    raise ValueError(msg)
                self.params[k] = np.unique(self.params[k])

    def get_ssp_filenames(self):
        """ Get the name of SSP filenames. """
        grid = np.array(np.meshgrid(*[self.params[k] for k in
                                     self.params.keys()]),
                        dtype=object).T.reshape(-1, 5)
        filenames = []
        for args in grid:
            fname = os.path.join(self.libpath, self.get_filename(*args))
            if not os.path.exists(fname):
                raise OSError(f"Model file does not found in libpath: {fname}")
            filenames.append(fname)
        return filenames

    def set_wave(self, wave):
        if wave is None:
            self.wave = self.wave_emiles
        elif hasattr(wave, "unit"):
            self.wave = wave.to(u.Angstrom).value
        else:
            self.wave = wave #Assumes units are Angstrom
        assert self.wave.min() >= 1680.2, "Minimum wavelength is 3501 Angstrom"
        assert self.wave.max() <= 49999.4, "Maximum wavelength is 25000 " \
                                           "Angstrom"
        return

    def get_filename(self, imf, metal, age, alpha, na):
        """ Returns the name of files for the EMILES library. """
        msign = "p" if metal >= 0. else "m"
        esign = "p" if alpha >= 0. else "m"
        azero = "0" if age < 10. else ""
        nasign = "p" if na >= 0. else "m"
        return "Ebi{0:.2f}Z{1}{2:.2f}T{3}{4:02.4f}_Afe{5}{6:2.1f}_NaFe{7}{" \
               "8:1.1f}.fits".format(imf, msign, abs(metal), azero, age, esign,
                                     abs(alpha), nasign, na)

    def process_ssps(self):
        """ Process SSP models. """
        ssp_files = self.get_ssp_filenames()
        # Check if all files exist
        check_files = all([os.path.exists(_) for _ in ssp_files])
        assert check_files, "There are missing model files in libpath."


        velscale = int(self.sigma / 4)
        kernel_sigma = np.sqrt(self.sigma**2 - 100 ** 2) / velscale
        ssps, params = [], []

        for fname in tqdm(ssp_files, desc="Processing SSP files"):
            data = fits.getdata(fname)
            ssps.append(data)
            continue
            spec = os.path.split(fname)[1]
            T = float(spec.split("_")[3][1:])
            Z = float(spec.split("_")[4][1:-8].replace("p", "+").replace(
                        "m", "-"))
            for i, (x1, x2) in enumerate(zip(x1s, x2s)):
                params.append(Table([[Z], [T], [x1], [x2]],
                                    names=["Z", "Age", "x1", "x2"]))
            data = np.loadtxt(fname)
            w = data[:,0]
            if self.sigma > 100:
                wvel = logspace_dispersion(w, velscale)
            ssp = data[:, 1:].T
            if self.sigma <= 100:
                newssp = spectres(self.wave, w, ssp, fill=0, verbose=False)
            else:
                ssp_rebin = spectres(wvel, w, ssp, fill=0, verbose=False)
                ssp_broad = gaussian_filter1d(ssp_rebin, kernel_sigma,
                                              mode="constant", cval=0.0)
                newssp = spectres(self.wave, wvel, ssp_broad, fill=0,
                                  verbose=False)

            ssps.append(newssp)
        ssps = np.array(ssps)
        print(ssps.shape)
        input()
        self.params = vstack(params)
        self.templates = np.vstack(ssps)

        self.fluxnorm = np.median(self.templates, axis=1) if self.norm else 1.
        self.templates /= self.fluxnorm[:, np.newaxis]
        return

    def build_model(self):
        """ Build model with paintbox SED methods. """
        ssp = pb.ParametricModel(self.wave, self.params, self.templates)
        self._limits = {}
        for p in self.params.colnames:
            vmin = self.params[p].data.min()
            vmax = self.params[p].data.max()
            self._limits[p] = (vmin, vmax)

        self._response_functions = {}
        for element in self.elements:
            rf = pb.ParametricModel(self.wave, self.rfpars[element], self.rfs[
                element])
            self.response_functions[element] = rf
            ssp = ssp * rf
            vmin = rf.params[element].data.min()
            vmax = rf.params[element].data.max()
            self._limits[element] = (vmin, vmax)
        if len(self.elements) > 0: # Update limits in case response functions
            # are used.
            for p in ["Age", "Z"]:
                vmin = rf.params[p].data.min()
                vmax = rf.params[p].data.max()
                self._limits[p] = (vmin, vmax)
        self._interpolator = ssp.constrain_duplicates()
        self._parnames = self._interpolator.parnames
        self._nparams = len(self.parnames)

    def __call__(self, theta):
        """ Returns a model for a given set of parameters theta. """
        return self._interpolator(theta)

    def save_templates(self):
        hdu0 = fits.PrimaryHDU()
        hdu1 = fits.BinTableHDU(Table([self.wave], names=["wave"]))
        hdu1.header["EXTNAME"] = "WAVE"
        hdu2 = fits.ImageHDU(self.templates * self.fluxnorm[:, None])
        hdu2.header["EXTNAME"] = "DATA.SSPS"
        params = Table(self.params)
        hdu3 = fits.BinTableHDU(params)
        hdu3.header["EXTNAME"] = "PARS.SSPS"
        hdulist = fits.HDUList([hdu0, hdu1, hdu2, hdu3])
        for element in self._all_elements:
            hdudata = fits.ImageHDU(self.rfs[element])
            hdudata.header["EXTNAME"] = f"DATA.{element}"
            hdulist.append(hdudata)
            hdutable = fits.BinTableHDU(self.rfpars[element])
            hdutable.header["EXTNAME"] = f"PARS.{element}"
            hdulist.append(hdutable)
        hdulist.writeto(self.store, overwrite=True)

    def load_templates(self):
        hdulist = fits.open(self.store)
        nhdus = len(hdulist)
        hdunum = np.arange(1, nhdus)
        hdunames = [hdulist[i].header["EXTNAME"] for i in hdunum]
        hdudict = dict(zip(hdunames, hdunum))
        self.wave = Table.read(self.store, hdu=hdudict["WAVE"])["wave"].data
        self.params = Table.read(self.store, hdu=hdudict["PARS.SSPS"])
        self.templates = hdulist[hdudict["DATA.SSPS"]].data

        self.rfs = {}
        self.rfpars = {}
        for e in self._all_elements:
            self.rfs[e] = hdulist[hdudict[f"DATA.{e}"]].data
            self.rfpars[e] = Table.read(self.store, hdu=hdudict[f"PARS.{e}"])
        self.fluxnorm = np.median(self.templates, axis=1) if self.norm else 1.
        self.templates /= self.fluxnorm[:, np.newaxis]
        return

    @property
    def limits(self):
        """ Lower and upper limits of the model parameters. """
        return self._limits

def prepare_xshooter(w1=3800, w2=18200, sigma=300):
    """ Prepare EMILES templates to be used with X-shooter data. """
    velscale = int(sigma/3)
    wave = pb.logspace_dispersion([w1, w2], velscale)
    libpath = context.emiles_dir
    templates_dir = os.path.join(context.home_dir, "templates")
    if not os.path.exists(templates_dir):
        os.mkdir(templates_dir)
    store = os.path.join(templates_dir,
                         f"templates/emiles_basti_interpolated_w{w1}_"
                         f"{w2}_sig{sigma}")
    params = {"gamma_b": np.array([0.8, 1.3, 1.5]),
              "age": np.linspace(1, 14, 14)}
    print(params)
    emiles_basti_interpolated = Emiles_Basti_Interpolated(libpath=libpath,
                                params=params)

if __name__ == "__main__":
    prepare_xshooter()