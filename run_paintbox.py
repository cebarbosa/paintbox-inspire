""" Run paintbox for INSPIRE test cases. """
import os
import glob
import shutil

from astropy.table import Table, vstack
import numpy as np
from scipy import stats
import paintbox as pb
from paintbox.utils import CvD18
from paintbox.utils.disp2vel import disp2vel
import emcee

import context

def make_priors(parnames, ssp_ranges, wranges):
    """ Define priors for the model. """
    priors = {}
    priors["Vsyst"] = stats.uniform(loc=-1000, scale=2000)
    priors["sigma"] = stats.uniform(loc=50, scale=200)
    priors["eta"] = stats.uniform(loc=1., scale=19)
    priors["nu"] = stats.uniform(loc=2, scale=20)
    for param in parnames:
        psplit = param.split("_")
        if len(psplit) > 1:
            pname, number = psplit
        else:
            pname = param
        if pname in ssp_ranges:
            vmin, vmax = ssp_ranges[pname]
            priors[param] = stats.uniform(loc=vmin, scale=vmax - vmin)
    for wr in wranges:
        # Scale of models
        p = "w{}".format(wr)
        if p in parnames:
            priors[p] = stats.uniform(loc=0, scale=2)
        # Polynomials
        pname = "p{}".format(wr)
        for i in range(1000):
            parname = "{}_{}".format(pname, i)
            if parname not in parnames:
                continue
            if i == 0:
                priors[parname] = stats.uniform(loc=0, scale=2)
            else:
                priors[parname] = stats.norm(0, 0.05)
    for param in parnames:
        if param not in priors:
            print("Missing prior for {}".format(param))
    return priors

def run_sampler(loglike, priors, outdb, nsteps=5000):
    ndim = len(loglike.parnames)
    nwalkers = 2 * ndim
    pos = np.zeros((nwalkers, ndim))
    logpdf = []
    for i, param in enumerate(loglike.parnames):
        logpdf.append(priors[param].logpdf)
        pos[:, i] = priors[param].rvs(nwalkers)
    def log_probability(theta):
        lp = np.sum([prior(val) for prior, val in zip(logpdf, theta)])
        if not np.isfinite(lp) or np.isnan(lp):
            return -np.inf
        ll = loglike(theta)
        if not np.isfinite(ll):
            # print("nu={}".format(theta[loglike.parnames.index("nu")]), ll)
            return -np.inf
        return lp + ll
    backend = emcee.backends.HDFBackend(outdb)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    backend=backend)
    sampler.run_mcmc(pos, nsteps, progress=True)
    return

def make_table(trace, outtab):
    data = np.array([trace[p].data for p in trace.colnames]).T
    v = np.percentile(data, 50, axis=0)
    vmax = np.percentile(data, 84, axis=0)
    vmin = np.percentile(data, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    tab = []
    for i, param in enumerate(trace.colnames):
        t = Table()
        t["param"] = [param]
        t["median"] = [round(v[i], 5)]
        t["lerr".format(param)] = [round(vlerr[i], 5)]
        t["uerr".format(param)] = [round(vuerr[i], 5)]
        tab.append(t)
    tab = vstack(tab)
    tab.write(outtab, overwrite=True)
    return tab

def run_testdata(sigma=300, elements=None, nsteps=5000, redo=False):
    """ Run paintbox on test galaxies. """
    elements = ["C", "N", "Na", "Mg", "Si", "Ca", "Ti", "Fe", "K", "Cr",
                "Mn", "Ba", "Ni", "Co", "Eu", "Sr", "V", "Cu"] if \
                elements is None else elements
    velscale = int(sigma / 3)
    # Prepare models for fitting
    cvd_data_dir = "/home/kadu/Dropbox/SSPs/CvD18"
    ssps_dir = os.path.join(cvd_data_dir, "VCJ_v8")
    ssp_files = glob.glob(os.path.join(ssps_dir, "VCJ*.s100"))
    rfs_dir = os.path.join(cvd_data_dir, "RFN_v3")
    rf_files = glob.glob(os.path.join(rfs_dir, "atlas_ssp*.s100"))
    models_dir = os.path.join(context.home_dir, "CvD18")
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    outname = "testdata_xshooter_sig{}".format(sigma)
    wave = disp2vel(np.array([3510, 21500], dtype="object"), velscale)
    ssp = CvD18(wave, ssp_files=ssp_files, rf_files=rf_files,
                outdir=models_dir, outname=outname,
                elements=elements)
    ssp_kin = pb.LOSVDConv(ssp, losvdpars=["Vsyst", "sigma"])
    # Perform fitting
    data_dir = os.path.join(context.data_dir, "test")
    galaxies = os.listdir(data_dir)
    wranges = ["UVB", "VIS", "NIR"]
    # Test values
    pssp = [0, 10, 2.5, 2.5]
    pelements = [0] * len(elements)
    pkin = [100, 150]
    for galaxy in galaxies:
        gal_dir = os.path.join(data_dir, galaxy)
        fname = os.path.join(gal_dir, "{}_sig{}.fits".format(galaxy, sigma))
        seds, logps = [], []
        norms = np.zeros((3,))
        for i, wr in enumerate(wranges):
            t = Table.read(fname, hdu=i + 1)
            wave = t["wave"].data
            flux = t["flam"].data
            fluxerr = t["flamerr"].data
            mask = t["mask"].data.astype(bool)
            # Normalize data to
            norm = np.median(flux)
            flux /= norm
            fluxerr /= norm
            norms[i] = norm
            # Determination of polynomial order
            porder = int((wave[-1] - wave[0]) / 200)
            poly = pb.Polynomial(wave, porder, zeroth=True,
                                 pname="p{}".format(wr))

            # Making paintbox model
            sed = pb.Resample(wave, ssp_kin) * poly
            ppoly = [1] * (porder + 1)
            # p0 = np.array(pssp + pelements + pkin + ppoly)
            # print(sed(p0))
            # Changing name of the flux parameters to avoid confusion
            seds.append(sed)
            logp = pb.StudT2LogLike(flux, sed, obserr=fluxerr, mask=mask)
            logps.append(logp)
        logp = logps[0] + logps[1] + logps[2]
        priors = make_priors(logp.parnames, ssp.limits, wranges)
        # Running fit
        dbname = "{}_studt2_{}.h5".format(galaxy, nsteps)
        # Run in any directory outside Dropbox to avoid conflicts
        tmp_db = os.path.join(os.getcwd(), dbname)
        if os.path.exists(tmp_db):
            os.remove(tmp_db)
        outdb = os.path.join(gal_dir, dbname)
        if not os.path.exists(outdb) or redo:
            run_sampler(logp, priors, tmp_db, nsteps=nsteps)
            shutil.move(tmp_db, outdb)
        # Load database and make a table with summary statistics
        reader = emcee.backends.HDFBackend(outdb)
        tracedata = reader.get_chain(discard=int(0.9 * nsteps),
                                     thin=100, flat=True)
        print(tracedata.shape)
        print(len(logp.parnames))
        input()
        trace = Table(tracedata, names=logp.parnames)
        outtab = os.path.join(outdb.replace(".h5", "_results.fits"))
        make_table(trace, outtab)

if __name__ == "__main__":
    run_testdata(redo=True, nsteps=10)
