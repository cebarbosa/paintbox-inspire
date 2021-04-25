""" Run paintbox for INSPIRE test cases. """
import os
import glob
import shutil
import pickle
import platform

from astropy.table import Table, vstack
import astropy.constants as const
import numpy as np
from scipy import stats
import paintbox as pb
from paintbox.utils import CvD18
from paintbox.utils.disp2vel import disp2vel
import emcee
from dynesty import NestedSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp

import context

def make_priors(parnames, ssp_ranges, wranges):
    """ Define priors for the model. """
    priors = {}
    priors["Vsyst"] = stats.uniform(loc=-1000, scale=2000)
    priors["sigma"] = stats.uniform(loc=50, scale=300)
    priors["eta"] = stats.uniform(loc=1., scale=100)
    priors["nu"] = stats.uniform(loc=2.01, scale=20)
    for param in parnames:
        psplit = param.split("_")
        if len(psplit) > 1:
            pname, number = psplit
        else:
            pname = param
        if pname in ssp_ranges:
            vmin, vmax = ssp_ranges[pname]
            priors[param] = stats.uniform(loc=vmin, scale=vmax - vmin)
    for i in range(1000):
        parname = "poly_{}".format(i)
        if parname not in parnames:
            continue
        if i == 0:
            sd = 1
            a = -1 / sd
            b = np.infty
            priors[parname] = stats.truncnorm(a, b, 1, sd)
        else:
            priors[parname] = stats.norm(0, 0.05)
    for param in parnames:
        if param not in priors:
            print("Missing prior for {}".format(param))
    return priors

def log_probability(theta):
    global priors
    global logp
    lp = np.sum([priors[p].logpdf(x) for p, x in zip(logp.parnames, theta)])
    if not np.isfinite(lp) or np.isnan(lp):
        return -np.inf
    ll = logp(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def run_sampler(outdb, nsteps=5000):
    global logp
    global priors
    ndim = len(logp.parnames)
    nwalkers = 2 * ndim
    pos = np.zeros((nwalkers, ndim))
    logpdf = []
    for i, param in enumerate(logp.parnames):
        logpdf.append(priors[param].logpdf)
        pos[:, i] = priors[param].rvs(nwalkers)
    backend = emcee.backends.HDFBackend(outdb)
    backend.reset(nwalkers, ndim)
    pool_size = 1
    if platform.node() in context.mp_pool_size:
        pool_size = context.mp_pool_size[platform.node()]
    pool = mp.Pool(pool_size)
    with pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                         backend=backend, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)
    return

class PriorTransform():
    def __init__(self, parnames, priors):
        self.parnames = parnames
        self.priors = priors

    def __call__(self, u):
        x = np.zeros(len(u))
        for i, param in enumerate(self.parnames):
            x[i] = self.priors[param].ppf(u[i])
        return x

def run_dynesty(logp, priors, dbname):
    """ Perform fitting with dynesty. """
    pool_size = 1
    pool = None
    if platform.node() in context.mp_pool_size:
        pool_size = context.mp_pool_size[platform.node()]
        pool = mp.Pool(pool_size)
        print("Pool size: ", pool_size)
    prior_transform = PriorTransform(logp.parnames, priors)
    ndim = len(logp.parnames)
    sampler = NestedSampler(logp, prior_transform, ndim, queue_size=pool_size,
                            pool=pool)
    sampler.run_nested()
    results = sampler.results
    with open(dbname, "wb") as f:
        pickle.dump(results, f)

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

def plot_fitting(wave, flux, fluxerr, mask, sed, trace, output,
                 skylines=None, bestfit=None):
    fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={
        'height_ratios': [2, 1]}, figsize=(2 * 3.54, 3))
    ax0 = fig.add_subplot(axs[0])
    t = np.array([trace[p].data for p in sed.parnames]).T
    k = np.array([trace.colnames.index(p) for p in sed.parnames])
    for i in range(len(logp.parnames)):
        print(logp.parnames[i], bestfit[i])
    n = len(t)
    flux = np.ma.masked_array(flux, mask=mask)
    fluxerr = np.ma.masked_array(fluxerr, mask=mask)
    # models = np.zeros((n, len(wave)))
    # for j in tqdm(range(len(trace)), desc="Generating models "
    #                                                  "for trace"):
    #     models[j] = seds[i](t[j])
    # y = np.percentile(models, 50, axis=(0,))
    # y = np.ma.masked_array(y, mask=masks[i])
    # yuerr = np.percentile(models, 84, axis=(0,)) - y
    # ylerr = y - np.percentile(models, 16, axis=(0,))
    theta = bestfit[:-2]
    y = np.ma.masked_array(sed(theta), mask=mask)
    ax0.errorbar(wave, flux, yerr=fluxerr, fmt="-",
                 ecolor="0.8", c="tab:blue")
    ax0.plot(wave, y, c="tab:orange")
    ax0.xaxis.set_ticklabels([])
    ax0.set_ylabel("Flux")
    ax1 = fig.add_subplot(axs[1])
    ax1.errorbar(wave, 100 * (flux - y) / flux, yerr=100 * fluxerr, \
                                                            fmt="-",
                 ecolor="0.8", c="tab:blue")
    ax1.plot(wave, 100 * (flux - y) / flux, c="tab:orange")
    ax1.set_ylabel("Res. (\%)")
    ax1.set_xlabel("$\lambda$ (Angstrom)")
    # ax1.set_ylim(-5, 5)
    ax1.axhline(y=0, ls="--", c="k")
    # Include sky lines shades
    if skylines is not None:
        for ax in [ax0, ax1]:
            w0, w1 = ax0.get_xlim()
            for skyline in skylines:
                if (skyline < w0) or (skyline > w1):
                    continue
                ax.axvspan(skyline - 3, skyline + 3, color="0.9",
                           zorder=-100)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.show()
    plt.clf()
    plt.close(fig)
    return

def run_testdata(sampler="emcee", redo=False, sigma=300, nsteps=5000,
                 elements=None):
    """ Run paintbox on test galaxies. """
    global logp
    global priors
    elements = ["C", "N", "Na", "Mg", "Si", "Ca", "Ti", "Fe", "K", "Cr",
                "Mn", "Ba", "Ni", "Co", "Eu", "Sr", "V", "Cu", "as/Fe"] if \
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
                elements=elements, norm=True)
    ssp_kin = pb.LOSVDConv(ssp, losvdpars=["Vsyst", "sigma"])
    # Perform fitting
    data_dir = os.path.join(context.data_dir, "test")
    galaxies = os.listdir(data_dir)
    wranges = ["UVB", "VIS", "NIR"]
    # Test values
    pssp = [0, 10, 1.5, 1.5]
    pelements = [0] * len(elements)
    pkin = [100, 150]
    p0s = []
    for galaxy in galaxies:
        print(galaxy)
        gal_dir = os.path.join(data_dir, galaxy)
        fname = os.path.join(gal_dir, "{}_sig{}.fits".format(galaxy, sigma))
        # Read tables
        ts = [Table.read(fname, hdu=i + 1) for i in range(len(wranges))]
        wave = np.hstack([t["wave"].data for t in ts])
        flam = np.hstack([t["flam"].data for t in ts])
        flamerr = np.hstack([t["flamerr"].data for t in ts])
        mask = np.hstack([t["mask"].data.astype(bool) for t in ts])
        # Making paintbox model
        porder = int((wave.max() - wave.min()) / 200)
        poly = pb.Polynomial(wave, porder, zeroth=True, pname="poly")
        # Making paintbox model
        sed = pb.Resample(wave, ssp_kin) * poly
        ppoly = [0] * (porder + 1)
        ppoly[0] = 1.
        # Vector for tests
        p0 = np.array(pssp + pelements + pkin + ppoly)
        norm = np.ma.median(np.ma.masked_array(flam, mask=mask))
        flam /= norm
        flamerr /= norm
        logp = pb.StudT2LogLike(flam, sed, obserr=flamerr, mask=mask)
        priors = make_priors(logp.parnames, ssp.limits, wranges)
        if sampler == "emcee":
            # Running fit
            dbname = "{}_studt2_{}.h5".format(galaxy, nsteps)
            # Run in any directory outside Dropbox to avoid conflicts
            tmp_db = os.path.join(os.getcwd(), dbname)
            if os.path.exists(tmp_db):
                os.remove(tmp_db)
            outdb = os.path.join(gal_dir, dbname)
            if not os.path.exists(outdb) or redo:
                run_sampler(tmp_db, nsteps=nsteps)
                shutil.move(tmp_db, outdb)
            # Load database and make a table with summary statistics
            reader = emcee.backends.HDFBackend(outdb)
            tracedata = reader.get_chain(discard=0,
                                         thin=5000, flat=True)
            trace = Table(tracedata, names=logp.parnames)
            bestfit = np.percentile(tracedata, 50, axis=(0,))
            outtab = os.path.join(outdb.replace(".h5", "_results.fits"))
            make_table(trace, outtab)
            # outimg = outdb.replace(".h5", "fitting.png")
            # plot_fitting(wave, flam, flamerr, mask, sed, trace, outimg,
            #              bestfit=bestfit)
        elif sampler == "dynesty":
            dbname = "{}_studt2_dynesty.pkl".format(galaxy)
            outdb = os.path.join(gal_dir, dbname)
            if not os.path.exists(dbname) or redo:
                run_dynesty(logp, priors, outdb)
            with open(outdb, "rb") as f:
                results = pickle.load(f)
            # samples = results.samples
if __name__ == "__main__":
    run_testdata()

