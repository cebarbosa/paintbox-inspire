""" Run paintbox for INSPIRE test cases. """
import os
import glob
import shutil
import platform
import copy

from astropy.table import Table, vstack, hstack
import numpy as np
from scipy import stats
import paintbox as pb
from paintbox.utils import CvD18, disp2vel
import emcee
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import multiprocessing as mp

import context

def make_paintbox_model(wave, store, dlam=150, nssps=1,
                        sigma=100, wmin=3501, wmax=21500):
    velscale = sigma / 3
    twave = disp2vel(np.array([wmin, wmax], dtype="object"), velscale)
    ssp = CvD18(twave, sigma=sigma, store=store, libpath=context.cvd_data_dir)
    limits = ssp.limits
    porder = int((wave[-1] - wave[0]) / dlam)
    if nssps > 1:
        for i in range(nssps):
            p0 = pb.Polynomial(twave, 0, pname="w")
            p0.parnames = [f"w_{i+1}"]
            s = copy.deepcopy(ssp)
            s.parnames = ["{}_{}".format(_, i+1) for _ in s.parnames]
            if i == 0:
                pop = p0 * s
            else:
                pop += (p0 * s)
    else:
        pop = ssp
    stars = pb.Resample(wave, pb.LOSVDConv(pop, losvdpars=["Vsyst", "sigma"]))
    # Adding a polynomial
    poly = pb.Polynomial(wave, porder, zeroth=True)
    sed = stars * poly
    return sed, limits

def set_priors(parnames, limits, vsyst=0, nssps=1):
    """ Defining prior distributions for the model. """
    priors = {}
    for parname in parnames:
        name = parname.split("_")[0]
        if name in limits:
            vmin, vmax = limits[name]
            delta = vmax - vmin
            priors[parname] = stats.uniform(loc=vmin, scale=delta)
        elif parname == "Vsyst":
            priors[parname] = stats.norm(loc=vsyst, scale=500)
        elif parname == "eta":
            priors["eta"] = stats.uniform(loc=1, scale=10)
        elif parname == "nu":
            priors["nu"] = stats.uniform(loc=2, scale=20)
        elif parname == "sigma":
            priors["sigma"] = stats.uniform(loc=50, scale=300)
        elif name == "w":
            priors[parname] = stats.uniform(loc=0, scale=1)
        elif name == "p":
            porder = int(parname.split("_")[1])
            if porder == 0:
                mu, sd = np.sqrt(2 * nssps), 1
                a, b = (0 - mu) / sd, (np.infty - mu) / sd
                priors[parname] = stats.truncnorm(a, b, mu, sd)
            else:
                priors[parname] = stats.norm(0, 0.05)
        else:
            raise ValueError(f"Parameter without prior: {parname}")
    return priors

def log_probability(theta):
    """ Calculates the probability of a model."""
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
    try:
        pool_size = context.mp_pool_size
    except:
        pool_size = 1
    pool = mp.Pool(pool_size)
    with pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                         backend=backend, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)
    return

def weighted_traces(parnames, trace, nssps):
    """ Combine SSP traces to have mass/luminosity weighted properties"""
    weights = np.array([trace["w_{}".format(i+1)].data for i in range(
        nssps)])
    wtrace = []
    for param in parnames:
        data = np.array([trace["{}_{}".format(param, i+1)].data
                         for i in range(nssps)])
        t = np.average(data, weights=weights, axis=0)
        wtrace.append(Table([t], names=["{}_weighted".format(param)]))
    return hstack(wtrace)

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

def plot_likelihood(logp, outdb):
    nsteps = int(outdb.split("_")[2].split(".")[0])
    reader = emcee.backends.HDFBackend(outdb)
    tracedata = reader.get_chain(discard=int(0.9 * nsteps), thin=500,
                                 flat=True)
    n = len(tracedata)
    llf = np.zeros(n)
    for i in tqdm(range(n)):
        llf[i] = logp(tracedata[i])
    plt.plot(llf)
    plt.show()

def plot_fitting(wave, flux, fluxerr, sed, trace, db, redo=True, sky=None,
                 print_pars=None, mask=None, lw=1, galaxy=None):
    outfig = "{}_fitting".format(db.replace(".h5", ""))
    if os.path.exists("{}.png".format(outfig)) and not redo:
        return
    galaxy = "Observed" if galaxy is None else galaxy
    mask = np.full_like(wave, True) if mask is None else mask
    pmask = mask.astype(np.bool)
    fig_width = 3.54  # inches - A&A template for 1 column
    print_pars = sed.parnames if print_pars is None else print_pars
    ssp_model = "CvD"
    labels = {"imf": r"$\Gamma_b$", "Z": "[Z/H]", "T": "Age (Gyr)",
              "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]",
              "Age": "Age (Gyr)", "x1": "$x_1$", "x2": "$x_2$", "Ca": "[Ca/H]",
              "Fe": "[Fe/H]",
              "Na": "[Na/Fe]" if ssp_model == "emiles" else "[Na/H]",
              "K": "[K/H]", "C": "[C/H]", "N": "[N/H]",
              "Mg": "[Mg/H]", "Si": "[Si/H]", "Ca": "[Ca/H]", "Ti": "[Ti/H]",
              "as/Fe": "[as/Fe]", "Vsyst": "$V$", "sigma": "$\sigma$",
              "Cr": "[Cr/H]", "Ba": "[Ba/H]", "Ni": "[Ni/H]", "Co": "[Co/H]",
              "Eu": "[Eu/H]", "Sr": "[Sr/H]", "V": "[V/H]", "Cu": "[Cu/H]",
              "Mn": "[Mn/H]"}
    # Getting numpy array with trace
    tdata = np.array([trace[p].data for p in sed.parnames]).T
    # Arrays for the clean plot
    w = np.ma.masked_array(wave, mask=pmask)
    f = np.ma.masked_array(flux, mask=pmask)
    ferr = np.ma.masked_array(fluxerr, mask=pmask)
    # Defining percentiles/colors for model plots
    percs = np.linspace(5, 85, 9)
    fracs = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    colors = [cm.get_cmap("Oranges")(f) for f in fracs]
    # Calculating models
    models = np.zeros((len(trace), len(wave)))
    for i in tqdm(range(len(trace)), desc="Loading spectra for plots"):
        models[i] = sed(tdata[i])
    m50 = np.median(models, axis=0)
    mperc = np.zeros((len(percs), len(wave)))
    for i, per in enumerate(percs):
        mperc[i] = np.percentile(models, per, axis=0)
    # Calculating sky model if necessary
    skyspec = np.zeros((len(trace), len(wave)))
    if sky is not None:
        idx = [i for i,p in enumerate(sed.parnames) if p.startswith("sky")]
        skytrace = trace[:, idx]
        for i in tqdm(range(len(skytrace)), desc="Loading sky models"):
            skyspec[i] = sky(skytrace[i])
    sky50 = np.median(skyspec, axis=0)
    s50 = np.ma.masked_array(sky50, mask=pmask)
    # Starting plot
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]},
                            figsize=(2 * fig_width, 3))
    # Main plot
    ax = fig.add_subplot(axs[0])
    ax.plot(w, f, "-", c="0.8", lw=lw)
    ax.fill_between(w, f + ferr, f - ferr, color="C0", alpha=0.7)
    ax.plot(w, f - s50, "-", label=galaxy, lw=lw)
    ax.plot(w, m50 - s50, "-", lw=lw, label="Model")
    for c, per in zip(colors, percs):
        y1 = np.ma.masked_array(np.percentile(models, per, axis=0) - sky50,
                                mask=pmask)
        y2 = np.ma.masked_array(np.percentile(models, per + 10, axis=0) - sky50,
                                mask=pmask)
        ax.fill_between(w, y1, y2, color=c)
    ax.set_ylabel("Normalized flux")
    ax.xaxis.set_ticklabels([])
    plt.legend()
    # Residual plot
    ax = fig.add_subplot(axs[1])
    for c, per in zip(colors, percs):
        y1 = 100 * (flux - np.percentile(models, per, axis=0)) / flux
        y2 = 100 * (flux - np.percentile(models, per + 10, axis=0)) / flux
        y1 = np.ma.masked_array(y1, mask=pmask)
        y2 = np.ma.masked_array(y2, mask=pmask)
        ax.fill_between(w, y1, y2, color=c)
    rmse = np.std((f - m50)/flux)
    ax.plot(w, 100 * (f - m50) / f, "-", lw=lw, c="C1",
            label="RMSE={:.1f}\%".format(100 * rmse))
    ax.axhline(y=0, ls="--", c="k", lw=1, zorder=1000)
    ax.set_xlabel(r"$\lambda$ (\r{A})")
    ax.set_ylabel("Residue (\%)")
    ax.set_ylim(-3 * 100 * rmse, 3 * 100 * rmse)
    plt.legend()
    plt.subplots_adjust(left=0.065, right=0.995, hspace=0.02, top=0.99,
                        bottom=0.11)
    plt.savefig("{}.png".format(outfig), dpi=250)
    plt.show()
    return


def run_dr1(sigma=300, ssps="CvD", nssps=2, nsteps=6000, redo=False,
            fit=False, eta=0.1, lltype="studt2"):
    """ Run paintbox on test galaxies. """
    global logp
    global priors
    pbdir = os.path.join(context.home_dir, f"paintbox")
    version =  f"dr1_sig{sigma}"
    wdir = os.path.join(pbdir, version)

    filenames = sorted([_ for _ in os.listdir(wdir) if
                        _.endswith(f"sig{sigma}.fits")])
    # Prepare models for fitting
    store = os.path.join(pbdir, "CvD18_INSPIRE_dr1.fits")
    for filename in filenames:
        # Read galaxy data
        galaxy = filename.split("_")[0]
        dbname = f"{galaxy}_{lltype}_nsteps{nsteps}.h5"
        outdb = os.path.join(wdir, dbname)
        if not os.path.exists(outdb) and fit is False:
            continue
        table = Table.read(os.path.join(wdir, filename))
        wave = table["wave"].data
        flux = table["flux"].data
        fluxerr = table["fluxerr"].data * eta
        mask = table["mask"].data.astype(np.int)
        wmin = wave[mask==0][0]
        wmax =  wave[mask==0][-1]
        # Cropping wavelenght
        if ssps == "CvD":
            wmin = np.maximum(wmin, 3520)
        idx = np.where((wave >= wmin) & (wave <= wmax))[0]
        wave = wave[idx]
        flux = flux[idx]
        fluxerr = fluxerr[idx]
        mask = mask[idx]
        sed, limits = make_paintbox_model(wave, store, sigma=sigma, nssps=nssps)
        if lltype == "normal2":
            logp = pb.Normal2LogLike(flux, sed, obserr=fluxerr, mask=mask)
        elif lltype == "studt2":
            logp = pb.StudT2LogLike(flux, sed, obserr=fluxerr, mask=mask)
        priors = set_priors(logp.parnames, limits, nssps=nssps)
        # Run in any directory outside Dropbox to avoid conflicts
        tmp_db = os.path.join(os.getcwd(), dbname)
        if os.path.exists(tmp_db):
            os.remove(tmp_db)
        outdb = os.path.join(wdir, dbname)
        if not os.path.exists(outdb) or redo:
            run_sampler(tmp_db, nsteps=nsteps)
            shutil.move(tmp_db, outdb)
        # Load database and make a table with summary statistics
        reader = emcee.backends.HDFBackend(outdb)
        tracedata = reader.get_chain(discard=int(nsteps * 0.94), flat=True,
                                     thin=100)
        trace = Table(tracedata, names=logp.parnames)
        if nssps > 1:
            ssp_pars = list(limits.keys())
            wtrace = weighted_traces(ssp_pars, trace, nssps)
            trace = hstack([trace, wtrace])
        outtab = os.path.join(outdb.replace(".h5", "_results.fits"))
        make_table(trace, outtab)
        if platform.node() == "kadu-Inspiron-5557":
            plot_fitting(wave, flux, fluxerr, sed, trace,  outdb,
                         mask=mask, galaxy=galaxy)


if __name__ == "__main__":
    fit = False if platform.node() == "kadu-Inspiron-5557" else True
    # run_testdata()
    run_dr1(fit=fit)

