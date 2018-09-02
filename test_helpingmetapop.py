# build packages
import cppimport
cppimport.set_quiet(False)
#cppimport.force_rebuild()
cppimport.imp("genetic_values")
cppimport.imp("wright_fisher_slocus_metapop")
helping = cppimport.imp("helping_metapop")
import numpy as np
import matplotlib.pyplot as plt
import fwdpy11 as fp11
import fwdpy11.model_params


def evolve_helping(ngens, N, m, x0, mu, sig, slope, b, bex, c, cex,
                   step=1, seed=42):
    from wright_fisher_metapop import evolve

    # Initialize a random number generator
    rng = fp11.GSLrng(seed)

    nd = m.shape[0]
    nlist = np.array([N]*ngens*nd, dtype=np.uint32).reshape(ngens, nd)
    pop = fp11.SlocusPop(N*nd)

    p = {'sregions': [fp11.GaussianS(0, 1, 1, sig, 1.0)],
         'recregions': [fp11.Region(0, 1, 1)],
         'nregions': [],
         'gvalue': helping.SlocusHelping(rng,
                                         b, bex, c, cex, slope, x0),
         'demography': (nlist, m),
         'rates': (0.0, mu, 0.0),
         'prune_selected': False}

    params = fp11.model_params.ModelParams(**p)

    sampler = helping.PhenotypeSampler(step)

    evolve(rng, pop, params, sampler)

    return pop, sampler


def plotEvolveDist(vals, bins=100, interpolation=None, vmin=None, vmax=None,
                   tmult=1, phenorange=[0, 1]):
    parr = np.array(vals)
    nruns = parr.shape[0]

    phist = np.zeros((nruns, bins))
    for i in range(nruns):
        ph, xe = np.histogram(parr[i].ravel(), bins=bins, range=phenorange)
        phist[i] = ph

    plt.imshow(phist, aspect='auto', interpolation=interpolation,
               origin='lower', cmap=plt.cm.binary,
               extent=(min(xe), max(xe), 0, nruns),
               vmin=vmin, vmax=vmax)
    plt.yticks(np.around(np.linspace(0, nruns, 11), -1),
               np.around(np.linspace(0, (nruns-1)*tmult+1, 11), -1).astype('int'))
    plt.show()


def islandmat(nd, m):
    return ((1-m) - m/(nd-1)) * np.eye(nd) + m / (nd-1) * np.ones((nd, nd))


### Examples in iPython

# %matplotlib

evolve_helping(1000, 4, islandmat(16,0.1), 0.5, 0.02, 0.1, 6,
               20.0, 1.0, 0.8, 10.0, step=10)

# time step=10; nstep=2000; pop, sampler = evolve_helping(nstep*step, 1000, 0.1, 0.01, 0.01, 6, 6, -1.4, 4.56, -1.6, step, 314)
#plotEvolveDist(sampler.phenotypes, vmax=100)
