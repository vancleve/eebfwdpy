import cppimport
cppimport.set_quiet(False)
cppimport.force_rebuild()
cppimport.imp("genetic_values")
cppimport.imp("wright_fisher_slocus_metapop")
helping = cppimport.imp("helping_metapop")

import unittest
import numpy as np
import matplotlib.pyplot as plt
import fwdpy11 as fp11
from wright_fisher_metapop import evolve


def evolve_helping(ngens, N, x0, mu, sig, slope, b, c,
                   step=1, seed=42):
    pop = fp11.SlocusPop(N)

    # Initialize a random number generator
    rng = fp11.GSLrng(seed)

    # nlist = np.array([N]*step,dtype=np.uint32)
    nlist = np.array([N]*ngens, dtype=np.uint32)

    p = {'sregions': [fp11.GaussianS(0, 1, 1, sig, 1.0)],
         'recregions': [fp11.Region(0, 1, 1)],
         'nregions': [],
         'gvalue': helping.SlocusHelpingMetapop(rng, b, c, slope, x0),
         'demography': nlist,
         'rates': (0.0, mu, 0.0),
         'prune_selected': False}

    params = fp11.model_params.ModelParams(**p)

    helping_sampler = helping.PhenotypeSampler(step)

    fp11.wright_fisher.evolve(rng, pop, params, helping_sampler)

    return pop, helping_sampler


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


### Examples in iPython

# %matplotlib

# time step=10; nstep=2000; pop, sampler = evolve_helping(nstep*step, 1000, 0.1, 0.01, 0.01, 6, 6, -1.4, 4.56, -1.6, step, 314)
#plotEvolveDist(sampler.phenotypes, vmax=100)
