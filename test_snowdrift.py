#We use cppimport to build the module.
#For the sake of unit testing, we force
#a rebuild every time, but that clearly
#wouldn't be needed for normal use.
import cppimport
cppimport.force_rebuild()
cppimport.set_quiet(False)
snowdrift = cppimport.imp("snowdrift")
import unittest
import numpy as np
import fwdpy11 as fp11
import fwdpy11.wright_fisher
import fwdpy11.model_params
import matplotlib.pyplot as plt


class phenotype_sampler(object):
    """
    Temporal sampler checks that one can hook
    a stateful fitness model to a sampler
    and access its data and that the data
    are as expected.
    """
    def __init__(self, step, f):
        self.step = step
        self.f = f
        self.samples = list()

    def __call__(self, pop):
        if pop.generation % self.step == 0:
            self.samples.append(self.f.phenotypes)


def evolve_snowdrift_dist(ngens, N, x0, mu, sig, slope, b1, b2, c1, c2, step=1, seed=42):
    pop = fp11.SlocusPop(N)

    # Initialize a random number generator
    rng = fp11.GSLrng(seed)
    snowdrift_fitness = snowdrift.SlocusSnowdrift(rng, b1, b2, c1, c2, slope, x0)

    nsteps = int(ngens / step)
    # nlist = np.array([N]*step,dtype=np.uint32)
    nlist = np.array([N]*ngens, dtype=np.uint32)

    p = {'sregions': [fp11.GaussianS(0, 1, 1, sig, 1.0)],
         'recregions': [fp11.Region(0, 1, 1)],
         'nregions': [],
         'gvalue': snowdrift_fitness,
         'demography': nlist,
         'rates': (0.0, mu, 0.0),
         'prune_selected': False}

    params = fp11.model_params.SlocusParams(**p)

    snowdrift_sampler = snowdrift.SamplerSnowdrift(step, snowdrift_fitness)
    # snowdrift_sampler = phenotype_sampler(step, snowdrift_fitness)

    fp11.wright_fisher.evolve(rng, pop, params, snowdrift_sampler)

    phenos = list()
    for step in range(nsteps):
        fp11.wright_fisher.evolve(rng, pop, params)

        # must append as 'list' otherwise weird buffer problem
        phenos.append(list(snowdrift_fitness.phenotypes))

    # return population, fitness, and phenotypes
    return pop, snowdrift_fitness, phenos, snowdrift_sampler


def plotEvolveDist(vals, bins=100, interpolation=None, vmin=None, vmax=None, tmult=1, phenorange=[0, 1]):
    parr = np.array(vals)
    nruns = parr.shape[0]

    phist = np.zeros((nruns, bins))
    for i in range(nruns):
        ph, xe = np.histogram(parr[i].ravel(), bins=bins, range=phenorange)
        phist[i] = ph

    plt.imshow(phist, aspect='auto', interpolation=interpolation,
               origin='lower', cmap=plt.cm.binary, extent=(min(xe),max(xe), 0, nruns),
               vmin=vmin, vmax=vmax)
    plt.yticks(np.around(np.linspace(0, nruns, 11), -1), np.around(np.linspace(0, (nruns-1)*tmult+1, 11), -1).astype('int'))
    plt.show()


### Examples in iPython

### Test Fig 1A in Doebeli and Hauert (2004, Science)
### The phenotype distirbution starts at 0.1 and should evolve towards 0.6
### and then branch due to disruptive selection. Middle branch are heterozygotes (not present in Doebeli & Hauert)

# time step=10; nstep=2000; pop, fit, phe = evolve_snowdrift_dist(nstep*step, 1000, 0.1, 0.01, 0.01, 6, 6, -1.4, 4.56, -1.6, step, 314)
# plotEvolveDist(phe,vmax=100)

# ### Test Fig 1B in Doebeli and Hauert (2004, Science)
# ### The phenotype distirbution starts at 0.1 and should evolve towards 0.6

# time step=10; nstep=2000; pop, fit, phe = evolve_snowdrift_dist(nstep*step, 1000, 0.1, 0.01, 0.01, 6, 7, -1.5, 4.6, -1, step, 314)
# plotEvolveDist(phe,vmax=100)

# ### Test Fig 1D in Doebeli and Hauert (2004, Science)
# ### The phenotype distirbution starts at 0.9 and should evolve towards 0.0

# time step=10; nstep=2000; pop, fit, phe = evolve_snowdrift_dist(nstep*step, 1000, 0.9, 0.01, 0.01, 6, 7, -1.5, 8, -1, step, 314)
# plotEvolveDist(phe,vmax=100)

# ### Test Fig 1E in Doebeli and Hauert (2004, Science)
# ### The phenotype distirbution starts at 0.1 and should evolve towards 1.0

# time step=10; nstep=2000; pop, fit, phe = evolve_snowdrift_dist(nstep*step, 1000, 0.1, 0.01, 0.01, 6, 7, -1.5, 2, -1, step, 314)
# plotEvolveDist(phe,vmax=100)
