#!/usr/bin/env python3
"""
test metapopulation simulation generation of neutral F_ST values

"""

import numpy as np
import pandas as pd
import fwdpy11 as fp11
from fwdpy11.model_params import ModelParams
import fwdpy11.sampling
import libsequence.polytable
import libsequence.fst


class record_fst(object):
    def __init__(self, nlist, step):
        self.step = step
        self.nlist = nlist
        self.data = []

    def __call__(self, pop):
        if pop.generation % self.step == 0:
            demes = list(2*self.nlist[pop.generation])
            dm = pop.sample([i for i in range(pop.N)])
            neutral, selected = fwdpy11.sampling.matrix_to_sample(dm)

            self.data.append((pop.generation,
                              self.calc_fst(neutral, demes, True),
                              self.calc_fst(selected, demes, False)))

    def calc_fst(self, sample, demes, neutral=True):
        seq = libsequence.polytable.SimData(sample)
        if (len(seq) > 0):
            fst = libsequence.fst.Fst(seq, demes)
            fst_val = fst.hsm()
        else:
            fst_val = float('nan')

        return fst_val


def evolve_pop(ngens, N, migm, z0, mu, rec, sig, slope, b, bex, c, cex,
               step=1, seed=42, build=False, **kwargs):

    from wright_fisher_metapop import evolve
    import helping_metapop as helping

    # Initialize a random number generator
    rng = fp11.GSLrng(seed)

    nd = migm.shape[0]
    nlist = np.array([N]*ngens*nd, dtype=np.uint32).reshape(ngens, nd)
    pop = fp11.SlocusPop(N*nd)

    p = {'sregions': [fp11.GaussianS(0, 1, 1, sig, 1.0)],
         'nregions': [fp11.Region(1, 2, 1)],
         'recregions': [fp11.Region(0, 2, 1)],
         'gvalue': helping.SlocusHelping(rng,
                                         b, bex, c, cex, slope, z0),
         'demography': (nlist, migm),
         'rates': (mu, mu, rec),
         'prune_selected': False}

    params = ModelParams(**p)

    sampler = record_fst(nlist, step)

    evolve(rng, pop, params, sampler)

    return pop, sampler, seed


def islandmat(nd, m):
    return ((1-m) - m/(nd-1)) * np.eye(nd) + m / (nd-1) * np.ones((nd, nd))


def read_params(parsdefault, json_file):
    import json
    if json_file:
        with open(json_file) as f:
            parf = json.load(f)

    pars = parsdefault.copy()
    rangfuncs = {'range': np.arange,
                 'linspace': np.linspace,
                 'logspace': np.logspace}

    for p in pars.keys():
        # if there is a json file,
        # then take parameters from that file
        if json_file:
            if p in parf:
                if 'value' in parf[p]:
                    pars[p] = [parf[p]['value']]
                    continue

                for fname, func in rangfuncs.items():
                    if fname in parf[p]:
                        pars[p] = func(**parf[p][fname]).tolist()
                        break
            # else:
            #     print(p, "not found")

        # make sure each parameter is list
        if type(pars[p]) != list:
            pars[p] = [pars[p]]

    return pars


def main():
    import sys
    from argparse import ArgumentParser
    import itertools as it
    from pathos.multiprocessing import ProcessingPool as Pool

    # commmand line arguments
    parser = ArgumentParser(prog='command', description='test FST values')
    parser.add_argument('--build', action="store_true", default=False,
                        help='build and exit')
    parser.add_argument('--cpus', type=int, default=1)
    parser.add_argument('--parfile', type=str)
    parser.add_argument('--output', type=str,
                        help='.feather with pandas DataFrame')
    args = parser.parse_args()

    if args.build:
        import cppimport
        cppimport.force_rebuild()
        cppimport.imp("genetic_values")
        cppimport.imp("wright_fisher_slocus_metapop")
        cppimport.imp("helping_metapop")
        sys.exit()

    # simulation parameters and their defaults
    param_default = {
        'mu': 0.01,
        'mig': 0.1,
        'rec': 0.01,
        'N': 10,
        'nd': 100,
        'ngens': 10**3,
        'z0': 0.5,
        'sig': 0.01,
        'slope': 6.0,
        'b': 0.0,
        'bex': 1.0,
        'c': 0.0,
        'cex': 1.0,
        'step': 100,
        'reps': 10,
        'mseed': 42
        }

    pvals = read_params(param_default, args.parfile)

    # set master seed
    np.random.seed(pvals['mseed'])

    # expand reps to individual replicates
    pvals['rep'] = list(np.arange(pvals.pop('reps')[0]))

    # generate full parameter set w/ replicates and seeds
    psets = list(it.product(*pvals.values()))
    psets = [p + (np.random.randint(0, 2**32),) for p in psets]

    def evomap(pset):
        argd = dict(zip(list(pvals.keys()) + ['seed'], pset))
        mig = argd.pop('mig')
        migm = islandmat(argd['nd'], mig)
        argd['migm'] = migm

        pop, sampler, seed = evolve_pop(**argd)

        # shape data to accumulate in DataFrame
        argd.pop('migm')
        data = np.array(sampler.data)
        print('--> ran sim:', argd)

        return {**argd,
                'gen': data[:, 0].astype('int'),
                'fst_n': data[:, 1],
                'fst_s': data[:, 2]}

    with Pool(args.cpus) as pool:
        results = pool.map(evomap, psets)

    data = pd.DataFrame(results)
    if args.output:
        data.to_hdf(args.output, 'fst_data', mode='w')

    return data


# Run main function
if __name__ == "__main__":
    data = main()
