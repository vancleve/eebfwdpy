#!/usr/bin/env python3
"""
test metapopulation simulation generation of neutral F_ST values

"""

import sys
from argparse import ArgumentParser
import numpy as np
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
            print(self.data[-1])

    def calc_fst(self, sample, demes, neutral=True):
        seq = libsequence.polytable.SimData(sample)
        if (len(seq) > 0):
            fst = libsequence.fst.Fst(seq, demes)
            fst_val = fst.hbk()
        else:
            fst_val = float('nan')

        return fst_val


def evolve_pop(ngens, N, migm, z0, mu, rec, sig, slope, b, bex, c, cex,
               step=1, seed=42, build=False):

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


def main():
    parser = ArgumentParser(prog='command', description='test FST values')

    parser.add_argument('--build', action="store_true", default=False,
                        help="increase output verbosity")
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--mig', type=float, default=0.1)
    parser.add_argument('--rec', type=float, default=0.01)
    parser.add_argument('--size_demes', type=int, default=10)
    parser.add_argument('--num_demes', type=int, default=100)
    parser.add_argument('--ngens', type=int, default=10**4)
    parser.add_argument('--z0', type=float, default=0.5)
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    if args.build:
        import cppimport
        cppimport.force_rebuild()
        cppimport.imp("genetic_values")
        cppimport.imp("wright_fisher_slocus_metapop")
        cppimport.imp("helping_metapop")
        sys.exit()

    np.random.seed(args.seed)
    pop, sampler = evolve_pop(args.ngens,
                              args.size_demes,
                              islandmat(args.num_demes, args.mig),
                              args.z0,
                              args.mu,
                              args.rec,
                              0.01,
                              6.0,
                              0.0, 1.0, 0.0, 1.0,
                              args.step,
                              np.random.randint(0, 2**32),
                              args.build)


# Run main function
if __name__ == "__main__":
    main()
