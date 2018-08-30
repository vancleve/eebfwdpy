#
# Copyright (C) 2017 Kevin Thornton <krthornt@uci.edu>
#
# This file is part of fwdpy11.
#
# fwdpy11 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# fwdpy11 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with fwdpy11.  If not, see <http://www.gnu.org/licenses/>.
#


def evolve(rng, pop, params, recorder=None):
    """
    Evolve a population

    :param rng: An instance of :class:`fwdpy11.GSLrng`
    :param pop: An instance of :class:`fwdpy11.SlocusPop`
    :param params: An instance of :class:`fwdpy11.model_params.SlocusParams`
    :param recorder: (None) A temporal sampler/data recorder.

    .. note::
        If recorder is None,
        then :class:`fwdpy11.temporal_samplers.RecordNothing` will be used.

    """
    import warnings
    # Test parameters while suppressing warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Will throw exception if anything is wrong:
        params.validate()

    from fwdpy11.internal import makeMutationRegions, makeRecombinationRegions
    from wright_fisher_slocus_metapop import WFSlocusMetapop
    pneutral = params.mutrate_n/(params.mutrate_n+params.mutrate_s)
    mm = makeMutationRegions(rng, pop, params.nregions,
                             params.sregions, pneutral)
    rm = makeRecombinationRegions(rng, params.recrate, params.recregions)

    if recorder is None:
        from fwdpy11.temporal_samplers import RecordNothing
        recorder = RecordNothing()

    WFSlocusMetapop(rng, pop, params.demography[0], params.demography[1], params.mutrate_n, params.mutrate_s,
                    params.recrate, mm, rm, params.make_gvalue(), recorder, params.pself, params.prune_selected)
