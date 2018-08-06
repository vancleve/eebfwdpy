//
// Copyright (C) 2017 Kevin Thornton <krthornt@uci.edu>
//
// This file is part of fwdpy11.
//
// fwdpy11 is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// fwdpy11 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with fwdpy11.  If not, see <http://www.gnu.org/licenses/>.
//
#ifndef EEBFWDPY_SLOCUSMETAPOP_GENERATION_HPP__
#define EEBFWDPY_SLOCUSMETAPOP_GENERATION_HPP__

#include <tuple>
#include <type_traits>
#include <fwdpp/internal/gamete_cleaner.hpp>
#include <fwdpp/insertion_policies.hpp>
#include <fwdpp/mutate_recombine.hpp>
#include <fwdpy11/rng.hpp>
#include <fwdpy11/types/SlocusPop.hpp>
#include <fwdpy11/genetic_values/SlocusPopGeneticValue.hpp>
#include <gsl/gsl_randist.h>

namespace eebfwdpy
{
    template <typename poptype, typename pick1_function,
              typename pick2_function, typename update_function,
              typename mutation_model, typename recombination_model>
    void
    evolve_generation(
        const GSLrng_t& rng, poptype& pop, const std::vector<std::size_t> N_next,
        const double mu,
        const mutation_model& mmodel, const recombination_model& recmodel,
        const pick1_function& pick1, const pick2_function& pick2,
        const update_function& update)
    {
        static_assert(
            std::is_same<typename poptype::popmodel_t,
                         fwdpp::sugar::SINGLELOC_TAG>::value,
            "Population type must be a single-locus type.");

        auto gamete_recycling_bin
            = fwdpp::fwdpp_internal::make_gamete_queue(pop.gametes);
        auto mutation_recycling_bin
            = fwdpp::fwdpp_internal::make_mut_queue(pop.mcounts);

        // Efficiency hit.  Unavoidable
        // in use case of a sampler looking
        // at the gametes themselves (even tho
        // gamete.n has little bearing on anything
        // beyond recycling).  Can revisit later
        for (auto&& g : pop.gametes)
            g.n = 0;

        // Total metapopulation size
        std::size_t N_next_tot = 0;
        for (auto& n : N_next)
            N_next_tot += n;

        decltype(pop.diploids) offspring(N_next_tot);
        decltype(pop.diploid_metadata) offspring_metadata(N_next_tot);
        // Generate the offspring
        std::size_t label = 0;
        for (auto& dip : offspring)
            {
                // Pass `label` to pick functions to identify deme
                auto p1 = pick1(label);
                auto p2 = pick2(label, p1);

                auto p1g1 = pop.diploids[p1].first;
                auto p1g2 = pop.diploids[p1].second;
                auto p2g1 = pop.diploids[p2].first;
                auto p2g2 = pop.diploids[p2].second;

                // Mendel
                if (gsl_rng_uniform(rng.get()) < 0.5)
                    std::swap(p1g1, p1g2);
                if (gsl_rng_uniform(rng.get()) < 0.5)
                    std::swap(p2g1, p2g2);

                // Update to fwdpp 0.5.7 API
                // in fwdpy11 0.1.4
                fwdpp::mutate_recombine_update(
                    rng.get(), pop.gametes, pop.mutations,
                    std::make_tuple(p1g1, p1g2, p2g1, p2g2), recmodel, mmodel,
                    mu, gamete_recycling_bin, mutation_recycling_bin, dip,
                    pop.neutral, pop.selected);

                assert(pop.gametes[dip.first].n);
                assert(pop.gametes[dip.second].n);
                offspring_metadata[label].label = label;
                update(offspring_metadata[label++], p1, p2,
                       pop.diploid_metadata);
            }

        fwdpp::fwdpp_internal::process_gametes(pop.gametes, pop.mutations,
                                               pop.mcounts);
        // This is constant-time
        pop.diploids.swap(offspring);
        pop.diploid_metadata.swap(offspring_metadata);
    }
} // namespace fwdpy11

#endif
