// Wright-Fisher simulation for a fwdpy11::SlocusPop
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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <functional>
#include <tuple>
#include <queue>
#include <cmath>
#include <string>
#include <stdexcept>
#include <fwdpp/diploid.hh>
#include <fwdpp/sugar/GSLrng_t.hpp>
#include <fwdpp/extensions/regions.hpp>
#include <fwdpy11/rng.hpp>
#include <fwdpy11/samplers.hpp>
#include <fwdpy11/sim_functions.hpp>
#include <fwdpy11/genetic_values/SlocusPopGeneticValue.hpp>
#include <fwdpy11/genetic_values/GeneticValueToFitness.hpp>
#include "SlocusMetapop.hpp"
#include "SlocusMetapop_generation.hpp"

namespace py = pybind11;

std::vector<double>
calculate_fitness(const fwdpy11::GSLrng_t &rng, eebfwdpy::SlocusMetapop &pop,
                  const fwdpy11::SlocusPopGeneticValue &genetic_value_fxn)
{
    // Calculate parental fitnesses
    std::vector<double> parental_fitnesses(pop.diploids.size());
    double sum_parental_fitnesses = 0.0;
    for (std::size_t i = 0; i < pop.diploids.size(); ++i)
        {
            auto g = genetic_value_fxn(i, pop);
            pop.diploid_metadata[i].g = g;
            pop.diploid_metadata[i].e = genetic_value_fxn.noise(
                rng, pop.diploid_metadata[i],
                pop.diploid_metadata[i].parents[0],
                pop.diploid_metadata[i].parents[1], pop);
            pop.diploid_metadata[i].w
                = genetic_value_fxn.genetic_value_to_fitness(
                    pop.diploid_metadata[i]);
            parental_fitnesses[i] = pop.diploid_metadata[i].w;
            sum_parental_fitnesses += parental_fitnesses[i];
        }
    // If the sum of parental fitnesses is not finite,
    // then the genetic value calculator returned a non-finite value/
    // Unfortunately, gsl_ran_discrete_preproc allows such values through
    // without raising an error, so we have to check things here.
    if (!std::isfinite(sum_parental_fitnesses))
        {
            throw std::runtime_error("non-finite fitnesses encountered");
        }

    return parental_fitnesses;
}

// For the moment being, this function will implement a group competition demography
// so that local competition is minimized (sensu Lehmann & Rousset 2010 PTRS)
//
// TODO: generalize for more kinds of demography
//
std::vector<fwdpp::fwdpp_internal::gsl_ran_discrete_t_ptr>
calculate_parent_sampling(const fwdpy11::GSLrng_t &rng,
			  eebfwdpy::SlocusMetapop &pop,
			  const std::vector<double> parental_fitnesses,
			  const py::array_t<std::size_t> metapopsizes,
                          const py::array_t<double> migrate)
{
    auto Ns_next = metapopsizes.at(pop.generation+1);
    auto nd_next = Ns_next.size();
    std::vector<double> deme_fitnesses(pop.nd);
    
    for (std::size_t i = 0; d < pop.N; ++i)
        {
            deme_fitnesses[pop.deme_map[i]] += pop.diploid_metadata[i].w;
        }
    
    auto deme_lookup = fwdpp::fwdpp_internal::gsl_ran_discrete_t_ptr(
        gsl_ran_discrete_preproc(deme_fitnesses.size(), deme_fitnesses.data()));


    // set parent demes according to deme fitnesses
    std::vector<std::size_t> deme_parents(nd_next);
    for (auto&& p : deme_parents)
        {
            p = gsl_ran_discrete(rng.get(), deme_lookup.get());
        }

    
    std::vector<fwdpp::fwdpp_internal::gsl_ran_discrete_t_ptr> lookups(nd_next);
    std::vector<double> post_migration(pop.N);

    // src and dest are in reference to migration after group competition
    for (std::size_t d_dest = 0; d_dest < nd_next; ++d_dest)
        {
            for (std::size_t i = 0; i < pop.N; ++i)
                {
                    post_migration[i] = 0;
                    for (std::size_t d_src = 0; d_src < nd_next; ++j)
                        {
                            post_migration[i] +=
                            (deme_parents[d_src] == pop.deme_map[i]) * Ns_next.at(d_src)
                                * parental_fitnesses[i] / deme_fitnesses[pop.deme_map[i]]
                                * migrate.at(d_dest, d_src);
                        }
                }
            
            auto rv = fwdpp::fwdpp_internal::gsl_ran_discrete_t_ptr(
                gsl_ran_discrete_preproc(post_migration.size(),
                                         post_migration.data()));
            if (rv == nullptr)
                {
                    // This is due to negative fitnesses
                    throw std::runtime_error(
                        "fitness lookup table for deme "
                        + std::to_string(d_dest) + " could not be generated");
                }
            lookups[d_dest] = rv;
        }
    
    return lookups;
}

void
handle_fixations(const bool remove_selected_fixations,
                 const std::uint32_t N_next, eebfwdpy::SlocusMetapop &pop)
{
    if (remove_selected_fixations)
        {
            fwdpp::fwdpp_internal::gamete_cleaner(pop.gametes, pop.mutations,
                                                  pop.mcounts, 2 * N_next,
                                                  std::true_type());
        }
    else
        {
            fwdpp::fwdpp_internal::gamete_cleaner(pop.gametes, pop.mutations,
                                                  pop.mcounts, 2 * N_next,
                                                  fwdpp::remove_neutral());
        }
    fwdpy11::update_mutations(pop.mutations, pop.fixations, pop.fixation_times,
                              pop.mut_lookup, pop.mcounts, pop.generation,
                              2 * pop.N, remove_selected_fixations);
}

void
wfSlocusMetapop(
    const fwdpy11::GSLrng_t &rng, eebfwdpy::SlocusMetapop &pop,
    py::array_t<std::size_t> metapopsizes, py::array_t<double> migrate, 
    const double mu_neutral, const double mu_selected, const double recrate,
    const fwdpp::extensions::discrete_mut_model<eebfwdpy::SlocusMetapop::mcont_t>
        &mmodel,
    const fwdpp::extensions::discrete_rec_model &rmodel,
    fwdpy11::SlocusPopGeneticValue &genetic_value_fxn,
    fwdpy11::SlocusPop_temporal_sampler recorder, const double selfing_rate,
    const bool remove_selected_fixations)
{
    //validate the input params
    if (!std::isfinite(mu_neutral))
        {
            throw std::invalid_argument("neutral mutation rate is not finite");
        }
    if (!std::isfinite(mu_selected))
        {
            throw std::invalid_argument(
                "selected mutation rate is not finite");
        }
    if (mu_neutral < 0.0)
        {
            throw std::invalid_argument(
                "neutral mutation rate must be non-negative");
        }
    if (mu_selected < 0.0)
        {
            throw std::invalid_argument(
                "selected mutation rate must be non-negative");
        }
    if (metapopsizes.ndim() != 2 || metapopsizes.shape()[0] == 1 || metapopsizes.shape()[1] == 1)
      {
	throw std::invalid_argument("metapopsizes must be 2D array for deme sizes for each generation");
      }
    if (migrate.ndim() != 2 || migrate.shape()[0] != migrate.shape()[1])
      {
	throw std::invalid_argument("migration rate matrix must be square 2D array");
      }
    if (metapopsizes.shape()[1] != migrate.shape()[0])
      {
	throw std::invalid_argument("number of demes must match migration matrix size");
      }

    const std::uint32_t num_generations = static_cast<std::uint32_t>(metapopsizes.shape()[0]);

    // E[S_{2N}] I got the expression from Ewens.
    // JVC: this is probably very conservative for a metapop
    pop.mutations.reserve(std::ceil(
        std::log(2 * pop.N)
        * (4. * double(pop.N) * (mu_neutral + mu_selected))
           + 0.667 * (4. * double(pop.N) * (mu_neutral + mu_selected))));

    const auto bound_mmodel = fwdpp::extensions::bind_dmm(rng.get(), mmodel);
    const auto bound_rmodel = [&rng, &rmodel]() { return rmodel(rng.get()); };

    // A stateful fitness model will need its data up-to-date,
    // so we must call update(...) prior to calculating fitness,
    // else bad stuff like segfaults could happen.
    genetic_value_fxn.update(pop);
    auto parental_fitnesses = calculate_fitness(rng, pop, genetic_value_fxn);
    auto lookups = calculate_parent_sampling(rng, pop,
                                             parental_fitnesses,
                                             metapopsizes, migrate);

    // Generate our fxns for picking parents

    // Because lambdas that capture by reference do a "late" binding of
    // params, this is safe w.r.to updating lookup after each generation.
    const auto pick_first_parent = [&rng, &lookup]() {
        return gsl_ran_discrete(rng.get(), lookup.get());
    };

    const auto pick_second_parent
        = [&rng, &lookup, selfing_rate](const std::size_t p1) {
              if (selfing_rate == 1.0
                  || (selfing_rate > 0.0
                      && gsl_rng_uniform(rng.get()) < selfing_rate))
                  {
                      return p1;
                  }
              return gsl_ran_discrete(rng.get(), lookup.get());
          };

    const auto generate_offspring_metadata
        = [&rng](fwdpy11::DiploidMetadata &offspring_metadata,
                 const std::size_t p1, const std::size_t p2,
                 const std::vector<fwdpy11::DiploidMetadata>
                     & /*parental_metadata*/) {
              offspring_metadata.deme = 0;
              offspring_metadata.parents[0] = p1;
              offspring_metadata.parents[1] = p2;
          };

    for (std::uint32_t gen = 0; gen < num_generations; ++gen)
        {
            ++pop.generation;
            const auto N_next = metapopsizes.at(gen);
	    // update deme sizes and deme_map to offspring generation
	    pop.update_deme_sizes(N_next); 
        
            fwdpy11::evolve_generation(
                rng, pop, N_next, mu_neutral + mu_selected, bound_mmodel,
                bound_rmodel, pick_first_parent, pick_second_parent,
                generate_offspring_metadata);
            handle_fixations(remove_selected_fixations, N_next, pop);

            // TODO: deal with random effects
            genetic_value_fxn.update(pop);
            lookup = calculate_fitness(rng, pop, genetic_value_fxn);
            recorder(pop); // The user may now analyze the pop'n
        }
}

PYBIND11_MODULE(wright_fisher_slocus_metapop, m)
{
    m.doc() = "Evolution under a Wright-Fisher model.";

    m.def("WFSlocusMetapop", &wfSlocusMetapop);
}
