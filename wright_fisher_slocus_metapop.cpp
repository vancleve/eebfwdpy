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

// ** Build with https://github.com/tbenthompson/cppimport **
// clang-format off
<% 
import fwdpy11 as fp11 
cfg['include_dirs'] = [ fp11.get_includes(), fp11.get_fwdpp_includes() ]
cfg['dependencies'] = ['SlocusMetapopGeneticValue.hpp', 'SlocusMetapop_generation.hpp']
cfg['compiler_args'].extend(['-std=c++11'])
setup_pybind11(cfg)
%>
// clang-format on

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <functional>
#include <tuple>
#include <queue>
#include <cmath>
#include <numeric>
#include <string>
#include <stdexcept>
#include <fwdpp/diploid.hh>
#include <fwdpp/sugar/GSLrng_t.hpp>
#include <fwdpp/extensions/regions.hpp>
#include <fwdpy11/rng.hpp>
#include <fwdpy11/samplers.hpp>
#include <fwdpy11/sim_functions.hpp>
#include <fwdpy11/genetic_values/GeneticValueToFitness.hpp>
#include "SlocusMetapopGeneticValue.hpp"
#include "SlocusMetapop_generation.hpp"

namespace py = pybind11;

std::vector<double>
calculate_fitness(const fwdpy11::GSLrng_t &rng, fwdpy11::SlocusPop &pop,
                  const fwdpy11::SlocusMetapopGeneticValue &genetic_value_fxn,
		  std::vector<std::size_t> &N_psum)
{
    // Calculate parental fitnesses
    std::vector<double> parental_fitnesses(pop.diploids.size());
    double sum_parental_fitnesses = 0.0;
    for (std::size_t i = 0; i < pop.diploids.size(); ++i)
        {
            auto g = genetic_value_fxn(i, pop);
            pop.diploid_metadata[i].g = g;
            pop.diploid_metadata[i].e
		= genetic_value_fxn.noise(
		    rng, pop.diploid_metadata[i],
		    pop.diploid_metadata[i].parents[0],
		    pop.diploid_metadata[i].parents[1], pop);
            pop.diploid_metadata[i].w
                = genetic_value_fxn.genetic_value_to_fitness(
                    pop.diploid_metadata[i],
		    N_psum);
            parental_fitnesses[i] = pop.diploid_metadata[i].w;
            sum_parental_fitnesses += parental_fitnesses[i];
        }
    if (!std::isfinite(sum_parental_fitnesses))
        {
            throw std::runtime_error("non-finite fitnesses encountered");
        }

    return parental_fitnesses;
}

// For the moment being, this function will implement a group competition demography
// so that local competition is minimized (sensu Lehmann & Rousset 2010 PTRS).
//
// Life cycle:
// - Fitness is calculated based on interactions within a deme
// - Demes compete for resources in the next generation based on total deme fitness
// - Demes are populated by offspring from only one parent deme
// - Offspring become adults who can migrate between demes
//
// TODO: generalize for more kinds of demography
//
std::vector<fwdpp::fwdpp_internal::gsl_ran_discrete_t_ptr>
calculate_parent_sampling(const fwdpy11::GSLrng_t &rng,
			  fwdpy11::SlocusPop &pop,
			  const std::vector<double> &parental_fitnesses,
			  const py::array_t<std::size_t> &metapopsizes,
                          const py::array_t<double> &migrate)
{
    // pybind11 access w/o bounds checking
    auto mp  = metapopsizes.unchecked<2>();
    auto mig = migrate.unchecked<2>();
    const std::size_t num_generations = static_cast<std::size_t>(mp.shape(0));
    const std::size_t nd = static_cast<std::size_t>(mp.shape(1));
    const std::size_t offgen = pop.generation+1;
    std::vector<double> deme_fitnesses(nd);

    if (offgen >= num_generations)
        {
            throw std::invalid_argument(
                "cannot calculate fitness sampling beyond generation " +
		std::to_string(offgen));
        }
    
    // calculate deme fitness and use for group competition
    for (std::size_t i = 0; i < pop.N; ++i)
        {
            deme_fitnesses[pop.diploid_metadata[i].deme] += pop.diploid_metadata[i].w;
        }
    
    auto deme_lookup = fwdpp::fwdpp_internal::gsl_ran_discrete_t_ptr(
        gsl_ran_discrete_preproc(deme_fitnesses.size(), deme_fitnesses.data()));

    // set parent demes according to deme fitnesses
    std::vector<std::size_t> deme_parents(nd);
    for (auto&& parent : deme_parents)
        {
            parent = gsl_ran_discrete(rng.get(), deme_lookup.get());
        }    
    
    std::vector<fwdpp::fwdpp_internal::gsl_ran_discrete_t_ptr> lookups(nd);
    std::vector<double> post_migration(pop.N);

    // src and dest are in reference to migration after group competition
    for (std::size_t d_dest = 0; d_dest < nd; ++d_dest)
        {
            for (std::size_t i = 0; i < pop.N; ++i)
                {
                    post_migration[i] = 0;
                    for (std::size_t d_src = 0; d_src < nd; ++d_src)
                        {
			    // for parent i living in deme d_src, calculate the
			    // number of offspring it sends to deme d_dest.
			    // for group competition, this is
			    // success of group d_src * popsize d_src * fitness[i] / deme_fitness[i] * migration
			    post_migration[i] +=
				(deme_parents[d_src] == pop.diploid_metadata[i].deme)
				* mp(offgen, d_src)
                                * parental_fitnesses[i] / deme_fitnesses[pop.diploid_metadata[i].deme]
                                * mig(d_dest, d_src);
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
            lookups[d_dest] = std::move(rv); // can only move std::unique_ptr
        }
    
    return lookups;
}
    
void
update_deme_map(std::vector<std::size_t> &label_to_deme, std::vector<std::size_t> &N_psum,
		py::array_t<std::size_t> &metapopsizes, std::size_t gen)
{
    auto mp = metapopsizes.unchecked<2>(); // pybind11 access w/o bounds checking
    const std::size_t nd = static_cast<std::size_t>(mp.shape(1));
    const std::size_t num_generations = static_cast<std::size_t>(mp.shape(0));

    if (gen >= num_generations)
        {
	    throw std::invalid_argument(
                "cannot update_deme_map sampling beyond generation " +
		std::to_string(gen));
        }
    if (N_psum.size() != nd + 1)
	{
	    throw std::invalid_argument("N_psum has wrong size");
	}
    
    // calculate partial sums
    N_psum[0] = 0;
    for (std::size_t d = 0; d < nd; ++d)
	{
	    N_psum[d+1] = N_psum[d] + mp(gen, d);
	}
	    
    // total population size (at generation `gen`)
    std::size_t N = N_psum[nd];   
    
    // update deme map
    label_to_deme.resize(N);
    std::size_t d = 0;
    for (std::size_t i = 0; i < N; ++i)
	{
	    if (i < N_psum[d+1])
		label_to_deme[i] = d;
	    else
		++d;
	}
}

void
handle_fixations(const bool remove_selected_fixations,
                 const std::size_t N_next, fwdpy11::SlocusPop &pop)
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
    const fwdpy11::GSLrng_t &rng, fwdpy11::SlocusPop &pop,
    py::array_t<std::size_t> metapopsizes, py::array_t<double> migrate, 
    const double mu_neutral, const double mu_selected, const double recrate,
    const fwdpp::extensions::discrete_mut_model<fwdpy11::SlocusPop::mcont_t> &mmodel,
    const fwdpp::extensions::discrete_rec_model &rmodel,
    fwdpy11::SlocusMetapopGeneticValue &genetic_value_fxn,
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

    auto mp = metapopsizes.unchecked<2>(); // pybind11 access w/o bounds checking
    const std::size_t nd = static_cast<std::size_t>(mp.shape(1));
    const std::size_t num_generations = static_cast<std::size_t>(mp.shape(0));

    // mapping from individual label to deme and
    // partial sums of population size for all demes
    // (used to locate all individuals in a specific deme)
    std::vector<std::size_t> label_to_deme(pop.N);
    std::vector<std::size_t> N_psum(nd + 1);
    
    // E[S_{2N}] I got the expression from Ewens.
    // JVC: this is probably very conservative for a metapop
    pop.mutations.reserve(std::ceil(
        std::log(2 * pop.N)
        * (4. * double(pop.N) * (mu_neutral + mu_selected))
           + 0.667 * (4. * double(pop.N) * (mu_neutral + mu_selected))));

    const auto bound_mmodel = fwdpp::extensions::bind_dmm(rng.get(), mmodel);
    const auto bound_rmodel = [&rng, &rmodel]() { return rmodel(rng.get()); };

    // Generation 0:
    // update deme labels, phenotypes, and fitnesses, and record.
    update_deme_map(label_to_deme, N_psum, metapopsizes, 0);
    genetic_value_fxn.update(pop);
    auto parental_fitnesses = calculate_fitness(rng, pop, genetic_value_fxn, N_psum);
    auto lookups = calculate_parent_sampling(rng, pop,
                                             parental_fitnesses,
                                             metapopsizes, migrate);
    recorder(pop);

    // Generate our fxns for picking parents
    //
    // Because lambdas that capture by reference do a "late" binding of
    // params, this is safe w.r.to updating lookup after each generation.
    const auto pick_first_parent
	= [&rng, &lookups, &label_to_deme](const std::size_t label)
	      {
		  return gsl_ran_discrete(rng.get(), lookups[label_to_deme[label]].get());
	      };

    const auto pick_second_parent
	= [&rng, &lookups, &label_to_deme, selfing_rate](const std::size_t label,
							 const std::size_t p1) {
	      if (selfing_rate == 1.0
		  || (selfing_rate > 0.0
		      && gsl_rng_uniform(rng.get()) < selfing_rate))
		  {
		      return p1;
		  }
	      return gsl_ran_discrete(rng.get(), lookups[label_to_deme[label]].get());
	  };

    const auto generate_offspring_metadata
        = [&rng, &label_to_deme](fwdpy11::DiploidMetadata &offspring_metadata,
				 const std::size_t p1, const std::size_t p2,
				 const std::vector<fwdpy11::DiploidMetadata>
				 & /*parental_metadata*/) {
              offspring_metadata.deme = label_to_deme[offspring_metadata.label];
              offspring_metadata.parents[0] = p1;
              offspring_metadata.parents[1] = p2;
          };

    for (std::size_t gen = 1; gen < num_generations; ++gen)
        {
            ++pop.generation;

	    // Total metapopulation size for offspring generation
	    std::size_t N_next = 0;
	    for (std::size_t i = 0; i < nd; ++i)
		N_next += mp(gen, i);

	    // update label_to_deme and N_psum to offspring generation
	    update_deme_map(label_to_deme, N_psum, metapopsizes, gen); 
        
            fwdpy11::evolve_generation(
                rng, pop, N_next, mu_neutral + mu_selected, bound_mmodel,
                bound_rmodel, pick_first_parent, pick_second_parent,
                generate_offspring_metadata);
            handle_fixations(remove_selected_fixations, N_next, pop);

            genetic_value_fxn.update(pop);
	    parental_fitnesses = calculate_fitness(rng, pop, genetic_value_fxn, N_psum);
	    lookups = calculate_parent_sampling(rng, pop,
						parental_fitnesses,
						metapopsizes, migrate);
            recorder(pop);
        }
}

PYBIND11_MODULE(wright_fisher_slocus_metapop, m)
{
    m.doc() = "Evolution in a metapopulation under a Wright-Fisher model.";

    m.def("WFSlocusMetapop", &wfSlocusMetapop);
}
