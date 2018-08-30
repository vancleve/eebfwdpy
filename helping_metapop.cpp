/* Helping in a metapopluation with fwdpy11.
 *
 * The module is built using cppimport:
 * https://github.com/tbenthompson/cppimport
 */
// clang-format off
<% 
import fwdpy11 as fp11 
cfg['include_dirs'] = [ fp11.get_includes(), fp11.get_fwdpp_includes() ]
cfg['dependencies'] = ['SlocusMetapopGeneticValue.hpp']
cfg['compiler_args'].extend(['-std=c++11'])
setup_pybind11(cfg)
%>
// clang-format on


#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fwdpy11/types/SlocusPop.hpp>
#include <fwdpp/fitness_models.hpp>
#include "SlocusMetapopGeneticValue.hpp"
#include <gsl/gsl_rng.h>
      
namespace py = pybind11;

struct helping : public fwdpy11::SlocusMetapopGeneticValue
/* This is our stateful fitness object.
 * It records the model parameters and holds a
 * vector to track individual phenotypes.
 *
 * Here, we publicly inherit from fwdpy11::SlocusMetapopGeneticValue,
 * which is defined in the header included above.  It is
 * an abstract class in C++ terms, and is reflected
 * as a Python Abstract Base Class (ABC) called
 * fwdpy11.genetic_values.SlocusMetapopGeneticValue.
 *
 * The phenotypes get updated each generation during
 * the simulation.
 *
 * The phenotypes will be the simple additive model,
 * calculated using fwdpp's machinery.
 */
{
    const fwdpy11::GSLrng_t &rng;
    const double b, c;
    const double sigslope, pheno0;
    std::vector<double> phenotypes;
	
    helping(const fwdpy11::GSLrng_t &rng_,
	    double b_, double c_,
	    double sigslope_, double pheno0_)
        : fwdpy11::SlocusMetapopGeneticValue{}, rng(rng_),
	  b(b_), c(c_), sigslope(sigslope_), pheno0(pheno0_), phenotypes()
    {
    }

    inline double
    operator()(const std::size_t diploid_index,
               const fwdpy11::SlocusPop & /*pop*/) const
    // The call operator must return the genetic value of an individual
    {
        return phenotypes[diploid_index];
    }

    inline double
    genetic_value_to_fitness(const fwdpy11::DiploidMetadata &metadata,
			     const std::vector<std::size_t> &N_psum) const
    // This function converts genetic value to fitness.
    {
        double fitness = 0.0;
        double zself = metadata.g;

	// get random partner k in current deme
	std::size_t k = N_psum[metadata.deme]
	  + gsl_rng_uniform_int(rng.get(), N_psum[metadata.deme+1] - N_psum[metadata.deme] - 1);
	if (k >= metadata.label)
	    {
		++k;
	    }
	double zother = phenotypes[k];
	
	// Payoff function
	fitness += 1 + b * std::sqrt(zother) - c * std::pow(zself, 2.0);

       	return std::max(fitness, 0.0);
    }

    inline double
    noise(const fwdpy11::GSLrng_t & /*rng*/,
          const fwdpy11::DiploidMetadata & /*offspring_metadata*/,
          const std::size_t /*parent1*/, const std::size_t /*parent2*/,
          const fwdpy11::SlocusPop & /*pop*/) const
    // This function may be used to model random effects...
    {
        //...but there are no random effects here.
        return 0.0;
    }

    inline void
    update(const fwdpy11::SlocusPop &pop)
    // A stateful fitness model needs updating.
    {
	double summut;
	double sig0;

        phenotypes.resize(pop.N);
        for (std::size_t i = 0; i < pop.N; ++i)
            {
                // A diploid tracks its index via
                // fwdpy11::DiploidMetadata::label
		summut = fwdpp::additive_diploid(2.0)(pop.diploids[i],
						      pop.gametes, pop.mutations) - 1.0;
		sig0 = 1.0 / sigslope * std::log(pheno0 / (1 - pheno0));
		phenotypes[pop.diploid_metadata[i].label]
		    = 1.0 / (1.0 + std::exp( - sigslope * (summut + sig0)));
            }
    }

    // no pickling for now but function definition is necessary
    py::object
    pickle() const
	{
	    return py::make_tuple(b, c, phenotypes);
	}
};

struct phenotype_sampler
{
    std::size_t sample_time;
    std::vector< std::vector<double> > phenotypes;
    
    phenotype_sampler(std::size_t sample_time_) : sample_time(sample_time_), phenotypes(0)
	{
	}

    void operator()(const fwdpy11::SlocusPop &pop)
    {
	if (pop.generation % sample_time == 0)
	    {
		std::vector<double> new_samples(pop.N);
		for (std::size_t i = 0; i < pop.N; ++i)
		    {
			new_samples[i] = pop.diploid_metadata[i].g;
		    }
		phenotypes.push_back(new_samples);
	    }
    }
    
};

PYBIND11_MODULE(helping_metapop, m)
{
  m.doc() = "Helping metapopulation model.";
    

  // We need to import the Python version of our base class:
  pybind11::object imported_helping_base_class_type
    = pybind11::module::import("genetic_values")
    .attr("SlocusMetapopGeneticValue");
  
  // Create a Python class based on our new type
  py::class_<helping, fwdpy11::SlocusMetapopGeneticValue>(m, "SlocusHelping")
    .def(py::init<const fwdpy11::GSLrng_t&, double, double, double, double>(),
	 py::arg("rng"),
	 py::arg("b"), py::arg("c"),
	 py::arg("sigslope"), py::arg("pheno0"))
    .def_readwrite("phenotypes", &helping::phenotypes, "helping phenotypes")
    .def("update", &helping::update, py::arg("pop"));
  
  py::class_<phenotype_sampler>(m, "PhenotypeSampler")
    .def(py::init<std::size_t>(),
	 py::arg("sample_time"))
    .def("__call__",
	 [](phenotype_sampler &f, const fwdpy11::SlocusPop &p) { return f(p); },
	 py::arg("pop"))
    .def_readwrite("phenotypes", &phenotype_sampler::phenotypes, "sampled phenotypes");

}
