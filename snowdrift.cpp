/* Implement a stateful fitness model.
 * We define a new C++ type that will be
 * wrapped as a fwdpy11.fitness.SlocusFitness
 * object.
 *
 * Such a fitness model is ultimately responsible
 * for generating a bound C++ callback whose signature
 * is fwdpy11::single_locus_fitness_fxn.
 *
 * The module is built using cppimport:
 * https://github.com/tbenthompson/cppimport
 */

/* The next block of code is used by cppimport
 * The formatting is important, so I protect it
 * from the auto-formatter that I use.
 */
// clang-format off
<% 
#import fwdpy11 so we can find its C++ headers
import fwdpy11 as fp11 
#add fwdpy11 header locations to the include path
cfg['include_dirs'] = [ fp11.get_includes(), fp11.get_fwdpp_includes() ]
setup_pybind11(cfg)  
#On OS X using clang, there is more work to do.  Using gcc on OS X
#gets rid of these requirements. The specifics sadly depend on how
#you initially built fwdpy11, and what is below assumes you used
#the provided setup.py + OS X + clang:
#cfg['compiler_args'].extend(['-stdlib=libc++','-mmacosx-version-min=10.7'])
#cfg['linker_args']=['-stdlib=libc++','-mmacosx-version-min=10.7']
#An alternative to the above is to add the first line to CPPFLAGS
#and the second to LDFLAGS when compiling a plugin on OS X using clang.
%>
// clang-format on

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fwdpy11/types.hpp>
#include <fwdpy11/fitness/fitness.hpp>
#include <gsl/gsl_rng.h>
      
namespace py = pybind11;

struct snowdrift_diploid
/* This is a function object implementing the snowdrift
 * fitness calculation
 */
{
    using result_type = double;
    inline result_type
    operator()(const fwdpy11::GSLrng_t& rng,
	       const fwdpy11::diploid_t &dip, const fwdpy11::gcont_t &gametes,
               const fwdpy11::mcont_t &mutations,
               const std::vector<double> &phenotypes, const double b1,
               const double b2, const double c1, const double c2) const
    /* The first 3 arguments will be passed in from fwdpp.
     * The phenotypes are the stateful part, and will get
     * passed in by reference.  The remaining params
     * define the payoff function and are also supplied externally.
     */
    {
        auto N = phenotypes.size();
        // A diploid tracks its index via
        // fwdpy11::diploid_t::label, which
        // is std::size_t.
        auto i = dip.label;
        double zself = phenotypes[i];

	result_type fitness = 0;
	// get random partner k
	std::size_t k = gsl_rng_uniform_int(rng.get(), N-1);
	if (k >= i) {
	    ++k;
	}	
	double zpair = zself + phenotypes[k];

	// Payoff function from Fig 1 of Doebeli, Hauert & Killingback (2004, Science)
	fitness += 1 + b1 * zpair + b2 * zpair * zpair
	    - c1 * zself - c2 * zself * zself;

       	return std::max(fitness, 0.0);
    }
};

struct snowdrift : public fwdpy11::single_locus_fitness
/* This is our stateful fitness object.
 * It records the model parameters and holds a
 * vector to track individual phenotypes.
 *
 * The C++ side of an SlocusFitness object must publicly
 * inherit from fwdpy11::single_locus_fitness.
 *
 * The phenotypes get updated each generation during
 * the simulation.
 *
 * The phenotypes will be the simple additive model,
 * calculated using fwdpp's machinery.
 */
{
    const fwdpy11::GSLrng_t& rng;
    double b1, b2, c1, c2;
    double sigslope;
    double pheno0;
    std::vector<double> phenotypes;

    snowdrift(const fwdpy11::GSLrng_t& rng_,
	      double b1_, double b2_, double c1_, double c2_,
	      double sigslope_, double pheno0_)
        : rng(rng_), b1(b1_), b2(b2_), c1(c1_), c2(c2_),
	  sigslope(sigslope_), pheno0(pheno0_), phenotypes(std::vector<double>())
    {
    }

    inline fwdpy11::single_locus_fitness_fxn
    callback() const final
    // A stateful fitness model must return a bound callable.
    {
        /* Please remember that std::bind will pass a copy unless you
         * explicity wrap with a reference wrapper.  Here,
         * we use std::cref to make sure that phenotypes are
         * passed as const reference.
         */
        return std::bind(snowdrift_diploid(), std::cref(rng),
			 std::placeholders::_1,
			 std::placeholders::_2,
			 std::placeholders::_3,
                         std::cref(phenotypes), b1, b2, c1, c2);
    }
    void
    update(const fwdpy11::singlepop_t &pop) final
    /* A stateful fitness model needs updating.
     * The base class defines this virtual function
     * to do nothing (for non-stateful models).
     * Here, we redefine it as needed.
     */
    {
	using __mtype = typename fwdpy11::mcont_t::value_type;
	double summut;
	double sig0;
	
        phenotypes.resize(pop.N);
        for (auto &&dip : pop.diploids)
            {
		// sum of mutant effects
		summut =
		    KTfwd::site_dependent_fitness()(
			pop.gametes[dip.first],
			pop.gametes[dip.second],
			pop.mutations,
			[&](double &fitness, const __mtype &mut) {
			    fitness += (2.0 * mut.s);
			},
			[&](double &fitness, const __mtype &mut) {
			    fitness += (mut.h * mut.s);
			},
			0.);
                // A diploid tracks its index via
                // fwdpy11::diploid_t::label

		// map mutant effects to unit interval with sigmoidal function
		sig0 = 1.0 / sigslope * std::log(pheno0 / (1 - pheno0));
		phenotypes[dip.label] = 1.0 / (1.0 + std::exp( - sigslope * (summut + sig0)));
            }
    };

    // A custom fitness function requires that
    // several pure virtual functions be defined.
    // These macros do it for you:
    SINGLE_LOCUS_FITNESS_CLONE_SHARED(snowdrift);
    SINGLE_LOCUS_FITNESS_CLONE_UNIQUE(snowdrift);
    // This one gets pass the C++ name of the
    // callback.  We use C++11's typid to
    // do that for us:
    SINGLE_LOCUS_FITNESS_CALLBACK_NAME(typeid(snowdrift_diploid).name());
};

struct phenotype_sampler
{
    std::size_t sample_time;
    std::vector< std::vector<double> > samples;
    const snowdrift &sd_fitness;
    
    phenotype_sampler(std::size_t sample_time_,
		      const snowdrift &sd_fitness_) : sample_time(sample_time_), samples(0), sd_fitness(sd_fitness_)
	{
	}

    void operator()(const fwdpy11::singlepop_t &pop)
    {
	if (pop.generation % sample_time == 0)
	    {
		samples.push_back(sd_fitness.phenotypes);
	    }
    }
    
};

PYBIND11_PLUGIN(snowdrift)
{
    pybind11::module m("snowdrift",
                       "Example of custom stateful fitness model.");

    // fwdpy11 provides a macro
    // to make sure that the Python wrapper
    // to fwdpy11::singlepop_fitness is visible
    // to this module.
    FWDPY11_SINGLE_LOCUS_FITNESS()

    // Create a Python class based on our new type
    py::class_<snowdrift, std::shared_ptr<snowdrift>,
               fwdpy11::single_locus_fitness>(m, "SlocusSnowdrift")
        .def(py::init<const fwdpy11::GSLrng_t&, double, double, double, double, double, double>(),
	     py::arg("rng"),
	     py::arg("b1"), py::arg("b2"), py::arg("c1"), py::arg("c2"),
	     py::arg("sigslope"), py::arg("pheno0"))
        .def_readwrite("phenotypes", &snowdrift::phenotypes, "snowdrift phenotypes")
	.def("update", &snowdrift::update, py::arg("pop"));

    py::class_<phenotype_sampler>(m, "SamplerSnowdrift")
	.def(py::init<std::size_t, const snowdrift&>(),
	     py::arg("sample_time"), py::arg("snowdrift_fitness"))
	.def("__call__",
	     [](phenotype_sampler& f, const fwdpy11::singlepop_t &p) { return f(p); },
	     py::arg("pop"))
	.def_readwrite("samples", &phenotype_sampler::samples, "sampled phenotypes");

    return m.ptr();
}
