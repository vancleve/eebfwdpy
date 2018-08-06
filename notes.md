# New code to add multiple demes to fwdpy11 dev_015/0.2

## Current program flow

1. Call `evolve` from `wright_fisher.py` with population, parameters, and recorder
2. `evolve` does the following
   - calls `validate` on parameter object
   - constructs mutation regions with neutral and selected mutation rates
   - Calls `WFSlocusPop` from `wright_fisher_slocus` compiled C++
3. `WFSlocusPop` (`wfSlocusPop` in C++) does the following
   - validates mutation input
   - builds recombination and mutation functions
   - calls genetic value function
   - calls `calculate_fitness`
   - builds functions for picking parents of a diploid
   - for loop for all generations
	   * set population size for next gen
	   * call `fwdpy11::evolve_generation` from `SlocusPop_generation.hpp`
		   Iterate over offspring with N_next individuals
		   For each offspring: 
			   pick parents, 
			   call mutation and recombination functions
			   update metadata
			   swap offspring and parental diploid vectors
	   * call genetic value function
	   * call `calculate_fitness`
	   * call recorder function

## Adding multiple demes

General thrust is to use the normal diploid vector and create a partition of that vector into multiple demes. Then, the pick parent functions will depend on the diploid index in such a way to represent migration among the demes.

Some notable facts:
1. `ModelParams.demography` is just a list of population sizes for each generation (and is how the number of generation is actually specified)
2. 

### TODO:

- [ ] Derived population class
- [ ] New pop class member: function to get individuals from deme?
- [ ] generate new version of `wright_fisher_slocus.cc`
- [ ] generate new version of `wright_fisher.py`
- [ ] generate new version of `SlocusPop_generation.hpp`

### Done:

- eebfwdpy::evolve_generation takes std::vector of deme sizes
- 
