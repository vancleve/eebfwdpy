
# eebfwdpy 

Ecology and evolutionary biology forward population genetic simulations in python 

## Background ##

This will be a collection of python modules for running evolutionary ecology simulations based on the [`fwdpy11`](http://github.com/molpopgen/fwdpy11) Python package for forward-time population genetic simulations (known pejoratively as "agent-based models" or ABMs). The package `fwdpy11` uses the [`fwdpp`](http://github.com/molpopgen/fwdpp) population genetics C++ template library.

## Modules ##

Currently there is only one example that implements the continuous snowdrift game modeled in [Doebeli, Hauert, and Killingback (2004, Science)](http://dx.doi.org/10.1126/science.1101456). A metapopulation version of this game studied in [Wakano and Lehmann (2014, Journal of Theoretical Biology)](http://dx.doi.org/10.1016/j.jtbi.2014.02.036) and code for that is in the works.

## Requirements ##

- fwdpy11 (see [here](https://github.com/vancleve/fwdpy11#installation) for installation)
- numpy and matplotlib for analysis and plotting
