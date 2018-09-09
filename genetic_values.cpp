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
cfg['dependencies'] = ['SlocusMetapopGeneticValue.hpp']
cfg['compiler_args'].extend(['-std=c++11'])
setup_pybind11(cfg)
%>
// clang-format on

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SlocusMetapopGeneticValue.hpp"

namespace py = pybind11;

PYBIND11_MODULE(genetic_values, m)
{
    py::class_<fwdpy11::SlocusMetapopGeneticValue>(
        m, "SlocusMetapopGeneticValue",
        "ABC for genetic value calculations for diploid members of "
        ":class:`fwdpy11.SlocusMetapop`")
        .def("__call__",
             [](const fwdpy11::SlocusMetapopGeneticValue &gv,
                const std::size_t diploid_index,
                const fwdpy11::SlocusPop &pop) {
                 return gv(diploid_index, pop);
             },
             R"delim(
             :param diploid_index: The index of the individual to calculate.
             :type diploid_index: int >= 0
             :param pop: The population object containing the individual.
             :type pop: :class:`fwdpy11.SlocusPop`
             :return: The genetic value of an individual.
             :rtype: float
             )delim",
             py::arg("diploid_index"), py::arg("pop"))
        .def("fitness",
             [](const fwdpy11::SlocusMetapopGeneticValue &gv,
                const std::size_t diploid_index,
                const fwdpy11::SlocusPop &pop,
		const std::vector<std::size_t> &N_psum) {
                 return gv.genetic_value_to_fitness(
                     pop.diploid_metadata[diploid_index],
		     N_psum);
             },
             R"delim(
             :param diploid_index: The index of the individual
             :type diploid_index: int >= 0
             :param pop: The population containing the individual
             :type pop: :class:`fwdpy11.SlocusPop`
             :return: The fitness of an individual.
             :rtype: float
             )delim",
             py::arg("diploid_index"), py::arg("pop"), py::arg("N_psum"));

}
