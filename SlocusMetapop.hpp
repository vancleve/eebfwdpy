#ifndef EEBFWDPY_SLOCUSMETAPOP_HPP__
#define EEBFWDPY_SLOCUSMETAPOP_HPP__

#include <numeric>
#include <functional>
#include <stdexcept>
#include <fwdpy11/types/SlocusPop.hpp>

namespace eebfwdpy
{
    class SlocusMetapop : public fwdpy11::SlocusPop
    {
      public:
        using Popbase = fwdpy11::SlocusPop;

        std::vector<fwdpp::uint_t> Ns;       // Vector of deme sizes
        std::vector<fwdpp::uint_t> deme_map; // map from individual label to deme
        fwdpp::uint_t nd;                    // number of demes
        
        SlocusMetapop(SlocusMetapop &&)                 = default; // Move constructor
        SlocusMetapop(const SlocusMetapop &)            = default; // Copy constructor
        SlocusMetapop &operator=(SlocusMetapop &&)      = default; // Move assignment operator
        SlocusMetapop &operator=(const SlocusMetapop &) = default; // Copy assignment operator

        // The abstraction here could be backwards in the long run:
        // the base class should allow for multiple demes and a derived class
        // is a special case. However, this should hopefully work in the interim.
        SlocusMetapop(const std:vector<fwdpp::uint_t> Ns_)
            : Popbase{ std::accumulate(Ns_.begin(), Ns_.end(), 0) },
              Ns( Ns_.size() ),
              deme_map( std::accumulate(Ns_.begin(), Ns_.end(), 0) ),
              nd{ Ns_.size }
        {
            if (std::accumulate(Ns_.begin(), Ns_.end(), 1, std::multiplies<int>()) <= 0)
                {
                    throw std::invalid_argument("all deme sizes must be > 0");
                }

            // update deme sizes and deme_map
            update_deme_sizes(Ns_);
            
            // update deme metadata
            for (fwdpp::uint_t i = 0; i < N; ++i)
                {
                    diploid_metadata[i].deme = deme_map[i];
                }
        }

        void update_deme_sizes(const std:vector<fwdpp::uint_t> Ns_)
        {
            // update deme sizes
            Ns = Ns_;

            // update number of demes
            nd = Ns.size();
            
            // update total population size
            N = std::accumulate(Ns.begin(), Ns.end(), 0);                

            // partial sums
            std::vector<fwdpp::uint_t> Ns_psum(N);
            std::partial_sum(Ns.begin(), Ns.end(), Ns_psum.begin());

            // update deme map
            fwdpp::uint_t d = 0;
            for (fwdpp::uint_t i = 0; i < N; ++i)
                {
                    if (i < Ns_psum[d])
                        deme_map[i] = d;
                    else
                        ++d;
                }
        }
        
    };
} // namespace eebfwdpy
#endif
