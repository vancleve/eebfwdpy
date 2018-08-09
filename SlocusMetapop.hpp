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

    using Population
        = PyPopulation<Mutation, std::vector<Mutation>,
                       std::vector<fwdpp::gamete>, std::vector<Mutation>,
                       std::vector<fwdpp::uint_t>,
                       std::unordered_multimap<double,fwdpp::uint_t>>;

    class SlocusPop : public Population
    {
      private:
        void
        process_individual_input()
        {
            std::vector<fwdpp::uint_t> gcounts(this->gametes.size(), 0);
            for (auto &&dip : diploids)
                {
                    this->validate_individual_keys(dip.first);
                    this->validate_individual_keys(dip.second);
                    gcounts[dip.first]++;
                    gcounts[dip.second]++;
                }
            this->validate_gamete_counts(gcounts);
        }

      public:
        using dipvector_t = std::vector<DiploidGenotype>;
        using diploid_t = dipvector_t::value_type;
        using popbase_t = Population;
        using popmodel_t = fwdpp::sugar::SINGLELOC_TAG;
        using fitness_t
            = fwdpp::traits::fitness_fxn_t<dipvector_t,
                                           typename popbase_t::gcont_t,
                                           typename popbase_t::mcont_t>;

        dipvector_t diploids;

        SlocusPop(SlocusPop &&) = default;
        SlocusPop(const SlocusPop &) = default;
        SlocusPop &operator=(SlocusPop &&) = default;
        SlocusPop &operator=(const SlocusPop &) = default;

        // Constructors for Python
        SlocusPop(const fwdpp::uint_t N)
            : Population{ N }, diploids(N, { 0, 0 })
        {
            if (!N)
                {
                    throw std::invalid_argument("population size must be > 0");
                }
            std::size_t label = 0;
            for (auto &&d : this->diploid_metadata)
                {
                    d.label = label++;
                    d.w = 1.0;
                }
        }

        template <typename diploids_input, typename gametes_input,
                  typename mutations_input>
        explicit SlocusPop(diploids_input &&d, gametes_input &&g,
                           mutations_input &&m)
            : Population(static_cast<fwdpp::uint_t>(d.size()),
                         std::forward<gametes_input>(g),
                         std::forward<mutations_input>(m), 100),
              diploids(std::forward<diploids_input>(d))
        //! Constructor for pre-determined population status
        {
            this->process_individual_input();
        }

        bool
        operator==(const SlocusPop &rhs) const
        {
            return this->diploids == rhs.diploids && popbase_t::is_equal(rhs);
        };

        void
        clear()
        {
            diploids.clear();
            popbase_t::clear_containers();
        }

        virtual std::vector<std::size_t>
        add_mutations(typename fwdpp_base::mcont_t &new_mutations,
                      const std::vector<std::size_t> &individuals,
                      const std::vector<short> &gametes)
        {
            std::unordered_set<double> poschecker;
            for (const auto &m : new_mutations)
                {
                    if (this->mut_lookup.find(m.pos) != this->mut_lookup.end())
                        {
                            throw std::invalid_argument(
                                "attempting to add new mutation at "
                                "already-mutated position");
                        }
                    if (poschecker.find(m.pos) != poschecker.end())
                        {
                            throw std::invalid_argument(
                                "attempting to add multiple mutations at the "
                                "same position");
                        }
                    poschecker.insert(m.pos);
                }
            std::vector<std::size_t> rv;

            for (auto &i : new_mutations)
                {
                    auto pos = i.pos;
                    // remaining preconditions get checked by fwdpp:
                    auto idx = fwdpp::add_mutation(*this, individuals, gametes,
                                                   std::move(i));

                    // fwdpp's function doesn't update the lookup:
                    this->mut_lookup.emplace(pos, idx);
                    rv.push_back(idx);
                }
            return rv;
        }

        fwdpp::data_matrix
        sample_individuals(const std::vector<std::size_t> &individuals,
                           const bool haplotype) const
        {
            return sample_individuals_details(*this, individuals, haplotype);
        }
            
        fwdpp::data_matrix
        sample_random_individuals(const GSLrng_t &rng,
                                  const std::uint32_t nsam,
                                  const bool haplotype) const
        {
            return sample_random_individuals_details(*this, rng, nsam,
                                                     haplotype);
        }
    };
