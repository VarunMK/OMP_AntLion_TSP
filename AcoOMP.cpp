#ifndef __ACO_OMP_CPP__
#define __ACO_OMP_CPP__

#include <stdint.h>
#include <vector>
#include <algorithm>
#include <omp.h> // Include OpenMP header

#include "Environment.cpp"
#include "Parameters.cpp"
#include "Ant.cpp"
#include "common.hpp"

template <typename T, typename D>
class AcoOMP
{

private:
    const Parameters<T> &params;
    Environment<T, D> &env;
    std::vector<Ant<T>> ants;

    void initEta(std::vector<T> &eta, const std::vector<T> &edges)
    {
        const size_t N = eta.size();
#pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
        {
            eta[i] = (edges[i] == 0.0 ? 0.0 : 1.0 / edges[i]);
        }
    }

    void calcFitness(std::vector<T> &fitness,
                     const std::vector<T> &pheromone,
                     const std::vector<T> &eta,
                     const T alpha,
                     const T beta)
    {
        const size_t N = fitness.size();
#pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
        {
            fitness[i] = pow(pheromone[i], alpha) * pow(eta[i], beta);
        }
    }

    void calcTour(const std::vector<T> &fitness,
                  const std::vector<T> &edges)
    {
#pragma omp parallel for
        for (size_t i = 0; i < ants.size(); ++i)
        {
            ants[i].constructTour(fitness, edges);
        }
    }

    void updateBestTour(std::vector<uint32_t> &bestTour,
                        T &bestTourLength)
    {
        const Ant<T> &bestAnt = *std::min_element(ants.begin(), ants.end());
        std::copy(bestAnt.getTabu().begin(), bestAnt.getTabu().end(), bestTour.begin());
        bestTourLength = bestAnt.getTourLength();
    }

    void updateDelta(std::vector<D> &delta,
                     const uint32_t nCities,
                     const T q)
    {
// Zero out delta
#pragma omp parallel for
        for (size_t i = 0; i < delta.size(); ++i)
        {
            delta[i] = 0.0;
        }

// Accumulate tau into delta (critical to avoid data races)
#pragma omp parallel for
        for (size_t i = 0; i < ants.size(); ++i)
        {
            const Ant<T> &ant = ants[i];
            const T tau = q / ant.getTourLength();
            const std::vector<uint32_t> &tabu = ant.getTabu();

            for (size_t j = 0; j < tabu.size() - 1; ++j)
            {
                uint32_t from = tabu[j];
                uint32_t to = tabu[j + 1];
#pragma omp atomic
                delta[from * nCities + to] += tau;
#pragma omp atomic
                delta[to * nCities + from] += tau;
            }

            uint32_t from = tabu.back();
            uint32_t to = tabu.front();
#pragma omp atomic
            delta[from * nCities + to] += tau;
#pragma omp atomic
            delta[to * nCities + from] += tau;
        }
    }

    void updatePheromone(std::vector<T> &pheromone,
                         const std::vector<D> &delta,
                         const T rho)
    {
        const size_t N = pheromone.size();
#pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
        {
            pheromone[i] = pheromone[i] * rho + delta[i];
        }
    }

public:
    AcoOMP(const Parameters<T> &params, Environment<T, D> &env) : params(params),
                                                                  env(env),
                                                                  ants(env.nAnts, Ant<T>(env.nCities))
    {
        initEta(env.eta, env.edges);
    }

    void solve()
    {
        uint32_t epoch = 0;
        for (int epoch = 0; epoch < params.maxEpoch; epoch++)
        {
            calcFitness(env.fitness, env.pheromone, env.eta, params.alpha, params.beta);
            calcTour(env.fitness, env.edges);
            updateBestTour(env.bestTour, env.bestTourLength);
            updateDelta(env.delta, env.nCities, params.q);
            updatePheromone(env.pheromone, env.delta, params.rho);
        }
    }

    ~AcoOMP() {}
};

#endif
