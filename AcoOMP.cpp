#ifndef __ACO_OMP_CPP__
#define __ACO_OMP_CPP__

#include <stdint.h>
#include <vector>
#include <algorithm>
#include <omp.h>

#include "Environment.cpp"
#include "Parameters.cpp"
#include "Ant.cpp"
#include "common.hpp"

template <typename T, typename D>
class AcoOMP {

private:
    const Parameters<T> & configRef;
    Environment<T, D> & contextRef;
    std::vector<Ant<T>> agentPool;

    void prepareEta(std::vector<T> & etaVals, const std::vector<T> & edgeVals)
    {
        const size_t size = etaVals.size();
#pragma omp parallel for
        for (size_t idx = 0; idx < size; ++idx)
        {
            etaVals[idx] = (edgeVals[idx] == 0.0 ? 0.0 : 1.0 / edgeVals[idx]);
        }
    }

    void evaluateFitness(std::vector<T> & fitnessArray,
                         const std::vector<T> & pheromoneArray,
                         const std::vector<T> & etaArray,
                         const T alphaParam,
                         const T betaParam)
    {
        const size_t size = fitnessArray.size();
#pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
        {
            fitnessArray[i] = pow(pheromoneArray[i], alphaParam) * pow(etaArray[i], betaParam);
        }
    }

    void performTourConstruction(const std::vector<T> & desirability,
                                 const std::vector<T> & edgeWeights)
    {
#pragma omp parallel for
        for (size_t i = 0; i < agentPool.size(); ++i)
        {
            agentPool[i].constructTour(desirability, edgeWeights);
        }
    }

    void extractBestPath(std::vector<uint32_t> & tourResult,
                         T & tourLength)
    {
        const Ant<T> & optimalAgent = *std::min_element(agentPool.begin(), agentPool.end());
        std::copy(optimalAgent.getTabu().begin(), optimalAgent.getTabu().end(), tourResult.begin());
        tourLength = optimalAgent.getTourLength();
    }

    void generateDeltaMatrix(std::vector<D> & deltaMatrix,
                             const uint32_t totalCities,
                             const T qVal)
    {
#pragma omp parallel for
        for (size_t i = 0; i < deltaMatrix.size(); ++i)
        {
            deltaMatrix[i] = 0.0;
        }

#pragma omp parallel for
        for (size_t k = 0; k < agentPool.size(); ++k)
        {
            const Ant<T> & antRef = agentPool[k];
            const T contribution = qVal / antRef.getTourLength();
            const std::vector<uint32_t> & route = antRef.getTabu();

            for (size_t j = 0; j < route.size() - 1; ++j)
            {
                uint32_t src = route[j];
                uint32_t dst = route[j + 1];
#pragma omp atomic
                deltaMatrix[src * totalCities + dst] += contribution;
#pragma omp atomic
                deltaMatrix[dst * totalCities + src] += contribution;
            }

            uint32_t last = route.back();
            uint32_t first = route.front();
#pragma omp atomic
            deltaMatrix[last * totalCities + first] += contribution;
#pragma omp atomic
            deltaMatrix[first * totalCities + last] += contribution;
        }
    }

    void adjustPheromones(std::vector<T> & pheromoneGrid,
                          const std::vector<D> & deltaGrid,
                          const T evaporation)
    {
        const size_t len = pheromoneGrid.size();
#pragma omp parallel for
        for (size_t i = 0; i < len; ++i)
        {
            pheromoneGrid[i] = pheromoneGrid[i] * evaporation + deltaGrid[i];
        }
    }

public:
    AcoOMP(const Parameters<T> & pRef, Environment<T, D> & eRef)
        : configRef(pRef), contextRef(eRef),
          agentPool(eRef.nAnts, Ant<T>(eRef.nCities))
    {
        prepareEta(contextRef.eta, contextRef.edges);
    }

    void solve()
    {
        for (uint32_t cycle = 0; cycle < configRef.maxEpoch; ++cycle)
        {
            evaluateFitness(contextRef.fitness, contextRef.pheromone, contextRef.eta, configRef.alpha, configRef.beta);
            performTourConstruction(contextRef.fitness, contextRef.edges);
            extractBestPath(contextRef.bestTour, contextRef.bestTourLength);
            generateDeltaMatrix(contextRef.delta, contextRef.nCities, configRef.q);
            adjustPheromones(contextRef.pheromone, contextRef.delta, configRef.rho);
        }
    }

    ~AcoOMP() {}
};

#endif
