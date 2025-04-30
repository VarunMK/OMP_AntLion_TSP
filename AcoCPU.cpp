#ifndef __ACO_CPU_CPP__
#define __ACO_CPU_CPP__

#include <stdint.h>
#include <vector>
#include <algorithm>

#include "Environment.cpp"
#include "Parameters.cpp"
#include "Ant.cpp"
#include "common.hpp"

template <typename T, typename D>
class AcoCPU {
    
private:
    const Parameters<T> & config;
    Environment<T, D>   & context;
    std::vector< Ant<T> > antGroup;
    
    void computeEta(std::vector<T> & etaVals, const std::vector<T> & edgeList)
    {
        auto etaIter = etaVals.begin();
        auto edgeIter = edgeList.begin();
        while (etaIter != etaVals.end()) {
            T & etaVal = *(etaIter++);
            const T & edgeVal = *(edgeIter++);
            etaVal = (edgeVal == 0.0 ? 0.0 : 1.0 / edgeVal);
        }
    }

    void computeFitness(std::vector<T> & fitnessVec,
                        const std::vector<T> & pheromoneVec,
                        const std::vector<T> & etaVec,
                        const T alphaVal,
                        const T betaVal)
    {
        auto fitIter = fitnessVec.begin();
        auto pherIter = pheromoneVec.begin();
        auto etaIter = etaVec.begin();

        while (fitIter != fitnessVec.end()) {
            T & fit = *(fitIter++);
            const T & pher = *(pherIter++);
            const T & eta = *(etaIter++);
            fit = pow(pher, alphaVal) * pow(eta, betaVal);
        }
    }
    
    void generateTours(const std::vector<T> & fitValues,
                       const std::vector<T> & graph)
    {
        for (Ant<T> & agent : antGroup) {
            agent.constructTour(fitValues, graph);
        }
    }
    
    void recordBestTour(std::vector<uint32_t> & pathResult,
                        T & pathLength)
    {
        const Ant<T> & topAnt = *std::min_element(antGroup.begin(), antGroup.end());
        std::copy(topAnt.getTabu().begin(), topAnt.getTabu().end(), pathResult.begin());
        pathLength = topAnt.getTourLength();
    }
     
    void computeDelta(std::vector<D> & deltaVec,
                      const uint32_t cityCount,
                      const T qParam)
    {
        for (T & d : deltaVec) {
            d = 0.0;
        }

        for (Ant<T> & agent : antGroup) {
            const T tauVal = qParam / agent.getTourLength();
            auto walkIter = agent.getTabu().begin();
            const auto startIter = walkIter;

            while (walkIter != agent.getTabu().end() - 1) {
                const uint32_t fromCity = *(walkIter++);
                const uint32_t toCity   = *(walkIter);
                deltaVec[fromCity * cityCount + toCity] += tauVal;
                deltaVec[toCity * cityCount + fromCity] += tauVal;
            }
            const uint32_t fromCity = *(walkIter);
            const uint32_t toCity   = *(startIter);
            deltaVec[fromCity * cityCount + toCity] += tauVal;
            deltaVec[toCity * cityCount + fromCity] += tauVal;
        }
    }
    
    void updatePheromones(std::vector<T> & pheromoneMap,
                          const std::vector<D> & deltaMap,
                          const T decayFactor)
    {
        auto pherIter = pheromoneMap.begin();
        auto deltaIter = deltaMap.begin();

        while (pherIter != pheromoneMap.end()) {
            T & pherVal = *(pherIter++);
            const T & deltaVal = *(deltaIter++);
            pherVal = pherVal * decayFactor + deltaVal;
        }
    }
    
public:
    
    AcoCPU(const Parameters<T> & configRef, Environment<T, D> & ctxRef)
        : config(configRef), context(ctxRef), antGroup(ctxRef.nAnts, Ant<T>(ctxRef.nCities))
    {
        computeEta(context.eta, context.edges);
    }

    void solve() {
        uint32_t currentEpoch = 0;
        do {
            computeFitness    (context.fitness, context.pheromone, context.eta, config.alpha, config.beta);
            generateTours     (context.fitness, context.edges);
            recordBestTour    (context.bestTour, context.bestTourLength);
            computeDelta      (context.delta, context.nCities, config.q);
            updatePheromones  (context.pheromone, context.delta, config.rho);
        } while (++currentEpoch < config.maxEpoch);
    }
    
    ~AcoCPU() {}
};

#endif
