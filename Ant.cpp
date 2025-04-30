#ifndef __ANT_CPP__
#define __ANT_CPP__

#include <stdint.h>
#include <vector>
#include <random>
#include <limits>

#include "common.hpp"

template <typename T>
class Ant {

private:
    
    const uint32_t totalCities;
    std::vector<uint32_t> pathMemory;
    std::vector<uint8_t> cityStatus;
    std::vector<T> cumulativeProb;
    T pathCost;

    T sampleRandom() {
        static thread_local std::mt19937 rngEngine(randomSeed);
        static thread_local std::uniform_real_distribution<T> dist(0.0f, 1.0f);
        return dist(rngEngine);
    }

    void markAllUnvisited() {
        for (uint8_t & flag : cityStatus) {
            flag = 1;
        }
    }

    void buildTourMemory(const std::vector<T> & desirability) {
        uint32_t current = sampleRandom() * totalCities;
        cityStatus[current] = 0;
        pathMemory[0] = current;

        for (uint32_t step = 1; step < totalCities; ++step) {

            T totalWeight = 0.f;

            auto pPtr = cumulativeProb.begin();
            auto visitPtr = cityStatus.begin();
            auto fitPtr = desirability.begin() + current * totalCities;

            while (pPtr != cumulativeProb.end()) {
                auto & probVal = *(pPtr++);
                const auto & visitFlag = *(visitPtr++);
                const auto & fitVal = *(fitPtr++);

                totalWeight += fitVal * visitFlag;
                probVal = totalWeight;
            }

            const T randVal = sampleRandom() * totalWeight;
            
            uint32_t low = 0, high = totalCities - 1;
            while (low < high) {
                uint32_t mid = (low + high) / 2;
                if (cumulativeProb[mid] < randVal) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }

            current = low;
            pathMemory[step] = current;
            cityStatus[current] = 0;
        }
    }

    void calculatePathCost(const std::vector<T> & edgeMatrix) {

        pathCost = 0.0;
        auto walker = pathMemory.begin();
        const auto start = walker;

        while (walker != pathMemory.end() - 1) {
            const uint32_t src = *(walker++);
            const uint32_t dst = *(walker);
            pathCost += edgeMatrix[src * totalCities + dst];
        }
        const uint32_t src = *(walker);
        const uint32_t dst = *(start);
        pathCost += edgeMatrix[src * totalCities + dst];
    }

public:

    Ant(const uint32_t cityCount):
        totalCities(cityCount),
        pathMemory(cityCount),
        cityStatus(cityCount),
        cumulativeProb(cityCount),
        pathCost(std::numeric_limits<T>::max()) 
    {}

    void constructTour(const std::vector<T> & desirability, const std::vector<T> & edgeMatrix) {
        markAllUnvisited();
        buildTourMemory(desirability);
        calculatePathCost(edgeMatrix);
    }

    const T getTourLength() const {
        return pathCost;
    }

    const std::vector<uint32_t> & getTabu() const {
        return pathMemory;
    }

    inline bool operator< (const Ant<T> & other) const {
        return pathCost < other.pathCost;
    }
};

#endif
