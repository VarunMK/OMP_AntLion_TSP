#include <iostream>
#include <fstream>
#include <float.h>
#include <cmath>
#include <climits>
#include <thread>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#include "common.hpp"
#include "TSP.cpp"

#define cudaCheck(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)
inline void cudaAssert(cudaError_t code, const char * file, uint32_t line, bool abort = true) {
    if (code != cudaSuccess) {
        std::cout <<  "cudaErrorAssert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

// Kernel for initializing curand state
__global__ void initCurand(curandStateXORWOW_t *state, uint64_t seed, uint32_t numStates) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t c = tid; c < numStates; c += gridDim.x * blockDim.x) {
        curand_init(seed, c, 0, &state[c]);
    }
}

__device__ __forceinline__ float randXOR(curandState *state) {
    return (float) curand_uniform(state);
}

// Kernel to initialize heuristic matrix (eta)
__global__ void initHeuristicMatrix(float *heuristicMatrix, const float *distanceMatrix, uint32_t rows, uint32_t cols) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = tid; c < cols; c += gridDim.x * blockDim.x) {
            uint32_t id = r * cols + c;
            float d = distanceMatrix[id];
            heuristicMatrix[id] = (d == 0.0f) ? 0.0f : __powf(d, -2.0f);
        }
    }
}

// Kernel to initialize pheromone matrix
__global__ void initPheromoneMatrix(float *pheromoneMatrix, float initialValue, uint32_t rows, uint32_t cols) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = tid; c < cols; c += gridDim.x * blockDim.x) {
            uint32_t id = r * cols + c;
            pheromoneMatrix[id] = initialValue;
        }
    }
}

// Kernel to initialize delta pheromone matrix
__global__ void initDeltaMatrix(float *deltaMatrix, uint32_t rows, uint32_t cols) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = tid; c < cols; c += gridDim.x * blockDim.x) {
            uint32_t id = r * cols + c;
            deltaMatrix[id] = 0.0f;
        }
    }
}

// Kernel to compute fitness matrix
__global__ void calcFitnessMatrix(float *fitnessMatrix, const float *pheromoneMatrix, const float *heuristicMatrix,
                                 float alpha, float beta, uint32_t rows, uint32_t cols) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = tid; c < cols; c += gridDim.x * blockDim.x) {
            uint32_t id = r * cols + c;
            float p = pheromoneMatrix[id];
            float e = heuristicMatrix[id];
            fitnessMatrix[id] = __powf(p, alpha) * e;
        }
    }
}

// Helper: Round up division
inline uint32_t divUp(uint32_t elems, uint32_t div) {
    return (elems + div - 1) / div;
}

// Helper: Get number of blocks
inline uint32_t numberOfBlocks(uint32_t elems, uint32_t blockSize) {
    return divUp(elems, blockSize);
}

// Helper: Align size to warp size * 4
inline uint32_t alignToWarp4(uint32_t elems) {
    return numberOfBlocks(elems, 128) * 128;
}

int main(int argc, char *argv[]) {

    // Parameters
    char * tspFilePath = new char[MAX_LEN];
    float alpha = 1.0f;
    float beta = 2.0f;
    float q = 1.0f;
    float rho = 0.5f;
    uint32_t maxEpochs = 1;
    uint32_t threadsPerBlock = 128;
    uint32_t numBlocks = 1;
    uint32_t numWarpsPerBlock = 1;

    if (argc < 7) {
        std::cout << "Usage:"
                  << " ./acogpu"
                  << " file.tsp"
                  << " alpha"
                  << " beta"
                  << " q"
                  << " rho"
                  << " maxEpoch"
                  << " [threadsPerBlock = " << threadsPerBlock << "]"
                  << " [numBlocks = " << numBlocks << "]"
                  << " [numWarpsPerBlock = " << numWarpsPerBlock << "]"
                  << std::endl;
        exit(-1);
    }

    tspFilePath      = argv[1];
    alpha            = parseArg<float>   (argv[2]);
    beta             = parseArg<float>   (argv[3]);
    q                = parseArg<float>   (argv[4]);
    rho              = parseArg<float>   (argv[5]);
    maxEpochs        = parseArg<uint32_t>(argv[6]);
    if (argc > 7) threadsPerBlock = parseArg<uint32_t>(argv[7]);
    if (argc > 8) numBlocks       = parseArg<uint32_t>(argv[8]);
    if (argc > 9) numWarpsPerBlock= parseArg<uint32_t>(argv[9]);

    TSP<float> tsp(tspFilePath);

    // Problem size
    const uint64_t randomSeed         = time(0);
    const uint32_t numAnts            = tsp.getNCities();
    const uint32_t numCities          = tsp.getNCities();
    const float    initialPheromone   = 1.0f / numCities;

    // Device arrays
    curandStateXORWOW_t * randomStates;
    float * distanceMatrix;
    float * heuristicMatrix;
    float * pheromoneMatrix;
    float * fitnessMatrix;
    float * deltaMatrix;
    uint32_t * tabuMatrix;
    float * tourLengths;
    uint32_t * bestTour;
    float * bestTourLength;

    // Matrix sizes (aligned)
    const uint32_t alignedAnts      = alignToWarp4(numAnts);
    const uint32_t alignedCities    = alignToWarp4(numCities);

    // Matrix dimensions
    const uint32_t randStateRows    = alignedAnts;
    const uint32_t randStateCols    = 1;
    const uint32_t distRows         = numCities;
    const uint32_t distCols         = alignedCities;
    const uint32_t heurRows         = numCities;
    const uint32_t heurCols         = alignedCities;
    const uint32_t pherRows         = numCities;
    const uint32_t pherCols         = alignedCities;
    const uint32_t fitRows          = numCities;
    const uint32_t fitCols          = alignedCities;
    const uint32_t deltaRows        = numCities;
    const uint32_t deltaCols        = alignedCities;
    const uint32_t tabuRows         = numAnts;
    const uint32_t tabuCols         = alignedCities;
    const uint32_t tourLenRows      = alignedAnts;
    const uint32_t tourLenCols      = 1;
    const uint32_t bestTourRows     = alignedCities;
    const uint32_t bestTourCols     = 1;

    // Element counts
    const uint32_t randStateElems   = randStateRows  * randStateCols;
    const uint32_t distElems        = distRows       * distCols;
    const uint32_t heurElems        = heurRows       * heurCols;
    const uint32_t pherElems        = pherRows       * pherCols;
    const uint32_t fitElems         = fitRows        * fitCols;
    const uint32_t deltaElems       = deltaRows      * deltaCols;
    const uint32_t tabuElems        = tabuRows       * tabuCols;
    const uint32_t tourLenElems     = tourLenRows    * tourLenCols;
    const uint32_t bestTourElems    = bestTourRows   * bestTourCols;

    // Memory requirements
    const float gmemRequired = (randStateElems  * sizeof(float)    +
                                distElems       * sizeof(float)    +
                                heurElems       * sizeof(float)    +
                                pherElems       * sizeof(float)    +
                                fitElems        * sizeof(float)    +
                                deltaElems      * sizeof(float)    +
                                tabuElems       * sizeof(uint32_t) +
                                tourLenElems    * sizeof(float)    +
                                bestTourElems   * sizeof(uint32_t) +
                                1               * sizeof(float)
                                ) / 1048576.0;

    const uint32_t smemRequired  = numWarpsPerBlock * alignedCities * 5;

    // CUDA device info
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cout << "There are no available device(s) that support CUDA" << std::endl;
        exit(-1);
    }
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));

    const float globalMemory = deviceProp.totalGlobalMem / 1048576.0;
    const uint32_t sharedMemory = deviceProp.sharedMemPerBlock;

    std::cout << "       Device: " << deviceProp.name           << std::endl
              << "Global memory: " << std::setw(8) << std::setprecision(2) << std::fixed << globalMemory     << " MB" << std::endl
              << "     required: " << std::setw(8) << std::setprecision(2) << std::fixed << gmemRequired     << " MB" << std::endl
              << "Shared memory: " << std::setw(8) << std::setprecision(2) << std::fixed << sharedMemory     << "  B" << std::endl
              << "     required: " << std::setw(8) << std::setprecision(2) << std::fixed << smemRequired     << "  B" << std::endl;

    // Allocate device memory
    cudaCheck(cudaMallocManaged(&randomStates,   randStateElems  * sizeof(curandStateXORWOW_t)));
    cudaCheck(cudaMallocManaged(&distanceMatrix, distElems       * sizeof(float)));
    cudaCheck(cudaMallocManaged(&heuristicMatrix,heurElems       * sizeof(float)));
    cudaCheck(cudaMallocManaged(&pheromoneMatrix,pherElems       * sizeof(float)));
    cudaCheck(cudaMallocManaged(&fitnessMatrix,  fitElems        * sizeof(float)));
    cudaCheck(cudaMallocManaged(&deltaMatrix,    deltaElems      * sizeof(float)));
    cudaCheck(cudaMallocManaged(&tabuMatrix,     tabuElems       * sizeof(uint32_t)));
    cudaCheck(cudaMallocManaged(&tourLengths,    tourLenElems    * sizeof(float)));
    cudaCheck(cudaMallocManaged(&bestTour,       bestTourElems   * sizeof(uint32_t)));
    cudaCheck(cudaMallocManaged(&bestTourLength, sizeof(float)));

    rho = 1.0f - rho;

    *bestTourLength = FLT_MAX;
    for (uint32_t i = 0; i < tourLenElems; ++i) {
        tourLengths[i] = FLT_MAX;
    }

    // Copy TSP distances to distanceMatrix (aligned)
    const std::vector<float> & tspEdges = tsp.getEdges();
    for (uint32_t i = 0; i < numCities; ++i) {
        for (uint32_t j = 0; j < alignedCities; ++j) {
            const uint32_t alignedId = i * alignedCities + j;
            const uint32_t id = i * numCities + j;
            distanceMatrix[alignedId] = (j < numCities) ? tspEdges[id] : 0.0;
        }
    }

    // CUDA kernel launch configurations
    const dim3 randBlock(threadsPerBlock);
    const dim3 randGrid(numberOfBlocks(randStateElems, randBlock.x));
    const dim3 heurBlock(threadsPerBlock);
    const dim3 heurGrid(numberOfBlocks(heurCols, heurBlock.x));
    const dim3 pherBlock(threadsPerBlock);
    const dim3 pherGrid(numberOfBlocks(pherCols, pherBlock.x));
    const dim3 deltaBlock(threadsPerBlock);
    const dim3 deltaGrid(numberOfBlocks(deltaCols, deltaBlock.x));
    const dim3 fitBlock(threadsPerBlock);
    const dim3 fitGrid(numberOfBlocks(fitCols, fitBlock.x));
    const dim3 tourGrid(numBlocks);
    const dim3 tourBlock(32 * numWarpsPerBlock);
    const uint32_t tourShared  = numWarpsPerBlock * (alignedCities  * sizeof(uint8_t) + alignedCities  * sizeof(float));
    const dim3 lenGrid(numAnts);
    const dim3 lenBlock(threadsPerBlock);
    const uint32_t lenShared = lenBlock.x / 32 * sizeof(float);
    const dim3 bestGrid(1);
    const dim3 bestBlock(32);
    const dim3 updateDeltaGrid(numAnts);
    const dim3 updateDeltaBlock(threadsPerBlock);
    const uint32_t updateDeltaShared = alignedCities * sizeof(uint32_t);
    const dim3 updatePheroBlock(threadsPerBlock);
    const dim3 updatePheroGrid(numberOfBlocks(pherCols, updatePheroBlock.x));

    // Occupancy check
    uint32_t threadsActive = 0;
    uint32_t realActiveBlocks = 0;
    uint32_t maxActiveBlocks = 0;
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor((int *)&maxActiveBlocks, calcTour, tourBlock.x, tourShared));
    realActiveBlocks = (tourGrid.x < maxActiveBlocks * deviceProp.multiProcessorCount) ?
                        tourGrid.x : maxActiveBlocks * deviceProp.multiProcessorCount;
    threadsActive = realActiveBlocks * tourBlock.x;

    if (tourShared > deviceProp.sharedMemPerBlock) {
        std::cout << "Shared memory is not enough. Please reduce numWarpsPerBlock." << std::endl;
        printResult(tsp.getName(),
                    0,
                    threadsActive,
                    maxEpochs,
                    0,
                    0,
                    numBlocks,
                    numWarpsPerBlock,
                    false);
        exit(-1);
    }

    if (numBlocks > divUp(numAnts, numWarpsPerBlock) + 1) {
        std::cout << "Too many resources will be wasted. Please reduce numBlocks and/or numWarpsPerBlock parameters." << std::endl;
        printResult(tsp.getName(),
                    0,
                    threadsActive,
                    maxEpochs,
                    0,
                    0,
                    numBlocks,
                    numWarpsPerBlock,
                    false);
        exit(-1);
    }

    // Initialization kernels
    initCurand<<<randGrid, randBlock>>>(randomStates, randomSeed, alignedAnts);
    cudaCheck(cudaGetLastError());
    initHeuristicMatrix<<<heurGrid, heurBlock>>>(heuristicMatrix, distanceMatrix, heurRows, heurCols);
    cudaCheck(cudaGetLastError());
    initPheromoneMatrix<<<pherGrid, pherBlock>>>(pheromoneMatrix, initialPheromone, pherRows, pherCols);
    cudaCheck(cudaGetLastError());

    // Timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    cudaCheck(cudaEventRecord(start, 0));

    // Main optimization loop
    uint32_t epoch = 0;
    do {
        initDeltaMatrix<<<deltaGrid, deltaBlock>>>(deltaMatrix, deltaRows, deltaCols);
        cudaCheck(cudaGetLastError());
        calcFitnessMatrix<<<fitGrid, fitBlock>>>(fitnessMatrix, pheromoneMatrix, heuristicMatrix, alpha, beta, fitRows, fitCols);
        cudaCheck(cudaGetLastError());
        calcTour<<<tourGrid, tourBlock, tourShared>>>(tabuMatrix, fitnessMatrix, numAnts, tabuRows, tabuCols, randomStates);
        cudaCheck(cudaGetLastError());
        calcTourLength<<<lenGrid, lenBlock, lenShared>>>(tourLengths, distanceMatrix, tabuMatrix, numAnts, alignedCities, numCities);
        cudaCheck(cudaGetLastError());
        updateBestTour<<<bestGrid, bestBlock>>>(bestTour, bestTourLength, tabuMatrix, tourLengths, numAnts, alignedCities);
        cudaCheck(cudaGetLastError());
        updateDelta<<<updateDeltaGrid, updateDeltaBlock, updateDeltaShared>>>(deltaMatrix, tabuMatrix, tourLengths, numAnts, alignedCities, numCities, q);
        cudaCheck(cudaGetLastError());
        updatePheromone<<<updatePheroGrid, updatePheroBlock>>>(pheromoneMatrix, deltaMatrix, pherRows, pherCols, rho);
        cudaCheck(cudaGetLastError());
    } while (++epoch < maxEpochs);

    cudaCheck(cudaEventRecord(stop, 0));
    cudaCheck(cudaEventSynchronize(stop));
    float msec;
    long usec;
    cudaCheck(cudaEventElapsedTime(&msec, start, stop));
    usec = msec * 1000;
    std::cout << "Compute time: " << msec << " ms " << usec << " usec " << std::endl;
    printMatrix("bestTour", bestTour, 1, numCities);
    printResult(tsp.getName(),
                realActiveBlocks,
                threadsActive,
                maxEpochs,
                msec,
                usec,
                *bestTourLength,
                *tourLengths,
                tsp.checkTour(bestTour));

    cudaFree(randomStates);
    cudaFree(distanceMatrix);
    cudaFree(heuristicMatrix);
    cudaFree(pheromoneMatrix);
    cudaFree(fitnessMatrix);
    cudaFree(deltaMatrix);
    cudaFree(tabuMatrix);
    cudaFree(tourLengths);
    cudaFree(bestTour);
    cudaFree(bestTourLength);

    return 0;
}