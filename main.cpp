#include <iostream>
#include <cmath>

#include "AcoCPU.cpp"
#include "AcoOMP.cpp"
#include "TSP.cpp"
#include "Parameters.cpp"
#include "Environment.cpp"
#include "common.hpp"

#ifndef D_TYPE
#define D_TYPE float
#endif

int main(int argc, char *argv[])
{
    char *inputFile = new char[MAX_LEN];
    D_TYPE paramAlpha = 1.0;
    D_TYPE paramBeta = 2.0;
    D_TYPE paramQ = 1.0;
    D_TYPE paramRho = 0.5;
    uint32_t numIterations = 10;

    if (argc < 7 || argc > 8)
    {
        std::cout << "Usage: ./acocpu file.tsp alpha beta q rho maxEpoch [mapWorkers farmWorkers]" << std::endl;
        exit(EXIT_ARGUMENTS_NUMBER);
    }

    argc--;
    argv++;
    inputFile = argv[0];
    paramAlpha = parseArg<D_TYPE>(argv[1]);
    paramBeta = parseArg<D_TYPE>(argv[2]);
    paramQ = parseArg<D_TYPE>(argv[3]);
    paramRho = parseArg<D_TYPE>(argv[4]);
    numIterations = parseArg<uint32_t>(argv[5]);

    int enableOMP = 0;
    if (argc == 7)
    {
        enableOMP = parseArg<uint32_t>(argv[6]);
    }

    TSP<D_TYPE> tspInstance(inputFile);
    Parameters<D_TYPE> config(paramAlpha, paramBeta, paramQ, paramRho, numIterations);

    if (!enableOMP)
    {
        std::cout << "***** RUNNING ACO ON CPU *****" << std::endl;
        Environment<D_TYPE, D_TYPE> simulationEnv(tspInstance.getNCities(),
                                                  tspInstance.getNCities(),
                                                  tspInstance.getEdges());

        AcoCPU<D_TYPE, D_TYPE> cpuSolver(config, simulationEnv);

        startTimer();
        cpuSolver.solve();
        stopAndPrintTimer();

        printMatrixV("bestTour", simulationEnv.getBestTour(), 1, simulationEnv.nCities, 0);
        printResult(tspInstance.getName(),
                    0,
                    0,
                    numIterations,
                    getTimerMS(),
                    getTimerUS(),
                    simulationEnv.getBestTourLength(),
                    tspInstance.calcTourLength(simulationEnv.getBestTour()),
                    tspInstance.checkTour(simulationEnv.getBestTour()));
    }
    else
    {
        std::cout << "***** RUNNING ACO WITH OPENMP *****" << std::endl;
        Environment<D_TYPE, D_TYPE> simulationEnv(tspInstance.getNCities(),
                                                  tspInstance.getNCities(),
                                                  tspInstance.getEdges());

        AcoOMP<D_TYPE, D_TYPE> ompSolver(config, simulationEnv);

        startTimer();
        ompSolver.solve();
        stopAndPrintTimer();

        printMatrixV("bestTour", simulationEnv.getBestTour(), 1, simulationEnv.nCities, 0);
        printResult(tspInstance.getName(),
                    0,
                    0,
                    numIterations,
                    getTimerMS(),
                    getTimerUS(),
                    simulationEnv.getBestTourLength(),
                    tspInstance.calcTourLength(simulationEnv.getBestTour()),
                    tspInstance.checkTour(simulationEnv.getBestTour()));
    }

    return 0;
}
