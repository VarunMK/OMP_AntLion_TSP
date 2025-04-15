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

    char *path = new char[MAX_LEN];
    D_TYPE alpha = 1.0;
    D_TYPE beta = 2.0;
    D_TYPE q = 1.0;
    D_TYPE rho = 0.5;
    uint32_t maxEpoch = 10;

    if (argc < 7 || argc > 8)
    {
        std::cout << "Usage: ./acocpu file.tsp alpha beta q rho maxEpoch [mapWorkers farmWorkers]" << std::endl;
        exit(EXIT_ARGUMENTS_NUMBER);
    }

    argc--;
    argv++;
    path = argv[0];
    alpha = parseArg<D_TYPE>(argv[1]);
    beta = parseArg<D_TYPE>(argv[2]);
    q = parseArg<D_TYPE>(argv[3]);
    rho = parseArg<D_TYPE>(argv[4]);
    maxEpoch = parseArg<uint32_t>(argv[5]);

    int parallelCondition = 0; // Set to false initially

    if (argc == 7)
    {
        parallelCondition = parseArg<uint32_t>(argv[6]);
    }

    TSP<D_TYPE> tsp(path);
    Parameters<D_TYPE> params(alpha, beta, q, rho, maxEpoch);

    if (!parallelCondition)
    {
        std::cout << "***** ACO CPU *****" << std::endl;
        Environment<D_TYPE, D_TYPE> env(tsp.getNCities(), tsp.getNCities(), tsp.getEdges());
        AcoCPU<D_TYPE, D_TYPE> acocpu(params, env);

        startTimer();
        acocpu.solve();
        stopAndPrintTimer();

        printMatrixV("bestTour", env.getBestTour(), 1, env.nCities, 0);
        printResult(tsp.getName(),
                    0,
                    0,
                    maxEpoch,
                    getTimerMS(),
                    getTimerUS(),
                    env.getBestTourLength(),
                    tsp.calcTourLength(env.getBestTour()),
                    tsp.checkTour(env.getBestTour()));
    }
    else
    {
        std::cout << "***** ACO OMP *****" << std::endl;
        Environment<D_TYPE, D_TYPE> env(tsp.getNCities(), tsp.getNCities(), tsp.getEdges());
        AcoOMP<D_TYPE, D_TYPE> acoomp(params, env);

        startTimer();
        acoomp.solve();
        stopAndPrintTimer();

        printMatrixV("bestTour", env.getBestTour(), 1, env.nCities, 0);
        printResult(tsp.getName(),
                    0,
                    0,
                    maxEpoch,
                    getTimerMS(),
                    getTimerUS(),
                    env.getBestTourLength(),
                    tsp.calcTourLength(env.getBestTour()),
                    tsp.checkTour(env.getBestTour()));
    }

    return 0;
}
