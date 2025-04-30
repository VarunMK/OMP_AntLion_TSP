CXX			= g++
CXXFLAGS	= -std=c++14 -O3 -Wall -pedantic -Xpreprocessor -fopenmp -lomp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include
LIBS		= -lpthread

OBJS	= main.cpp TSP.o Environment.o Parameters.o Ant.o AcoCPU.o AcoOMP.o
ACOCPUOMP	= acocpuomp
ACOGPU	= acogpu
STATS	= stats

$(ACOCPUOMP): $(OBJS)
	$(CXX) $(CXXFLAGS) main.cpp -o $(ACOCPUOMP) $(LIBS)

$(ACOGPU): AcoGPU.cu TSP.cpp
	nvcc -Xptxas="-v" -O3 -c TSP.cpp -o TSP.o
	nvcc -Xptxas="-v" -O3 AcoGPU.cu -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	$(RM) *.o *~ $(ACOCPUOMP) $(ACOGPU) $(STATS)
