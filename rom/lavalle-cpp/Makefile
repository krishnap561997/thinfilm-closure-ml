# The HiPerGator hdf5/1.10.5 wrapper defaults to a conda compiler and ships no
# static libs, so force g++ underneath and link the shared libraries.
export HDF5_CXX ?= g++
export HDF5_CLINKER ?= g++
CXX = h5c++ -shlib
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic -Iinclude

SOURCES := $(shell find src -name '*.cpp')
TARGET := thinfilm

# Simulation output files
OUTPUTS := thinfilm_training_data.h5 \
           inlet_signal.csv \
           final_state.csv

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@

# Remove compiled executable
clean:
	rm -f $(TARGET)
	find . -type f -name '*.o' -delete

# Remove executable and all simulation outputs
purge: clean
	rm -f $(OUTPUTS)

.PHONY: all clean purge

# Check the ML hook against Python (see rom/lavalle-cpp/README.md, "ML closure")
ml_check: tools/ml_check.cpp src/ml/ml_closure.cpp src/numerics/derivatives.cpp include/ml/ml_closure.hpp
	g++ $(CXXFLAGS) tools/ml_check.cpp src/ml/ml_closure.cpp src/numerics/derivatives.cpp -o $@
