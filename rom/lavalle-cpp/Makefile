CXX = h5c++
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
