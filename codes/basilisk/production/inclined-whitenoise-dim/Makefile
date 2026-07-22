# ============================================================
# Basilisk Production Makefile
#
# Usage:
#   make                  # Compile filmflow0.c
#   make default          # Compile filmflow0.c
#   make restart          # Compile filmflow_restart.c
#   make run              # Submit filmflow
#   make run-restart      # Submit filmflow_restart
#   make clean
#   make purge
# ============================================================

SHELL := /bin/bash

# ============================================================
# Source files and executables
# ============================================================

DEFAULT_SRC    := filmflow0.c
DEFAULT_TARGET := filmflow

RESTART_SRC    := filmflow_restart.c
RESTART_TARGET := filmflow_restart

# ============================================================
# Modules and compiler configuration
# ============================================================

MODULES := ufrc gcc openmpi imagemagick hdf5

CC99 := gcc \
	-std=c99 \
	-D_XOPEN_SOURCE=700 \
	-D_GNU_SOURCE=1 \
	-fopenmp

QCCFLAGS := \
	-O3 \
	-fno-strict-aliasing \
	-march=x86-64-v3 \
	-disable-dimensions

# ============================================================
# Include paths and libraries
# ============================================================

CUSTOM_HEADERS := \
	/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/headers

OPENGLIBS := \
	-L$(BASILISK)/gl \
	-lglutils \
	-lfb_tiny

HDFLIBS := \
	-I$$HPC_HDF5_INC \
	-L$$HPC_HDF5_LIB \
	-lhdf5 \
	-lhdf5_hl

LIBS := \
	-I$(BASILISK) \
	-I$(CUSTOM_HEADERS) \
	$(OPENGLIBS) \
	$(HDFLIBS) \
	-lm

# ============================================================
# Build targets
# ============================================================

all: default

default: $(DEFAULT_TARGET)

restart: $(RESTART_TARGET)

$(DEFAULT_TARGET): $(DEFAULT_SRC)
	module purge; \
	module load $(MODULES); \
	CC99='$(CC99)' qcc $(QCCFLAGS) \
		$(DEFAULT_SRC) \
		-o $(DEFAULT_TARGET) \
		$(LIBS)

$(RESTART_TARGET): $(RESTART_SRC)
	module purge; \
	module load $(MODULES); \
	CC99='$(CC99)' qcc $(QCCFLAGS) \
		$(RESTART_SRC) \
		-o $(RESTART_TARGET) \
		$(LIBS)

# ============================================================
# Run targets
# ============================================================

run: $(DEFAULT_TARGET)
	sbatch run.sh $(DEFAULT_TARGET)

run-restart: $(RESTART_TARGET)
	sbatch run.sh $(RESTART_TARGET)

# ============================================================
# Cleanup
# ============================================================

clean:
	rm -f $(DEFAULT_TARGET) $(RESTART_TARGET)
	rm -f *.o _*.c

purge:
	rm -f *.dat
	rm -f interface*
	rm -f infc*.dat
	rm -f dump*
	rm -rf images
	rm -rf output
	rm -rf infc
	rm -rf logs
	rm -rf run_*

.PHONY: all default restart run run-restart clean purge
