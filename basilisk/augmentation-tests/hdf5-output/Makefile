# ============================================================
# Basilisk Makefile
# Usage:
#   make SRC=filmflow.c TARGET=filmflow
#   make clean
# ============================================================

SHELL := /bin/bash

SRC    ?= filmflow.c
TARGET ?= filmflow

#HiPerGator Modules
MODULES = ufrc gcc openmpi imagemagick hdf5


CC99 = gcc -std=c99 -D_XOPEN_SOURCE=700 -D_GNU_SOURCE=1 -fopenmp

QCCFLAGS = -O2 -g -Wall -disable-dimensions

HDF5_FLAGS = $(shell pkg-config --cflags hdf5 2>/dev/null)
HDF5_LIBS  = $(shell pkg-config --libs hdf5 2>/dev/null)

BASILISK_LIBS = \
	-L$(BASILISK)/gl \
	-lglutils \
	-lfb_tiny \
	$(HDF5_LIBS) \
	-lm

all: $(TARGET)

$(TARGET): $(SRC)
	module purge; \
	module load $(MODULES); \
	HDF5_SHOW="$$(h5cc -show)"; \
	HDF5_FLAGS="$$(echo "$$HDF5_SHOW" | grep -o '\-I[^ ]*')"; \
	HDF5_LIBS="$$(echo "$$HDF5_SHOW" | grep -o '\-L[^ ]*\|\-lhdf5[^ ]*\|\-lz\|\-ldl')"; \
	echo "HDF5_FLAGS = $$HDF5_FLAGS"; \
	echo "HDF5_LIBS  = $$HDF5_LIBS"; \
	CC99='$(CC99)' qcc $(QCCFLAGS) $(HDF5_FLAGS) $(SRC) -o $(TARGET) $(BASILISK_LIBS)

run: $(TARGET)
	sbatch run.sh $(TARGET)

clean:
	rm -f $(TARGET) *.o _*.c

purge:
	rm -f *.dat
	rm -f interface*
	rm -f dump-*
	rm -rf images
	rm -rf run_*

.PHONY: all run clean purge
