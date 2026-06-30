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

BASILISK_LIBS = \
	-L$(BASILISK)/gl \
	-lglutils \
	-lfb_tiny \
	-lm

all: $(TARGET)

$(TARGET): $(SRC)
	module purge; \
	module load $(MODULES); \
	CC99='$(CC99)' qcc $(QCCFLAGS) $(SRC) -o $(TARGET) $(BASILISK_LIBS)

run: $(TARGET)
	sbatch run.sh

clean:
	rm -f $(TARGET) *.o _*.c

purge:
	rm -f *.dat
	rm -f interface
	rm -f dump-*
	rm -rf images

.PHONY: all run clean purge
