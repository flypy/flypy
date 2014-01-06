# Makefile for creating our standalone Cython program
PYVERSION=2.7
PYPREFIX=$(shell python -c 'import sys; print(sys.exec_prefix)')
INCLUDES=-I$(PYPREFIX)/include/python$(PYVERSION)
LIBS=-L$(PYPREFIX)/lib

python: python.o
	gcc -o python python.o $(shell python-config --ldflags) -lgc

python.o: python.c Makefile
	gcc -c python.c $(INCLUDES) $(shell python-config --cflags)

python.c: python.pyx
	cython --embed python.pyx

all: python
