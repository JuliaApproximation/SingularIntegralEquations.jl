all: liblhelmfs.dylib

liblhelmfs.dylib: fundsol.o
	gcc-4.9 -shared -o liblhelmfs.dylib fundsol.o -lm -lgomp

.c.o:
	gcc-4.9 -fPIC -fopenmp -O3 -std=c99 -c $<