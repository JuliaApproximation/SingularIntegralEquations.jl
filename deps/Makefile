ifeq ($(OS), Windows_NT)
	SLIB = dll
else
	UNAME := $(shell uname)
	ifeq ($(UNAME), Darwin)		
		SLIB = dylib
	else
		SLIB = so
	endif
endif

OBJ = fundsol.c
CFLAGS = -shared -fPIC -fopenmp -O3 -std=c99 -lm -lgomp

all:
	gcc-5 $(CFLAGS) $(OBJ) -o liblhelmfs.$(SLIB)