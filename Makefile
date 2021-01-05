CC = cc
CFLAGS_COMMON = -fopenmp -std=c90 -Wall

# macOS clang does not build against omp symbols without
# some additional flags :(
OS := $(shell uname -s)
ifeq ($(OS), Darwin)
	CFLAGS = -Xpreprocessor $(CFLAGS_COMMON) -lomp
else
	CFLAGS = $(CFLAGS_COMMON)
endif

C := $(CC) $(CFLAGS)

main: main.o
	mv main.o main

main.o:
	echo CFLAGS
	$(C) main.c rk.c -o main.o

.PHONY: main clean

clean:
	rm -f *.o
	rm -f main