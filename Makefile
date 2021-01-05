CC = mpicc
CFLAGS = -std=c90 -Wall

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