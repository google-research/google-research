CFLAGS = -O3 -Wall -march=native -fopenmp

all: sue_main

sue_main: sue_main.cc
	g++ -pthread -I eigen-3.3.9 ${CFLAGS} -c sue_main.cc -o lib/sue_main.o
	g++ -pthread ${CFLAGS} lib/sue_main.o -o bin/sue_main

