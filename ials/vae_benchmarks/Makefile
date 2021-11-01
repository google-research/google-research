CFLAGS = -O3 -Wall -march=native

all: ialspp_main ials_main icd_main popularity_main

ialspp_main: ialspp_main.cc
	g++ -pthread -I eigen-3.3.9 ${CFLAGS} -c ialspp_main.cc -o lib/ialspp_main.o
	g++ -pthread ${CFLAGS} lib/ialspp_main.o -o bin/ialspp_main

ials_main: ials_main.cc
	g++ -pthread -I eigen-3.3.9 ${CFLAGS} -c ials_main.cc -o lib/ials_main.o
	g++ -pthread ${CFLAGS} lib/ials_main.o -o bin/ials_main

icd_main: icd_main.cc
	g++ -pthread -I eigen-3.3.9 ${CFLAGS} -c icd_main.cc -o lib/icd_main.o
	g++ -pthread ${CFLAGS} lib/icd_main.o -o bin/icd_main

popularity_main: popularity_main.cc
	g++ -pthread -I eigen-3.3.9 ${CFLAGS} -c popularity_main.cc -o lib/popularity_main.o
	g++ -pthread ${CFLAGS} lib/popularity_main.o -o bin/popularity_main
