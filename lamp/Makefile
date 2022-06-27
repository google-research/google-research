CC=c++
# PROF=-pg
OPT=-O3
# Use at least c++14 to guarantee that erasing does change the order of iterators
# See http://www.cplusplus.com/reference/unordered_map/unordered_map/erase/
CFLAGS=-g $(OPT) -std=c++20 -Wall -Werror -c $(PROF)
LDFLAGS=-g $(OPT) $(PROF)
LIBS=-lgflags
OBJECTS=gradient.o common.o io.o lamp.o
EXECUTABLE=lamp

# gflags needs this on Mac
UNAME_S = $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
  CFLAGS += -I/usr/local/Cellar/gflags/2.2.2/include
  LDFLAGS += -L/usr/local/Cellar/gflags/2.2.2/lib
endif

all: $(EXECUTABLE)

common.o: common.h common.cc
	$(CC) $(CFLAGS) common.cc

gradient.o: common.h gradient.h gradient.cc
	$(CC) $(CFLAGS) gradient.cc

io.o: common.h io.h io.cc
	$(CC) $(CFLAGS) io.cc

lamp.o: common.h io.h gradient.h lamp.cc
	$(CC) $(CFLAGS) lamp.cc

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $(LIBS) -o $@

clean:
	rm -f *.o lamp
