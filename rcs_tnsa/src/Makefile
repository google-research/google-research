CXX=g++
CXXFLAGS=-I. -O3 -Wno-narrowing

.PHONY: all
all: optimize

# Tensor networks.
optimize: main.o sa_optimizer.o tensor_network.o utils.o
	$(CXX) $(CXXFLAGS) -o $@ $^

main.o: main.cpp tensor_network.cpp tensor_network.h sa_optimizer.cpp sa_optimizer.h
	$(CXX) $(CXXFLAGS) -c $<

sa_optimizer.o: sa_optimizer.cpp sa_optimizer.h tensor_network.cpp tensor_network.h
	$(CXX) $(CXXFLAGS) -c $<

tensor_network.o: tensor_network.cpp tensor_network.h
	$(CXX) $(CXXFLAGS) -c $<

utils.o: utils.cpp utils.h
	$(CXX) $(CXXFLAGS) -c $<

.PHONY: clean
clean:
	rm -f optimize *.o *.gch
