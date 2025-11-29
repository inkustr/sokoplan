# Makefile for Festival Sokoban Solver

CXX = g++
CXXFLAGS = -O3 -pthread -DTHREADS -DLINUX -std=c++11 -mcmodel=medium
TARGET = festival
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean

