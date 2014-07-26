O ?= 2

CXXFLAGS = -std=c++0x -W -Wall -Wno-unused -Wno-unused-parameter -g
CXXFLAGS += -DDEBUG -g
CC=g++

all: cffCompressor
cffCompressor: cffCompressor.o
	$(CC) $(CXXFLAGS) -o $@ $^
%.o: %.cc
	$(CC) $(CXXFLAGS) -o $@ -c -O$(O) $^
clean:
	rm -f cffCompressor.o cffCompressor
