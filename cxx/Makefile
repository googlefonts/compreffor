O ?= 2

CXXFLAGS = -std=c++0x -W -Wall -Wno-unused -Wno-unused-parameter -g
CXXFLAGS += -DDEBUG -g
CC=g++

all: cffCompressor
cffCompressor: cffCompressor.o
	$(CC) $(CXXFLAGS) -O$(O) -o $@ $^
clean:
	rm -f cffCompressor.o cffCompressor
