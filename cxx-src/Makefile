O ?= 2

CXXFLAGS = -std=c++0x -pthread -W -Wall -Wno-unused -Wno-unused-parameter
CXXFLAGS += -DDEBUG -g
CXXSOFLAGS = $(CXXFLAGS) -shared
CC = g++
DEPS = cffCompressor.h

all: cffCompressor lib
cffCompressor: cffCompressor.o
	$(CC) $(CXXFLAGS) -o $@ $^
%.o: %.cc $(DEPS)
	$(CC) $(CXXFLAGS) -o $@ -c -O$(O) $<
lib: libcompreff.so
libcompreff.so: cffCompressor.cc $(DEPS)
	$(CC) $(CXXSOFLAGS) -o $@ -O$(O) -fPIC $<
clean:
	rm -f cffCompressor.o cffCompressor libcompreff.so
ALL: clean all
