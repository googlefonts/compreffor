from libc.stdint cimport uint32_t


cdef extern from "../cxx/cffCompressor.h":
    uint32_t* _compreff "compreff" (
        unsigned char* dataStream, int numRounds,
        unsigned& outputLength) except +
    void unload(uint32_t* response)


def compreff(bytes dataStream, int numRounds):
    cdef unsigned outputLength = 0
    cdef uint32_t* raw_output = _compreff(dataStream, numRounds, outputLength)
    cdef list output = []
    cdef unsigned i
    for i in range(outputLength):
        output.append(raw_output[i])
    if raw_output != NULL:
        unload(raw_output)
    return output
