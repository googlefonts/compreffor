#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <map>
#include <iostream>
#include <vector>
#include <string>

typedef std::map<std::string, unsigned int> tokmap_t;
typedef uint32_t int_type;
const unsigned int int_size = sizeof(int_type);

class token_t {
public:
  token_t (int_type value_ = 0) : value(value_) {}
  token_t (token_t &other) : value(other.value) {}
  token_t (const token_t &other) : value(other.value) {}

  unsigned int size() {
    return (value & 0xff000000) >> 24;
  }

private:
  int_type value;
};

class charstring_pool_t {
public:
  charstring_pool_t (unsigned int nCharstrings) 
    : quarkMap(), nextQuark(0), offset(nCharstrings), fdselect(nCharstrings) {}

  void readCharString(std::istream &stream, unsigned int len) {
    unsigned int nToks = 0;
    for (int csPos = 0; csPos < len; ++csPos) {
      unsigned char first;
      stream.read((char*) &first, 1);
      unsigned int tokSize;
      if (first < 12)
        // operators
        tokSize = 1;
      else if (first < 13)
        // escape + addl operator code
        tokSize = 2;
      else if (first < 19)
        // operators
        tokSize = 1;
      else if (first < 21)
        // hintmask/cntrmask
        assert(false); // TODO not implemented
      else if (first < 28)
        // operators
        tokSize = 1;
      else if (first < 29)
        // 16-bit signed
        tokSize = 3;
      else if (first < 32)
        // operators
        tokSize = 1;
      else if (first < 247)
        // -107 to 107
        tokSize = 1;
      else if (first < 251)
        // +108 to +1131
        tokSize = 2;
      else if (first < 255)
        // -108 to -1131
        tokSize = 2;
      else
        // 4-byte floating point
        tokSize = 5;

      csPos += (tokSize - 1);

      unsigned char rawTok[tokSize];
      rawTok[0] = first;
      stream.read((char*) rawTok + 1, tokSize - 1);

      addRawToken(rawTok, tokSize);

      ++nToks;
    }

    offset.push_back(nToks);
  }

  void setFDSelect() {

  }

private:
  tokmap_t quarkMap;
  unsigned int nextQuark;
  std::vector<token_t> pool;
  std::vector<token_t> offset;
  std::vector<token_t> fdselect;

  inline unsigned int quarkFor(unsigned char* data, unsigned int len) {
    // TODO: verify using a string key isn't a time problem
    std::string key((const char*) data, (size_t) len);
    if (quarkMap.find(key) == quarkMap.end()) {
      assert(nextQuark < 65536);
      unsigned int q = nextQuark++;
      quarkMap[key] = q;
      return q;
    }
    else {
      return quarkMap[key];
    }
  }

  void addRawToken(unsigned char* data, unsigned int len) {
    assert(len > 0);
    assert(len < 256);
    int_type v;
    if (len < int_size) {
      unsigned int i = 0;
      v = len;
      while (i++ < len) {
        v <<= 8;
        v |= data[i];
      }
      v <<= 8 * (int_size - len - 1);
    }
    else {
      unsigned int q = quarkFor(data, len);
      v = len;
      v <<= 8;
      v |= data[0];
      v <<= 16;
      v |= q;
      // std::cout << "QUARK: " << q << std::endl;
    }
    pool.push_back(token_t(v));
  }
};

charstring_pool_t charstringPoolFactory(std::istream &instream) {
  uint16_t count;
  unsigned char countBuffer[2];
  instream.read((char*) countBuffer, 2);
  count = (countBuffer[0] << 8) | (countBuffer[1]); // XXX
  std::cout << "count: " << count << std::endl;

  unsigned char offSize;
  instream.read((char*) &offSize, 1);
  std::cout << "offSize: " << (int) offSize << std::endl;

  uint32_t offset[count + 1];
  unsigned char offsetBuffer[(count + 1) * offSize];
  instream.read((char*) offsetBuffer, (count + 1) * offSize);
  for (int i = 0; i < count + 1; ++i) {
    offset[i] = 0;
    for (int j = 0; j < offSize; ++j) {
      offset[i] += offsetBuffer[i * offSize + j] << ((offSize - j - 1) * 8);
    }
    offset[i] -= 1; // CFF is 1-indexed(-ish)
  }
  assert(offset[0] == 0);
  std::cout << "offset loaded" << std::endl;

  charstring_pool_t csPool(count);

  for (int i = 0; i < count; ++i) {
    csPool.readCharString(instream, offset[i + 1] - offset[i]);
  }

  std::cout << "loaded " << offset[count] << " bytes of charstrings" << std::endl;

  unsigned char fdCount;
  unsigned char* fdselect;
  instream.read((char *) &fdCount, 1);
  if (fdCount > 1) {
    unsigned char buf[count];
    instream.read((char*) buf, count);
    fdselect = buf;
    std::cout << "loaded FDSelect" << std::endl;
  }
  else {
    fdselect = NULL;
    std::cout << "no FDSelect loaded" << std::endl;
  }

  return csPool;
}


int main(int argc, const char* argv[]) {
  charstring_pool_t csPool = charstringPoolFactory(std::cin);

  return 0;
}