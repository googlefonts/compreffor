#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <map>
#include <iostream>
#include <vector>
#include <string>

typedef std::map<std::string, unsigned int> tokmap_t;

class token_t {
public:
  typedef uint32_t int_type;

  token_t (int_type value_ = 0) : value(value_) {}
  token_t (token_t &other) : value(other.value) {}
  token_t (const token_t &other) : value(other.value) {}
  token_t (unsigned char* data, unsigned int len) {
    assert(len > 0);
    assert(len < 256);
    if (len < int_size) {
      unsigned int i = 0;
      int_type v = len;
      while (i++ < len) {
        v <<= 8;
        v |= data[i];
      }
      v <<= 8 * (int_size - len - 1);
      value = v;
    }
    else {
      unsigned int q = quarkFor(data, len);
      int_type v = len;
      v <<= 8;
      v |= data[0];
      v <<= 16;
      v |= q;
      value = v;
      // std::cout << "QUARK: " << q << std::endl;
    }
  }

  unsigned int size();

private:
  static const unsigned int int_size = sizeof (int_type);
  int_type value;

  static tokmap_t quarkmap;

  static inline unsigned int quarkFor(unsigned char* data, unsigned int len) {
    // TODO: verify using a string key isn't a time problem
    std::string key((const char*) data, (size_t) len);
    if (quarkmap.find(key) == quarkmap.end()) {
      assert(next_quark < 65536);
      unsigned int q = next_quark++;
      quarkmap[key] = q;
      return q;
    }
    else {
      return quarkmap[key];
    }
  }

  static unsigned int next_quark;
};

unsigned int token_t::next_quark = 0;
tokmap_t token_t::quarkmap;

unsigned int token_t::size() {
  return (value & 0xff000000) >> 24;
}

std::vector<token_t> tokenPool;


int main(int argc, const char* argv[]) {
  uint16_t count;
  unsigned char countBuffer[2];
  std::cin.read((char*) countBuffer, 2);
  count = (countBuffer[0] << 8) | (countBuffer[1]); // XXX
  std::cout << "count: " << count << std::endl;

  unsigned char offSize;
  std::cin.read((char*) &offSize, 1);
  std::cout << "offSize: " << (int) offSize << std::endl;

  uint32_t offset[count + 1];
  unsigned char offsetBuffer[(count + 1) * offSize];
  std::cin.read((char*) offsetBuffer, (count + 1) * offSize);
  for (int i = 0; i < count + 1; ++i) {
    offset[i] = 0;
    for (int j = 0; j < offSize; ++j) {
      offset[i] += offsetBuffer[i * offSize + j] << ((offSize - j - 1) * 8);
    }
    offset[i] -= 1; // CFF is 1-indexed(-ish)
  }
  assert(offset[0] == 0);
  std::cout << "offset loaded" << std::endl;

  unsigned int csOffset[count];
  unsigned int csSize, nToks;
  for (int i = 0; i < count; ++i) {
    csSize = offset[i + 1] - offset[i];
    nToks = 0;
    for (int csPos = 0; csPos < csSize; ++csPos) {
      unsigned char first;
      std::cin.read((char*) &first, 1);
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
      std::cin.read((char*) rawTok + 1, tokSize - 1);

      token_t nextTok(rawTok, tokSize);
      tokenPool.push_back(nextTok);

      ++nToks;
    }

    if (i > 0)
      csOffset[i] = csOffset[i - 1] + nToks;
    else
      csOffset[i] = 0;
  }

  char fdselectIndicator;
  unsigned char* fdselect;
  std::cin.read(&fdselectIndicator, 1);
  if (fdselectIndicator == 'y') {
    unsigned char buf[count];
    std::cin.read((char*) buf, count);
    fdselect = buf;
    std::cout << "loaded FDSelect" << std::endl;
  }
  else {
    fdselect = NULL;
    std::cout << "no FDSelect provided" << std::endl;
  }

  return 0;
}