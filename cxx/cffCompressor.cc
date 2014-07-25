#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <algorithm>
#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class token_t;
struct charstring_t;
class charstring_pool_t;

typedef std::map<std::string, unsigned int> tokmap_t;
typedef std::vector<token_t>::iterator tokiter_t;
typedef uint32_t int_type;
const unsigned int int_size = sizeof(int_type);

class token_t {
public:
  token_t (int_type value_ = 0) : value(value_) {}
  token_t (const token_t &other) : value(other.value) {}

  inline int_type getValue() const {
    return value;
  }

  inline unsigned int size() const {
    return part(0);
  }

  inline unsigned int part(unsigned int idx) const {
    assert(idx < 4);
    char shift = (int_size - idx - 1) * 8;
    return (value & (0xff << shift)) >> shift;
  }

  bool operator<(const token_t &other) {
    return value < other.value;
  }

  bool operator!=(const token_t &other) {
    return value != other.value;
  }

private:
  int_type value;
};

std::ostream& operator<<(std::ostream &stream, const token_t &tok) {
  return stream << "token_t(" << tok.part(0) << ", " << tok.part(1) << 
                   ", " << tok.part(2) << ", " << tok.part(3) << ")";
}

struct charstring_t {
  tokiter_t begin;
  tokiter_t end;
  unsigned char fd;
};

class charstring_pool_t {
public:
  charstring_pool_t (unsigned int nCharstrings) 
    : nextQuark(0), fdSelectTrivial(true), count(nCharstrings),
      finalized(false) {
      pool.reserve(nCharstrings);
      offset.reserve(nCharstrings + 1);
      offset.push_back(0);
    }

  void generateSuffixes() {
    assert(suffixes.size() == 0 && finalized);
    for (unsigned int i = 0; i < pool.size(); ++i)
      suffixes.push_back(i);

    std::sort(suffixes.begin(), suffixes.end(), suffixSortFunctor(pool, offset, rev));
  }

  charstring_t getCharstring(unsigned int idx) {
    charstring_t cs;
    cs.begin = pool.begin() + offset[idx];
    cs.end = pool.begin() + offset[idx + 1];
    if (fdSelectTrivial)
      cs.fd = 0;
    else
      cs.fd = fdSelect[idx];
    return cs;
  }

  void addRawCharstring(char* data, unsigned int len) {
    unsigned int nToks = 0;
    for (unsigned int csPos = 0; csPos < len; ++csPos) {
      unsigned char first = data[csPos];
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

      unsigned char rawTok[tokSize];
      rawTok[0] = first;
      memcpy(rawTok + 1, data + csPos + 1, tokSize - 1);
      csPos += (tokSize - 1);

      addRawToken(rawTok, tokSize);

      ++nToks;
    }
  
    offset.push_back(offset.back() + nToks);
  }

  void setFDSelect(unsigned char* rawFD) {
    if (rawFD == NULL) {
      fdSelectTrivial = true;
    }
    else {
      fdSelectTrivial = false;
      std::vector<unsigned char> fdSelect(rawFD, rawFD + count);
    }
  }

  void finalize() {
    rev.reserve(pool.size());
    int cur = 0;
    for (unsigned int i = 0; i < pool.size(); ++i) {
      if (i >= offset[cur + 1])
        ++cur;
      rev.push_back(cur);
    }

    finalized = true;
  }

private:
  tokmap_t quarkMap;
  unsigned int nextQuark;
  std::vector<token_t> pool;
  std::vector<unsigned int> offset;
  std::vector<unsigned char> fdSelect;
  std::vector<unsigned int> suffixes;
  std::vector<unsigned int> rev;
  bool fdSelectTrivial;
  unsigned int count;
  bool finalized;

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
      for (; i < len; ++i) {
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

  struct suffixSortFunctor {
    const std::vector<token_t> &pool;
    const std::vector<unsigned int> &offset;
    const std::vector<unsigned int> &rev;
    suffixSortFunctor(const std::vector<token_t> &_pool,
                      const std::vector<unsigned int> &_offset,
                      const std::vector<unsigned int> &_rev) 
                    : pool(_pool), offset(_offset), rev(_rev) {}
    bool operator()(unsigned int a, unsigned int b) {
      int aLen = offset[rev[a] + 1] - a;
      int bLen = offset[rev[b] + 1] - b;
      if (aLen == bLen) {
        auto aIter = pool.begin() + a;
        auto bIter = pool.begin() + b;
        for (int i = 0; i < aLen; ++i) {
          if (((token_t) *aIter) != *bIter)
            return ((token_t) *aIter) < *bIter;

          ++aIter;
          ++bIter;
        }
        return false;
      }
      else {
        return aLen < bLen;
      }
    }
  };
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

  unsigned int len;
  for (int i = 0; i < count; ++i) {
    unsigned int len = offset[i + 1] - offset[i];
    char data[len];
    instream.read(data, len);
    csPool.addRawCharstring(data, len);
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
  csPool.setFDSelect(fdselect);

  csPool.finalize();

  return csPool;
}


int main(int argc, const char* argv[]) {
  charstring_pool_t csPool = charstringPoolFactory(std::cin);

  csPool.generateSuffixes();

  return 0;
}