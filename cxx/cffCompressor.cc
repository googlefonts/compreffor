#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <algorithm>
#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

class token_t;
struct charstring_t;
class charstring_pool_t;

typedef std::map<std::string, unsigned> tokmap_t;
typedef std::vector<token_t>::iterator tokiter_t;
typedef uint32_t int_type;
const unsigned int_size = sizeof(int_type);

class token_t {
public:
  token_t (int_type value_ = 0) : value(value_) {}
  token_t (const token_t &other) : value(other.value) {}

  inline int_type getValue() const {
    return value;
  }

  inline unsigned size() const {
    return part(0);
  }

  inline unsigned part(unsigned idx) const {
    assert(idx < 4);
    char shift = (int_size - idx - 1) * 8;
    return (value & (0xff << shift)) >> shift;
  }

  std::string toString() const {
    std::ostringstream os;
    os << "token_t(" << part(0) << ", " << part(1) << 
          ", " << part(2) << ", " << part(3) << ")";
    return os.str();
  }

  bool operator<(const token_t &other) const {
    return value < other.value;
  }

  bool operator!=(const token_t &other) const {
    return value != other.value;
  }

  bool operator==(const token_t &other) const {
    return value == other.value;
  }

private:
  int_type value;
};

std::ostream& operator<<(std::ostream &stream, const token_t &tok) {
  return stream << tok.toString();
}

typedef struct charstring_t {
  tokiter_t begin;
  tokiter_t end;
  unsigned char fd;
} charstring_t;

class charstring_pool_t {
public:
  charstring_pool_t (unsigned nCharstrings) 
    : nextQuark(0), fdSelectTrivial(true), count(nCharstrings),
      finalized(false) {
      pool.reserve(nCharstrings);
      suffixes.reserve(nCharstrings);
      lcp.reserve(nCharstrings);
      offset.reserve(nCharstrings + 1);
      offset.push_back(0);
    }

  void getSubstrings() {
    if (!finalized)
      finalize();

    generateSuffixes();
    generateLCP();
  }

  charstring_t getCharstring(unsigned idx) {
    charstring_t cs;
    cs.begin = pool.begin() + offset[idx];
    cs.end = pool.begin() + offset[idx + 1];
    if (fdSelectTrivial)
      cs.fd = 0;
    else
      cs.fd = fdSelect[idx];
    return cs;
  }

  void addRawCharstring(char* data, unsigned len) {
    if (finalized)
      throw std::runtime_error("Attempted to add a charstring to a closed pool.");

    unsigned nToks = 0;
    for (unsigned csPos = 0; csPos < len; ++csPos) {
      unsigned char first = data[csPos];
      unsigned tokSize;
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
    for (unsigned i = 0; i < pool.size(); ++i) {
      if (i >= offset[cur + 1])
        ++cur;
      rev.push_back(cur);
    }

    finalized = true;
  }
  std::vector<unsigned> lcp;
private:
  tokmap_t quarkMap;
  unsigned nextQuark;
  std::vector<token_t> pool;
  std::vector<unsigned> offset;
  std::vector<unsigned char> fdSelect;
  std::vector<unsigned> suffixes;
  
  std::vector<unsigned> rev;
  bool fdSelectTrivial;
  unsigned count;
  bool finalized;

  inline unsigned quarkFor(unsigned char* data, unsigned len) {
    // TODO: verify using a string key isn't a time problem
    std::string key((const char*) data, (size_t) len);
    if (quarkMap.find(key) == quarkMap.end()) {
      assert(nextQuark < 65536);
      unsigned q = nextQuark++;
      quarkMap[key] = q;
      return q;
    }
    else {
      return quarkMap[key];
    }
  }

  void addRawToken(unsigned char* data, unsigned len) {
    assert(len > 0);
    assert(len < 256);
    int_type v = genValue(data, len);
    pool.push_back(token_t(v));
  }

  int_type genValue(unsigned char* data, unsigned len) {
    int_type v;
    if (len < int_size) {
      v = len;
      for (unsigned i = 0; i < len; ++i) {
        v <<= 8;
        v |= data[i];
      }
      v <<= 8 * (int_size - len - 1);
    }
    else {
      unsigned q = quarkFor(data, len);
      v = len;
      v <<= 8;
      v |= data[0];
      v <<= 16;
      v |= q;
      // std::cout << "QUARK: " << q << std::endl;
    }
    return v;
  }

  void generateSuffixes() {
    assert(finalized);
    assert(suffixes.size() == 0);
    for (unsigned i = 0; i < pool.size(); ++i)
      suffixes.push_back(i);

    std::sort(suffixes.begin(), suffixes.end(), suffixSortFunctor(pool, offset, rev));
  }

  struct suffixSortFunctor {
    const std::vector<token_t> &pool;
    const std::vector<unsigned> &offset;
    const std::vector<unsigned> &rev;
    suffixSortFunctor(const std::vector<token_t> &_pool,
                      const std::vector<unsigned> &_offset,
                      const std::vector<unsigned> &_rev) 
                    : pool(_pool), offset(_offset), rev(_rev) {}
    bool operator()(unsigned a, unsigned b) {
      int aLen = offset[rev[a] + 1] - a;
      int bLen = offset[rev[b] + 1] - b;
      if (aLen == bLen) {
        auto aFirst = pool.begin() + a;
        auto aLast = pool.begin() + offset[rev[a] + 1];
        auto bFirst = pool.begin() + b;
        auto p = std::mismatch(aFirst, aLast, bFirst);
        if (p.first == aLast)
          return false;
        else
          return *p.first < *p.second;
      }
      else {
        return aLen < bLen;
      }
    }
  };

  void generateLCP() {
    assert(finalized);
    assert(suffixes.size() == pool.size());

    unsigned rank[pool.size()];
    lcp.assign(pool.size(), 0);
 
    for (unsigned i = 0; i < pool.size(); ++i) {
      unsigned idx = suffixes[i];
      rank[idx] = i;
    }

    for (std::vector<unsigned>::iterator ch = offset.begin(); ch != offset.end() - 1; ++ch) {
      unsigned start = *ch;
      unsigned end = *(ch + 1);
      unsigned curH = 0;
      for (unsigned tokIdx = start; tokIdx < end; ++tokIdx) {
        unsigned curRank = rank[tokIdx];
        if (curRank > 0) {
          unsigned befInSuffixes = suffixes[curRank - 1];
          unsigned befEnd = offset[rev[befInSuffixes] + 1];
          while (befInSuffixes + curH < befEnd 
                 && tokIdx + curH < end
                 && pool[befInSuffixes + curH] == pool[tokIdx + curH])
            ++curH;
          lcp[curRank] = curH;

          if (curH > 0)
            --curH;
        }
      }
    }
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

  unsigned len;
  for (int i = 0; i < count; ++i) {
    unsigned len = offset[i + 1] - offset[i];
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

  csPool.getSubstrings();
  std::cout << "finished getSubstrings()" << std::endl;

  return 0;
}