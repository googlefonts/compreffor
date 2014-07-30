#include "cffCompressor.h"

const unsigned int_size = sizeof(int_type);
const float K = 0.1;
const float ALPHA = 0.1;
const unsigned NUM_THREADS = 100;

// token_t ============
token_t::token_t (int_type value_) : value(value_) {}
token_t::token_t (const token_t &other) : value(other.value) {}

int_type token_t::getValue() const {
  return value;
}

inline unsigned token_t::size() const {
  return part(0);
}

inline unsigned token_t::part(unsigned idx) const {
  assert(idx < 4);
  char shift = (int_size - idx - 1) * 8;
  return (value & (0xff << shift)) >> shift;
}

std::string token_t::toString() const {
  std::ostringstream os;
  os << "token_t(" << part(0) << ", " << part(1) << 
        ", " << part(2) << ", " << part(3) << ")";
  return os.str();
}

bool token_t::operator<(const token_t &other) const {
  return value < other.value;
}

bool token_t::operator!=(const token_t &other) const {
  return value != other.value;
}

bool token_t::operator==(const token_t &other) const {
  return value == other.value;
}

std::ostream& operator<<(std::ostream &stream, const token_t &tok) {
  return stream << tok.toString();
}
// end token_t ===============


// light_substring_t =========
bool light_substring_t::operator<(const light_substring_t &other) const {
  // ordering is by start pos, then len
  if (start == other.start)
    return len < other.len;
  else
    return start < other.start;
}
// end light_substring_t =====


// substring_t ===============
substring_t::substring_t (unsigned _len, unsigned _start, unsigned _freq)
  : start(_start), len(_len), freq(_freq), _cost(0) {}

substring_t::substring_t (const substring_t &other)
  : start(other.start), len(other.len), freq(other.freq), _cost(0) {}

const_tokiter_t substring_t::begin(const charstring_pool_t &chPool) const {
  return chPool.getTokenIter(start);
}

const_tokiter_t substring_t::end(const charstring_pool_t &chPool) const {
  return begin(chPool) + len;
}

std::string substring_t::toString(const charstring_pool_t &chPool) {
  std::ostringstream os;
  os << "[";
  auto it = begin(chPool);
  for (; it != end(chPool) - 1; ++it) {
    os << *it << ", ";
  }
  ++it;
  os << *it << "]";
  return os.str();
}

uint16_t substring_t::cost(const charstring_pool_t &chPool) {
  if (_cost != 0) {
    return _cost;
  }
  else {
    // call other cost
    int sum = doCost(chPool);
    _cost = sum;
    return _cost;
  }
}

uint16_t substring_t::cost(const charstring_pool_t &chPool) const {
  if (_cost != 0) {
    return _cost;
  }
  else {
    return doCost(chPool);
  }
}

uint16_t substring_t::doCost(const charstring_pool_t &chPool) const {
  int sum = 0;
  for (auto it = begin(chPool); it != end(chPool); ++it) {
    sum += it->size();
  }
  return sum;
}

int substring_t::subrSaving(const charstring_pool_t &chPool) { // XXX needs use_usages and true_cost, (and call_cost and subr_overhead params)
  return doSubrSaving(cost(chPool));
}

int substring_t::subrSaving(const charstring_pool_t &chPool) const { // XXX needs use_usages and true_cost, (and call_cost and subr_overhead params)
  return doSubrSaving(cost(chPool));
}

int substring_t::doSubrSaving(int subCost) const {
  int amt = freq;
  int callCost = 5;
  int subrOverhead = 3;

  return   subCost * amt
         - subCost
         - callCost * amt
         - subrOverhead;
}

substring_t& substring_t::operator=(const substring_t &other) {
  if (*this != other) {
    start = other.start;
    len = other.len;
    freq = other.freq;
    _cost = other._cost;
  }
  return *this;
}

bool substring_t::operator<(const substring_t &other) const {
  // ordering is by start pos, then len
  if (start == other.start)
    return len < other.len;
  else
    return start < other.start;
}

bool substring_t::operator==(const substring_t &other) const {
  return start == other.start && len == other.len;
}

bool substring_t::operator!=(const substring_t &other) const {
  return !(*this == other);
}

inline uint32_t substring_t::size() {
  return len;
}

inline uint32_t substring_t::getStart() {
  return start;
}

void substring_t::updatePrice() {
  float margCost = (float) adjCost / (freq + K);
  price = margCost * ALPHA + price * (1 - ALPHA);
}

inline void substring_t::resetFreq() {
  freq = 0;
}

inline void substring_t::incrementFreq() {
  ++freq;
}

inline uint16_t substring_t::getPrice() const {
  return price;
}

inline void substring_t::setPrice(uint16_t newPrice) {
  price = newPrice;
}
// end substring_t ============


// charstring_pool_t ==========
charstring_pool_t::charstring_pool_t (unsigned nCharstrings)
  : nextQuark(0), fdSelectTrivial(true), count(nCharstrings),
    finalized(false) {
  pool.reserve(nCharstrings);
  offset.reserve(nCharstrings + 1);
  offset.push_back(0);
}

void charstring_pool_t::subroutinize() {
  std::vector<substring_t> substrings = getSubstrings();
  std::map<light_substring_t, substring_t*> substrMap;
  
  // set up map with initial values
  for (substring_t &substr : substrings) {
    substrMap[light_substring_t(substr.getStart(), substr.size())] = &substr;
  }

  unsigned substringChunkSize = substrings.size() / NUM_THREADS + 1;
  unsigned glyphChunkSize = count / NUM_THREADS + 1;
  std::vector<encoding_list> substrEncodings;
  std::vector<encoding_list> glyphEncodings;
  std::vector<std::future<std::vector<encoding_list>>> futures;

  for (int runCount = 0; runCount < 4; ++runCount) {
    // update market
    for (substring_t &substr : substrings) {
      substr.updatePrice();
    }

    // minimize cost of substrings
    futures.clear();
    substrEncodings.clear();
    for (unsigned i = 0; i < NUM_THREADS; ++i) {
      unsigned stop = (i + 1) * substringChunkSize;
      if (stop > substrings.size())
        stop = substrings.size();

      futures.push_back(std::async(std::launch::async,
                            optimizeSubstrings,
                            std::ref(substrMap),
                            std::ref(*this),
                            i * substringChunkSize,
                            stop,
                            std::ref(substrings)));
    }
    for (auto threadIt = futures.begin(); threadIt != futures.end(); ++threadIt) {
      std::vector<encoding_list> res = threadIt->get();
      substrEncodings.insert(substrEncodings.end(), res.begin(), res.end());
    }

    // minimize cost of glyphstrings
    futures.clear();
    glyphEncodings.clear();
    for (unsigned i = 0; i < NUM_THREADS; ++i) {
      unsigned stop = (i + 1) * glyphChunkSize;
      if (stop > count)
        stop = count;

      futures.push_back(std::async(std::launch::async,
                            optimizeGlyphstrings,
                            std::ref(substrMap),
                            std::ref(*this),
                            i * glyphChunkSize,
                            stop));
    }
    for (auto threadIt = futures.begin(); threadIt != futures.end(); ++threadIt) {
      std::vector<encoding_list> res = threadIt->get();
      glyphEncodings.insert(glyphEncodings.end(), res.begin(), res.end());
    }

    // update usages
    for (substring_t &substr : substrings) {
      substr.resetFreq();
    }

    for (encoding_list &encList : substrEncodings) {
      for (encoding_item &enc : encList) {
        enc.substr->incrementFreq();
      }
    }
    for (encoding_list &encList : substrEncodings) {
      for (encoding_item &enc : encList) {
        enc.substr->incrementFreq();
      }
    }

    std::cout << "Round " << runCount + 1 << " Done!" << std::endl;

    // TODO: cutdown
  }
}

std::vector<encoding_list> optimizeSubstrings(std::map<light_substring_t, substring_t*> &substrMap,
                        charstring_pool_t &csPool,
                        unsigned start,
                        unsigned stop,
                        std::vector<substring_t> &substrings) {
  std::vector<encoding_list> result;
  for (auto it = substrings.begin() + start; it != substrings.begin() + stop; ++it) {
    auto ans = optimizeCharstring(it->begin(csPool), it->end(csPool), substrMap);
    result.push_back(ans.first);
    it->setPrice(ans.second);
  }

  return result;
}

std::vector<encoding_list> optimizeGlyphstrings(std::map<light_substring_t, substring_t*> &substrMap,
                          charstring_pool_t &csPool,
                          unsigned start,
                          unsigned stop) {
  std::vector<encoding_list> result;
  for (unsigned i = start; i < stop; ++i) {
    charstring_t cs = csPool.getCharstring(i);
    result.push_back(optimizeCharstring(cs.begin, cs.end, substrMap).first);
  }
  return result;
}

std::pair<encoding_list, uint16_t> optimizeCharstring(
      const_tokiter_t begin, const_tokiter_t end,
      std::map<light_substring_t, substring_t*> &substrMap) {
  uint16_t lenCharstring = end - begin;
  std::vector<uint16_t> results(lenCharstring + 1);
  std::vector<int> nextEncIdx(lenCharstring, -1);
  std::vector<substring_t*> nextEncSubstr(lenCharstring, NULL);

  for (int i = lenCharstring - 1; i >= 0; --i) {
    int minOption = -1;
    int minEncIdx = lenCharstring;
    substring_t* minEncSubstr = NULL;
    int curCost = 0;

    const_tokiter_t curToken = begin + i;
    for (int j = i + 1; j <= lenCharstring; ++j, ++curToken) {
      curCost += curToken->size();

      auto entryIt = substrMap.find(light_substring_t(i, j - i));
      substring_t* substr;
      uint16_t option;
      if (entryIt != substrMap.end()) {
        // TODO check to not subroutinize with yourself
        substr = entryIt->second;
        option = substr->getPrice() + results[j];
      }
      else {
        substr = NULL;
        option = curCost + results[j];
      }

      if (option < minOption || minOption == -1) {
        minOption = option;
        minEncIdx = j;
        minEncSubstr = NULL;
      }
    }

    results[i] = minOption;
    nextEncIdx[i] = minEncIdx;
    nextEncSubstr[i] = minEncSubstr;
  }

  uint16_t marketCost = results[0];
  encoding_list ans;
  int curEncIdx = 0;

  while (curEncIdx < lenCharstring) {
    uint16_t lastIdx = curEncIdx;
    substring_t* curEncSubstr = nextEncSubstr[curEncIdx];
    curEncIdx = nextEncIdx[curEncIdx];

    if (curEncSubstr != NULL) {
      encoding_item item;
      item.pos = lastIdx;
      item.substr = curEncSubstr;
      ans.push_back(item);
    }
  }

  return std::pair<encoding_list, uint16_t>(ans, marketCost);
}

std::vector<substring_t> charstring_pool_t::getSubstrings() {
  if (!finalized)
    finalize();

  std::vector<unsigned> suffixes = generateSuffixes();
  std::vector<unsigned> lcp = generateLCP(suffixes);
  std::vector<substring_t> substrings = generateSubstrings(suffixes, lcp);

  return substrings;
}

charstring_t charstring_pool_t::getCharstring(unsigned idx) {
  charstring_t cs;
  cs.begin = pool.begin() + offset[idx];
  cs.end = pool.begin() + offset[idx + 1];
  if (fdSelectTrivial)
    cs.fd = 0;
  else
    cs.fd = fdSelect[idx];
  return cs;
}

void charstring_pool_t::addRawCharstring(char* data, unsigned len) {
  assert(!finalized);

  uint32_t numHints = 0;
  uint32_t stackSize = 0;

  unsigned nToks = 0;
  for (unsigned csPos = 0; csPos < len; ++csPos) {
    unsigned char first = data[csPos];
    unsigned tokSize;
    if (first < 28 || (first >= 29 && first < 32)) {
      if (first < 12){
        // operators 0-11
        tokSize = 1;
      }
      else if (first == 12) {
        // escape (12) + addl operator code
        tokSize = 2;
      }
      else if (first < 19) {
        // operators 13-18
        if (first == 18 || first == 23) {
          // hstemhm/vstemhm
          numHints += stackSize / 2;
        }

        tokSize = 1;
      }
      else if (first < 21) {
        // hintmask/cntrmask (19/20)
        tokSize = 1 + numHints / 4 + (numHints % 4 != 0) ? 1 : 0;
      }
      else if (first < 28) {
        // operators 21-27
        tokSize = 1;
      }
      else {
        // operators 29-31
        tokSize = 1;
      }

      stackSize = 0;
    }
    else {
      stackSize += 1;

      if (first < 29) {
        // 16-bit signed
        tokSize = 3;
      }
      else if (first < 247) {
        // -107 to 107
        tokSize = 1;
      }
      else if (first < 251) {
        // +108 to +1131
        tokSize = 2;
      }
      else if (first < 255) {
        // -108 to -1131
        tokSize = 2;
      }
      else {
        // 4-byte floating point
        tokSize = 5;
      }
    }

    unsigned char rawTok[tokSize];
    rawTok[0] = first;
    memcpy(rawTok + 1, data + csPos + 1, tokSize - 1);
    csPos += (tokSize - 1);

    addRawToken(rawTok, tokSize);

    ++nToks;
  }

  offset.push_back(offset.back() + nToks);
}

void charstring_pool_t::setFDSelect(unsigned char* rawFD) {
  if (rawFD == NULL) {
    fdSelectTrivial = true;
  }
  else {
    fdSelectTrivial = false;
    for (unsigned i = 0; i < count; ++i)
      fdSelect.push_back(rawFD[i]);
  }
}

void charstring_pool_t::finalize() {
  rev.reserve(pool.size());
  int cur = 0;
  for (unsigned i = 0; i < pool.size(); ++i) {
    if (i >= offset[cur + 1])
      ++cur;
    rev.push_back(cur);
  }

  finalized = true;
}

const_tokiter_t charstring_pool_t::getTokenIter(unsigned idx) const {
  return pool.begin() + idx;
}

inline unsigned charstring_pool_t::quarkFor(unsigned char* data, unsigned len) {
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

void charstring_pool_t::addRawToken(unsigned char* data, unsigned len) {
  assert(len > 0);
  assert(len < 256);
  int_type v = generateValue(data, len);
  pool.push_back(token_t(v));
}

int_type charstring_pool_t::generateValue(unsigned char* data, unsigned len) {
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

struct charstring_pool_t::suffixSortFunctor {
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

std::vector<unsigned> charstring_pool_t::generateSuffixes() {
  assert(finalized);

  std::vector<unsigned> suffixes;
  suffixes.reserve(pool.size());

  for (unsigned i = 0; i < pool.size(); ++i)
    suffixes.push_back(i);

  std::sort(suffixes.begin(), suffixes.end(), suffixSortFunctor(pool, offset, rev));
  return suffixes;
}

std::vector<unsigned> charstring_pool_t::generateLCP(std::vector<unsigned> &suffixes) {
  assert(finalized);
  assert(suffixes.size() == pool.size());

  std::vector<uint32_t> lcp(pool.size(), 0);
  std::vector<uint32_t> rank(pool.size(), 0);

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

  return lcp;
}

std::vector<substring_t> charstring_pool_t::generateSubstrings
              (std::vector<unsigned> &suffixes, std::vector<unsigned> &lcp) {
  assert(finalized);
  assert(suffixes.size() == lcp.size());
  assert(lcp.size() == pool.size());

  std::vector<substring_t> substrings;
  std::list<std::pair<unsigned, unsigned>> startIndices;

  for (unsigned i = 0; i < suffixes.size(); ++i) {
    while (!startIndices.empty() && startIndices.back().first > lcp[i]) {
      std::pair<unsigned, unsigned> cur = startIndices.back();
      unsigned len = cur.first;
      unsigned startIdx = cur.second;
      startIndices.pop_back();

      unsigned freq = i - startIdx;
      assert(freq >= 2); // NOTE: python allows different min_freq

      substring_t subr(len, suffixes[startIdx], freq);
      if (len > 1 && subr.subrSaving(*this) > 0) { // NOTE: python allows turning this check off
        substrings.push_back(subr);
      }
    }

    if (startIndices.empty() || lcp[i] > startIndices.back().first) {
      startIndices.push_back(std::pair<unsigned, unsigned>(lcp[i], i - 1));
    }
  }

  // NOTE: python also allows sorting by length
  std::sort(substrings.begin(), substrings.end(),
    [this](const substring_t a, const substring_t b) {return a.subrSaving(*this) > b.subrSaving(*this);});

  return substrings;
}
// end charstring_pool_t ========= 



charstring_pool_t CharstringPoolFactory(std::istream &instream) {
  uint16_t count;
  unsigned char countBuffer[2];
  instream.read((char*) countBuffer, 2);
  count = (countBuffer[0] << 8) | (countBuffer[1]);
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
  instream.read((char *) &fdCount, 1);
  if (fdCount > 1) {
    unsigned char* buf = (unsigned char*) malloc(count);
    instream.read((char*) buf, count);
    csPool.setFDSelect(buf);
    free(buf);
    std::cout << "loaded FDSelect" << std::endl;
  }
  else {
    csPool.setFDSelect(NULL);
    std::cout << "no FDSelect loaded" << std::endl;
  }

  csPool.finalize();

  return csPool;
}


int main(int argc, const char* argv[]) {
  charstring_pool_t csPool = CharstringPoolFactory(std::cin);

  csPool.subroutinize();
  std::cout << "finished subroutinize()" << std::endl;

  return 0;
}