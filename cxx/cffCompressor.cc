#include "cffCompressor.h"

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
// end token_t =========


// substring_t =========
substring_t::substring_t (charstring_pool_t &_chPool, unsigned _len, unsigned _start, unsigned _freq)
  : chPool(_chPool), start(_start), len(_len), freq(_freq), _cost(-1) {}

substring_t::substring_t (const substring_t &other)
  : chPool(other.chPool), start(other.start), len(other.len), freq(other.freq),
    _cost(-1) {}

const_tokiter_t substring_t::begin() const {
  return chPool.getTokenIter(start);
}

const_tokiter_t substring_t::end() const {
  return begin() + len;
}

std::string substring_t::toString() {
  std::ostringstream os;
  os << "[";
  auto it = begin();
  for (; it != end() - 1; ++it) {
    os << *it << ", ";
  }
  ++it;
  os << *it << "]";
  return os.str();
}

int substring_t::cost() {
  if (_cost != -1) {
    return _cost;
  }
  else {
    // call other cost
    int sum = ((const substring_t) *this).cost();
    _cost = sum;
    return _cost;
  }
}

int substring_t::cost() const {
  if (_cost != -1) {
    return _cost;
  }
  else {
    int sum = 0;
    const_tokiter_t it = chPool.getTokenIter(start);
    const_tokiter_t end = it + len;
    for (; it != end; ++it) {
      sum += (*it).size();
    }
    return sum;
  }
}

int substring_t::subrSaving() { // XXX needs use_usages and true_cost, (and call_cost and subr_overhead params)
  return doSubrSaving(cost());
}

int substring_t::subrSaving() const { // XXX needs use_usages and true_cost, (and call_cost and subr_overhead params)
  return doSubrSaving(cost());
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
    assert(&chPool == &other.chPool);
    start = other.start;
    len = other.len;
    freq = other.freq;
    _cost = other._cost;
    left = other.left;
    right = other.right;
  }
  return *this;
}

bool substring_t::operator==(const substring_t &other) const {
  return &chPool == &other.chPool && start == other.start && len == other.len;
}

bool substring_t::operator!=(const substring_t &other) const {
  return !(*this == other);
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

void charstring_pool_t::getSubstrings() {
  if (!finalized)
    finalize();

  std::vector<unsigned> suffixes = generateSuffixes();
  std::vector<unsigned> lcp = generateLCP(suffixes);
  std::vector<substring_t> substrings = generateSubstrings(suffixes, lcp);
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

void charstring_pool_t::setFDSelect(unsigned char* rawFD) {
  if (rawFD == NULL) {
    fdSelectTrivial = true;
  }
  else {
    fdSelectTrivial = false;
    std::vector<unsigned char> fdSelect(rawFD, rawFD + count);
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

  std::vector<unsigned> lcp(pool.size(), 0);
  unsigned rank[pool.size()];

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

      uint32_t curLen;
      if (startIndices.empty())
        curLen = 0;
      else
        curLen = startIndices.back().first + 1;

      for (; curLen <= len; ++curLen) {
        substring_t subr(*this, curLen, suffixes[startIdx], freq);
        if (curLen > 1) {
          subr.setLeft(&substrings.back());
        // if (subr.subrSaving() > 0) // NOTE: python allows turning this check off
          substrings.push_back(subr);
        }
      }
    }

    if (startIndices.empty() || lcp[i] > startIndices.back().first) {
      startIndices.push_back(std::pair<unsigned, unsigned>(lcp[i], i - 1));
    }
  }

  // NOTE: python also allows sorting by length
  std::sort(substrings.begin(), substrings.end(), 
    [](const substring_t a, const substring_t b) {return a.subrSaving() > b.subrSaving();});

  std::cout << substrings.size() << std::endl;
  return substrings;
}
// end charstring_pool_t ========= 



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