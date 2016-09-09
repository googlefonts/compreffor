/*
 * Copyright 2015 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cffCompressor.h"

// needed for Windows's "_setmode" to enable binary mode for stdin/stdout
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

const unsigned int_size = sizeof(int_type);
const float K = 0.1;
const float ALPHA = 0.1;
const unsigned hardware_threads = std::thread::hardware_concurrency();
const unsigned NUM_THREADS = hardware_threads ? hardware_threads : 1;
const unsigned DEFAULT_NUM_ROUNDS = 4;

// token_t ============
token_t::token_t(int_type value_) : value(value_) {}
token_t::token_t(const token_t &other) : value(other.value) {}

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
  /// compares actual tokens

  // optimization if they are literally pointing to the same thing
  if (begin == other.begin && end == other.end)
    return false;  // they are equal

  unsigned thisLen = end - begin;
  unsigned otherLen = other.end - other.begin;

  if (thisLen < otherLen) {
    auto p = std::mismatch(begin, end, other.begin);
    if (p.first == end)
      return true;
    else
      return *p.first < *p.second;
  } else {  // thisLen >= otherLen
    auto p = std::mismatch(other.begin, other.end, begin);
    if (p.first == other.end)
      return false;
    else
      return *p.second < *p.first;
  }
}

light_substring_t::light_substring_t(uint32_t start, uint32_t len,
                                            charstring_pool_t* pool) {
  begin = pool->get(start);
  end = begin + len;
}
// end light_substring_t =====


// substring_t ===============
substring_t::substring_t(unsigned _len, unsigned _start, unsigned _freq)
  :  pos(0), flatten(true), start(_start), len(_len), freq(_freq), _cost(0) {}

substring_t::substring_t(const substring_t &other)
  :  pos(0), flatten(other.flatten), start(other.start), len(other.len),
    freq(other.freq), _cost(0) {}

const_tokiter_t substring_t::begin(const charstring_pool_t &chPool) const {
  return chPool.get(start);
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
  } else {
    // call other cost
    int sum = doCost(chPool);
    _cost = sum;
    return _cost;
  }
}

uint16_t substring_t::cost(const charstring_pool_t &chPool) const {
  if (_cost != 0) {
    return _cost;
  } else {
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

int substring_t::subrSaving(const charstring_pool_t &chPool) {
  // XXX needs use_usages and true_cost, (and call_cost and subr_overhead params)
  return doSubrSaving(cost(chPool));
}

int substring_t::subrSaving(const charstring_pool_t &chPool) const {
  // XXX needs use_usages and true_cost, (and call_cost and subr_overhead params)
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

std::vector<unsigned char> substring_t::getTranslatedValue(
                                    const charstring_pool_t& chPool) const {
  std::vector<unsigned char> ans;

  for (auto it = begin(chPool); it != end(chPool); ++it) {
    std::vector<unsigned char> transTok = chPool.translateToken(*it);
    ans.insert(ans.end(), transTok.begin(), transTok.end());
  }

  return ans;
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

inline uint32_t substring_t::size() const {
  return len;
}

inline uint32_t substring_t::getStart() const {
  return start;
}

inline void substring_t::setAdjCost(float val) {
  assert(val > 0);
  adjCost = val;
}

inline void substring_t::syncPrice() {
  price = adjCost;
}

void substring_t::updatePrice() {
  float margCost = static_cast<float>(adjCost) / (freq + K);
  price = margCost * ALPHA + price * (1 - ALPHA);
}

inline uint32_t substring_t::getFreq() const {
  return freq;
}

inline void substring_t::resetFreq() {
  freq = 0;
}

inline void substring_t::incrementFreq() {
  ++freq;
}

inline void substring_t::increaseFreq(unsigned amt) {
  freq += amt;
}

inline void substring_t::decrementFreq() {
  assert(freq != 0);
  --freq;
}

inline float substring_t::getPrice() const {
  return price;
}

inline void substring_t::setPrice(float newPrice) {
  price = newPrice;
}
// end substring_t ============


// charstring_pool_t ==========
charstring_pool_t::charstring_pool_t(unsigned nCharstrings)
  : nextQuark(0), fdSelectTrivial(true), count(nCharstrings),
    finalized(false), numRounds(DEFAULT_NUM_ROUNDS) {
  pool.reserve(nCharstrings);
  offset.reserve(nCharstrings + 1);
  offset.push_back(0);
}

charstring_pool_t::charstring_pool_t(unsigned nCharstrings, int _nrounds)
  : nextQuark(0), fdSelectTrivial(true), count(nCharstrings),
    finalized(false), numRounds(_nrounds) {
  pool.reserve(nCharstrings);
  offset.reserve(nCharstrings + 1);
  offset.push_back(0);
}

void charstring_pool_t::writeEncoding(
                              const encoding_list& enc,
                              const std::map<const substring_t*, uint32_t>& index,
                              std::ostream& outFile) {
  // write the number of subrs called
  assert(enc.size() < 128);
  outFile.put(enc.size());
  // write each call
  for (const encoding_item& enc_item : enc) {
    outFile.write(
              reinterpret_cast<const char*>(&enc_item.pos),
              sizeof(enc_item.pos));  // 4 bytes
    auto it = index.find(enc_item.substr);
    assert(it != index.end());
    uint32_t subrIndex = it->second;
    outFile.write(reinterpret_cast<const char*>(&subrIndex), 4);
  }
}

void charstring_pool_t::writeSubrs(
              std::list<substring_t>& subrs,
              std::vector<encoding_list>& glyphEncodings,
              std::ostream& outFile) {
  /// write subrs
  // write number of subrs
  uint32_t numSubrs = (uint32_t) subrs.size();
  outFile.write(reinterpret_cast<const char*>(&numSubrs), 4);

  // number subrs
  std::map<const substring_t*, uint32_t> index;

  // write each subr's representative glyph and offset in that charstring
  uint32_t curIndex = 0;
  for (const substring_t& subr : subrs) {
    index[&subr] = curIndex++;
    uint32_t glyphIdx = rev[subr.getStart()];
    uint32_t glyphOffset = subr.getStart() - offset[glyphIdx];
    uint32_t subrLength = subr.size();
    outFile.write(reinterpret_cast<const char*>(&glyphIdx), 4);
    outFile.write(reinterpret_cast<const char*>(&glyphOffset), 4);
    outFile.write(reinterpret_cast<const char*>(&subrLength), 4);
  }

  // after producing `index`, write subr encodings
  for (const substring_t& subr : subrs) {
    writeEncoding(subr.encoding, index, outFile);
  }

  /// write glyph encoding instructions
  for (const encoding_list& glyphEnc : glyphEncodings) {
    writeEncoding(glyphEnc, index, outFile);
  }
}

unsigned charstring_pool_t::packEncoding(
                            const encoding_list& enc,
                            const std::map<const substring_t*, uint32_t>& index,
                            uint32_t* buffer) {
  unsigned pos = 0;

  // write the number of subrs called
  buffer[pos++] = enc.size();
  // write each call
  for (const encoding_item& enc_item : enc) {
    buffer[pos++] = enc_item.pos;
    auto it = index.find(enc_item.substr);
    assert(it != index.end());
    uint32_t subrIndex = it->second;
    buffer[pos++] = subrIndex;
  }

  return pos;
}

uint32_t* charstring_pool_t::getResponse(
              std::list<substring_t>& subrs,
              std::vector<encoding_list>& glyphEncodings,
              unsigned& outputLength) {
  unsigned length = 1 + subrs.size() * 3;
  for (const substring_t& subr : subrs) {
    length += 1 + subr.encoding.size() * 2;
  }
  for (const encoding_list& glyphEnc : glyphEncodings) {
    length += 1 + glyphEnc.size() * 2;
  }
  outputLength = length;

  uint32_t* buffer = new uint32_t[length];
  unsigned pos = 0;

  /// write subrs
  // write number of subrs
  uint32_t numSubrs = (uint32_t) subrs.size();
  buffer[pos++] = numSubrs;

  // number subrs
  std::map<const substring_t*, uint32_t> index;

  // write each subr's representative glyph and offset in that charstring
  uint32_t curIndex = 0;
  for (const substring_t& subr : subrs) {
    index[&subr] = curIndex++;
    uint32_t glyphIdx = rev[subr.getStart()];
    uint32_t glyphOffset = subr.getStart() - offset[glyphIdx];
    uint32_t subrLength = subr.size();
    buffer[pos++] = glyphIdx;
    buffer[pos++] = glyphOffset;
    buffer[pos++] = subrLength;
  }

  // after producing `index`, write subr encodings
  for (const substring_t& subr : subrs) {
    pos += packEncoding(subr.encoding, index, buffer + pos);
  }

  /// write glyph encoding instructions
  for (const encoding_list& glyphEnc : glyphEncodings) {
    pos += packEncoding(glyphEnc, index, buffer + pos);
  }

  return buffer;
}

std::vector<unsigned char> charstring_pool_t::formatInt(int num) {
  std::vector<unsigned char> ret;
  if (num >= -107 && num <= 107) {
    ret.push_back((unsigned char) num + 139);
  } else if (num >= 108 && num <= 1131) {
    unsigned char first = (num - 108) / 256;
    unsigned char second = num - 108 - first * 256;
    assert((static_cast<int>(first)) * 256 + static_cast<int>(second) + 108);
    ret.push_back(first + 247);
    ret.push_back(second);
  } else if (num >= -1131 && num <= -108) {
    unsigned char first = (num + 108) / 256;
    unsigned char second = -num - 108 - first * 256;
    assert(-(static_cast<int>(first)) * 256 - static_cast<int>(second) - 108);
    ret.push_back(first + 251);
    ret.push_back(second);
  } else {
    assert(num >= -32768 && num <= 32767);

    ret.push_back((unsigned char) 28);
    ret.push_back((unsigned char) ((num & 0xff00) >> 8));
    ret.push_back((unsigned char) (num & 0xff));
  }
  return ret;
}

void charstring_pool_t::subroutinize(
              std::list<substring_t>& substrings,
              std::vector<encoding_list>& glyphEncodings) {  // TODO: testMode
  std::map<light_substring_t, substring_t*> substrMap;

  /// set up map with initial values
  for (substring_t &substr : substrings) {
    substr.setAdjCost(substr.cost(*this));
    substr.syncPrice();
    light_substring_t key(substr.begin(*this), substr.end(*this));
    substrMap[key] = &substr;
  }

  unsigned substringChunkSize = substrings.size() / NUM_THREADS + 1;
  unsigned glyphChunkSize = count / NUM_THREADS + 1;
  std::vector<std::thread> threads;
  std::vector<std::vector<encoding_list> > results((count+glyphChunkSize-1)/glyphChunkSize);

  for (int runCount = 0; runCount < numRounds; ++runCount) {
    /// update market
    for (substring_t& substr : substrings) {
      substr.updatePrice();
    }

    /// minimize cost of substrings
    // XXX consider redoing substringChunkSize
    threads.clear();
    auto curSubstr = substrings.begin();
    for (unsigned i = 0; i < NUM_THREADS; ++i) {
      if (i * substringChunkSize >= substrings.size())
        break;

      unsigned step = substringChunkSize;
      if ((i + 1) * substringChunkSize > substrings.size())
        step = substrings.size() - i * substringChunkSize;

      auto start = curSubstr;
      std::advance(curSubstr, step);

      threads.push_back(std::thread(optimizeSubstrings,
                            std::ref(substrMap),
                            std::ref(*this),
                            start,
                            curSubstr));
    }
    for (auto threadIt = threads.begin(); threadIt != threads.end(); ++threadIt) {
      threadIt->join();
    }

    // minimize cost of glyphstrings
    threads.clear();
    glyphEncodings.clear();
    for (unsigned i = 0; i < NUM_THREADS; ++i) {
      if (i * glyphChunkSize >= count)
        break;

      unsigned stop = (i + 1) * glyphChunkSize;
      if (stop > count)
        stop = count;

      results[i].clear();
      threads.push_back(std::thread(optimizeGlyphstrings,
                            std::ref(substrMap),
                            std::ref(*this),
                            i * glyphChunkSize,
                            stop,
                            std::ref(results[i])));
    }
    for (auto threadIt = threads.begin(); threadIt != threads.end(); ++threadIt) {
      threadIt->join();
    }

    for (std::vector<encoding_list> &res : results) {
      glyphEncodings.insert(glyphEncodings.end(), res.begin(), res.end());
    }

    // update usages
    for (substring_t& substr : substrings) {
      substr.resetFreq();
    }

    for (substring_t& substr : substrings) {
      for (encoding_item& enc : substr.encoding) {
        enc.substr->incrementFreq();
      }
    }
    for (encoding_list& encList : glyphEncodings) {
      for (encoding_item& enc : encList) {
        enc.substr->incrementFreq();
      }
    }

    /// cutdown
    if (runCount <= numRounds - 2) {  // NOTE: python checks for testMode
      auto substrIt = substrings.begin();
      for (; substrIt != substrings.end();) {
        if (substrIt->subrSaving(*this) <= 0) {
          light_substring_t key(substrIt->begin(*this), substrIt->end(*this));
          size_t response = substrMap.erase(key);
          // heuristic:
          for (encoding_list::iterator encItem = substrIt->encoding.begin();
                  encItem != substrIt->encoding.end(); ++encItem) {
            encItem->substr->increaseFreq(substrIt->getFreq() - 1);
          }

          substrIt = substrings.erase(substrIt);
        } else {
          ++substrIt;
        }
      }
    }
  }
}

void optimizeSubstrings(std::map<light_substring_t, substring_t*> &substrMap,
                        charstring_pool_t &csPool,
                        std::list<substring_t>::iterator begin,
                        std::list<substring_t>::iterator end) {
  for (auto it = begin; it != end; ++it) {
    auto ans = optimizeCharstring(
                    it->begin(csPool),
                    it->size(),
                    substrMap,
                    csPool,
                    true);
    it->encoding = ans.first;
    it->setAdjCost(ans.second);
  }
}

void optimizeGlyphstrings(
                          std::map<light_substring_t, substring_t*> &substrMap,
                          charstring_pool_t &csPool,
                          unsigned start,
                          unsigned stop,
                          std::vector<encoding_list>& result) {
  for (unsigned i = start; i < stop; ++i) {
    charstring_t cs = csPool.getCharstring(i);
    result.push_back(optimizeCharstring(
                              cs.begin,
                              cs.len,
                              substrMap,
                              csPool,
                              false)
                        .first);
  }
}

std::pair<encoding_list, float> optimizeCharstring(
      const_tokiter_t begin, uint32_t len,
      std::map<light_substring_t, substring_t*> &substrMap,
      charstring_pool_t& csPool, bool isSubstring) {
  std::vector<float> results(len + 1);
  std::vector<int> nextEncIdx(len, -1);
  std::vector<substring_t*> nextEncSubstr(len, NULL);

  for (int i = len - 1; i >= 0; --i) {
    float minOption = -1;
    int minEncIdx = len;
    substring_t* minEncSubstr = NULL;
    int curCost = 0;

    const_tokiter_t curToken = begin + i;
    for (unsigned j = i + 1; j <= len; ++j, ++curToken) {
      curCost += curToken->size();

      light_substring_t key(begin + i, begin + j);
      auto entryIt = substrMap.find(key);
      substring_t* substr;
      float option;
      if (!(i == 0 && j == len) && entryIt != substrMap.end()) {
        // TODO: check to not subroutinize with yourself
        substr = entryIt->second;
        option = substr->getPrice() + results[j];
      } else {
        substr = NULL;
        option = curCost + results[j];
      }

      if (option < minOption || minOption == -1) {
        minOption = option;
        minEncIdx = j;
        minEncSubstr = substr;
      }
    }

    results[i] = minOption;
    nextEncIdx[i] = minEncIdx;
    nextEncSubstr[i] = minEncSubstr;
  }

  encoding_list ans;
  unsigned curEncIdx = 0;

  while (curEncIdx < len) {
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

  return std::pair<encoding_list, float>(ans, results[0]);
}

std::list<substring_t> charstring_pool_t::getSubstrings() {
  if (!finalized)
    finalize();

  std::vector<unsigned> suffixes = generateSuffixes();
  std::vector<unsigned> lcp = generateLCP(suffixes);
  std::list<substring_t> substrings = generateSubstrings(suffixes, lcp);

  return substrings;
}

charstring_t charstring_pool_t::getCharstring(unsigned idx) {
  charstring_t cs;
  cs.begin = pool.begin() + offset[idx];
  cs.len = offset[idx + 1] - offset[idx];
  if (fdSelectTrivial)
    cs.fd = 0;
  else
    cs.fd = fdSelect[idx];
  return cs;
}

void charstring_pool_t::addRawCharstring(unsigned char* data, unsigned len) {
  assert(!finalized);

  uint32_t numHints = 0;
  uint32_t stackSize = 0;

  unsigned nToks = 0;
  for (unsigned csPos = 0; csPos < len; ++csPos) {
    unsigned char first = data[csPos];
    unsigned tokSize;
    if (first < 28 || (first >= 29 && first < 32)) {
      if (first < 12) {
        // operators 1-11
        if (first == 1 || first == 3) {
          // hstem/vstem
          numHints += stackSize / 2;
        }
        tokSize = 1;
      } else if (first == 12) {
        // escape (12) + addl operator code
        tokSize = 2;
      } else if (first < 19) {
        // operators 13-18
        if (first == 18) {
          // hstemhm
          numHints += stackSize / 2;
        }
        tokSize = 1;
      } else if (first < 21) {
        // hintmask/cntrmask (19/20)
        if (stackSize != 0) {
          // account for additonal vhints on stack (assuming legal program)
          numHints += stackSize / 2;
        }
        tokSize = 1 + numHints / 8 + ((numHints % 8 != 0) ? 1 : 0);
      } else if (first < 28) {
        // operators 21-27
        if (first == 23) {
          // vstemhm
          numHints += stackSize / 2;
        }
        tokSize = 1;
      } else {
        // operators 29-31
        tokSize = 1;
      }

      stackSize = 0;
    } else {
      stackSize += 1;

      if (first == 28) {
        // 16-bit signed
        tokSize = 3;
      } else if (first < 247) {
        // -107 to 107
        tokSize = 1;
      } else if (first < 251) {
        // +108 to +1131
        tokSize = 2;
      } else if (first < 255) {
        // -108 to -1131
        tokSize = 2;
      } else {
        // 4-byte floating point
        tokSize = 5;
      }
    }

    unsigned char* rawTok = new unsigned char[tokSize];
    rawTok[0] = first;
    memcpy(rawTok + 1, data + csPos + 1, tokSize - 1);
    csPos += (tokSize - 1);

    addRawToken(rawTok, tokSize);
    delete[] rawTok;

    ++nToks;
  }

  offset.push_back(offset.back() + nToks);
}

void charstring_pool_t::setFDSelect(uint8_t* rawFD) {
  if (rawFD == NULL) {
    fdSelectTrivial = true;
  } else {
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

const_tokiter_t charstring_pool_t::get(unsigned idx) const {
  const_tokiter_t x = pool.begin() + idx;
  return x;
}

inline uint16_t charstring_pool_t::quarkFor(unsigned char* data, unsigned len) {
  // TODO: verify using a string key isn't a time problem
  std::string key((const char*) data, (size_t) len);
  auto it = quarkMap.find(key);
  if (it == quarkMap.end()) {
    assert(nextQuark < 65536);
    assert(revQuark.size() == nextQuark);
    unsigned q = nextQuark++;
    quarkMap[key] = q;
    revQuark.push_back(key);
    return (uint16_t) q;
  } else {
    return (uint16_t) it->second;
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
  } else {
    uint16_t q = quarkFor(data, len);
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
    auto aFirst = pool.begin() + a;
    auto bFirst = pool.begin() + b;

    if (aLen < bLen) {
      auto aLast = pool.begin() + offset[rev[a] + 1];
      auto p = std::mismatch(aFirst, aLast, bFirst);
      if (p.first == aLast)
        return true;
      else
        return *p.first < *p.second;
    } else {  // aLen >= bLen
      auto bLast = pool.begin() + offset[rev[b] + 1];
      auto p = std::mismatch(bFirst, bLast, aFirst);
      if (p.first == bLast)
        return false;
      else
        return *p.second < *p.first;
    }
  }
};

std::vector<unsigned> charstring_pool_t::generateSuffixes() {
  assert(finalized);

  std::vector<unsigned> suffixes;
  suffixes.reserve(pool.size());

  for (unsigned i = 0; i < pool.size(); ++i)
    suffixes.push_back(i);

  std::stable_sort(
              suffixes.begin(),
              suffixes.end(),
              suffixSortFunctor(pool, offset, rev));

  return suffixes;
}

std::vector<unsigned> charstring_pool_t::generateLCP(
                              const std::vector<unsigned> &suffixes) {
  assert(finalized);
  assert(suffixes.size() == pool.size());

  std::vector<uint32_t> lcp(pool.size(), 0);
  std::vector<uint32_t> rank(pool.size(), 0);

  for (unsigned i = 0; i < pool.size(); ++i) {
    unsigned idx = suffixes[i];
    rank[idx] = i;
  }

  for (std::vector<unsigned>::iterator ch = offset.begin();
          ch != offset.end() - 1; ++ch) {
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

bool charstring_pool_t::verify_lcp(
                  std::vector<unsigned>& lcp,
                  std::vector<unsigned>& suffixes) {
  for (unsigned i = 1; i < pool.size(); ++i) {
    auto thisCur = pool.begin() + suffixes[i];
    auto befCur = pool.begin() + suffixes[i - 1];
    auto thisEnd = pool.begin() + offset[rev[suffixes[i]] + 1];
    auto befEnd = pool.begin() + offset[rev[suffixes[i - 1]] + 1];
    for (unsigned j = 0; j < lcp[i]; ++j) {
      assert(*thisCur == *befCur);
      ++thisCur;
      ++befCur;
    }
    assert(*thisCur != *befCur || thisCur == thisEnd || befCur == befEnd);
  }

  return true;
}

std::list<substring_t> charstring_pool_t::generateSubstrings(
                              std::vector<unsigned> &suffixes,
                              std::vector<unsigned> &lcp) {
  assert(finalized);
  assert(suffixes.size() == lcp.size());
  assert(lcp.size() == pool.size());

  std::list<substring_t> substrings;
  std::list<std::pair<unsigned, unsigned>> startIndices;

  for (unsigned i = 0; i < suffixes.size(); ++i) {
    while (!startIndices.empty() && startIndices.back().first > lcp[i]) {
      unsigned len = startIndices.back().first;
      unsigned startIdx = startIndices.back().second;
      startIndices.pop_back();

      unsigned freq = i - startIdx;
      assert(freq >= 2);  // NOTE: python allows different min_freq

      substring_t subr(len, suffixes[startIdx], freq);
      // NOTE: python allows turning this check off --
      if (len > 1 && subr.subrSaving(*this) > 0) {
        substrings.push_back(subr);
      }
    }

    if (startIndices.empty() || lcp[i] > startIndices.back().first) {
      startIndices.push_back(std::make_pair(lcp[i], i - 1));
    }
  }

  // NOTE: python sorts by length or saving

  return substrings;
}

std::vector<unsigned char> charstring_pool_t::translateToken(const token_t& tok) const {
  size_t tokLen = tok.size();

  if (tokLen < int_size) {
    std::vector<unsigned char> ans;
    for (unsigned i = 0; i < tokLen; ++i)
      ans.push_back(tok.part(i + 1));
    return ans;
  } else {
    uint16_t q = (tok.part(2) << 8) + tok.part(3);
    std::string orig = revQuark.at(q);
    std::vector<unsigned char> ans(orig.begin(), orig.end());
    return ans;
  }
}
// end charstring_pool_t =========



charstring_pool_t CharstringPoolFactory(
                          std::istream &instream,
                          int numRounds) {
  uint16_t count;
  unsigned char countBuffer[2];
  instream.read(reinterpret_cast<char*>(countBuffer), 2);
  count = (countBuffer[0] << 8) | (countBuffer[1]);

  unsigned char offSize;
  instream.read(reinterpret_cast<char*>(&offSize), 1);

  uint32_t* offset = new uint32_t[count + 1];

  unsigned char* offsetBuffer = new unsigned char[(count + 1) * offSize];
  instream.read(reinterpret_cast<char*>(offsetBuffer), (count + 1) * offSize);
  for (int i = 0; i < count + 1; ++i) {
    offset[i] = 0;
    for (int j = 0; j < offSize; ++j) {
      offset[i] += offsetBuffer[i * offSize + j] << ((offSize - j - 1) * 8);
    }
    offset[i] -= 1;  // CFF is 1-indexed(-ish)
  }
  delete[] offsetBuffer;
  assert(offset[0] == 0);

  charstring_pool_t csPool(count, numRounds);

  unsigned len;
  for (int i = 0; i < count; ++i) {
    unsigned len = offset[i + 1] - offset[i];
    char* data = new char[len];
    instream.read(data, len);
    csPool.addRawCharstring(reinterpret_cast<unsigned char*>(data), len);
    delete[] data;
  }

  unsigned char fdCount;
  instream.read(reinterpret_cast<char*>(&fdCount), 1);
  if (fdCount > 1) {
    uint8_t* buf = new uint8_t[count];
    instream.read(reinterpret_cast<char*>(buf), count);
    csPool.setFDSelect(buf);
    delete[] buf;
  } else {
    csPool.setFDSelect(NULL);
  }

  delete[] offset;
  csPool.finalize();

  return csPool;
}

void charstring_pool_t::printSuffix(unsigned idx, bool printVal) {
  std::cerr << "[";
  auto start = pool.begin() + idx;
  auto end = pool.begin() + offset[rev[idx] + 1];
  for (auto it = start; it != end; ++it) {
    if (printVal)
      std::cerr << it->getValue();
    else
      std::cerr << *it;

    if (it + 1 != end)
      std::cerr << ", ";
  }
  std::cerr << "]" << std::endl;
}

charstring_pool_t CharstringPoolFactoryFromString(
                          unsigned char* buffer,
                          int numRounds) {
  unsigned pos = 0;

  uint16_t count;
  count = (buffer[pos] << 8) | (buffer[pos + 1]);
  pos += 2;

  unsigned char offSize = buffer[pos++];

  uint32_t* offset = new uint32_t[count + 1];

  unsigned char* offsetBuffer = &buffer[pos];
  pos += (count + 1) * offSize;
  for (int i = 0; i < count + 1; ++i) {
    offset[i] = 0;
    for (int j = 0; j < offSize; ++j) {
      offset[i] += offsetBuffer[i * offSize + j] << ((offSize - j - 1) * 8);
    }
    offset[i] -= 1;  // CFF is 1-indexed(-ish)
  }
  assert(offset[0] == 0);

  charstring_pool_t csPool(count, numRounds);

  unsigned len;
  for (int i = 0; i < count; ++i) {
    unsigned len = offset[i + 1] - offset[i];
    csPool.addRawCharstring(buffer + pos, len);
    pos += len;
  }

  unsigned char fdCount = buffer[pos++];
  if (fdCount > 1) {
    csPool.setFDSelect(buffer + pos);
    pos += count;
  } else {
    csPool.setFDSelect(NULL);
  }

  delete[] offset;
  csPool.finalize();

  return csPool;
}

extern "C" uint32_t* compreff(unsigned char* dataStream, int numRounds, unsigned& outputLength) {
  charstring_pool_t csPool = CharstringPoolFactoryFromString(dataStream,
                                                             numRounds);
  std::list<substring_t> subrs = csPool.getSubstrings();
  std::vector<encoding_list> glyphEncodings;
  csPool.subroutinize(subrs, glyphEncodings);
  return csPool.getResponse(subrs, glyphEncodings, outputLength);
}

extern "C" void unload(uint32_t* response) {
  free(response);
}

int main(int argc, const char* argv[]) {
  int numRounds = DEFAULT_NUM_ROUNDS;

  unsigned argIdx = 1;
  while (argIdx < static_cast<unsigned>(argc)) {
    if (strcmp(argv[argIdx], "--nrounds") == 0) {
      numRounds = atoi(argv[argIdx + 1]);
      argIdx += 2;
    } else {
      std::cerr << "Unrecognized argument: " << argv[argIdx] << std::endl;
      return 1;
    }
  }

#ifdef _WIN32
  if (_setmode(_fileno(stdin), _O_BINARY) == -1) {
    std::cerr << "Cannot set stdin to binary mode" << std::endl;
    return 1;
  }
  if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
    std::cerr << "Cannot set stdout to binary mode" << std::endl;
    return 1;
  }
#endif

  charstring_pool_t csPool = CharstringPoolFactory(
                                      std::cin,
                                      numRounds);

  std::list<substring_t> subrs = csPool.getSubstrings();
  std::vector<encoding_list> glyphEncodings;
  csPool.subroutinize(subrs, glyphEncodings);

  csPool.writeSubrs(subrs, glyphEncodings, std::cout);

  return 0;
}
