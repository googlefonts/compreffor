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

#ifndef CFFCOMPRESSOR_H_
#define CFFCOMPRESSOR_H_

#include <assert.h>
#include <forward_list>
#include <stdint.h>
#include <string.h>
#include <thread>

/* If MinGW GCC is compiled with "win32" threads instead of "posix"
 * it lacks the C++11 standard threading classes, so we need to include
 * the "mingw-std-threads" header-only library from:
 *
 *     https://github.com/meganz/mingw-std-threads
 */
#if defined(__MINGW32__) && !defined(__WINPTHREADS_VERSION)
#include "mingw-std-threads/mingw.thread.h"
#endif

#include <algorithm>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <queue>
#include <utility>
#include <vector>

class token_t;
struct charstring_t;
class substring_t;
class charstring_pool_t;

typedef uint32_t int_type;
typedef std::map<std::string, unsigned> tokmap_t;
typedef std::vector<token_t>::iterator tokiter_t;
typedef std::vector<token_t>::const_iterator const_tokiter_t;

class token_t {
  public:
    explicit token_t(int_type value_ = 0);
    token_t(const token_t &other);
    inline int_type getValue() const;
    inline unsigned size() const;
    inline unsigned part(unsigned idx) const;
    std::string toString() const;
    bool operator<(const token_t &other) const;
    bool operator!=(const token_t &other) const;
    bool operator==(const token_t &other) const;

  private:
    int_type value;
};

typedef struct charstring_t {
  tokiter_t begin;
  uint32_t len;
  unsigned char fd;
} charstring_t;

class light_substring_t {
  public:
    light_substring_t(const_tokiter_t _begin, const_tokiter_t _end)
          : begin(_begin), end(_end) {}
    light_substring_t(uint32_t start, uint32_t len, charstring_pool_t* pool);
    light_substring_t& operator=(const light_substring_t &other) {
      begin = other.begin;
      end = other.end;
      return *this;
    };
    bool operator<(const light_substring_t &other) const;

    const_tokiter_t begin;
    const_tokiter_t end;
};

typedef struct encoding_item {
  uint32_t pos;
  substring_t* substr;
} encoding_item;

typedef std::vector<encoding_item> encoding_list;

class substring_t {
  public:
    substring_t(unsigned _len, unsigned _start, unsigned _freq);
    substring_t(const substring_t &other);
    const_tokiter_t begin(const charstring_pool_t &chPool) const;
    const_tokiter_t end(const charstring_pool_t &chPool) const;
    uint16_t cost(const charstring_pool_t &chPool);
    int subrSaving(const charstring_pool_t &chPool);
    uint16_t cost(const charstring_pool_t &chPool) const;
    int subrSaving(const charstring_pool_t &chPool) const;
    std::string toString(const charstring_pool_t &chPool);
    bool operator<(const substring_t &other) const;
    bool operator==(const substring_t &other) const;
    bool operator!=(const substring_t &other) const;
    substring_t& operator=(const substring_t &other);
    inline uint32_t size() const;
    inline uint32_t getStart() const;
    void updatePrice();
    uint32_t getFreq() const;
    void resetFreq();
    void incrementFreq();
    void increaseFreq(unsigned amt);
    void decrementFreq();
    float getPrice() const;
    void setPrice(float newPrice);
    void setAdjCost(float value);
    void syncPrice();
    std::vector<unsigned char> getTranslatedValue(
            const charstring_pool_t& chPool) const;

    uint16_t pos;
    bool flatten;
    encoding_list encoding;

  private:
    uint32_t start;
    uint32_t len;
    uint32_t freq;
    uint16_t _cost;
    float adjCost;
    float price;

    int doSubrSaving(int subCost) const;
    uint16_t doCost(const charstring_pool_t &chPool) const;
};

typedef std::pair<std::vector<encoding_list>, std::vector<substring_t> >
        subr_pair;

void optimizeSubstrings(
                    std::map<light_substring_t,
                    substring_t*> &substrMap,
                    charstring_pool_t &csPool,
                    std::list<substring_t>::iterator begin,
                    std::list<substring_t>::iterator end);

void optimizeGlyphstrings(
                    std::map<light_substring_t,
                    substring_t*> &substrMap,
                    charstring_pool_t &csPool,
                    unsigned start,
                    unsigned stop,
                    std::vector<encoding_list>& result);

std::pair<encoding_list, float> optimizeCharstring(
                    const_tokiter_t begin,
                    uint32_t len,
                    std::map<light_substring_t, substring_t*> &substrMap,
                    charstring_pool_t& csPool,
                    bool isSubstring);

class charstring_pool_t {
  public:
    explicit charstring_pool_t(unsigned nCharstrings);
    charstring_pool_t(unsigned nCharstrings, int numRounds);
    void writeSubrs(
                std::list<substring_t>& substrings,
                std::vector<encoding_list>& glyphEncodings,
                std::ostream& outFile);
    uint32_t* getResponse(
                std::list<substring_t>& substrings,
                std::vector<encoding_list>& glyphEncodings,
                unsigned& outputLength);
    std::vector<unsigned char> formatInt(int num);
    void subroutinize(
                std::list<substring_t>& substrings,
                std::vector<encoding_list>& glyphEncodings);
    std::list<substring_t> getSubstrings();
    charstring_t getCharstring(unsigned idx);
    void addRawCharstring(unsigned char* data, unsigned len);
    void setFDSelect(uint8_t* rawFD);
    void finalize();
    const_tokiter_t get(unsigned idx) const;
    std::vector<unsigned char> translateToken(const token_t& tok) const;

    void printSuffix(unsigned idx, bool printVal = false);
    bool verify_lcp(std::vector<unsigned>& lcp, std::vector<unsigned>& suffixes);

  private:
    tokmap_t quarkMap;
    unsigned nextQuark;
    std::vector<std::string> revQuark;
    std::vector<token_t> pool;
    std::vector<unsigned> offset;
    std::vector<uint8_t> fdSelect;
    std::vector<unsigned> rev;
    bool fdSelectTrivial;
    unsigned count;
    bool finalized;
    int numRounds;

    inline uint16_t quarkFor(unsigned char* data, unsigned len);
    void addRawToken(unsigned char* data, unsigned len);
    int_type generateValue(unsigned char* data, unsigned len);
    std::vector<unsigned> generateSuffixes();
    struct suffixSortFunctor;
    std::vector<unsigned> generateLCP(const std::vector<unsigned>& suffixes);
    std::list<substring_t> generateSubstrings(
                                        std::vector<unsigned> &suffixes,
                                        std::vector<unsigned> &lcp);
    encoding_list getUpdatedEncoding(substring_t* subr);
    void writeEncoding(
              const encoding_list& enc,
              const std::map<const substring_t*, uint32_t>& index,
              std::ostream& outFile);
    unsigned packEncoding(
              const encoding_list& enc,
              const std::map<const substring_t*, uint32_t>& index,
              uint32_t* buffer);
};

charstring_pool_t CharstringPoolFactory(
                        std::istream& instream,
                        int numRounds);

charstring_pool_t CharstringPoolFactoryFromString(
                        unsigned char* buffer,
                        int numRounds);

extern "C" uint32_t* compreff(unsigned char* dataStream, int numRounds, unsigned& outputLength);
extern "C" void unload(uint32_t* response);

#endif
