#ifndef CFFCOMPRESSOR_H_
#define CFFCOMPRESSOR_H_

#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <algorithm>
#include <map>
#include <iostream>
#include <fstream>
#include <future>
#include <vector>
#include <list>
#include <string>
#include <sstream>
#include <stdexcept>
#include <thread>

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
  token_t (int_type value_ = 0);
  token_t (const token_t &other);
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
  tokiter_t end;
  unsigned char fd;
} charstring_t;

class light_substring_t {
public:
  light_substring_t (const_tokiter_t _begin, const_tokiter_t _end)
        : begin(_begin), end(_end) {}
  light_substring_t (uint32_t start, uint32_t len, charstring_pool_t* pool);
  light_substring_t& operator=(const light_substring_t &other) {
    begin = other.begin;
    end = other.end;
    return *this;
  };
  bool operator<(const light_substring_t &other) const;

  const_tokiter_t begin;
  const_tokiter_t end;
};

class substring_t {
public:
  substring_t (unsigned _len, unsigned _start, unsigned _freq);
  substring_t (const substring_t &other);
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
  inline uint32_t size();
  inline uint32_t getStart();
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

typedef struct encoding_item {
  uint16_t pos;
  substring_t* substr;
} encoding_item;

typedef std::vector<encoding_item> encoding_list;

typedef struct subr_set {
  std::vector<encoding_list> glyphEncodings;
  std::vector<substring_t> gsubrs;
  std::vector<encoding_list> gsubrEncodings; 
} subr_set;

std::vector<encoding_list> optimizeSubstrings(std::map<light_substring_t, substring_t*> &substrMap,
                        charstring_pool_t &csPool,
                        unsigned start,
                        unsigned stop,
                        std::vector<substring_t> &substrings);
std::vector<encoding_list> optimizeGlyphstrings(std::map<light_substring_t, substring_t*> &substrMap,
                        charstring_pool_t &csPool,
                        unsigned start,
                        unsigned stop);
std::pair<encoding_list, float> optimizeCharstring(const_tokiter_t begin, const_tokiter_t end,
                        std::map<light_substring_t, substring_t*> &substrMap,
                        charstring_pool_t& csPool);

class charstring_pool_t {
public:
  charstring_pool_t (unsigned nCharstrings);
  subr_set subroutinize();
  charstring_t getCharstring(unsigned idx);
  void addRawCharstring(char* data, unsigned len);
  void setFDSelect(unsigned char* rawFD);
  void finalize();
  const_tokiter_t get(unsigned idx) const;

private:
  tokmap_t quarkMap;
  unsigned nextQuark;
  std::vector<token_t> pool;
  std::vector<unsigned> offset;
  std::vector<unsigned char> fdSelect;
  std::vector<unsigned> rev;
  bool fdSelectTrivial;
  unsigned count;
  bool finalized;

  inline unsigned quarkFor(unsigned char* data, unsigned len);
  void addRawToken(unsigned char* data, unsigned len);
  int_type generateValue(unsigned char* data, unsigned len);
  std::vector<substring_t> getSubstrings();
  std::vector<unsigned> generateSuffixes();
  struct suffixSortFunctor;
  std::vector<unsigned> generateLCP(std::vector<unsigned> &suffixes);
  std::vector<substring_t> generateSubstrings(std::vector<unsigned> &suffixes,
                                              std::vector<unsigned> &lcp);
};

charstring_pool_t CharstringPoolFactory(std::istream &instream);

#endif