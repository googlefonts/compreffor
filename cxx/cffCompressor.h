#ifndef CFFCOMPRESSOR_H_
#define CFFCOMPRESSOR_H_

#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <algorithm>
#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <string>
#include <sstream>
#include <stdexcept>

class token_t;
struct charstring_t;
class substring_t;
class charstring_pool_t;

typedef uint32_t int_type;
const unsigned int_size = sizeof(int_type);
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

class substring_t {
public:
  substring_t (charstring_pool_t &_chPool, unsigned _len, unsigned _start, unsigned _freq);
  substring_t (const substring_t &other);
  const_tokiter_t begin() const;
  const_tokiter_t end() const;
  int cost();
  int subrSaving();
  int cost() const;
  int subrSaving() const;
  std::string toString();
  bool operator==(const substring_t &other) const;
  bool operator!=(const substring_t &other) const;
  substring_t& operator=(const substring_t &other);
  void setLeft(substring_t *subr) { left = subr; }
  void setRight(substring_t *subr) { left = subr; }

private:
  const charstring_pool_t &chPool;
  uint32_t start;
  uint32_t len;
  uint32_t freq;
  short _cost;
  substring_t *left;
  substring_t *right;

  int doSubrSaving(int subCost) const;
};

class charstring_pool_t {
public:
  charstring_pool_t (unsigned nCharstrings);
  void getSubstrings();
  charstring_t getCharstring(unsigned idx);
  void addRawCharstring(char* data, unsigned len);
  void setFDSelect(unsigned char* rawFD);
  void finalize();
  const_tokiter_t getTokenIter(unsigned idx) const;

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
  std::vector<unsigned> generateSuffixes();
  struct suffixSortFunctor;
  std::vector<unsigned> generateLCP(std::vector<unsigned> &suffixes);
  std::vector<substring_t> generateSubstrings(std::vector<unsigned> &suffixes,
                                              std::vector<unsigned> &lcp);

};

#endif