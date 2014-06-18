#!/usr/bin/env python

"""Tool to subroutinize a CFF table"""

import argparse, heapq
from collections import deque
# import numpy as np
from fontTools.ttLib import TTFont
from fontTools.misc import psCharStrings

class CharSubStringSet(object):
    """
    Records a substring of a charstring that is generally
    repeated throughout many glyphs.
    """
    length = None # length of substring
    locations = None # list tuples of form (glyph_idx, start_pos)
    freq = None # number of times it appears
    chstrings = None # chstrings from whence this substring came
    rev_keymap = None

    def __init__(self, length=None, locs=None, chstrings=None, rev_keymap=None):
        if length == None:
            self.length = 0
        else:
            self.length = length

        if locs == None:
            self.locations = []
            self.freq = 0
        else:
            self.locations = locs
            self.freq = len(locs)

        self.chstrings = chstrings
        self.rev_keymap = rev_keymap

    def __len__(self):
        return self.length

    def value(self):
        """Returns the actual substring"""
        try:
            return self.chstrings[self.locations[0][0]][self.locations[0][1]:(self.locations[0][1] + self.length)]
        except IndexError: # there are no locations
            return None

    def add_location(self, location):
        """Add a location where this substring appears"""
        self.locations.append(location)
        self.freq += 1

    def cost(self):
        """Return the size (in bytes) that the bytecode for this takes up"""
        try:
            if not hasattr(self, '__cost'):
                self.__cost = self.string_cost(self.value(), self.rev_keymap)
            return self.__cost
        except:
            raise Exception('Translated token not recognized') 

    def subr_saving(self, call_cost=5, subr_overhead=3):
        """
        Return the savings that will be realized by subroutinizing
        this substring.

        Arguments:
        call_cost -- the cost to call a subroutine
        subr_overhead -- the cost to define a subroutine
        """
        #TODO:
        # - If substring ends in "endchar", we need no "return"
        #   added and as such subr_overhead will be one byte
        #   smaller.
        return (  self.cost() * self.freq # avoided copies
                - self.cost() # cost of subroutine body
                - call_cost * self.freq # cost of calling
                - subr_overhead) # cost of subr definition

    @staticmethod
    def tokenCost(token):
        """Calculate the bytecode size of a T2 Charstring token"""
        tp = type(token)
        if issubclass(tp, basestring):
            if token[:8] in ('hintmask', 'cntrmask'):
                return 1 + len(token[8:])
            return len(psCharStrings.T2CharString.opcodes[token])
        elif tp == int:
            return len(psCharStrings.encodeIntT2(token))
        elif tp == float:
            return len(psCharStrings.encodeFixed(token))
        assert 0

    @staticmethod
    def string_cost(charstring, rev_keymap):
        return sum([CharSubStringSet.tokenCost(rev_keymap[token]) for token in charstring])


    sort_on = lambda self: -self.subr_saving()

    def __lt__(self, other):
        return self.sort_on() < other.sort_on()

    def __le__(self, other):
        return self.sort_on() <= other.sort_on()

    def __eq__(self, other):
        return self.sort_on() == other.sort_on()

    def __ne__(self, other):
        return self.sort_on() != other.sort_on()

    def __ge__(self, other):
        return self.sort_on() >= other.sort_on()

    def __gt__(self, other):
        return self.sort_on() > other.sort_on()


    def __repr__(self):
        return "<SubStringSet: %d x %d>" % (self.length, self.freq)

class SubstringFinder(object):
    """Builds a sorted suffix array from a glyph set"""

    suffixes = None
    data = None
    # data is a 2-level array of charstrings:
    #   The first level separates by glyph
    #   The second level separates by token
    #       in a glyph's charstring
    bucket_for = None
    alphabet_size = None
    length = None
    keymap = None
    rev_keymap = None

    _completed = False

    def __init__(self, glyph_set):
        self.keymap = {} # maps charstring tokens -> simple integer alphabet
        self.rev_keymap = [] # reversed keymap
        #TODO: make above a numpy array
        self.data = []
        self.suffixes = []

        self.process_chstrings(glyph_set)

        self.length = 0
        for glyph_idx in xrange(len(self.data)):
            chstr_len = len(self.data[glyph_idx])
            self.length += chstr_len
            self.suffixes.extend(
                map(lambda x: (glyph_idx, x), range(chstr_len))
                )

        self.bucket_for = [[None for _ in xrange(len(self.data[i]))] \
                                for i in xrange(len(self.data))]


    def process_chstrings(self, glyph_set):
        """Remap the charstring alphabet and put into self.data"""

        glyph_set_keys = glyph_set.keys()

        next_key = 0

        for k in glyph_set_keys:
            char_string = glyph_set[k]
            char_string.decompile()
            program = []
            piter = iter(enumerate(char_string.program))
            for i, tok in piter:
                assert tok not in ("callsubr", "callgsubr", "return", "endchar") \
                        or tok == "endchar" and i == len(char_string.program) - 1
                if tok in ("hintmask", "cntrmask"):
                    # Attach next token to this, as a subroutine
                    # call cannot be placed between this token and
                    # the following.
                    _, tokennext = next(piter)
                    token = '%s %s' % (token, tokennext)
                if not tok in self.keymap:
                    self.keymap[tok] = next_key
                    self.rev_keymap.append(tok)
                    next_key += 1
                program.append(self.keymap[tok])

            self.data.append(tuple(program))

        self.alphabet_size = next_key

    def get_suffixes(self):
        """Return the sorted suffix array"""

        import time
        if self._completed:
            return self.suffixes

        print("Gettings suffixes via Python sort"); start_time = time.time()

        self.suffixes.sort(key=lambda idx: self.data[idx[0]][idx[1]:])
        self._completed = True

        print("Took %gs" % (time.time() - start_time))
        return self.suffixes

    def get_lcp(self):
        """Returns the LCP array"""
        rank = [[0 for _ in xrange(len(d_list))] for d_list in self.data]
        lcp = [0 for _ in xrange(self.length)]

        # compute rank array
        for i in range(self.length):
            glyph_idx, tok_idx = self.suffixes[i]
            rank[glyph_idx][tok_idx] = i

        for glyph_idx in xrange(len(self.data)):
            cur_h = 0
            chstring = self.data[glyph_idx]
            for tok_idx in xrange(len(chstring)):
                cur_rank = rank[glyph_idx][tok_idx]
                if cur_rank > 0:
                    last_glidx, last_tidx = self.suffixes[cur_rank - 1]
                    last_chstring = self.data[last_glidx]
                    while last_tidx + cur_h < len(last_chstring) and \
                          tok_idx + cur_h < len(chstring) and \
                          last_chstring[last_tidx + cur_h] == self.data[glyph_idx][tok_idx + cur_h]:
                        cur_h += 1
                    lcp[cur_rank] = cur_h

                    if cur_h > 0:
                        cur_h -= 1

        return lcp

    def get_substrings_initial(self, branching=True, min_freq=2):
        """Return the sorted substring sets (with freq >= min_freq)"""

        self.get_suffixes()

        # initally just do this naively for comparison purposes
        # will use LCP later
        print("Extracting substrings"); import time; start_time = time.time()

        spin_time = 0

        start_indices = []
        self.substrings = []
        previous = ()
        # import pdb; pdb.set_trace()
        for i, (glyph_idx, tok_idx) in enumerate(self.suffixes):
            current = self.data[glyph_idx][tok_idx:]

            if current == previous:
                continue

            spin_start = time.time()
            max_l = min(len(previous), len(current))
            min_l = max_l
            for l in range(max_l):
                if previous[l] != current[l]:
                    min_l = l
                    break
            spin_time += time.time() - spin_start

            # First min_l items are still the same.

            # Pop the rest from previous and account for.
            #TODO: don't allow overlapping substrings into the same set
            for l in range(min_l, len(previous)):
                freq = i - start_indices[l]
                if freq < min_freq:
                    # print 'Freq reject: ', previous[:l]
                    break
                if branching and l + 1 < len(previous) and freq == i - start_indices[l+1]:
                    # This substring is redundant since the substring
                    # one longer has the same frequency.  Ie., this one
                    # is not "branching".
                    continue
                substr = CharSubStringSet(l + 1, 
                                          [self.suffixes[j] for j 
                                              in range(start_indices[l], i)],
                                          self.data,
                                          self.rev_keymap)
                if substr.subr_saving() > 0:
                    self.substrings.append(substr)
                # else:
                #     print 'Savings reject: ', current[:l]

            previous = current
            start_indices = start_indices[:min_l]
            start_indices += [i] * (len(current) - min_l)

        print("Spin time: %gs" % spin_time)

        print("Took %gs" % (time.time() - start_time)); start_time = time.time()
        print("Sorting")
        self.substrings.sort(key=lambda s: s.subr_saving(), reverse=True)
        print("Took %gs" % (time.time() - start_time))
        return self.substrings

    def get_substrings(self, min_freq=2, check_positive=True):
        """Return the sorted substring sets (with freq >= min_freq)"""

        self.get_suffixes()

        print("Extracting substrings"); import time; start_time = time.time()

        print("Getting lcp"); lcp_time = time.time()

        lcp = self.get_lcp()

        print("Took %gs (to get lcp array)" % (time.time() - lcp_time))

        start_indices = deque()
        self.substrings = []

        for i, (glyph_idx, tok_idx) in enumerate(self.suffixes):
            current = self.data[glyph_idx][tok_idx:]

            min_l = lcp[i]
            # First min_l items are still the same.

            # Pop the rest from previous and account for.
            # Note: non-branching substrings aren't included
            #TODO: don't allow overlapping substrings into the same set
            # import pdb; pdb.set_trace()
            while start_indices and start_indices[-1][0] > min_l:
                l, start_idx = start_indices.pop()
                freq = i - start_idx
                if freq < min_freq:
                    continue
                
                substr = CharSubStringSet(l,
                                          [self.suffixes[j] for j 
                                              in range(start_idx, i)],
                                          self.data,
                                          self.rev_keymap)
                if substr.subr_saving() > 0 or not check_positive:
                    self.substrings.append(substr)

            if not start_indices or min_l > start_indices[-1][0]:
                start_indices.append((min_l, i - 1))

        print("Took %gs (to extract substrings)" % (time.time() - start_time)); start_time = time.time()
        print("Sorting")
        self.substrings.sort(key=lambda s: s.subr_saving(), reverse=True)
        print("Took %gs (to sort)" % (time.time() - start_time))
        return self.substrings


def dynamic_allocate(glyph_set):
    ALPHA = 0.1
    K = 1

    import pdb; pdb.set_trace()

    # generate substrings for marketplace
    sf = SubstringFinder(glyph_set)
    substrings = sf.get_substrings(0, False)
    substr_dict = {}

    # set up dictionary with initial values
    for substr in substrings:
        # XXX this could poss. work? just using substr frequency rather than usage
        substr.price = substr.cost() / substr.freq
        substr.usages = 0
        substr_dict[substr.value()] = substr

    # encoding array to store chosen encodings
    encodings = [None] * len(sf.data)

    for i in range(100):
        # calibrate prices
        for substr in substr_dict.values():
            marg_cost = substr.cost() / (substr.usages + K)
            substr.price = marg_cost * ALPHA + substr.price * (1 - ALPHA)

        # minimize costs in current market through DP
        for idx, charstring in enumerate(sf.data):
            results = [0] * (len(charstring) + 1)
            next_idx = [None] * len(charstring)
            for i in reversed(range(len(charstring))):
                min_option = float('inf')
                min_idx = -1
                for j in range(i + 1, len(charstring) + 1):
                    if charstring[i:j] in substr_dict:
                        option = substr_dict[charstring[i:j]].price + results[j]
                    else:
                        option = \
                            CharSubStringSet.string_cost(charstring[i:j], sf.rev_keymap) + \
                            results[j]
                    
                    if option < min_option:
                        min_option = option
                        min_idx = j

                results[i] = min_option
                next_idx[i] = min_idx

            market_cost = results[0]
            encoding = []
            cur_idx = next_idx[0]
            while cur_idx != None and cur_idx != len(charstring):
                encoding.append(cur_idx)
                cur_idx = next_idx[cur_idx]
            encoding.append(len(charstring))

            encodings[idx] = tuple(encoding)

            print "Charstring ", charstring, ": market_cost=", \
                    market_cost, ", encoding=", encoding


        # update substring frequencies based on cost minimization
        for substr in substr_dict.values():
            substr.usages = 0
        for enc in encodings:
            for idx in range(1, len(enc)):
                substr = tuple(sf.data[enc[idx - 1]:enc[idx]])
                if substr in substr_dict:
                    substr = substr_dict[substr]
                    substr.usages += 1

        print "Round Done!"
        print




def _test():
    """
    >>> from testData import *
    >>> sf = SubstringFinder(glyph_set)
    >>> sf.get_suffixes() # doctest: +ELLIPSIS
    [(0, 0), (1, 1), (0, 1), (0, 2), (1, 0), (1, 2)]


    # test CharSubStringSet.cost() somehow!
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subroutinize a font.')
    parser.add_argument('filename', help='Where to find the font')
    parser.add_argument('-t', required=False, action='store_true',
                        dest='test', default=False)
    parser.add_argument('-v', required=False, action='store_true',
                        dest='verbose_test', default=False)

    args = parser.parse_args()
    font = TTFont(args.filename)

    if args.test:
        import doctest
        doctest.testmod(verbose=args.verbose_test)

    sab = SubstringFinder(font.getGlyphSet())
    substrings = sab.get_substrings()
    print(len(substrings))

