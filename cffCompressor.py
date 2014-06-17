#!/usr/bin/env python

"""Tool to subroutinize a CFF table"""

import argparse, heapq
# import numpy as np
from fontTools.ttLib import TTFont
from fontTools.misc import psCharStrings

# OP_MAP = dict((k, v) for (v, k) in t2Operators)

S_TYPE = True
L_TYPE = False

TIMSORT_THRESHOLD = 300
MAX_TOUCHES = 3
SORT_DEPTH = 10

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
                self.__cost = sum([self.tokenCost(self.rev_keymap[token]) for token in self.value()])
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

    def get_substrings(self, branching=True, min_freq=2):
        """Return the sorted substring sets (with freq >= min_freq)"""

        self.get_suffixes()

        # initally just do this naively for comparison purposes
        # will use LCP later
        print("Extracting substrings"); import time; start_time = time.time()

        spin_time = 0

        start_indices = []
        self.substrings = []
        previous = ()
        import pdb; pdb.set_trace()
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
                    print 'Freq reject: ', previous[:l]
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
                else:
                    print 'Savings reject: ', current[:l]

            previous = current
            start_indices = start_indices[:min_l]
            start_indices += [i] * (len(current) - min_l)

        print("Spin time: %gs" % spin_time)

        print("Took %gs" % (time.time() - start_time)); start_time = time.time()
        print("Sorting")
        self.substrings.sort(key=lambda s: s.subr_saving(), reverse=True)
        print("Took %gs" % (time.time() - start_time))
        return self.substrings




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

