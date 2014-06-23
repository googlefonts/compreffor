#!/usr/bin/env python

"""Tool to subroutinize a CFF table"""

import argparse, heapq, unittest
from collections import deque
# import numpy as np
from fontTools.ttLib import TTFont
from fontTools.misc import psCharStrings

class CandidateSubr(object):
    """
    Records a substring of a charstring that is generally
    repeated throughout many glyphs.
    """

    length = None # length of substring
    locations = None # list tuples of form (glyph_idx, start_pos)
    freq = None # number of times it appears
    chstrings = None # chstrings from whence this substring came
    rev_keymap = None # array from simple alphabet -> actual token

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
        """Return the number of tokens in this substring"""

        return self.length

    def value(self):
        """Returns the actual substring value"""

        try:
            return self.chstrings[self.locations[0][0]][self.locations[0][1]:(self.locations[0][1] + self.length)]
        except IndexError: # there are no locations
            return None

    def add_location(self, location):
        """Add a location where this substring appears (2-tuple specifying
            position in chstrings)"""

        self.locations.append(location)
        self.freq += 1

    def cost(self):
        """Return the size (in bytes) that the bytecode for this takes up"""

        try:
            if not hasattr(self, '__cost'):
                self.__cost = self.string_cost([self.rev_keymap[t] for t in self.value()])
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
    def string_cost(charstring):
        """Calculate the bytecode size of a T2 Charstring substring. Note:
        tokens are taken literally and are not remapped."""

        global STRING_COST_CACHE
        charstring = tuple(charstring)

        try:
            cached = charstring in STRING_COST_CACHE
        except NameError:
            STRING_COST_CACHE = {}
            cached = False

        if cached:
            return STRING_COST_CACHE[charstring]
        else:
            ret = sum(map(CandidateSubr.tokenCost, charstring))
            STRING_COST_CACHE[charstring] = ret
            return ret


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
        return "<CandidateSubr: %d x %d>" % (self.length, self.freq)

class SubstringFinder(object):
    """
    This class facilitates the finding of repeated substrings
    within a glyph_set. Typical usage involves creation of an instance
    and then calling `get_substrings`, which returns a sorted list
    of `CandidateSubr`s.
    """

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
    glyph_set_keys = None

    _completed_suffixes = False

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

        self.glyph_set_keys = glyph_set.keys()

        next_key = 0

        for k in self.glyph_set_keys:
            char_string = glyph_set[k]
            char_string.decompile()
            program = []
            piter = iter(enumerate(char_string.program))
            for i, tok in piter:
                assert tok not in ("callsubr", "callgsubr", "return", "endchar") or \
                       tok in ("callsubr", "callgsubr", "return", "endchar") and \
                            i == len(char_string.program) - 1
                if tok in ("hintmask", "cntrmask"):
                    # Attach next token to this, as a subroutine
                    # call cannot be placed between this token and
                    # the following.
                    _, tokennext = next(piter)
                    tok = '%s %s' % (tok, tokennext)
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
        if self._completed_suffixes:
            return self.suffixes

        print("Gettings suffixes via Python sort"); start_time = time.time()

        self.suffixes.sort(key=lambda idx: self.data[idx[0]][idx[1]:])
        self._completed_suffixes = True

        print("Took %gs" % (time.time() - start_time))
        return self.suffixes

    def get_lcp(self):
        """Returns the LCP array"""

        if not self._completed_suffixes:
            self.get_suffixes()

        assert self._completed_suffixes

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
        """
        Return repeated substrings (type CandidateSubr) from the charstrings
        sorted by subroutine savings with freq >= min_freq using the initial
        algorithm (no LCP). This is here for comparison with get_substrings to
        see the improvement.

        Arguments:
        branching -- if True, only include "branching" substrings (see Kasai et al)
        min_freq -- the minimum frequency required to include a substring
        """

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
                substr = CandidateSubr(l + 1, 
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

    def get_substrings(self, min_freq=2, check_positive=True, sort_by_length=False):
        """
        Return repeated substrings (type CandidateSubr) from the charstrings
        sorted by subroutine savings with freq >= min_freq using the LCP array. 

        Arguments:
        min_freq -- the minimum frequency required to include a substring
        check_positive -- if True, only allow substrings with positive subr_saving
        """

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

            while start_indices and start_indices[-1][0] > min_l:
                l, start_idx = start_indices.pop()
                freq = i - start_idx
                if freq < min_freq:
                    continue
                
                substr = CandidateSubr(l,
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
        if sort_by_length:
            self.substrings.sort(key=lambda s: len(s))
        else:
            self.substrings.sort(key=lambda s: s.subr_saving(), reverse=True)
        print("Took %gs (to sort)" % (time.time() - start_time))
        return self.substrings


def iterative_encode(glyph_set, verbose=True, test_mode=False):
    """
    Choose a subroutinization encoding for all charstrings in
    `glyph_set` using an iterative Dynamic Programming algorithm.
    Initially uses the results from SubstringFinder and then
    iteratively optimizes.

    Arguments:
    glyph_set -- the set of charstrings to encode
    verbose -- if True, print miscellanous info during iterations
    test_mode -- disables some checks (such as positive subr_saving)

    Returns:
    An encoding dictionary which specifies how to break up each charstring.
    Encoding[i] describes how to encode glyph i. Each entry is something
    like [(x_1, y_1), (x_2, y_2), ..., (x_k, y_k)], which means that the
    glyph_set[i][x_1:y_1], glyph_set[i][x_2:y_2], ..., glyph_set[i][x_k:y_k]
    should each be subroutinized.
    """

    ALPHA = 0.1
    K = 1.0

    # generate substrings for marketplace
    sf = SubstringFinder(glyph_set)
    substrings = sf.get_substrings(min_freq=0, check_positive=(not test_mode), sort_by_length=True)
    substr_dict = {}

    import time; start_time = time.time()

    # set up dictionary with initial values
    for substr in substrings:
        substr._adjusted_cost = substr.cost()
        substr._price = substr._adjusted_cost
        substr._usages = substr.freq # this is the frequency that the substring appears, 
                                    # not necessarily used
        substr_dict[substr.value()] = substr

    # encoding array to store chosen encodings
    encodings = [None] * len(sf.data)

    for run_count in range(2):
        # calibrate prices
        for substr in substr_dict.values():
            marg_cost = float(substr._adjusted_cost) / (substr._usages + K)
            substr._price = marg_cost * ALPHA + substr._price * (1 - ALPHA)

        # minimize charstring costs in current market through DP
        for idx, charstring in enumerate(sf.data):
            ans = optimize_charstring(charstring, sf.rev_keymap, substr_dict)
            
            encodings[idx] = tuple(ans["encoding"])

            if verbose:
                print "Charstring %s: market_cost=%f, encoding=%s" % \
                        (charstring, ans["market_cost"], ans["encoding"])

        # minimize substring costs
        for substr in substrings:
            ans = optimize_charstring(substr.value(), sf.rev_keymap, substr_dict)
            substr._encoding = ans["encoding"]
            substr._adjusted_cost = ans["market_cost"]

            if verbose:
                print "Substring %s: market_cost=%f, encoding=%s" % \
                        (substr.value(), ans["market_cost"], ans["encoding"])


        # update substring frequencies based on cost minimization
        for substr in substr_dict.values():
            substr._usages = 0
        for glyph_idx, enc in enumerate(encodings):
            for start, stop, substr in enc:
                if substr:
                    substr._usages += 1

        print "Round %d Done!" % (run_count + 1)
        print

    print("Took %gs (to run iterative_encode)" % (time.time() - start_time))

    return dict((sf.glyph_set_keys[i], encodings[i]) for i in xrange(len(encodings)))

class EncodingItem:
    idx = -1
    substr = None

    def __init__(self, idx, substr=None):
        self.idx = idx
        self.substr = substr

    def __len__(self):
        return len(self.substr) if self.substr else 1

def optimize_charstring(charstring, rev_keymap, substr_dict):
    """Optimize a charstring (encoded using inverse ofrev_keymap) using
    the substrings in substr_dict. This is the Dynamic Programming portion
    of `iterative_encode`."""

    results = [0] * (len(charstring) + 1)
    next_enc = [None] * len(charstring)
    for i in reversed(range(len(charstring))):
        min_option = float('inf')
        min_enc = EncodingItem(len(charstring))
        for j in range(i + 1, len(charstring) + 1):
            if charstring[i:j] in substr_dict:
                substr = substr_dict[charstring[i:j]]
                option = substr._price + results[j]
            else:
                # note: must not be branching, so just make _price actual cost
                substr = None
                option = \
                    CandidateSubr.string_cost([rev_keymap[t] \
                      for t in charstring[i:j]]) + results[j]
            
            if option < min_option:
                min_option = option
                min_enc = EncodingItem(j, substr)

        results[i] = min_option
        next_enc[i] = min_enc

    market_cost = results[0]
    encoding = []
    last_idx = 0 
    cur_enc = next_enc[0]
    while cur_enc != None and cur_enc.idx < len(next_enc):
        if cur_enc.idx - last_idx > 1:
            encoding.append((last_idx, cur_enc.idx, cur_enc.substr))
        last_idx = cur_enc.idx
        cur_enc = next_enc[cur_enc.idx]

    return {"encoding": encoding, "market_cost": market_cost}

def apply_encoding(font, glyph_encoding):
    """Apply the result of iterative_encode to a TTFont"""

    assert len(font['CFF '].cff.topDictIndex) == 1
    top_dict = font['CFF '].cff.topDictIndex[0]

    subrs = set()
    subrs.update([it[2] for enc in glyph_encoding.values() for it in enc])
    subrs = sorted(list(subrs), key=lambda s: s._usages, reverse=True)

    for subr in subrs:
        subr._position = len(top_dict.Private.Subrs)
        program = [subr.rev_keymap[tok] for tok in subr.value()]
        if program[-1] not in ("endchar", "return"):
            program.append("return")
        subr._program = program
        item = psCharStrings.T2CharString(program=program)
        top_dict.Private.Subrs.append(item)

    bias = psCharStrings.calcSubrBias(top_dict.Private.Subrs)

    for glyph, enc in glyph_encoding.iteritems():
        charstring = top_dict.CharStrings[glyph]
        offset = 0
        for item in enc:
            charstring.program[(item[0] - offset):(item[1] - offset)] = [item[2]._position - bias, "callsubr"]
            offset += item[1] - item[0] - 2
        if not (charstring.program[-1] == "endchar" \
            or charstring.program[-1] == "callsubr" and enc[-1][2]._program[-1] == "endchar"):
            charstring.program.append("endchar")


def compress_cff(font, out_file="compressed.otf"):
    """Compress a font using the iterative method and output result"""

    encoding = iterative_encode(font.getGlyphSet(), verbose=False)
    apply_encoding(font, encoding)
    font.save(out_file)

def _test():
    """
    >>> from testData import *
    >>> sf = SubstringFinder(glyph_set)
    >>> sf.get_suffixes() # doctest: +ELLIPSIS
    G...
    [(0, 0), (1, 1), (0, 1), (0, 2), (1, 0), (1, 2)]
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subroutinize a font.')
    parser.add_argument('filename', help='Where to find the font', nargs='?')
    parser.add_argument('-t', required=False, action='store_true',
                        dest='test', default=False)
    parser.add_argument('-d', required=False, action='store_true',
                        dest='doctest', default=False)
    parser.add_argument('-v', required=False, action='store_true',
                        dest='verbose_test', default=False)

    args = parser.parse_args()

    import doctest
    doctest.testmod(verbose=args.verbose_test)

    if args.test:
        from testCffCompressor import TestCffCompressor
        test_suite = unittest.TestLoader().loadTestsFromTestCase(TestCffCompressor)
        unittest.TextTestRunner().run(test_suite)

    if args.filename:
        font = TTFont(args.filename)
        sf = SubstringFinder(font.getGlyphSet())
        substrings = sf.get_substrings()
        print("%d substrings found" % len(substrings))
        print
        print("Running dynamic_allocate:")
        dynamic_allocate(font.getGlyphSet())

