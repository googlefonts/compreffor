#!/usr/bin/env python

"""Tool to subroutinize a CFF table"""

import os
import argparse
import unittest
import functools
import sys
from collections import deque
from multiprocessing import Pool
from multiprocessing.managers import BaseManager, BaseProxy
from fontTools.ttLib import TTFont
from fontTools.misc import psCharStrings

SINGLE_BYTE_OPS = set(['hstem',
                       'vstem',
                       'vmoveto',
                       'rlineto',
                       'hlineto',
                       'vlineto',
                       'rrcurveto',
                       'callsubr',
                       'return',
                       'endchar',
                       'blend',
                       'hstemhm',
                       'hintmask',
                       'cntrmask',
                       'rmoveto',
                       'hmoveto',
                       'vstemhm',
                       'rcurveline',
                       'rlinecurve',
                       'vvcurveto',
                       'hhcurveto',
                     # 'shortint',  # not really an operatr
                       'callgsubr',
                       'vhcurveto',
                       'hvcurveto'])

POOL_CHUNKRATIO = None

class CandidateSubr(object):
    """
    Records a substring of a charstring that is generally
    repeated throughout many glyphs.
    """

    __slots__ = ["length", "location", "freq", "chstrings", "cost_map", "_CandidateSubr__cost",
                 "_adjusted_cost", "_price", "_usages", "_list_idx", "_position", "_encoding",
                 "_program", "_reachable"]

    # length -- length of substring
    # location -- tuple of form (glyph_idx, start_pos) where a ref string starts
    # freq -- number of times it appears
    # chstrings -- chstrings from whence this substring came
    # cost_map -- array from simple alphabet -> actual token

    def __init__(self, length, ref_loc, freq=0, chstrings=None, cost_map=None):
        self.length = length
        self.location = ref_loc
        self.freq = freq
        self.chstrings = chstrings
        self.cost_map = cost_map

        self._reachable = False

    def __len__(self):
        """Return the number of tokens in this substring"""

        return self.length

    def value(self):
        """Returns the actual substring value"""

        assert self.chstrings != None

        return self.chstrings[self.location[0]][self.location[1]:(self.location[1] + self.length)]

    def cost(self):
        """Return the size (in bytes) that the bytecode for this takes up"""

        assert self.cost_map != None

        try:
            if not hasattr(self, '__cost'):
                self.__cost = sum([self.cost_map[t] for t in self.value()])
            return self.__cost
        except:
            raise Exception('Translated token not recognized') 

    def subr_saving(self, use_usages=False, true_cost=False, call_cost=5, subr_overhead=3):
        """
        Return the savings that will be realized by subroutinizing
        this substring.

        Arguments:
        call_cost -- the cost to call a subroutine
        subr_overhead -- the cost to define a subroutine
        """

        if use_usages:
            amt = self._usages
        else:
            amt = self.freq

        cost = self.cost()

        if true_cost and hasattr(self, "_encoding"):
            # account for subroutine calls
            cost -= sum([it[1].cost() - call_cost for it in self._encoding])

        #TODO:
        # - If substring ends in "endchar", we need no "return"
        #   added and as such subr_overhead will be one byte
        #   smaller.
        return (  cost * amt # avoided copies
                - cost # cost of subroutine body
                - call_cost * amt # cost of calling
                - subr_overhead) # cost of subr definition

    @staticmethod
    def tokenCost(token):
        """Calculate the bytecode size of a T2 Charstring token"""

        tp = type(token)
        if issubclass(tp, basestring):
            if token[:8] in ('hintmask', 'cntrmask'):
                return 1 + len(token[8:])
            elif token in SINGLE_BYTE_OPS:
                return 1
            else:
                return 2
        elif tp == int:
            if -107 <= token <= 107:
                return 1
            elif 108 <= token <= 1131 or -1131 <= token <= -108:
                return 2
            else:
                return 5
        elif tp == float:
            return 5
        assert 0


    def __eq__(self, other):
        if type(other) != CandidateSubr:
            return NotImplemented
        return self.length == other.length and self.location == other.location

    def __ne__(self, other):
        if type(other) != CandidateSubr:
            return NotImplemented
        return self.length != other.length or self.location != other.location

    def __repr__(self):
        return "<CandidateSubr: %d x %dreps>" % (self.length, self.freq)

class SubstringFinder(object):
    """
    This class facilitates the finding of repeated substrings
    within a glyph_set. Typical usage involves creation of an instance
    and then calling `get_substrings`, which returns a sorted list
    of `CandidateSubr`s.
    """

    __slots__ = ["suffixes", "data", "alphabet_size", "length", "substrings",
                 "rev_keymap", "glyph_set_keys", "_completed_suffixes",
                 "cost_map"]

    # suffixes -- sorted array of suffixes
    # data --
    #   A 2-level array of charstrings:
    #     - The first level separates by glyph
    #     - The second level separates by token
    #         in a glyph's charstring
    # alphabet_size -- size of alphabet
    # length -- sum of the lengths of the individual glyphstrings
    # rev_keymap -- map from simple alphabet -> original tokens
    # cost_map -- map from simple alphabet -> bytecost of token
    # glyph_set_keys -- glyph_set_keys[i] gives the glyph id for data[i]
    # _completed_suffixes -- boolean whether the suffix array is ready and sorted

    def __init__(self, glyph_set):
        self.rev_keymap = []
        self.cost_map = []
        self.data = []
        self.suffixes = []
        self.length = 0

        self.process_chstrings(glyph_set)

        self._completed_suffixes = False


    def process_chstrings(self, glyph_set):
        """Remap the charstring alphabet and put into self.data"""

        self.glyph_set_keys = glyph_set.keys()

        keymap = {} # maps charstring tokens -> simple integer alphabet

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
                if not tok in keymap:
                    keymap[tok] = next_key
                    self.rev_keymap.append(tok)
                    self.cost_map.append(CandidateSubr.tokenCost(tok))
                    next_key += 1
                program.append(keymap[tok])

            program = tuple(program)
            chstr_len = len(program)
            self.length += chstr_len
            glyph_idx = len(self.data)
            self.suffixes.extend(
                    map(lambda x: (glyph_idx, x), range(chstr_len))
                )
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
                                       self.suffixes[start_indices[l]],
                                       i - start_indices[l],
                                       self.data,
                                       self.cost_map)
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
                                       self.suffixes[start_idx],
                                       i - start_idx,
                                       self.data,
                                       self.cost_map)
                if substr.subr_saving() > 0 or not check_positive:
                    self.substrings.append(substr)

            if not start_indices or min_l > start_indices[-1][0]:
                start_indices.append((min_l, i - 1))

        print("Took %gs (to extract substrings)" % (time.time() - start_time)); start_time = time.time()
        print("%d substrings found" % len(self.substrings))
        print("Sorting...")
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

    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    SINGLE_PROCESS = False

    ALPHA = 0.1
    K = 0.1
    PROCESSES = 12
    NROUNDS = 4
    global POOL_CHUNKRATIO
    if POOL_CHUNKRATIO == None:
        # POOL_CHUNKRATIO = 0.05 # for latin
        POOL_CHUNKRATIO = 0.11 # for logotype

    # generate substrings for marketplace
    sf = SubstringFinder(glyph_set)
    if test_mode:
        substrings = sf.get_substrings(min_freq=0, check_positive=False, sort_by_length=True)
    else:
        substrings = sf.get_substrings(min_freq=2, check_positive=True, sort_by_length=True)
    substr_dict = {}

    if not SINGLE_PROCESS:
        pool = Pool(processes=PROCESSES)
    else:
        class DummyPool: pass
        pool = DummyPool()
        pool.map = lambda f, *l, **kwargs: map(f, *l)

    data = sf.data
    rev_keymap = sf.rev_keymap
    cost_map = sf.cost_map
    glyph_set_keys = sf.glyph_set_keys

    sf = None # garbage collect unnecessary stuff

    import time; start_time = time.time()

    print "dots=%d" % (len(data) + len(substrings))

    # set up dictionary with initial values
    for idx, substr in enumerate(substrings):
        substr._adjusted_cost = substr.cost()
        substr._price = substr._adjusted_cost
        substr._usages = substr.freq # this is the frequency that the substring appears, 
                                    # not necessarily used
        substr._list_idx = idx
        substr_dict[substr.value()] = (idx, substr._price) # XXX avoid excess data copying on fork
                                                           # probably can just pass substr
                                                           # if threading instead

    for run_count in range(NROUNDS):
        # calibrate prices
        for idx, substr in enumerate(substrings):
            marg_cost = float(substr._adjusted_cost) / (substr._usages + K)
            substr._price = marg_cost * ALPHA + substr._price * (1 - ALPHA)
            substr_dict[substr.value()] = (idx, substr._price)

        # minimize charstring costs in current market through DP
        encodings = pool.map(functools.partial(optimize_charstring,
                                               cost_map=cost_map,
                                               substr_dict=substr_dict,
                                               verbose=verbose),
                             zip(glyph_set_keys, data),
                             chunksize=int(POOL_CHUNKRATIO*len(data)) + 1)
        encodings = [[(enc_item[0], substrings[enc_item[1]]) for enc_item in i["encoding"]] for i in encodings]

        # minimize substring costs
        substr_encodings = pool.map(functools.partial(optimize_charstring, 
                                                      cost_map=cost_map,
                                                      substr_dict=substr_dict,
                                                      verbose=verbose),
                                    enumerate([s.value() for s in substrings]),
                                    chunksize=int(POOL_CHUNKRATIO*len(substrings)) + 1)

        for substr, result in zip(substrings, substr_encodings):
            substr._encoding = [(enc_item[0], substrings[enc_item[1]]) for enc_item in result["encoding"]]
            substr._adjusted_cost = result["market_cost"]
        substr_encodings = None # attempt to garbage collect this

        # update substring frequencies based on cost minimization
        for substr in substrings:
            substr._usages = 0

        for calling_substr in substrings:
            for start, substr in calling_substr._encoding:
                if substr:
                    substr._usages += 1
        for glyph_idx, enc in enumerate(encodings):
            for start, substr in enc:
                if substr:
                    substr._usages += 1

        print "Round %d Done!" % (run_count + 1)

        if run_count <= NROUNDS - 2 and not test_mode:
            cutdown_time = time.time()
            bad_substrings = [s for s in substrings if s.subr_saving(use_usages=True, true_cost=(run_count==NROUNDS-2)) <= 0]
            substrings = [s for s in substrings if s.subr_saving(use_usages=True, true_cost=(run_count==NROUNDS-2)) > 0]
            for substr in bad_substrings:
                # potential heuristic:
                # for idx, called_substr in substr._encoding:
                #     called_substr._usages += substr._usages - 1
                del substr_dict[substr.value()]
            for idx, s in enumerate(substrings):
                s._list_idx = idx
            if verbose:
                print "%d substrings with non-positive savings removed" % len(bad_substrings)
                print "(%d had positive usage)" % len([s for s in bad_substrings if s._usages > 0])
            print "Took %gs to cutdown" % (time.time() - cutdown_time)

        print

    bad_substrings = len([s for s in substrings if s.subr_saving(use_usages=True, true_cost=True) <= 0])
    print "%d useless substrings survived the reaping" % bad_substrings


    print("Took %gs (to run iterative_encode)" % (time.time() - start_time))

    def mark_reachable(cand_subr):
        cand_subr._reachable = True
        for it in cand_subr._encoding:
            mark_reachable(it[1])
    for encoding in encodings:
        for it in encoding:
            mark_reachable(it[1])

    subrs = set()
    # subrs.update([it[1] for enc in encodings for it in enc])
    subrs.update([subr for subr in substrings if subr._usages > 0 and subr._reachable])
    subrs = sorted(list(subrs), key=lambda s: s._usages, reverse=True)

    bias = psCharStrings.calcSubrBias(subrs)

    def update_position(idsubr): idsubr[1]._position = idsubr[0]
    map(update_position, enumerate(subrs))

    for subr in subrs:
        program = [rev_keymap[tok] for tok in subr.value()]
        if program[-1] not in ("endchar", "return"):
            program.append("return")
        update_program(program, subr._encoding, bias)
        subr._program = program

    # pr.disable()
    # pr.create_stats()
    # pr.dump_stats("totalstats")

    return {"glyph_encodings": dict(zip(glyph_set_keys, encodings)),
            "subroutines": subrs}

def optimize_charstring(charstring, cost_map, substr_dict, verbose=False):
    """Optimize a charstring (encoded using keymap) using
    the substrings in substr_dict. This is the Dynamic Programming portion
    of `iterative_encode`."""
    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()

    if len(charstring) > 1 and type(charstring[1]) == tuple:
        if type(charstring[0]) == int:
            skip_idx = charstring[0]
            charstring = charstring[1]
            glyph_key = None
        else:
            glyph_key = charstring[0] # XXX remove this testing thing!
            charstring = charstring[1]
            skip_idx = None
    else:
        skip_idx = None

    results = [0 for _ in xrange(len(charstring) + 1)]
    next_enc_idx = [None for _ in xrange(len(charstring))]
    next_enc_substr = [None for _ in xrange(len(charstring))]
    for i in reversed(range(len(charstring))):
        min_option = float('inf')
        min_enc_idx = len(charstring)
        min_enc_substr = None
        cur_cost = 0
        for j in range(i + 1, len(charstring) + 1):
            cur_cost += cost_map[charstring[j - 1]]

            if charstring[i:j] in substr_dict:
                substr = substr_dict[charstring[i:j]]
                if substr[0] != skip_idx:
                    option = substr[1] + results[j]
                    substr = substr[0]
                else:
                    substr = None
                    option = cur_cost + results[j]
            else:
                # note: must not be branching, so just make _price actual cost
                substr = None
                option = cur_cost + results[j]
            
            if option < min_option:
                min_option = option
                min_enc_idx = j
                min_enc_substr = substr

        results[i] = min_option
        next_enc_idx[i] = min_enc_idx
        next_enc_substr[i] = min_enc_substr

    market_cost = results[0]
    encoding = []
    cur_enc_idx = 0
    last = len(next_enc_idx)
    while cur_enc_idx < last:
        last_idx = cur_enc_idx
        cur_enc_substr = next_enc_substr[cur_enc_idx]
        cur_enc_idx = next_enc_idx[cur_enc_idx]

        if cur_enc_substr != None:
            encoding.append((last_idx, cur_enc_substr))
        elif cur_enc_idx - last_idx > 1:
            print "Weird charstring: %s" % (charstring,)
            print "Weird index: %d" % (cur_enc_idx,)

    # pr.disable()
    # pr.create_stats()
    # pr.dump_stats("stats")

    if verbose:
        sys.stdout.write("."); sys.stdout.flush()
    return {"encoding": encoding, "market_cost": market_cost}

def has_endchar(subr):
    return (subr._program[-1] == "endchar"
            or ((subr._program[-1] == "callsubr" or subr._program[-1] == "callgsubr")
                and has_endchar(subr._encoding[-1][1])))
    
def apply_encoding(font, glyph_encodings, subrs):
    """Apply the result of iterative_encode to a TTFont"""

    assert len(font['CFF '].cff.topDictIndex) == 1
    top_dict = font['CFF '].cff.topDictIndex[0]

    bias = psCharStrings.calcSubrBias(subrs)

    for subr in subrs:
        item = psCharStrings.T2CharString(program=subr._program)
        top_dict.GlobalSubrs.append(item)

    for glyph, enc in glyph_encodings.iteritems():
        charstring = top_dict.CharStrings[glyph]
        update_program(charstring.program, enc, bias)

def update_program(program, encoding, bias):
    offset = 0
    for item in encoding:
        assert hasattr(item[1], "_position"), "CandidateSubr without position in Subrs encountered"
        program[(item[0] - offset):(item[0] + item[1].length - offset)] = [item[1]._position - bias, "callgsubr"]
        offset += item[1].length - 2
    return program

def compress_cff(font, out_file="compressed.otf"):
    """Compress a font using the iterative method and output result"""

    # from guppy import hpy
    # hp = hpy()

    ans = iterative_encode(font.getGlyphSet(), verbose=True)
    encoding = ans["glyph_encodings"]
    subrs = ans["subroutines"]
    apply_encoding(font, encoding, subrs)
    font.save(out_file)

    # print hp.heap()

def human_size(num):
    """Return a number of bytes in human-readable units"""

    num = float(num)
    for s in ['bytes', 'KB', 'MB']:
        if num < 1024.0:
            return '%3.1f %s' % (num, s)
        else:
            num /= 1024.0
    return '%3.1f %s' % (num, 'GB')

def _test():
    """
    >>> from testData import *
    >>> sf = SubstringFinder(glyph_set)
    >>> sf.get_suffixes() # doctest: +ELLIPSIS
    G...
    [(0, 0), (1, 1), (0, 1), (0, 2), (1, 0), (1, 2)]
    """

def main(filename=None, test=False, doctest=False, verbose_test=False, check=False):
    if doctest:
        import doctest
        doctest.testmod(verbose=verbose_test)

    if test:
        from testCffCompressor import TestCffCompressor
        test_suite = unittest.TestLoader().loadTestsFromTestCase(TestCffCompressor)
        unittest.TextTestRunner().run(test_suite)

    if filename and len(filename) == 1:
        font = TTFont(filename[0])
        orig_size = os.path.getsize(filename[0])
        # sf = SubstringFinder(font.getGlyphSet())
        # substrings = sf.get_substrings()
        # print("%d substrings found" % len(substrings))
        # print
        print("Compressing font through iterative_encode:")
        out_name = "%s.compressed%s" % os.path.splitext(filename[0])

        compress_cff(font, out_name)

        comp_size = os.path.getsize(out_name)
        print("Saved %s!" % human_size(orig_size - comp_size))

    if check:
        from testCffCompressor import test_compression_integrity, test_call_depth
        assert len(filename) <= 2

        if len(filename) == 1:
            test_compression_integrity(filename[0], out_name)
            test_call_depth(out_name)
        else:
            test_compression_integrity(*filename)
            test_call_depth(filename[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subroutinize a font.')
    parser.add_argument('filename', help='Where to find the font', nargs='*')
    parser.add_argument('-t', required=False, action='store_true',
                        dest='test', default=False)
    parser.add_argument('-d', required=False, action='store_true',
                        dest='doctest', default=False)
    parser.add_argument('-v', required=False, action='store_true',
                        dest='verbose_test', default=False)
    parser.add_argument('-c', required=False, action='store_true',
                        dest='check', default=False)
    parser.add_argument('--chunkratio', required=False, type=float)

    kwargs = vars(parser.parse_args())

    POOL_CHUNKRATIO = kwargs.pop("chunkratio", None)

    main(**kwargs)
