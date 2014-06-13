#!/usr/bin/env python

"""Tool to subroutinize a CFF table"""

import argparse
# import numpy as np
from fontTools.ttLib import TTFont

# OP_MAP = dict((k, v) for (v, k) in t2Operators)

S_TYPE = True
L_TYPE = False

class CharSubStringSet(object):
    """
    Records a substring of a charstring that is generally
    repeated throughout many glyphs.
    """
    length = None # length of substring
    locations = None # list tuples of form (glyph_idx, start_pos)
    freq = None # number of times it appears

    def __init__(self, length=None, locs=None):
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

    def __len__(self):
        return self.length

    def value(self, chstrings):
        """Returns the actual substring"""
        try:
            return chstrings[self.locations[0][0]][self.locations[0][1]:self.length]
        except IndexError: # there are no locations
            return None

    def add_location(self, location):
        """Add a location where this substring appears"""
        self.locations.append(location)
        self.freq += 1

    def cost(self):
        """Return the size that the bytecode for this takes up"""
        return len(self)

    def subr_saving(self, call_cost=2, subr_overhead=2):
        """
        Return the savings that will be realized by subroutinizing
        this substring.

        Arguments:
        call_cost -- the cost to call a subroutine
        subr_overhead -- the cost to define a subroutine
        """

        return (  self.cost() * self.freq
                - self.cost() # include cost of subroutine
                - call_cost * self.freq # include cost of calling
                - subr_overhead) # include cost of subr definition

    def __repr__(self):
        return "<SubStringSet: %d x %d>" % (self.length, self.freq)

def get_suffix_array(chstrings):
    """Generate the sorted suffix array for `chstrings`."""

    # make type (S or L) array
    t = []
    for chstr in reversed(chstrings):
        t_run = []
        last = float('-inf')
        last_was_s = True #TODO: should this be initialized to True or False?
        for i in reversed(chstr):
            if i < last or (i == last and last_was_s):
                t_run.insert(0, S_TYPE)
                last_was_s = True
            else:
                t_run.insert(0, L_TYPE)
                last_was_s = False
            last = i
        t.insert(0, t_run)

    k = sum([len(chstr) for chstr in chstrings])
    suf_arr = [(-1, -1) for x in xrange(k)]
    buckets = [-1] * k

    sorted_lms = induce_sort_lms(chstrings, suf_arr, t, buckets, k)
    print(sorted_lms[:10]) # XXX PRINT ANSWER for now

def mark_buckets(chstrings, buckets, k, ends=False, counts=None):
    """
    Find the start (or end) of buckets based on the first token
    of suffixes in chstrings. Put this into `buckets` such that
    `buckets[x]` tells you the start (or end) of the bucket for
    alphabet element x.

    Arguments:
    chstrings -- 2-level array of charstrings (integer alphabet assumed)
    buckets -- list of size k to use for holding bucket indices
                (the effective output goes here)
    k -- size of alphabet
    ends -- True to get the end of bucket, false to get beginning
    counts -- Include this to supply counts of the number of elements
              beginning with each alphabet token. This is not necessary.

    Returns: nothing. output goes into `buckets`.
    """
    if not counts:
        # use `buckets` to store the bucket sizes (space optimization)
        for i in range(k):
            buckets[i] = 0 # 0 out relevant buckets

        for chstr in chstrings:
            for i in chstr:
                buckets[i] += 1
        counts = buckets

    cur_sum = 0
    if ends:
        for i in range(k):
            cur_sum += counts[i]
            buckets[i] = cur_sum - 1
    else:
        for i in range(k):
            cur_sum += counts[i]
            buckets[i] = cur_sum - counts[i]

def induce_sort_lms(chstrings, suf_arr, t, buckets, k):
    """ 
    Use the induced-sort strategy to sort the Left Most S-type
    substrings.

    Arguments:
    chstrings -- 2-level array of charstrings (integer alphabet assumed)
    suf_arr -- array to use for temporary storage during computation
    t -- 2-level array mirroring chstrings that contains the types of each token
    buckets -- list of size k to use for holding bucket indices
    k -- size of alphabet

    Returns: a list of identifiers for LMS substrings in sorted order
    """
    # import pdb; pdb.set_trace()

    # find lms_strings by looping through type array
    lms_strings = []
    is_lms = [] # mirrors chstrings for determinging whether a char is LMS
    #TODO: use numpy for above
    for chidx, chstr_row in enumerate(t):
        is_lms.append([False])
        for idx in range(1, len(chstr_row)):
            if chstr_row[idx] == S_TYPE and chstr_row[idx - 1] == L_TYPE:
                lms_strings.append((chidx, idx))
                is_lms[-1].append(True)
            else:
                is_lms[-1].append(False)

    # clear out the suffix array for use here
    for i in range(len(suf_arr)):
        suf_arr[i] = (-1, -1)

    # put lms suffixes into suf_arr in order of discovery
    mark_buckets(chstrings, buckets, k, True)
    for chidx, i in lms_strings:
        suf_arr[buckets[chstrings[chidx][i]]] = (chidx, i)
        buckets[chstrings[chidx][i]] -= 1

    # import pdb; pdb.set_trace()
    # put L-type prefixes into suf_arr in sorted order
    # inducing from the LMS-prefixes
    mark_buckets(chstrings, buckets, k)

    # handle last elt first
    chidx = len(chstrings) - 1 # last charstring
    i = len(chstrings[-1]) # one past the last elt
    if t[chidx][i - 1] == L_TYPE:
        other = chstrings[chidx][i - 1]
        suf_arr[buckets[other]] = (chidx, i - 1)
        buckets[other] += 1
    for chidx, i in suf_arr:
        if i > 0:
            if t[chidx][i - 1] == L_TYPE: # if other is l-type
                other = chstrings[chidx][i - 1]
                suf_arr[buckets[other]] = (chidx, i - 1)
                buckets[other] += 1
        elif i == 0 and chidx > 0:
            last_pos = len(chstrings[chidx - 1]) - 1
            if t[chidx - 1][-1] == L_TYPE: # if other is l-type
                other = chstrings[chidx - 1][-1]
                suf_arr[buckets[other]] = (chidx - 1, last_pos)
                buckets[other] += 1


    # Induced sort all S-types
    mark_buckets(chstrings, buckets, k, True)
    for chidx, i in reversed(suf_arr):
        if i > 0:
            if t[chidx][i - 1] == S_TYPE:
                other = chstrings[chidx][i - 1]
                suf_arr[buckets[other]] = (chidx, i - 1)
                buckets[other] -= 1
        elif i == 0 and chidx > 0:
            last_pos = len(chstrings[chidx - 1]) - 1
            if t[chidx - 1][-1] == S_TYPE: # if other is l-type
                other = chstrings[chidx - 1][-1]
                suf_arr[buckets[other]] = (chidx - 1, last_pos)
                buckets[other] -= 1

    # handle last elt last
    chidx = len(chstrings) - 1 # last charstring
    i = len(chstrings[-1]) # one past the last elt
    if t[chidx][i - 1] == S_TYPE:
        other = chstrings[chidx][i - 1]
        suf_arr[buckets[other]] = (chidx, i - 1)
        buckets[other] -= 1


    # run through (now-sorted) suf_arr and
    # write down the LMS-substrings in order
    # using lms_strings for storage
    pos = 0
    for chidx, i in suf_arr:
        if is_lms[chidx][i]:
            lms_strings[pos] = (chidx, i)
            pos += 1
    
    return lms_strings

    
def find_subrs(font):
    """
    Find subroutines using Suffix Arrays.

    Returns: nothing for now
    """
    # 2-level array of charstrings:
    #   The first level separates by glyph
    #   The second level separates by token
    #       in a glyph's charstring 
    chstrings = []

    glyph_set = font.getGlyphSet()
    glyph_set_keys = glyph_set.keys()

    keymap = {} # maps charstring tokens -> simple integer alphabet
    rev_keymap = [] # reversed keymap
    next_key = 0

    for k in glyph_set_keys:
        char_string = glyph_set[k]
        char_string.decompile()
        program = []
        for tok in char_string.program:
            if not tok in keymap:
                keymap[tok] = next_key
                rev_keymap.append(tok)
                next_key += 1
            program.append(keymap[tok])

        chstrings.append(tuple(program))

    suf_arr = get_suffix_array(chstrings)





# BASIC VERSION:
class CharSubString(object):
    """
    Legacy. Contains a substring of a charstring 
    for use in the simple subroutinizing algorithm.
    """
    glyph_idx = None
    # these are the start and stop of the substrings
    # relative to the start of the glyph's program
    start = None
    stop = None # exclusive

    def __init__(self, glyph_idx=None, start=None, stop=None):
        self.glyph_idx = glyph_idx
        self.start = start
        self.stop = stop
        super(CharSubString, self).__init__()

    def is_valid(self):
        return self.glyph_idx != None and self.start != None and self.stop != None

    def is_empty(self):
        return self.stop <= self.start

    def __repr__(self):
        return "<SubString: %d, %d:%d>" % (self.glyph_idx, self.start, self.stop)

class GlyphCharString(object):
    """
    Legacy. Holds both the internized charstring along with the
    glyph id from whence it come (ie cid1234).
    """
    glyph_key = None # glyh id (string)
    charstring = None # charstring (tuple)

    def __init__(self, glyph_key, program):
        self.glyph_key = glyph_key
        self.charstring = program
        super(GlyphCharString, self).__init__()

    def __iter__(self):
        return self.charstring.__iter__()

    def __getitem__(self, key):
        return self.charstring[key]

    def __repr__(self):
        return self.glyph_key

def simple_find_subrs(font):
    """
    Finds subroutines using simple FontForge-style algorithm

    Returns: a sorted list of `CharSubStringSet`s 
    """
    bytecode = []
    chstrings = []
    glyph_set = font.getGlyphSet()

    for k in glyph_set.keys():
        char_string = glyph_set[k]
        bytecode.append(char_string.bytecode)
        char_string.decompile()
        chstrings.append(GlyphCharString(k, char_string.program))

    substrings = []
    for glyph_idx, prgm in enumerate(chstrings):
        cur_span = CharSubString(glyph_idx, 0)
        break_after_next = False
        last_op = -1
        for pos, tok in enumerate(prgm):
            if type(tok) == str and tok[-6:] == 'moveto':
                cur_span.stop = last_op + 1
                if not cur_span.is_empty():
                    substrings.append(cur_span)
                cur_span = CharSubString(glyph_idx, pos + 1)
            elif tok == 'hintmask':
                break_after_next = True
            elif type(tok) == str or break_after_next:
                last_op = pos
        if last_op - cur_span.start >= 0:
            cur_span.stop = last_op + 1
            substrings.append(cur_span)

    matches = {}
    for substr in substrings:
        part = tuple(chstrings[substr.glyph_idx][substr.start:substr.stop])
        if part in matches:
            matches[part].locations.append((substr.glyph_idx, substr.start))
        else:
            matches[part] = CharSubStringSet(substr.stop - substr.start, 
                                [(substr.glyph_idx, substr.start)])

    match_items = matches.items()
    match_items.sort(key=lambda i: i[1].length * len(i[1].locations), reverse=True)

    return match_items

# ---------------------------


def _test():
    """
    >>> from testData import *
    >>> induce_sort_lms(chstrings, suf_arr, t, buckets, k)
    [(1, 5), (1, 1), (0, 2)]
    >>> induce_sort_lms(chstrings2, suf_arr, t2, buckets, k)
    [(0, 10), (0, 6), (0, 2)]
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

    # simple_subrs = simple_find_subrs(font)
    # print('Simple answer:')
    # print([(i[0], i[1].length, len(i[1].locations)) for i in simple_subrs[:10]])

    subrs = find_subrs(font) # prints some stuff

