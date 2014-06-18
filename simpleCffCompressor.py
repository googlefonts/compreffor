from cffCompressor import CharSubStringSet

TIMSORT_THRESHOLD = 300
MAX_TOUCHES = 3
SORT_DEPTH = 10

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

    simple_subrs = simple_find_subrs(font)
    print('Simple answer:')
    print([(i[0], i[1].length, len(i[1].locations)) for i in simple_subrs[:10]])
