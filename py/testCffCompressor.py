import unittest, random, sys
import cffCompressor
from fontTools.ttLib import TTFont
from testDummy import DummyGlyphSet

class TestCffCompressor(unittest.TestCase):

    def setUp(self):
        self.glyph_set = DummyGlyphSet({'a': (0, 1, 20, 21, 22, 2), 'b': (7, 0, 1, 20, 21, 22, 2), 'c': (0, 1, 20, 21, 22, 9, 3, 17)})
        self.sf = cffCompressor.SubstringFinder(self.glyph_set)

        self.short_sf = cffCompressor.SubstringFinder(DummyGlyphSet({'a': (1, 2, 3), 'b': (8, 1, 4)}))

        self.rand_gs = DummyGlyphSet()
        num_glyphs = random.randint(5, 20)
        for i in range(num_glyphs):
            length = random.randint(2, 30)
            self.rand_gs[i] = tuple(random.randint(0, 100) for _ in range(length))
        self.random_sf = cffCompressor.SubstringFinder(DummyGlyphSet(self.rand_gs))

        length = 3
        locations = [(0, 0), (1, 4)]
        charstrings = [(348, 374, 'rmoveto', 'endchar'), (123, -206, -140, 'hlineto', 348, 374, 'rmoveto', 'endchar')]

        self.cand_subr = cffCompressor.CandidateSubr(length, locations[0], 2, charstrings)

        self.empty_compreffor = cffCompressor.Compreffor(None, test_mode=True)

    def test_iterative_encode(self):
        """Test iterative_encode function"""

        ans = self.empty_compreffor.iterative_encode(self.glyph_set)
        self.assertIsInstance(ans, dict)

        encs = ans["glyph_encodings"]

        expected_subr_length = 5 # subr is (0, 1, 20, 21, 22)

        for glyph_enc in encs.itervalues():
            self.assertTrue(any(cs[1].length == expected_subr_length for cs in glyph_enc))


    def test_get_substrings_all(self):
        """Test get_substrings without restrictions"""

        ans = [s.value() for s in self.sf.get_substrings(0, False)]

        expected_values = [(0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4), (1, 2, 3, 4, 5), (1, 2, 3, 4), \
                            (2, 3, 4, 5), (2, 3, 4), (3, 4, 5), (3, 4), (4, 5), (4,), (5,)]

        self.assertEqual(ans, expected_values)

    def test_get_substrings_standard(self):
        """Check to make sure all substrings have freq >= 2 and positive savings"""
        ans = self.sf.get_substrings()

        for substr in ans:
            self.assertTrue(substr.freq >= 2)
            self.assertTrue(substr.subr_saving() > 0)

    def test_get_suffixes(self):
        """Test the results of suffix array construction."""

        ans = self.short_sf.get_suffixes()

        self.assertEqual(ans, [(0, 0), (1, 1), (0, 1), (0, 2), (1, 0), (1, 2)])

    def test_get_suffixes_random(self):
        """Check suffix array invariants on random input"""

        ans = self.random_sf.get_suffixes()

        # check there are the right number of suffixes
        expected_num = sum([len(chstring) for chstring in self.rand_gs.values()])
        actual_num = len(ans)
        self.assertEqual(actual_num, expected_num)

        # check that the order is correct
        last_glidx, last_tidx = ans[0]
        last = self.random_sf.data[last_glidx][last_tidx:]
        for glyph_idx, tok_idx in ans[1:]:
            current = self.random_sf.data[glyph_idx][tok_idx:]
            self.assertTrue(last <= current)

    def test_get_lcp(self):
        """Test the lcp array generation"""

        expected = [0, 6, 5, 0, 5, 4, 0, 4, 3, 0, 3, 2, 0, 2, 1, 0, 1, 0, 0, 0, 0]

        self.assertEqual(self.sf.get_lcp(), expected)

    def test_human_size(self):
        """Test the human_size function for various numbers of bytes"""

        human_size = cffCompressor.human_size

        self.assertEqual(human_size(2), "2.0 bytes")
        self.assertEqual(human_size(2050), "2.0 KB")
        self.assertEqual(human_size(3565158), "3.4 MB")
        self.assertEqual(human_size(6120328397), "5.7 GB")

    def test_update_program_local(self):
        """Test update_program with only one replacement"""

        program = [7, 2, 10, 4, 8, 7, 0]
        substr = cffCompressor.CandidateSubr(3, (0, 1))
        substr._position = 5
        substr._fdidx = [0]
        substr._global = False
        encoding = [(1, substr)]
        bias = 0

        self.empty_compreffor.update_program(program, encoding, bias, [bias], 0)

        self.assertEqual(program, [7, 5, "callsubr", 8, 7, 0])

    def test_update_program_global(self):
        """Test update_program with only one replacement"""

        program = [7, 2, 10, 4, 8, 7, 0]
        substr = cffCompressor.CandidateSubr(3, (0, 1))
        substr._position = 5
        substr._fdidx = [0]
        substr._global = True
        encoding = [(1, substr)]
        bias = 0

        self.empty_compreffor.update_program(program, encoding, bias, [bias], 0)

        self.assertEqual(program, [7, 5, "callgsubr", 8, 7, 0])

    def test_update_program_multiple(self):
        """Test update_program with two replacements"""

        program = [7, 2, 10, 4, 8, 7, 0]
        substr = cffCompressor.CandidateSubr(3, (0, 1))
        substr._position = 5
        substr._global = True
        substr2 = cffCompressor.CandidateSubr(2, (0, 5))
        substr2._position = 21
        substr2._global = True
        encoding = [(1, substr), (5, substr2)]
        bias = 0

        self.empty_compreffor.update_program(program, encoding, bias, [bias], 0)

        self.assertEqual(program, [7, 5, "callgsubr", 8, 21, "callgsubr"])

    # TODO: make this test actually work
    def test_multiple_nested_subr_calls(self):
        """Test to make sure we can handle nested subrs. This is really just
        a case to make check we're encoding optimally."""

        glyph_set = {'a': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20),
                     'b': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21),
                     'c': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22),
                     'd': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 23),
                     'e': (0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 24),
                     'f': (0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 25),
                     'g': (0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 26),}
        glyph_set = DummyGlyphSet(glyph_set)

        ans = self.empty_compreffor.iterative_encode(glyph_set)
        print(ans["glyph_encodings"])
        print(ans["lsubrs"])
        print([s._encoding for s in ans["lsubrs"][0]])

    def test_expand_hintmask_single_middle(self):
        """Non-edge usage of expand_hintmask"""

        data = [1, 2, 3, 4, 5, ('hintmask', 7), 8, 9, 10]
        self.empty_compreffor.expand_hintmask(data)
        self.assertEqual(data, [1, 2, 3, 4, 5, 'hintmask', 7, 8, 9, 10])

    def test_expand_hintmask_multi_middle(self):
        """Non-edge usage of expand_hintmask with two items"""

        data = [1, ('hintmask', 3), 4, 5, ('hintmask', 7), 8, 9, 10]
        self.empty_compreffor.expand_hintmask(data)
        self.assertEqual(data, [1, 'hintmask', 3, 4, 5, 'hintmask', 7, 8, 9, 10])

    def test_expand_hintmask_multi_end(self):
        """Non-edge usage of expand_hintmask with two items, one at end"""

        data = [1, 2, 3, 4, 5, ('hintmask', 7), 8, ('hintmask', 10)]
        self.empty_compreffor.expand_hintmask(data)
        self.assertEqual(data, [1, 2, 3, 4, 5, 'hintmask', 7, 8, 'hintmask', 10])

    def test_collapse_hintmask_single_middle(self):
        """Non-edge usage of collapse_hintmask"""

        data = [1, 2, 3, 4, 5, 'hintmask', 7, 8, 9, 10]
        self.empty_compreffor.collapse_hintmask(data)
        self.assertEqual(data, [1, 2, 3, 4, 5, ('hintmask', 7), 8, 9, 10])

    def test_collapse_hintmask_multi_middle(self):
        """Non-edge usage of collapse_hintmask with two items"""

        data = [1, 'hintmask', 3, 4, 5, 'hintmask', 7, 8, 9, 10]
        self.empty_compreffor.collapse_hintmask(data)
        self.assertEqual(data, [1, ('hintmask', 3), 4, 5, ('hintmask', 7), 8, 9, 10])

    def test_collapse_hintmask_multi_end(self):
        """Non-edge usage of collapse_hintmask with two items, one at end"""

        data = [1, 2, 3, 4, 5, 'hintmask', 7, 8, 'hintmask', 10]
        self.empty_compreffor.collapse_hintmask(data)
        self.assertEqual(data, [1, 2, 3, 4, 5, ('hintmask', 7), 8, ('hintmask', 10)])

    def test_tokenCost(self):
        """Make sure single tokens can have their cost calculated"""

        tokenCost = cffCompressor.tokenCost

        self.assertEqual(tokenCost('hlineto'), 1)
        self.assertEqual(tokenCost('flex'), 2)
        self.assertEqual(tokenCost(107), 1)
        self.assertEqual(tokenCost(108), 2)

    def test_candidatesubr_len(self):
        """Make sure len returns the correct length"""

        self.assertEqual(len(self.cand_subr), 3)

    def test_candidatesubr_value(self):
        """Make sure the value is correct"""

        expected_value = (348, 374, 'rmoveto')

        self.assertEqual(self.cand_subr.value(), expected_value)




def test_compression_integrity(orignal_file, compressed_file):
    """Compares two fonts to confirm they are functionally equivalent"""

    import fontTools
    import fontTools.subset

    orig_font = fontTools.ttLib.TTFont(orignal_file)
    orig_gset = orig_font.getGlyphSet()
    comp_font = fontTools.ttLib.TTFont(compressed_file)
    comp_gset = comp_font.getGlyphSet()

    assert orig_gset.keys() == comp_gset.keys()

    # decompress the compressed font
    options = fontTools.subset.Options()
    options.decompress = True
    subsetter = fontTools.subset.Subsetter(options=options)
    subsetter.populate(glyphs=comp_font.getGlyphOrder())
    subsetter.subset(orig_font)
    subsetter.subset(comp_font)

    passed = True
    for g in orig_gset.keys():
        orig_gset[g].decompile()
        if not (orig_gset[g].program == comp_gset[g].program):
            print "Difference found in glyph '%s'" % (g,)
            passed = False

    if passed:
        print "Fonts match!"
        return True
    else:
        print "Fonts have differences :("
        return False

def test_call_depth(compressed_file):
    """Runs `check_cff_call_depth` on a file"""

    import fontTools

    f = fontTools.ttLib.TTFont(compressed_file)

    return check_cff_call_depth(f["CFF "].cff)

def check_cff_call_depth(cff):
    """Checks that the Charstrings in the provided CFFFontSet
    obey the rules for subroutine nesting."""

    import fontTools
    psCharStrings = fontTools.misc.psCharStrings

    SUBR_NESTING_LIMIT = 10

    assert len(cff.topDictIndex) == 1

    td = cff.topDictIndex[0]

    class track_info: pass

    track_info.max_for_all = 0

    gsubrs = cff.GlobalSubrs
    gbias = psCharStrings.calcSubrBias(gsubrs)

    def follow_program(program, depth, subrs):
        bias = psCharStrings.calcSubrBias(subrs)

        if len(program) > 0:
            last = program[0]
            for tok in program[1:]:
                if tok == "callsubr":
                    assert type(last) == int
                    next_subr = subrs[last + bias]
                    if (not hasattr(next_subr, "_max_call_depth") or 
                            next_subr._max_call_depth < depth + 1):
                        increment_subr_depth(next_subr, depth + 1, subrs)
                elif tok == "callgsubr":
                    assert type(last) == int
                    next_subr = gsubrs[last + gbias]
                    if (not hasattr(next_subr, "_max_call_depth") or 
                            next_subr._max_call_depth < depth + 1):
                        increment_subr_depth(next_subr, depth + 1, subrs)
                last = tok
        else:
            print "Compiled subr encountered"

    def increment_subr_depth(subr, depth, subrs=None):
        if not hasattr(subr, "_max_call_depth") or subr._max_call_depth < depth:
            subr._max_call_depth = depth

        if subr._max_call_depth > track_info.max_for_all:
            track_info.max_for_all = subr._max_call_depth

        program = subr.program
        follow_program(program, depth, subrs)        

    for cs in td.CharStrings.values():
        cs.decompile()
        follow_program(cs.program, 0, cs.private.Subrs)

    if track_info.max_for_all <= SUBR_NESTING_LIMIT:
        print "Subroutine nesting depth ok! [max nesting depth of %d]" % track_info.max_for_all
        return track_info.max_for_all
    else:
        print "Subroutine nesting depth too deep :( [max nesting depth of %d]" % track_info.max_for_all
        return track_info.max_for_all
