import unittest, random, sys
import cffCompressor
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

    def test_iterative_encode(self):
        """Test iterative_encode function"""

        ans = cffCompressor.iterative_encode(self.glyph_set, test_mode=True)
        self.assertIsInstance(ans, dict)

        # don't care about CandidateSubr objects, just take their length
        encs = ans["glyph_encodings"]
        for k in encs.keys():
            enc = [(i[0], i[1].length) for i in encs[k]]
            encs[k] = tuple(enc)

        self.assertEqual(encs, {'a': ((0, 5),), 'b': ((1, 5),), 'c': ((0, 5),)})

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

    def test_update_program(self):
        """Test update_program with only one replacement"""

        program = [7, 2, 10, 4, 8, 7, 0]
        substr = cffCompressor.CandidateSubr(3, (0, 1))
        substr._position = 5
        encoding = [(1, substr)]
        bias = 0

        cffCompressor.update_program(program, encoding, bias)

        self.assertEqual(program, [7, 5, "callgsubr", 8, 7, 0])

    def test_update_program_multiple(self):
        """Test update_program with two replacements"""

        program = [7, 2, 10, 4, 8, 7, 0]
        substr = cffCompressor.CandidateSubr(3, (0, 1))
        substr._position = 5
        substr2 = cffCompressor.CandidateSubr(2, (0, 5))
        substr2._position = 21
        encoding = [(1, substr), (5, substr2)]
        bias = 0

        cffCompressor.update_program(program, encoding, bias)

        self.assertEqual(program, [7, 5, "callgsubr", 8, 21, "callgsubr"])

    # ---

    def test_tokenCost(self):
        """Make sure single tokens can have their cost calculated"""

        tokenCost = cffCompressor.CandidateSubr.tokenCost

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
