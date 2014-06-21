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

        self.csss = cffCompressor.CandidateSubr(length, locations, charstrings)

    def test_iterative_encode(self):
        """Test iterative_encode function"""

        ans = cffCompressor.iterative_encode(self.glyph_set, test_mode=True)
        self.assertIsInstance(ans, dict)

        # don't care about CandidateSubr objects, delete them
        for k in ans.keys():
            enc = [i[:2] for i in ans[k]]
            ans[k] = tuple(enc)

        self.assertEqual(ans, {'a': ((0, 5),), 'b': ((1, 6),), 'c': ((0, 5),)})

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

    def test_tokenCost(self):
        """Make sure single tokens can have their cost calculated"""

        tokenCost = cffCompressor.CandidateSubr.tokenCost

        self.assertEqual(tokenCost('hlineto'), 1)
        self.assertEqual(tokenCost('flex'), 2)
        self.assertEqual(tokenCost(107), 1)
        self.assertEqual(tokenCost(108), 2)

    def test_string_cost(self):
        """Ensure an entire string can have its cost calculated"""

        string_cost = cffCompressor.CandidateSubr.string_cost

        self.assertEqual(string_cost((108, 'endchar')), 3)

    def test_candidatesubr_add_location(self):
        """Test the add location function in CandidateSubr"""

        old_loc_len = len(self.csss.locations)
        old_freq = self.csss.freq
        self.assertEqual(old_loc_len, 2)
        self.assertTrue((5, 2) not in self.csss.locations)

        self.csss.add_location((5, 2))

        self.assertEqual(len(self.csss.locations) - old_loc_len, 1)
        self.assertTrue((5, 2) in self.csss.locations)
        self.assertEqual(self.csss.freq - old_freq, 1)

    def test_candidatesubr_len(self):
        """Make sure len returns the correct length"""

        self.assertEqual(len(self.csss), 3)

    def test_candidatesubr_value(self):
        """Make sure the value is correct"""

        expected_value = (348, 374, 'rmoveto')

        self.assertEqual(self.csss.value(), expected_value)
