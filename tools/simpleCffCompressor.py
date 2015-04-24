#!/usr/bin/env python
#
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from compreffor import pyCompressor

class SimpleSubstringFinder(pyCompressor.SubstringFinder):
    def get_substrings(self, min_freq=2, check_positive=True, sort_by_length=False):
        movetos = set()
        for idx, tok in enumerate(self.rev_keymap):
            if isinstance(tok, basestring) and tok[-6:] == "moveto":
                movetos.add(idx)
        try:
            hmask = self.rev_keymap.index("hintmask")
        except ValueError:
            hmask = None

        matches = {}

        for glyph_idx, program in enumerate(self.data):
            cur_start = 0
            last_op = -1
            for pos, tok in enumerate(program):
                if tok in movetos:
                    stop = last_op + 1
                    if stop - cur_start > 0:
                        if program[cur_start:stop] in matches:
                            matches[program[cur_start:stop]].freq += 1
                        else:
                            span = pyCompressor.CandidateSubr(stop - cur_start,
                                                               (glyph_idx, cur_start),
                                                               1,
                                                               self.data,
                                                               self.cost_map)
                            matches[program[cur_start:stop]] = span
                    cur_start = pos + 1
                elif tok == hmask:
                    last_op = pos + 1
                elif type(self.rev_keymap[tok]) == str:
                    last_op = pos

        constraints = lambda s: (s.freq >= min_freq and 
                                (s.subr_saving() > 0 or not check_positive))
        self.substrings = filter(constraints, matches.values())
        if sort_by_length:
            self.substrings.sort(key=lambda s: len(s))
        else:
            self.substrings.sort(key=lambda s: s.subr_saving(), reverse=True)
        return self.substrings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subroutinize a font.')
    parser.add_argument('filename', help='Where to find the font', nargs='*')
    parser.add_argument('-t', required=False, action='store_true',
                        dest='test', default=False)
    parser.add_argument('-v', required=False, action='store_true',
                        dest='verbose_test', default=False)
    parser.add_argument('-c', required=False, action='store_true',
                        dest='check', default=False)

    kwargs = vars(parser.parse_args())

    pyCompressor.SubstringFinder = SimpleSubstringFinder
    pyCompressor.main(**kwargs)
