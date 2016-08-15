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

from fontTools import ttLib
from fontTools.misc import psCharStrings
import fontTools.subset


def check_compression_integrity(orignal_file, compressed_file):
    """Compares two fonts to confirm they are functionally equivalent"""

    orig_font = ttLib.TTFont(orignal_file)
    orig_gset = orig_font.getGlyphSet()
    comp_font = ttLib.TTFont(compressed_file)
    comp_gset = comp_font.getGlyphSet()

    assert orig_gset.keys() == comp_gset.keys()

    # decompress the compressed font
    options = fontTools.subset.Options()
    options.desubroutinize = True
    subsetter = fontTools.subset.Subsetter(options=options)
    subsetter.populate(glyphs=comp_font.getGlyphOrder())
    subsetter.subset(orig_font)
    subsetter.subset(comp_font)

    passed = True
    for g in orig_gset.keys():
        orig_glyph = orig_gset[g]._glyph
        comp_glyph = comp_gset[g]._glyph
        orig_glyph.decompile()
        if not (orig_glyph.program == comp_glyph.program):
            print("Difference found in glyph '%s'" % (g,))
            passed = False

    if passed:
        print("Fonts match!")
        return True
    else:
        print("Fonts have differences :(")
        return False


def check_call_depth(compressed_file):
    """Runs `check_cff_call_depth` on a file"""

    f = ttLib.TTFont(compressed_file)

    return check_cff_call_depth(f["CFF "].cff)


def check_cff_call_depth(cff):
    """Checks that the Charstrings in the provided CFFFontSet
    obey the rules for subroutine nesting."""

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
            print("Compiled subr encountered")

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
        print("Subroutine nesting depth ok! [max nesting depth of %d]" % track_info.max_for_all)
        return track_info.max_for_all
    else:
        print("Subroutine nesting depth too deep :( [max nesting depth of %d]" % track_info.max_for_all)
        return track_info.max_for_all
