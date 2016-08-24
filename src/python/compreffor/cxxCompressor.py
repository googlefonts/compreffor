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

"""
Tool to subroutinize a CFF OpenType font. Backed by a C++ binary.

This file is a bootstrap for the C++ edition of the FontTools compreffor.
It prepares the input data for the extension and reads back in the results,
applying them to the input font.

Usage (command line):
>> ./cxxCompressor.py /path/to/font.otf
# font written to /path/to/font.compressed.otf

Usage (python):
>> font = TTFont("/path/to/font.otf")
>> cxxCompressor.compreff(font)
>> font.save("/path/to/output.otf")
"""

import argparse
import array
import struct
import sys
import time
import os
from compreffor.pyCompressor import (
    Compreffor, CandidateSubr, tokenCost, human_size)
from compreffor.test.util import (
    check_compression_integrity, check_call_depth)
from compreffor import _compreffor as lib
from fontTools.misc.py23 import BytesIO
from fontTools.ttLib import TTFont


# default values:
NSUBRS_LIMIT = 65533
SUBR_NEST_LIMIT  = 10

class IdKeyMap(object):
    """A map that where every key's value is itself. Used
    as a map from simplified key space to actual key space
    in pyCompressor"""

    def __getitem__(self, tok):
        return tok

class SimpleCandidateSubr(CandidateSubr):
    """A reimplimentation of CandidateSubr to be more
    compatible with results from C++"""

    def __init__(self, length, ref_loc):
        self.length = length
        self.location = ref_loc
        self.freq = 0
        self._flatten = False
        self._global = False

    def usages(self):
        return self.freq

    frequency = usages

    def cost(self):
        try:
            return self.__cost
        except AttributeError:
            self.__cost = sum(map(tokenCost, self.value()))
            return self.__cost

    def encoding(self):
        return self._encoding

def write_data(td):
    """Writes CharStrings and FDSelect from the TopDict td into a string
    that is easily readable."""

    out = BytesIO()
    td.CharStrings.charStringsIndex.getCompiler(td.strings, None).toFile(out)
    try:
        fdselect = struct.pack('B', len(td.FDArray)) + array.array('B', list(td.FDSelect)).tostring()
    except AttributeError:
        fdselect = struct.pack('B', 1)
    out.write(fdselect)
    return out.getvalue()

def get_encoding(data_buffer, subrs):
    """Read a charstring's encoding stream out of a string buffer response
    from cffCompressor.cc"""

    pos = 0
    num_calls = data_buffer[pos]
    pos += 1
    enc = []
    for j in range(num_calls):
        insertion_pos = struct.unpack_from('<I', data_buffer[pos:pos+4])[0]
        pos += 4
        subr_index = struct.unpack_from('<I', data_buffer[pos:pos+4])[0]
        pos += 4
        subrs[subr_index].freq += 1
        enc.append((insertion_pos, subrs[subr_index]))
    return enc, pos

def read_data(td, result_string):
    """Read the output of cffCompressor.cc into Python data
    structures."""

    results = array.array("B", result_string)
    num_subrs = struct.unpack_from('<I', results[:4])[0]

    # process subrs
    subrs = []
    pos = 4
    for i in range(num_subrs):
        glyph_idx = struct.unpack_from('<I', results[pos:pos+4])[0]
        pos += 4
        tok_idx = struct.unpack_from('<I', results[pos:pos+4])[0]
        pos += 4
        subr_len = struct.unpack_from('<I', results[pos:pos+4])[0]
        pos += 4
        subrs.append(SimpleCandidateSubr(subr_len, (glyph_idx, tok_idx)))
    for i in range(num_subrs):
        enc, num_read = get_encoding(results[pos:], subrs)
        pos += num_read
        subrs[i]._encoding = enc

    # process glyph encodings
    glyph_encodings = []
    for i in range(len(td.CharStrings)):
        enc, num_read = get_encoding(results[pos:], subrs)
        pos += num_read
        glyph_encodings.append(enc)

    assert pos == len(results)
    return (subrs, glyph_encodings)

def interpret_data(td, results):
    """Interpret the result array from a lib.compreff call to
    produce Python data structures."""

    class MutableSpace: pass
    MutableSpace.pos = 0
    def pop_result():
        ans = results[MutableSpace.pos]
        MutableSpace.pos += 1
        return ans

    num_subrs = pop_result()

    # process subrs
    subrs = []
    for i in range(num_subrs):
        glyph_idx = pop_result()
        tok_idx = pop_result()
        subr_len = pop_result()
        subrs.append(SimpleCandidateSubr(subr_len, (glyph_idx, tok_idx)))

    def pop_encoding():
        num_calls = pop_result()
        enc = []
        for j in range(num_calls):
            insertion_pos = pop_result()
            subr_index = pop_result()
            subrs[subr_index].freq += 1
            enc.append((insertion_pos, subrs[subr_index]))
        return enc

    for i in range(num_subrs):
        enc = pop_encoding()
        subrs[i]._encoding = enc

    # process glyph encodings
    glyph_encodings = []
    for i in range(len(td.CharStrings)):
        enc = pop_encoding()
        glyph_encodings.append(enc)

    return (subrs, glyph_encodings)

def compreff(font, verbose=False, **kwargs):
    """Main function that compresses `font`, a TTFont object,
    in place.
    """

    full_start_time = start_time = time.time()

    assert len(font['CFF '].cff.topDictIndex) == 1

    td = font['CFF '].cff.topDictIndex[0]

    if verbose:
        print("Preparing external call...")
        start_time = time.time()

    call = [os.path.join(os.path.abspath(os.path.dirname(__file__)), "cffCompressor")]

    if 'nrounds' in kwargs and kwargs.get('nrounds') != None:
        call.extend(['--nrounds', str(kwargs.get('nrounds'))])

    max_subrs = NSUBRS_LIMIT
    if 'nsubrs_limit' in kwargs and kwargs.get('nsubrs_limit') != None:
        max_subrs = kwargs.get('nsubrs_limit')

    input_data = write_data(td)
    if verbose:
        print("Produced data for C++ (delta %gs)" % (time.time() - start_time))
        start_time = time.time()
    results = lib.compreff(input_data, 4)
    if verbose:
        print("Lib call returned (delta %gs)" % (time.time() - start_time))
        start_time = time.time()
    subrs, glyph_encodings = interpret_data(td, results)

    if verbose:
        print("Extracted results (delta %gs)" % (time.time() - start_time))
        start_time = time.time()

    for cs in td.CharStrings.values():
        cs.decompile()

    if verbose:
        print("Decompiled charstrings (delta %gs)" % (time.time() - start_time))
        start_time = time.time()

    # in order of charset
    chstrings = [x.program for x in td.CharStrings.values()]
    for cs in chstrings:
        Compreffor.collapse_hintmask(cs)

    for s in subrs:
        s.chstrings = chstrings

    if hasattr(td, 'FDSelect'):
        fdselect = lambda g: td.CharStrings.getItemAndSelector(g)[1]
        fdlen = len(td.FDArray)
    else:
        fdselect = None
        fdlen = 1

    gsubrs, lsubrs = Compreffor.process_subrs(
                            td.charset,
                            glyph_encodings,
                            fdlen,
                            fdselect,
                            subrs,
                            IdKeyMap(),
                            max_subrs,
                            SUBR_NEST_LIMIT)

    encoding = dict(zip(td.charset, glyph_encodings))

    Compreffor.apply_subrs(td, encoding, gsubrs, lsubrs)

    if verbose:
        print("Finished post-processing (delta %gs)" % (time.time() - start_time))
        print("Total time: %gs" % (time.time() - full_start_time))

def main(filename=None, comp_fname=None, test=False, decompress=False,
         verbose=False, check=False, generate_cff=False, recursive=False,
         **comp_kwargs):
    if test:
        pass

    if filename and comp_fname == None:
        def handle_font(font_name):
            font = TTFont(font_name)

            td = font['CFF '].cff.topDictIndex[0]
            no_subrs = lambda fd: hasattr(fd, 'Subrs') and len(fd.Subrs) > 0
            priv_subrs = (hasattr(td, 'FDArray') and
                          any(no_subrs(fd) for fd in td.FDArray))
            if len(td.GlobalSubrs) > 0 or priv_subrs:
                print("Warning: There are subrs in %s" % font_name)

            orig_size = os.path.getsize(font_name)

            if decompress:
                from fontTools import subset
                options = subset.Options()
                options.desubroutinize = True
                subsetter = subset.Subsetter(options=options)
                subsetter.populate(glyphs=font.getGlyphOrder())
                subsetter.subset(font)

            out_name = "%s.compressed%s" % os.path.splitext(font_name)

            compreff(font, verbose=verbose, **comp_kwargs)

            # save compressed font
            start_time = time.time()
            font.save(out_name)
            if verbose:
                print("Compiled and saved (took %gs)" % (time.time() - start_time))

            if generate_cff:
                # save CFF version
                font['CFF '].cff.compile(open("%s.cff" % os.path.splitext(out_name)[0], 'w'), None)

            comp_size = os.path.getsize(out_name)
            print("Compressed to %s -- saved %s" %
                    (os.path.basename(out_name), human_size(orig_size - comp_size)))

            if check:
                check_compression_integrity(filename, out_name)
                check_call_depth(out_name)

        if recursive:
            for root, dirs, files in os.walk(filename):
                for fname in files:
                    if os.path.splitext(fname)[1] == '.otf':
                        handle_font(fname)
        else:
            handle_font(filename)

    if check and comp_fname != None:
        check_compression_integrity(filename, comp_fname)
        check_call_depth(comp_fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description="""FontTools Compreffor will take a CFF-flavored
                                       OpenType font and automatically detect
                                       repeated routines and generate subroutines
                                       to minimize the disk space needed to
                                       represent a font.""")
    parser.add_argument('filename', help="the path to the font file", nargs='?')
    parser.add_argument('comp_fname', nargs='?', metavar='compressed-file',
                        help="the path to the compressed file. if this is given"
                             " with the -c flag, it will be checked against "
                             " `filename`.")
    # no tests yet :(
    # parser.add_argument('-t', '--test', required=False, action='store_true',
    #                     default=False, help="run test cases")
    parser.add_argument('-v', '--verbose', required=False, action='store_true',
                        dest='verbose', default=False)
    parser.add_argument('-c', '--check', required=False, action='store_true',
                        help="verify that the outputted font is valid and "
                             "functionally equivalent to the input")
    parser.add_argument('-d', '--decompress', required=False, action='store_true',
                        help="decompress source before compressing (necessary if "
                             "there are subroutines in the source)")
    parser.add_argument('-r', '--recursive', required=False, action='store_true',
                        default=False)
    parser.add_argument('-n', '--nrounds', required=False, type=int,
                        help="the number of iterations to run the algorithm"
                             " (defaults to 4)")
    parser.add_argument('-m', '--maxsubrs', required=False, type=int,
                        dest='nsubrs_limit', help="limit to the number of "
                                                  " subroutines per INDEX"
                                                  " (defaults to 64K)")
    parser.add_argument('--generatecff', required=False, action='store_true',
                        dest='generate_cff', default=False)

    kwargs = vars(parser.parse_args())

    main(**kwargs)
