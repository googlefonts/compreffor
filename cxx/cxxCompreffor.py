#!/usr/bin/env python

import StringIO
import argparse
import array
import struct
import subprocess
import sys
import os
from fontTools.ttLib import TTFont

from cffCompressor import Compreffor, CandidateSubr, tokenCost, human_size

NSUBRS_LIMIT = 65533
SUBR_NEST_LIMIT  = 10

class IdKeyMap(object):
        def __getitem__(self, tok):
            return tok

class SimpleCandidateSubr(CandidateSubr):
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
    """Writes CharStrings and FDSelect from the TopDict td into a string buffer."""
    out = StringIO.StringIO()
    td.CharStrings.charStringsIndex.getCompiler(td.strings, None).toFile(out)
    try:
        fdselect = struct.pack('B', len(td.FDArray)) + array.array('B', list(td.FDSelect)).tostring()
    except AttributeError:
        fdselect = struct.pack('B', 1)
    out.write(fdselect)
    return out.getvalue()

def get_encoding(data_buffer, subrs):
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
    print("PYTHON>>> %d subrs read" % num_subrs)

    # process glyph encodings
    glyph_encodings = []
    for i in range(len(td.CharStrings)):
        enc, num_read = get_encoding(results[pos:], subrs)
        pos += num_read
        glyph_encodings.append(enc)

    assert pos == len(results)
    return (subrs, glyph_encodings)

def compreff(font):
    td = font['CFF '].cff.topDictIndex[0]
    print("PYTHON>>> # of charstrings == %d" % len(td.CharStrings))

    p = subprocess.Popen(
                        [os.path.join(os.path.dirname(__file__), 'cffCompressor')],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE)
    input_data = write_data(td)
    results, _ = p.communicate(input=input_data)
    subrs, glyph_encodings = read_data(td, results)

    for cs in td.CharStrings.values():
        cs.decompile()

    # in order of charset
    chstrings = map(lambda x: x.program, td.CharStrings.values())
    map(lambda x: Compreffor.collapse_hintmask(x), chstrings)

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
                            NSUBRS_LIMIT,
                            SUBR_NEST_LIMIT)

    encoding = dict(zip(td.charset, glyph_encodings))

    Compreffor.apply_subrs(td, encoding, gsubrs, lsubrs)

def main(filename=None, comp_fname=None, test=False, decompress=False,
         verbose=False, check=False, comp_func=None, **comp_kwargs):
    if test:
        pass

    if filename and comp_fname == None:
        font = TTFont(filename)
        orig_size = os.path.getsize(filename)

        if decompress:
            from fontTools import subset
            options = subset.Options()
            options.decompress = True
            subsetter = subset.Subsetter(options=options)
            subsetter.populate(glyphs=font.getGlyphOrder())
            subsetter.subset(font)

        if verbose:
            print("Compressing font through iterative_encode:")
        out_name = "%s.compressed%s" % os.path.splitext(filename)

        compreff(font)

        # save compressed font
        font.save(out_name)

        # save CFF version
        font['CFF '].cff.compile(open("%s.cff" % os.path.splitext(out_name)[0], 'w'), None)

        comp_size = os.path.getsize(out_name)
        print("Compressed to %s -- saved %s" % 
                (os.path.basename(out_name), human_size(orig_size - comp_size)))

    if check:
        from testCffCompressor import test_compression_integrity, test_call_depth

        if comp_fname == None:
            test_compression_integrity(filename, out_name)
            test_call_depth(out_name)
        else:
            test_compression_integrity(filename, comp_fname)
            test_call_depth(comp_fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description="""FontTools Compreffor will take a CFF-flavored
                                       OpenType font and automatically detect
                                       repeated routines and generate subroutines
                                       to minimize the disk space needed to
                                       represent a font.""")
    parser.add_argument("filename", help="the path to the font file", nargs="?")
    parser.add_argument("comp_fname", nargs="?", metavar="compressed-file",
                        help="the path to the compressed file. if this is given"
                             " with the -c flag, it will be checked against "
                             " `filename`.")
    # parser.add_argument("-t", "--test", required=False, action="store_true",
    #                     default=False, help="run test cases")
    # parser.add_argument("-s", "--status", required=False, action="store_true",
    #                     dest="print_status", default=False)
    # parser.add_argument("-v", "--verbose", required=False, action="store_true",
    #                     dest="verbose", default=False)
    parser.add_argument("-c", "--check", required=False, action="store_true",
                        help="verify that the outputted font is valid and "
                             "functionally equivalent to the input")
    parser.add_argument("-d", "--decompress", required=False, action="store_true",
                        help="decompress source before compressing (necessary if "
                             "there are subroutines in the source)")
    # parser.add_argument("-n", "--nrounds", required=False, type=int,
    #                     help="the number of iterations to run the algorithm"
    #                          " (defaults to 4)")
    # parser.add_argument("-m", "--maxsubrs", required=False, type=int,
    #                     dest="nsubrs_limit", help="limit to the number of "
    #                                               " subroutines per INDEX"
    #                                               " (defaults to 64K)")

    kwargs = vars(parser.parse_args())

    main(**kwargs)
