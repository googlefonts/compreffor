#!/usr/bin/env python

import StringIO
import array
import struct
import subprocess
import sys
import os
from fontTools.ttLib import TTFont

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

def read_data(td, result_string):
    results = array.array("B", result_string)
    num_subrs = struct.unpack_from('<I', results[:4])[0]

    # process subrs
    subr_code = []
    pos = 4
    while results[pos] != 0:
        cur_buffer = ""
        while results[pos] != 0:
            cur_buffer += chr(results[pos])
            pos += 1
        subr_code.append(cur_buffer)
        pos += 1
    print("PYTHON>>> %d subrs read" % num_subrs)

    # process glyph encodings
    glyph_encodings = []
    for i in range(len(td.CharStrings)):
        num_calls = results[pos]
        pos += 1
        enc = []
        for j in range(num_calls):
            insertion_pos = struct.unpack_from('<I', results[pos:pos+4])[0]
            pos += 4
            subr_index = struct.unpack_from('<I', results[pos:pos+4])[0]
            pos += 4
            enc.append((insertion_pos, subr_index))
        glyph_encodings.append(enc)

    assert pos == len(results)
    return (subr_code, glyph_encodings)

def compreff(font):
    td = font['CFF '].cff.topDictIndex[0]
    print("PYTHON>>> # of charstrings == %d" % len(td.CharStrings))

    p = subprocess.Popen(
                        [os.path.join(os.path.dirname(__file__), 'cffCompressor')],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE)
    input_data = write_data(td)
    results, _ = p.communicate(input=input_data)
    subr_code, glyph_encodings = read_data(td, results)
    print glyph_encodings

if __name__ == '__main__':
    f = TTFont(sys.argv[1])
    compreff(f)
