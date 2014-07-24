#!/usr/bin/env python

import StringIO
import array
import struct
import subprocess
import sys
from fontTools.ttLib import TTFont

def write_data(td, f):
    """Writes CharStrings and FDSelect from the TopDict td into the file f."""
    td.CharStrings.charStringsIndex.getCompiler(td.strings, None).toFile(f)
    try:
        fdselect = struct.pack('b', len(td.FDArray)) + array.array('B', list(td.FDSelect)).tostring()
    except AttributeError:
        fdselect = struct.pack('b', 1) + array.array('B', [-2]).tostring()
    f.write(fdselect)

if __name__ == '__main__':
    f = TTFont(sys.argv[1])
    td = f['CFF '].cff.topDictIndex[0]
    print("# of charstrings == %d" % len(td.CharStrings))

    p = subprocess.Popen(['./cffCompressor'], stdin=subprocess.PIPE)
    write_data(td, p.stdin)
    p.communicate()
