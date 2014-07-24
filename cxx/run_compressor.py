#!/usr/bin/env python

import StringIO
import array
import subprocess
import sys
from fontTools.ttLib import TTFont

if __name__ == '__main__':
    f = TTFont(sys.argv[1])
    td = f['CFF '].cff.topDictIndex[0]
    print("# of charstrings == %d" % len(td.CharStrings))

    cs_file = StringIO.StringIO()
    td.CharStrings.charStringsIndex.getCompiler(td.strings, None).toFile(cs_file)
    cstrings = cs_file.getvalue()
    cs_file.close() 
    print("# of bytes == %d" % len(cstrings))

    try:
        fdselect = 'y' + array.array('B', list(td.FDSelect)).tostring()
    except AttributeError:
        fdselect = 'n' + array.array('B', [-2]).tostring()

    p = subprocess.Popen(['./cffCompressor'], stdin=subprocess.PIPE)
    p.stdin.write(cstrings)
    p.stdin.write(fdselect)
