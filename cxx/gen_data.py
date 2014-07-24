#!/usr/bin/env python

import StringIO
import array
import subprocess
import sys
from fontTools.ttLib import TTFont

if __name__ == '__main__':
    f = TTFont(sys.argv[1])
    with open(sys.argv[2], 'w') as out_f:
        td = f['CFF '].cff.topDictIndex[0]
        print("# of charstrings == %d" % len(td.CharStrings))

        td.CharStrings.charStringsIndex.getCompiler(td.strings, None).toFile(out_f)

        try:
            fdselect = 'y' + array.array('B', list(td.FDSelect)).tostring()
        except AttributeError:
            fdselect = 'n' + array.array('B', [-2]).tostring()
            
        out_f.write(fdselect)