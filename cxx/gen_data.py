#!/usr/bin/env python

import StringIO
import array
import subprocess
import sys
from fontTools.ttLib import TTFont

from run_compressor import write_data

if __name__ == '__main__':
    f = TTFont(sys.argv[1])
    td = f['CFF '].cff.topDictIndex[0]
    with open(sys.argv[2], 'w') as out_f:
        print("# of charstrings == %d" % len(td.CharStrings))

        write_data(td, out_f)
