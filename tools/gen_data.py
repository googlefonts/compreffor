#!/usr/bin/env python

"""
This script generates a data source file for the C++ compressor.

Usage:
>>> ./gen_data.py /path/to/font.otf /path/to/output
"""

import StringIO
import array
import subprocess
import sys
from fontTools.ttLib import TTFont

from compreffor.cxxCompressor import write_data

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "missing arguments"
    else:
        f = TTFont(sys.argv[1])
        td = f['CFF '].cff.topDictIndex[0]
        with open(sys.argv[2], 'w') as out_f:
            data = write_data(td)
            out_f.write(data)
