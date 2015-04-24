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
Quick analysis tool for the compreffor. Compresses a directory of
fonts and writes a CSV containing various stats about the compression.

Usage (command line):
>>> ./analyzer.py /path/to/font/dir
...

A CSV will be written to the current working directory with
the name data.csv.
"""

import csv
import os
import time
from compreffor import cxxCompressor

from fontTools import subset
from fontTools.ttLib import TTFont

def sum_subrs(font):
   td = font['CFF '].cff.topDictIndex[0]
   ans = len(td.GlobalSubrs)
   try:
      ans += sum(len(fd.Private.Subrs) if hasattr(fd.Private, 'Subrs') else 0 for fd in td.FDArray)
   except AttributeError:
      pass
   return ans

if __name__ == '__main__':
   names = []
   times = []
   orig_sizes = []
   full_sizes = []
   compressed_sizes = []
   nsubrs = []
   nsubrs_orig = []

   for root, dirs, files in os.walk(os.argv[1]):
      for filename in files:
         if os.path.splitext(filename)[1] == '.otf':
            fname = os.path.join(root, filename)
            print "Handling %s" % filename

            # decompress
            print("\tDecompressing...")
            font = TTFont(fname)
            orig_subrs = sum_subrs(font)
            orig_size = os.path.getsize(fname)

            options = subset.Options()
            options.decompress = True
            subsetter = subset.Subsetter(options=options)
            subsetter.populate(glyphs=font.getGlyphOrder())
            subsetter.subset(font)
            name_parts = os.path.splitext(fname)
            new_fname = name_parts[0] + '-decomp' + name_parts[1]
            font.save(new_fname)
            full_size = os.path.getsize(new_fname)

            print("\tSubroutinizing...")
            print("----")
            start_time = time.time()
            cxxCompressor.main(filename=new_fname, verbose=True)
            times.append(time.time() - start_time)
            print("----")

            print("\tTabulating results...")
            comp_fname = name_parts[0] + '-decomp.compressed' + name_parts[1]
            comp_subrs = sum_subrs(TTFont(comp_fname))
            comp_size = os.path.getsize(comp_fname)

            orig_sizes.append(orig_size)
            full_sizes.append(full_size)
            compressed_sizes.append(comp_size)
            names.append(filename)
            nsubrs_orig.append(orig_subrs)
            nsubrs.append(comp_subrs)

   with open('data.csv', 'w') as csvf:
      nwriter = csv.writer(csvf)
      nwriter.writerow(['Name', 'Time to compress', 'Original Size',
                        'Expanded Size', 'Compressed Size',
                        'Original # of Subrs', 'Compressed # of Subrs'])
      nwriter.writerows(zip(names, times, orig_sizes, full_sizes, 
                            compressed_sizes, nsubrs_orig, nsubrs))
