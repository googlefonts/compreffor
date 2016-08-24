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
==== TTX/FontTools Compreffor ====
This module automatically subroutines the CFF table in
a TTFont object, for the purposes of compressing the
outputted font file. In addition to providing a Python
interface, this tool can be used on the command line.

Usage (python):
>> font = TTFont(filename)
>> options = { ... }
>> comp = Compreffor(font, method=compreffor.Methods.Cxx, **options)
>> comp.compress()
>> font.save(filename)

Options:
When initializing a Compreffor object, options can be set using
the options kwargs. They are:
    - verbose (boolean) -- print status messages during compression
    - nrounds (integer) -- the number of market iterations to run
    - nsubrs_limit (integer) -- limit to number of subrs per INDEX
With Methods.Py, the following additional options are available:
    - print_status (boolean) -- printing level lower than verbose
    - chunk_ratio (float) -- set the percentage of charstrings
                             to be run by each process
    - single_process (boolean) -- disable multiprocessing
    - processes (integer) -- the number of simultaneous processes
                             to run

Compression Backends:
There are 2 different ways the compreffor can be run.
    - First is a pure python approach, which can be selected from this module
      by passing method=Methods.Py. This backend is significantly slower than
      the other backend (~10-20x). The logic for this backend can be found
      in pyCompressor.py.
    - Second is C++ extension module backed. With this method, python calls
      the relevant functions by importing directly from `_compreffor.so`. This
      is the default method=Methods.Cxx. The logic is in cxxCompressor.py,
      cffCompressor.h, cffCompressor.cc and _compreffor.pyx.

Usage (command line):
To use on the command line, pyCompressor.py or cxxCompressor.py must be called
directly rather than through this file. The two offer almost identical options,
which can be described in the following way:
>> ./pyCompressor.py -h
...
>> ./cxxCompressor.py -h
...

In both versions, the output goes to a file in the same directory
as the original, but with .compressed appended just before the file
extension. Example usage:
>> ./cxxCompressor.py /path/to/font.otf
...
# font written to /path/to/font.compressed.otf
"""

from compreffor import cxxCompressor, pyCompressor


class Methods:
    Py, Cxx = range(2)


class Compreffor(object):
    def __init__(self, font, method=Methods.Cxx, **options):
        self.font = font
        self.method = method
        self.options = options

    def compress(self):
        if self.method == Methods.Py:
            self.run_py()
        elif self.method == Methods.Cxx:
            self.run_cxx()
        else:
            raise ValueError("Invalid method: %r" % self.method)

    def run_py(self):
        compreffor = pyCompressor.Compreffor(self.font, **self.options)
        compreffor.compress()

    def run_cxx(self):
        cxxCompressor.compreff(self.font, **self.options)
