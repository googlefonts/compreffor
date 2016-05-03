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
>>> font = TTFont(filename)
>>> options = { ... }
>>> comp = Compreffor(font, method=compreffor.Methods.CxxLib, **options)
>>> comp.compress()
>>> font.save(filename)

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
There are 3 different ways the compreffor can be run.
    - First is a pure python approach, which can be selected from this module
      by passing method=Methods.Py. This backend is significantly slower than
      the other 2 backends (~10-20x). The logic for this backend can be found
      in pyCompressor.py.
    - Second is a C++ executable backed approach. With this method, data is
      prepared in python and then the executable `cffCompressor` is started
      and data piped in. It's response is read back in over another pipe and
      python writes the result. This can be selected here using
      Methods.CxxExecutable and logic can be found in cxxCompressor.py,
      cffCompressor.h, and cffCompressor.cc.
    - Third is C++ dynamic library (shared object) backed. With this method,
      python calls the relevant functions directly from `libcompreff.so` rather
      than starting an executable in another process. This is currently slower
      than the second approach. This can be selected here using Methods.CxxLib
      and again logic is in cxxCompressor.py, cffCompressor.h, and
      cffCompressor.cc.
If a Compreffor is initialized with no method, Methods.NoPref is chosen. This
method automatically chooses the fastest available backend (order is exe, lib,
py). Availability is dependent on the presence of the compiled binary and library.

Usage (command line):
To use on the command line, pyCompressor.py or cxxCompressor.py must be called
directly rather than through this file. The two offer almost identical options,
which can be described in the following way:
>>> ./pyCompressor.py -h
...
>>> ./cxxCompressor.py -h
...

In both versions, the output goes to a file in the same directory
as the original, but with .compressed appended just before the file
extension. Example usage:
>>> ./cxxCompressor.py /path/to/font.otf
...
# font written to /path/to/font.compressed.otf
"""

from compreffor import cxxCompressor, pyCompressor
import os
import sys


# platform-specific executable name
EXE_NAME = 'cffCompressor'
if sys.platform == 'win32':
    EXE_NAME += '.exe'

# platform-specific shared library name
if sys.platform == 'win32':
    LIB_NAME = 'compreff.dll'
else:
    LIB_NAME = 'libcompreff.so'


class Methods:
    NoPref, Py, CxxExecutable, CxxLib = range(4)

class Compreffor(object):
    def __init__(self, font, method=Methods.NoPref, **options):
        self.font = font
        self.method = method
        self.options = options
        self.exe_path = os.path.join(os.path.dirname(__file__), EXE_NAME)
        self.lib_path = os.path.join(os.path.dirname(__file__), LIB_NAME)

    def compress(self):
        if self.method == Methods.NoPref:
            # choose fastest available method
            if os.path.exists(self.exe_path):
                self.run_executable()
            elif os.path.exists(self.lib_path):
                self.run_lib()
            else:
                self.run_py()
        elif self.method == Methods.Py:
            self.run_py()
        elif self.method == Methods.CxxExecutable:
            self.run_executable()
        elif self.method == Methods.CxxLib:
            self.run_lib()
        else:
            assert 0

    def run_py(self):
        compreffor = pyCompressor.Compreffor(self.font, **self.options)
        compreffor.compress()

    def run_executable(self):
        assert os.path.exists(self.exe_path)
        cxxCompressor.compreff(self.font, use_lib=False, **self.options)

    def run_lib(self):
        assert os.path.exists(self.lib_path)
        cxxCompressor.compreff(self.font, use_lib=True, **self.options)
