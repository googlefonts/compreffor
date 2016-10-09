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


def compress(ttFont, method_python=False, **options):
    """ Subroutinize TTFont instance in-place using the C++ Compreffor.
    If 'method_python' is True, use the slower, pure-Python Compreffor.
    If the font already contains subroutines, it is first decompressed.
    """
    if has_subrs(ttFont):
        # there are subroutines in font; must decompress it first
        decompress(ttFont)
    if method_python:
        pyCompressor.compreff(ttFont, **options)
    else:
        cxxCompressor.compreff(ttFont, **options)


def decompress(ttFont, **kwargs):
    """ Use the FontTools Subsetter to desubroutinize the font's CFF table.
    Any keyword arguments are passed on as options to the Subsetter.
    Skip if the font contains no subroutines.
    """
    if not has_subrs(ttFont):
        return  # Nothing to decompress

    from fontTools import subset

    # The FontTools subsetter modifies many tables by default; here
    # we only want to desubroutinize, so we run the subsetter on a
    # temporary copy and extract the resulting CFF table from it
    make_temp = kwargs.pop('make_temp', True)
    if make_temp:
        from io import BytesIO
        from fontTools.ttLib import TTFont, newTable

        stream = BytesIO()
        ttFont.save(stream, reorderTables=None)
        stream.flush()
        stream.seek(0)
        tmpfont = TTFont(stream)
    else:
        tmpfont = ttFont  # run subsetter on the original font

    options = subset.Options(**kwargs)
    options.desubroutinize = True
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(glyphs=tmpfont.getGlyphOrder())
    subsetter.subset(tmpfont)

    if make_temp:
        # copy modified CFF table to original font
        data = tmpfont['CFF '].compile(tmpfont)
        table = newTable('CFF ')
        table.decompile(data, ttFont)
        ttFont['CFF '] = table
        tmpfont.close()


def has_subrs(ttFont):
    """ Return True if the font's CFF table contains any subroutines. """
    if 'CFF ' not in ttFont:
        raise ValueError("Invalid font: no 'CFF ' table found")
    td = ttFont['CFF '].cff.topDictIndex[0]
    priv_subrs = (hasattr(td, 'FDArray') and
                  any((hasattr(fd, 'Subrs') and len(fd.Subrs) > 0)
                      for fd in td.FDArray))
    return len(td.GlobalSubrs) > 0 or priv_subrs


def check(original_file, compressed_file):
    """ Compare the original and compressed font files to confirm they are
    functionally equivalent. Also check that the Charstrings in the compressed
    font's CFFFontSet don't exceed the maximum subroutine nesting level.
    Return True if all checks pass, else return False.
    """
    from compreffor.test.util import check_compression_integrity
    from compreffor.test.util import check_call_depth
    rv = check_compression_integrity(original_file, compressed_file)
    rv &= check_call_depth(compressed_file)
    return rv


# The `Methods` and `Compreffor` classes are now deprecated, but we keep
# them here for backward compatibility


class Methods:
    Py, Cxx = range(2)


class Compreffor(object):
    def __init__(self, font, method=Methods.Cxx, **options):
        import warnings
        warnings.warn("'Compreffor' class is deprecated; use 'compress' function "
                      "instead", UserWarning)
        self.font = font
        self.method = method
        self.options = options

    def compress(self):
        if self.method == Methods.Py:
            compress(self.font, method_python=True, **self.options)
        elif self.method == Methods.Cxx:
            compress(self.font, method_python=False, **self.options)
        else:
            raise ValueError("Invalid method: %r" % self.method)
