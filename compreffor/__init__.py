"""
TTX/FontTools Compreffor
"""

import os

import pyCompressor
import cxxCompressor

class Methods:
    NoPref, Py, CxxExecutable, CxxLib = range(4)

class Compreffor(object):
    def __init__(self, font, method=Methods.NoPref, **options):
        self.font = font
        self.method = method
        self.options = options
        self.exe_path = os.path.join(os.path.dirname(__file__), "cffCompressor")
        self.lib_path = os.path.join(os.path.dirname(__file__), "libcompreff.so")

    def compreff(self):
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
