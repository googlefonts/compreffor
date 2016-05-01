#!/usr/bin/env python
from __future__ import print_function
from setuptools import setup
from distutils.command.build import build
from distutils.command.install_lib import install_lib
from distutils.command.clean import clean
from distutils.cmd import Command
import sys
import os
import subprocess

try:
    import fontTools
except:
    print("*** Warning: compreffor requires fontTools, see:")
    print("    https://github.com/behdad/fonttools")

CURR_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
CXX_SOURCES = os.path.join(CURR_DIR, 'cxx-src')

# platform-specific executable name
EXE_NAME = 'cffCompressor'
if sys.platform == 'win32':
    EXE_NAME += '.exe'

# platform-specific shared library name
if sys.platform == 'win32':
    LIB_NAME = 'compreff.dll'
else:
    LIB_NAME = 'libcompreff.so'

# make sure 'build_cxx' is called as part of 'build' command
build.sub_commands.insert(0, ('build_cxx', lambda *a: True))


class BuildCXX(Command):
    description = "Build 'compreffor' C++ executable and shared library."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        subprocess.check_call(['make'], cwd=CXX_SOURCES)
        executable = os.path.join(CXX_SOURCES, "..", "compreffor", EXE_NAME)
        library = os.path.join(CXX_SOURCES, "..", "compreffor", LIB_NAME)
        assert os.path.exists(executable)
        assert os.path.exists(library)
        build_py = self.get_finalized_command('build_py')
        dest = os.path.join(build_py.build_lib, 'compreffor')
        self.mkpath(dest)
        self.copy_file(executable, dest)
        self.copy_file(library, dest)


class CleanCommand(clean):

    user_options = clean.user_options + [
        ('cxx', None, "run 'make clean' to remove C++ build output")
    ]

    def initialize_options(self):
        clean.initialize_options(self)
        self.cxx = None

    def run(self):
        clean.run(self)
        if self.cxx or self.all:
            subprocess.check_call(['make', 'clean'], cwd=CXX_SOURCES)


class InstallLibCommand(install_lib):

    def build(self):
        install_lib.build(self)
        if not self.skip_build:
            self.run_command('build_cxx')


setup(
    name="compreffor",
    version="0.1.0",
    description="A CFF subroutinizer for fontTools.",
    author="Sam Fishman",
    license="Apache 2.0",
    packages=["compreffor"],
    package_data={"compreffor": [EXE_NAME, LIB_NAME]},
    cmdclass={
           'build_cxx': BuildCXX,
           'clean': CleanCommand,
           'install_lib': InstallLibCommand,
        },
    zip_safe=False,
)
