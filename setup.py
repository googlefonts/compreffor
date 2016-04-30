#!/usr/bin/env python
from __future__ import print_function
from setuptools import setup

try:
    import fontTools
except:
    print("*** Warning: compreffor requires fontTools, see:")
    print("    https://github.com/behdad/fonttools")


setup(
    name="compreffor",
    version="0.1.0",
    description="A CFF subroutinizer for fontTools.",
    author="Sam Fishman",
    license="Apache 2.0",
    packages=["compreffor"],
)
