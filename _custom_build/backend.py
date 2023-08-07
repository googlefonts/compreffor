"""
This is a custom PEP 517 build backend that extends setuptool's build backend
in order to provide different dependencies for wheel and sdist builds.

Please see the following link for documentation about this technique:

https://setuptools.pypa.io/en/latest/build_meta.html#dynamic-build-dependencies-and-other-build-meta-tweaks
"""

from setuptools import build_meta as _orig
from setuptools.build_meta import *


def get_requires_for_build_sdist(config_settings=None):
    return _orig.get_requires_for_build_sdist(config_settings) + [
        # Finds all git tracked files including submodules, when
        # making sdist MANIFEST
        "setuptools-git-ls-files"
    ]
