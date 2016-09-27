#!/usr/bin/env python
from __future__ import print_function
from setuptools import setup, find_packages, Extension
import os
from distutils.errors import DistutilsSetupError
from distutils import log
from distutils.dep_util import newer_group
import pkg_resources
import platform
import sys


needs_pytest = {'pytest', 'test'}.intersection(sys.argv)
pytest_runner = ['pytest_runner'] if needs_pytest else []
needs_wheel = {'bdist_wheel'}.intersection(sys.argv)
wheel = ['wheel'] if needs_wheel else []


# use Cython if available, else try use pre-generated .cpp sources
cython_min_version = '0.24'
try:
    pkg_resources.require("cython >= %s" % cython_min_version)
except pkg_resources.ResolutionError:
    with_cython = False
    log.info('Distribution mode: Compiling from Cython-generated .cpp sources.')
    from setuptools.command.build_ext import build_ext
else:
    with_cython = True
    log.info('Development mode: Compiling Cython modules from .pyx sources.')
    from Cython.Distutils import build_ext


class custom_build_ext(build_ext):
    """ Custom 'build_ext' command which allows to pass compiler-specific
    'extra_compile_args', 'define_macros' and 'undef_macros' options.
    """

    def finalize_options(self):
        build_ext.finalize_options(self)
        if self.compiler is None:
            # we use this variable with tox to build using GCC on Windows.
            # https://bitbucket.org/hpk42/tox/issues/274/specify-compiler
            self.compiler = os.environ.get("DISTUTILS_COMPILER", None)
        if self.compiler == "mingw32":
            # workaround for virtualenv changing order of libary_dirs on
            # Windows, which makes gcc fail to link with the correct libpython
            # https://github.com/mingwpy/mingwpy.github.io/issues/31
            self.library_dirs.insert(0, os.path.join(sys.exec_prefix, 'libs'))

    def build_extension(self, ext):
        sources = ext.sources
        if sources is None or not isinstance(sources, (list, tuple)):
            raise DistutilsSetupError(
                  "in 'ext_modules' option (extension '%s'), "
                  "'sources' must be present and must be "
                  "a list of source filenames" % ext.name)
        sources = list(sources)

        ext_path = self.get_ext_fullpath(ext.name)
        depends = sources + ext.depends
        if not (self.force or newer_group(depends, ext_path, 'newer')):
            log.debug("skipping '%s' extension (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' extension", ext.name)

        # do compiler specific customizations
        compiler_type = self.compiler.compiler_type

        if isinstance(ext.extra_compile_args, dict):
            extra_args_dict = ext.extra_compile_args or {}
            if compiler_type in extra_args_dict:
                extra_args = extra_args_dict[compiler_type]
            else:
                extra_args = extra_args_dict.get("default", [])
        else:
            extra_args = ext.extra_compile_args or []

        if isinstance(ext.define_macros, dict):
            macros_dict = ext.define_macros or {}
            if compiler_type in macros_dict:
                macros = macros_dict[compiler_type]
            else:
                macros = macros_dict.get("default", [])
        else:
            macros = ext.define_macros or []

        if isinstance(ext.undef_macros, dict):
            undef_macros_dict = ext.undef_macros
            for tp, undef in undef_macros_dict.items():
                if tp == compiler_type:
                    macros.append((undef,))
        else:
            for undef in ext.undef_macros:
                macros.append((undef,))

        # compile the source code to object files.
        objects = self.compiler.compile(sources,
                                        output_dir=self.build_temp,
                                        macros=macros,
                                        include_dirs=ext.include_dirs,
                                        debug=self.debug,
                                        extra_postargs=extra_args,
                                        depends=ext.depends)

        # Now link the object files together into a "shared object"
        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        # TODO: do compiler-specific extra link args?
        extra_args = ext.extra_link_args or []

        # Detect target language, if not provided
        language = ext.language or self.compiler.detect_language(sources)

        self.compiler.link_shared_object(
            objects, ext_path,
            libraries=self.get_libraries(ext),
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=extra_args,
            export_symbols=self.get_export_symbols(ext),
            debug=self.debug,
            build_temp=self.build_temp,
            target_lang=language)


extensions = [
    Extension(
        "compreffor._compreffor",
        sources=[
            os.path.join('src', 'cython', (
                '_compreffor' + ('.pyx' if with_cython else '.cpp'))),
            os.path.join('src', 'cxx', "cffCompressor.cc"),
        ],
        depends=[os.path.join('src', 'cxx', 'cffCompressor.h')],
        extra_compile_args={
            "default": [
                "-std=c++0x", "-pthread",
                "-Wextra", "-Wno-unused", "-Wno-unused-parameter",
                # pass extra compiler flags on OS X to enable support for C++11
            ] + (["-stdlib=libc++", "-mmacosx-version-min=10.7"]
                 if platform.system() == "Darwin" else []),
            "msvc": ["/EHsc", "/Zi"],
        },
        define_macros={
            # On Windows Python 2.7, pyconfig.h defines "hypot" as "_hypot",
            # This clashes with GCC's cmath, and causes compilation errors when
            # building under MinGW: http://bugs.python.org/issue11566
            "mingw32": [("_hypot", "hypot")],
        },
        language="c++",
    ),
]


setup_params = dict(
    name="compreffor",
    version="0.3.0",
    description="A CFF subroutinizer for fontTools.",
    author="Sam Fishman",
    license="Apache 2.0",
    package_dir={'': 'src/python'},
    packages=find_packages('src/python'),
    ext_modules=extensions,
    cmdclass={
        'build_ext': custom_build_ext,
    },
    setup_requires=pytest_runner + wheel,
    tests_require=[
        'pytest>=2.8',
    ],
    install_requires=[
        "fonttools>=3.1",
    ],
    zip_safe=False,
)


if __name__ == "__main__":
    setup(**setup_params)
