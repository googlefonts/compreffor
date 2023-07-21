#!/usr/bin/env python
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
cython_min_version = '0.29.30'
try:
    pkg_resources.require("cython >= %s" % cython_min_version)
except pkg_resources.ResolutionError:
    with_cython = False
    print('Distribution mode: Compiling from Cython-generated .cpp sources.')
    from setuptools.command.build_ext import build_ext
else:
    with_cython = True
    print('Development mode: Compiling Cython modules from .pyx sources.')
    from Cython.Distutils.old_build_ext import old_build_ext as build_ext


class custom_build_ext(build_ext):
    """ Custom 'build_ext' command which allows to pass compiler-specific
    'extra_compile_args', 'define_macros' and 'undef_macros' options.
    """

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
        language="c++",
    ),
]


with open('README.rst', 'r') as f:
    long_description = f.read()

setup_params = dict(
    name="compreffor",
    use_scm_version={"write_to": "src/python/compreffor/_version.py"},
    description="A CFF subroutinizer for fontTools.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Sam Fishman",
    license="Apache 2.0",
    package_dir={'': 'src/python'},
    packages=find_packages('src/python'),
    ext_modules=extensions,
    cmdclass={
        'build_ext': custom_build_ext,
    },
    setup_requires=(
        ["setuptools_scm", "setuptools_git_ls_files"] + pytest_runner + wheel
    ),
    tests_require=[
        'pytest>=2.8',
    ],
    install_requires=[
        "fonttools>=4",
    ],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            "compreffor = compreffor.__main__:main",
        ]
    },
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    ],
)


if __name__ == "__main__":
    setup(**setup_params)
