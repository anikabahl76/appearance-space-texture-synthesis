# Numair Khan, James Tompkin
# [2020-10-02] Initial setup file from lab3 run on 2020-10-01
# 
# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext

# ext_modules=[ Extension("filt",
#                         ["filt_cython.pyx"],
#                         libraries=["m"],
#                         extra_compile_args = ["-ffast-math"])]

# setup(
#       name = "filt",
#       cmdclass = {"build_ext": build_ext},
#       ext_modules = ext_modules)


# [2020-10-02] More up to date using 
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import sys

if sys.platform == "win32":
      ext_mods = [ Extension("kc_search",
                              ["kc_search.pyx"],
                              )]
else:
      ext_mods = [ Extension("kc_search",
                         ["kc_search.pyx"],
                         # Compiling on Windows, we have commented out the next two lines.
                         libraries=["m"],
                         extra_compile_args = ["-ffast-math"] # This will work on gcc; other C compilers would have different flags, but this isn't required. Visual C will not recognize it.
                         )]

setup(
      name = "kc_search",
      # language_level="3str",
      ext_modules = cythonize(ext_mods, language_level=3)
)