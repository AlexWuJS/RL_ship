from setuptools import setup
from Cython.Build import cythonize

setup(name='path_plan',
      ext_modules=cythonize("motion_plan_lib.pyx"))
