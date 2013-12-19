#!/usr/bin/python

from distutils.core import setup, Extension
import os


def list_src(dir_path):
    full_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    return [f for f in full_paths if f.endswith('.cpp') or f.endswith('.hpp')]

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_src = os.path.join(current_dir, 'asmlib-opencv', 'src', 'lib')
binding_src = os.path.join(current_dir, 'bindings_src')

src_files = list_src(lib_src) + list_src(binding_src)

pyasm_module = Extension('pyasm',
                         sources=src_files,
                         include_dirs=['/usr/include/opencv', lib_src, binding_src],
                         libraries=['opencv_core', 'opencv_highgui'],
                         library_dirs=['/usr/local/lib'],
                         extra_compile_args=[])

setup(name='pyasm',
      version='1.0',
      description='An asmlib-opencv binding package',
      ext_modules=[pyasm_module])