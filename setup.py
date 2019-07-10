#!/usr/bin/env python
# encoding: utf-8

from numpy.distutils.core import setup, Extension

module1 = Extension('position_constraints', sources=['src/plantenergy/position_constraints.f90',
                                       'src/plantenergy/adStack.c',
                                       'src/plantenergy/adBuffer.f'],
                    extra_compile_args=['-O2', '-c'])

setup(
    name='PlantEnergy',
    version='0.0.1',
    description='Wind farm optimization interface allowing wake models to be switched out',
    install_requires=['openmdao>=1.7.3', 'florisse'],
    package_dir={'': 'src'},
    dependency_links=['http://github.com/OpenMDAO/OpenMDAO.git@master', 'https://github.com/WISDEM/FLORISSE.git@develop'],
    packages=['plantenergy'],
    ext_modules=[module1],
    license='Apache License, Version 2.0',
)

