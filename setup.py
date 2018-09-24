#!/usr/bin/env python
# encoding: utf-8

from numpy.distutils.core import setup, Extension

setup(
    name='PlantEnergy',
    version='0.0.1',
    description='Wind farm optimization interface allowing wake models to be switched out',
    install_requires=['openmdao>=1.7.3', 'florisse'],
    package_dir={'': 'src'},
    dependency_links=['http://github.com/OpenMDAO/OpenMDAO.git@master', 'https://github.com/WISDEM/FLORISSE.git@develop'],
    packages=['plantenergy'],
    license='Apache License, Version 2.0',
)
