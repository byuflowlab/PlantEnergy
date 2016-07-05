"""
larsen.py
Created by Jared J. Thomas, Jul. 2016.
Brigham Young University
"""

from openmdao.api import IndepVarComp, Component, Group
import numpy as np


def add_jensen_params_IndepVarComps(openmdao_object, nTurbines, datasize):

    # add variable tree and indep-var stuff for Jensen
    openmdao_object.add('jp0', IndepVarComp('model_params:alpha', 0.1, pass_by_obj=True,
                                            desc='parameter for controlling wake velocity deficit'),
                        promotes=['*'])
    openmdao_object.add('jp1', IndepVarComp('model_params:spread_angle', 20.0, units='deg', pass_by_obj=True,
                                            desc='wake spreading angle in degrees. angle for one side of wake)'),
                        promotes=['*'])
