"""
gauss.py
Created by Jared J. Thomas, Jul. 2016.
Brigham Young University
"""
from __future__ import print_function, division, absolute_import

import numpy as np

import openmdao.api as om

from gaussianwake.gaussianwake import GaussianWake


def add_gauss_params_IndepVarComps(openmdao_group, nRotorPoints=1):

    ivc = openmdao_group.add_subsystem('model_params', om.IndepVarComp(), promotes_outputs=['*'])

    # ivc.add_discrete_output('model_params:ke', 0.052)
    # ivc.add_discrete_output('model_params:rotation_offset_angle', val=1.56, units='deg')
    # ivc.add_discrete_output('model_params:spread_angle', val=5.84, units='deg')

    # params for Bastankhah model with yaw
    ivc.add_discrete_output('model_params:ky', 0.022)
    ivc.add_discrete_output('model_params:kz', 0.022)
    ivc.add_discrete_output('model_params:alpha', 2.32)
    ivc.add_discrete_output('model_params:beta', 0.154)
    ivc.add_discrete_output('model_params:I', 0.075)
    ivc.add_discrete_output('model_params:wake_combination_method', 0)

    ivc.add_discrete_output('model_params:ti_calculation_method', 0
                            )
    ivc.add_discrete_output('model_params:calc_k_star', False)
    ivc.add_discrete_output('model_params:sort', True)
    ivc.add_discrete_output('model_params:RotorPointsY', val=np.zeros(nRotorPoints),
                            desc='rotor swept area sampling Y points centered at (y,z)=(0,0) normalized by rotor radius')

    ivc.add_discrete_output('model_params:RotorPointsZ', val=np.zeros(nRotorPoints),
                            desc='rotor swept area sampling Z points centered at (y,z)=(0,0) normalized by rotor radius')

    ivc.add_discrete_output('model_params:z_ref', val=80.0,
                            desc='wind speed measurement height')
    ivc.add_discrete_output('model_params:z_0', val=0.0,
                            desc='ground height')
    ivc.add_discrete_output('model_params:shear_exp', val=0.15,
                            desc='wind shear calculation exponent')

    ivc.add_discrete_output('model_params:wec_factor', val=1.0,
                            desc='opt_exp_fac')
    ivc.add_discrete_output('model_params:print_ti', val=False,
                            desc='print TI values to a file for use in plotting etc')
    ivc.add_discrete_output('model_params:wake_model_version', val=2016,
                            desc='choose whether to use Bastankhah 2014 or 2016')
    ivc.add_discrete_output('model_params:sm_smoothing', val=700.0,
                            desc='adjust degree of smoothing for TI smooth max')

    ivc.add_discrete_output('model_params:wec_spreading_angle', val=0.0,
                            desc='adjust spreading angle of the wake')

    # ivc.add_discrete_output('model_params:yaw_mode', val='bastankhah')
    # ivc.add_discrete_output('model_params:spread_mode', val='bastankhah')


class gauss_wrapper(om.Group):

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")
        self.options.declare('direction_id', types=int, default=0,
                             desc="Direction index.")
        self.options.declare('wake_model_options', types=dict, default=None, allow_none=True,
                             desc="Wake Model instantiation parameters.")

    def setup(self):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']
        wake_model_options = opt['wake_model_options']

        self.add_subsystem('f_1', GaussianWake(nTurbines=nTurbines, direction_id=direction_id, options=wake_model_options),
                           promotes=['*'])
