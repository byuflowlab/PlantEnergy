"""
gauss.py
Created by Jared J. Thomas, Jul. 2016.
Brigham Young University
"""

from openmdao.api import IndepVarComp, Group
from gaussianwake.gaussianwake import GaussianWake
import numpy as np


def add_gauss_params_IndepVarComps(openmdao_object, nRotorPoints=1):

    # openmdao_object.add('bp0', IndepVarComp('model_params:ke', 0.052, pass_by_object=True))
    # openmdao_object.add('bp1', IndepVarComp('model_params:rotation_offset_angle', val=1.56, units='deg',
                                        #    pass_by_object=True))
    # openmdao_object.add('bp2', IndepVarComp('model_params:spread_angle', val=5.84, units='deg', pass_by_object=True))

    # params for Bastankhah model with yaw
    openmdao_object.add('bp3', IndepVarComp('model_params:ky', 0.022, pass_by_object=True), promotes=['*'])
    openmdao_object.add('bp4', IndepVarComp('model_params:kz', 0.022, pass_by_object=True), promotes=['*'])
    openmdao_object.add('bp5', IndepVarComp('model_params:alpha', 2.32, pass_by_object=True), promotes=['*'])
    openmdao_object.add('bp6', IndepVarComp('model_params:beta', 0.154, pass_by_object=True), promotes=['*'])
    openmdao_object.add('bp7', IndepVarComp('model_params:I', 0.075, pass_by_object=True), promotes=['*'])
    openmdao_object.add('bp8', IndepVarComp('model_params:wake_combination_method', 0, pass_by_object=True),
                        promotes=['*'])
    openmdao_object.add('bp9', IndepVarComp('model_params:ti_calculation_method', 0, pass_by_object=True),
                        promotes=['*'])
    openmdao_object.add('bp10', IndepVarComp('model_params:calc_k_star', False, pass_by_object=True), promotes=['*'])
    openmdao_object.add('bp11', IndepVarComp('model_params:sort', True, pass_by_object=True), promotes=['*'])
    openmdao_object.add('bp12', IndepVarComp('model_params:RotorPointsY', val=np.zeros(nRotorPoints), pass_by_object=True,
                   desc='rotor swept area sampling Y points centered at (y,z)=(0,0) normalized by rotor radius'),
                        promotes=['*'])
    openmdao_object.add('bp13', IndepVarComp('model_params:RotorPointsZ', val=np.zeros(nRotorPoints), pass_by_object=True,
                   desc='rotor swept area sampling Z points centered at (y,z)=(0,0) normalized by rotor radius'),
                        promotes=['*'])
    openmdao_object.add('bp14', IndepVarComp('model_params:z_ref', val=80.0, pass_by_object=True,
                   desc='wind speed measurement height'), promotes=['*'])
    openmdao_object.add('bp15', IndepVarComp('model_params:z_0', val=0.0, pass_by_object=True,
                   desc='ground height'), promotes=['*'])
    openmdao_object.add('bp16', IndepVarComp('model_params:shear_exp', val=0.15, pass_by_object=True,
                   desc='wind shear calculation exponent'), promotes=['*'])

    openmdao_object.add('bp17', IndepVarComp('model_params:opt_exp_fac', val=1.0, pass_by_object=True,
                   desc='opt_exp_fac'), promotes=['*'])
    openmdao_object.add('bp18', IndepVarComp('model_params:print_ti', val=False, pass_by_object=True,
                                             desc='print TI values to a file for use in plotting etc'), promotes=['*'])
    openmdao_object.add('bp19', IndepVarComp('model_params:wake_model_version', val=2016, pass_by_object=True,
                                             desc='choose whether to use Bastankhah 2014 or 2016'), promotes=['*'])
    openmdao_object.add('bp20', IndepVarComp('model_params:sm_smoothing', val=700.0, pass_by_object=True,
                                             desc='adjust degree of smoothing for TI smooth max'), promotes=['*'])

    # openmdao_object.add('bp8', IndepVarComp('model_params:yaw_mode', val='bastankhah', pass_by_object=True))
    # openmdao_object.add('bp9', IndepVarComp('model_params:spread_mode', val='bastankhah', pass_by_object=True))

class gauss_wrapper(Group):

    def __init__(self, nTurbs, direction_id=0, wake_model_options=None):
        super(gauss_wrapper, self).__init__()

        self.add('f_1', GaussianWake(nTurbines=nTurbs, direction_id=direction_id, options=wake_model_options),
                 promotes=['*'])
