"""
gauss.py
Created by Jared J. Thomas, Jul. 2016.
Brigham Young University
"""

from openmdao.api import IndepVarComp, Group
from gaussianwake.gaussianwake import GaussianWake


def add_gauss_params_IndepVarComps(openmdao_object):

    # openmdao_object.add('bp0', IndepVarComp('model_params:ke', 0.052, pass_by_object=True))
    # openmdao_object.add('bp1', IndepVarComp('model_params:rotation_offset_angle', val=1.56, units='deg',
                                        #    pass_by_object=True))
    # openmdao_object.add('bp2', IndepVarComp('model_params:spread_angle', val=5.84, units='deg', pass_by_object=True))

    # params for Bastankhah model with yaw
    openmdao_object.add('bp3', IndepVarComp('model_params:ky', 0.022, pass_by_object=True))
    openmdao_object.add('bp4', IndepVarComp('model_params:kz', 0.022, pass_by_object=True))
    openmdao_object.add('bp5', IndepVarComp('model_params:alpha', 2.32, pass_by_object=True))
    openmdao_object.add('bp6', IndepVarComp('model_params:beta', 0.154, pass_by_object=True))
    openmdao_object.add('bp7', IndepVarComp('model_params:I', 0.075, pass_by_object=True))
    # openmdao_object.add('bp8', IndepVarComp('model_params:yaw_mode', val='bastankhah', pass_by_object=True))
    # openmdao_object.add('bp9', IndepVarComp('model_params:spread_mode', val='bastankhah', pass_by_object=True))

class gauss_wrapper(Group):

    def __init__(self, nTurbs, direction_id=0, wake_model_options=None):
        super(gauss_wrapper, self).__init__()

        self.add('f_1', GaussianWake(nTurbines=nTurbs, direction_id=direction_id, options=wake_model_options),
                 promotes=['*'])
