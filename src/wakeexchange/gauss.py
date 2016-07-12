"""
gauss.py
Created by Jared J. Thomas, Jul. 2016.
Brigham Young University
"""

from openmdao.api import IndepVarComp, Group
from gaussianwake.gaussianwake import GaussianWake


def add_gauss_params_IndepVarComps(openmdao_object):

    openmdao_object.add('p0', IndepVarComp('model_params:ke', 0.052, pass_by_object=True))
    openmdao_object.add('p1', IndepVarComp('model_params:rotation_offset_angle', val=1.87, units='deg',
                                           pass_by_object=True))
    openmdao_object.add('p2', IndepVarComp('model_params:spread_angle', val=6.37, units='deg', pass_by_object=True))


class gauss_wrapper(Group):

    def __init__(self, nTurbs, direction_id=0, wake_model_options=None):
        super(gauss_wrapper, self).__init__()

        self.add('f_1', GaussianWake(nTurbines=nTurbs, direction_id=direction_id, options=wake_model_options),
                 promotes=['*'])
