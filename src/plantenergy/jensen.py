"""
jensen.py
Created by Jared J. Thomas, Jul. 2016.
Brigham Young University
"""

from openmdao.api import IndepVarComp, Group
from jensen3d.JensenOpenMDAOconnect import Jensen

def add_jensen_params_IndepVarComps(openmdao_object, use_angle=False):

    # add variable tree and indep-var stuff for Jensen
    openmdao_object.add('jp0', IndepVarComp('model_params:alpha', 0.1, pass_by_obj=True,
                                            desc='parameter for controlling wake velocity deficit'),
                        promotes=['*'])
    if use_angle:
        openmdao_object.add('jp1', IndepVarComp('model_params:spread_angle', 20.0, units='deg', pass_by_obj=True,
                                                desc='wake spreading angle in degrees. angle for one side of wake)'),
                            promotes=['*'])

    openmdao_object.add('bp17', IndepVarComp('model_params:wec_factor', val=1.0, pass_by_object=True,
                   desc='wec_factor'), promotes=['*'])


class jensen_wrapper(Group):
    #Group with all the components for the Jensen model

    def __init__(self, nTurbs, direction_id=0, wake_model_options=None):
        super(jensen_wrapper, self).__init__()

        try:
            wake_model_options['variant']
        except:
            wake_model_options['variant'] = 'TopHat'

        self.add('jensen', Jensen(nTurbs, direction_id, model_options=wake_model_options), promotes=['*'])
