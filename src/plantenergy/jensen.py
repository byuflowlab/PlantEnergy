"""
jensen.py
Created by Jared J. Thomas, Jul. 2016.
Brigham Young University
"""

import openmdao.api as om
from jensen3d.JensenOpenMDAOconnect import Jensen


def add_jensen_params_IndepVarComps(om_group, use_angle=False):

    ivc = om_group.add_subsystem('model_params', om.IndepVarComp(), promotes=['*'])

    # add variable tree and indep-var stuff for Jensen
    ivc.add_discrete_output('model_params:alpha', 0.1,
                            desc='parameter for controlling wake velocity deficit')

    if use_angle:
        ivc.add_discrete_output('model_params:spread_angle', 20.0,
                                desc='wake spreading angle in degrees. angle for one side of wake) (deg)')

    ivc.add_discrete_output('model_params:wec_factor', val=1.0,
                            desc='wec_factor')


class jensen_wrapper(om.Group):
    #Group with all the components for the Jensen model

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
        nTurbs = opt['nTurbines']
        direction_id = opt['direction_id']
        wake_model_options = opt['wake_model_options']

        try:
            wake_model_options['variant']
        except:
            print("no wake model variant select, setting to 'TopHat'")
            wake_model_options['variant'] = 'TopHat'

        self.add_subsystem('jensen', Jensen(nTurbines=nTurbs, direction_id=direction_id,
                                            model_options=wake_model_options), promotes=['*'])
