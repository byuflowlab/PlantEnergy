"""
jensen.py
Created by Jared J. Thomas, Jul. 2016.
Brigham Young University
"""

from openmdao.api import IndepVarComp, Component, Group
from jensen3d.JensenOpenMDAOconnect import wakeOverlap, effectiveVelocity, effectiveVelocityCosineOverlap, \
    effectiveVelocityCosineNoOverlap, effectiveVelocityConference, JensenCosineYaw, JensenCosineYawIntegral


def add_jensen_params_IndepVarComps(openmdao_object, use_angle=False):

    # add variable tree and indep-var stuff for Jensen
    openmdao_object.add('jp0', IndepVarComp('model_params:alpha', 0.1, pass_by_obj=True,
                                            desc='parameter for controlling wake velocity deficit'),
                        promotes=['*'])
    if use_angle:
        openmdao_object.add('jp1', IndepVarComp('model_params:spread_angle', 20.0, units='deg', pass_by_obj=True,
                                                desc='wake spreading angle in degrees. angle for one side of wake)'),
                            promotes=['*'])


class jensen_wrapper(Group):
    #Group with all the components for the Jensen model

    def __init__(self, nTurbs, direction_id=0, wake_model_options=None):
        super(jensen_wrapper, self).__init__()

        try:
            wake_model_options['variant']
        except:
            wake_model_options = {'variant': 'Original'}

        if wake_model_options['variant'] is 'Original':
            self.add('jensen_1', wakeOverlap(nTurbs, direction_id=direction_id), promotes=['*'])
            self.add('jensen_2', effectiveVelocity(nTurbs, direction_id=direction_id), promotes=['*'])
        elif wake_model_options['variant'] is 'Cosine':
            self.add('jensen_1', wakeOverlap(nTurbs, direction_id=direction_id), promotes=['*'])
            self.add('jensen_2', effectiveVelocityCosineOverlap(nTurbs, direction_id=direction_id), promotes=['*'])
        elif wake_model_options['variant'] is 'CosineNoOverlap_1R' or wake_model_options['variant'] is \
                'CosineNoOverlap_2R':
            self.add('jensen_1', effectiveVelocityCosineNoOverlap(nTurbs, direction_id=direction_id,
                                                             options=wake_model_options),
                     promotes=['*'])
        elif wake_model_options['variant'] is 'Conference':
            self.add('jensen_1', effectiveVelocityConference(nTurbines=nTurbs, direction_id=direction_id), promotes=['*'])
        elif wake_model_options['variant'] is 'CosineYaw_1R' or wake_model_options['variant'] is 'CosineYaw_2R':
            self.add('jensen_1', JensenCosineYaw(nTurbines=nTurbs, direction_id=direction_id, options=wake_model_options),
                     promotes=['*'])
        elif wake_model_options['variant'] is 'CosineYawIntegral':
            self.add('f_1', JensenCosineYawIntegral(nTurbines=nTurbs, direction_id=direction_id, options=wake_model_options),
                     promotes=['*'])
        elif wake_model_options['variant'] is 'CosineYaw':
            self.add('f_1', JensenCosineYaw(nTurbines=nTurbs, direction_id=direction_id, options=wake_model_options),
                     promotes=['*'])
