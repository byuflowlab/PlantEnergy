"""
floris.py
Created by Jared J. Thomas, Jul. 2016.
Brigham Young University
"""

import numpy as np
from openmdao.api import Group, Component, Problem, IndepVarComp
from florisse.floris import Floris
import re


def add_floris_params_IndepVarComps(openmdao_object, use_rotor_components=False):

    print(use_rotor_components)
    # permanently alter defaults here

    # ##################   wake deflection   ##################

    # ## parameters
    # original model
    openmdao_object.add('fp00', IndepVarComp('model_params:kd', 0.15 if not use_rotor_components else 0.17,
                                             pass_by_obj=True,
                                             desc='model parameter that defines the sensitivity of the wake deflection '
                                                  'to yaw'),
                        promotes=['*'])
    openmdao_object.add('fp01', IndepVarComp('model_params:initialWakeDisplacement', -4.5, pass_by_obj=True,
                                             desc='defines the wake at the rotor to be slightly offset from the rotor. '
                                                  'This is necessary for tuning purposes'),
                        promotes=['*'])
    openmdao_object.add('fp02', IndepVarComp('model_params:bd', -0.01, pass_by_obj=True,
                                             desc='defines rate of wake displacement if initialWakeAngle is not used'),
                        promotes=['*'])
    # added
    openmdao_object.add('fp03', IndepVarComp('model_params:initialWakeAngle', 1.5, pass_by_obj=True,
                                             desc='sets how angled the wake flow should be at the rotor'),
                        promotes=['*'])

    # ## flags
    openmdao_object.add('fp04', IndepVarComp('model_params:useWakeAngle', False if not use_rotor_components else True,
                                             pass_by_obj=True,
                                             desc='define whether an initial angle or initial offset should be used for'
                                                  'wake center. If True, then bd will be ignored and initialWakeAngle '
                                                  'will be used. The reverse is also true'),
                        promotes=['*'])

    # ##################   wake expansion   ##################

    # ## parameters
    # original model
    openmdao_object.add('fp05', IndepVarComp('model_params:ke', 0.065 if not use_rotor_components else 0.05,
                                             pass_by_obj=True,
                                             desc='parameter defining overall wake expansion'),
                        promotes=['*'])
    openmdao_object.add('fp06', IndepVarComp('model_params:me', np.array([-0.5, 0.22, 1.0]) if not use_rotor_components else np.array([-0.5, 0.3, 1.0]),
                                             pass_by_obj=True,
                                             desc='parameters defining relative zone expansion. Mixing zone (me[2]) '
                                                  'must always be 1.0'),
                        promotes=['*'])

    # ## flags
    openmdao_object.add('fp07', IndepVarComp('model_params:adjustInitialWakeDiamToYaw',
                                             False, pass_by_obj=True,
                                             desc='if True then initial wake diameter will be set to '
                                                  'rotorDiameter*cos(yaw)'),
                        promotes=['*'])


    # ##################   wake velocity   ##################

    # ## parameters
    # original model
    openmdao_object.add('fp08', IndepVarComp('model_params:MU', np.array([0.5, 1.0, 5.5]), pass_by_obj=True,
                                             desc='velocity deficit decay rates for each zone. Middle zone must always '
                                                  'be 1.0'),
                        promotes=['*'])
    openmdao_object.add('fp09', IndepVarComp('model_params:aU', 5.0 if not use_rotor_components else 12.0, units='deg',
                                             pass_by_obj=True,
                                             desc='zone decay adjustment parameter independent of yaw'),
                        promotes=['*'])
    openmdao_object.add('fp10', IndepVarComp('model_params:bU', 1.66 if not use_rotor_components else 1.3,
                                             pass_by_obj=True,
                                             desc='zone decay adjustment parameter dependent yaw'),
                        promotes=['*'])
    # added
    openmdao_object.add('fp11', IndepVarComp('model_params:cos_spread', 2.0, pass_by_obj=True,
                                             desc='spread of cosine smoothing factor (multiple of sum of wake and '
                                                  'rotor radii)'),
                        promotes=['*'])


    openmdao_object.add('fp12', IndepVarComp('model_params:keCorrArray', 0.0, pass_by_obj=True,
                                             desc='multiplies the ke value by 1+keCorrArray*(sum of rotors relative '
                                                  'overlap with inner two zones for including array affects'),
                        promotes=['*'])
    openmdao_object.add('fp13', IndepVarComp('model_params:keCorrCT', 0.0, pass_by_obj=True,
                                             desc='adjust ke by adding a precentage of the difference of CT and ideal '
                                                  'CT as defined in Region2CT'),
                        promotes=['*'])
    openmdao_object.add('fp14', IndepVarComp('model_params:Region2CT', 4.0*(1.0/3.0)*(1.0-(1.0/3.0)), pass_by_obj=True,
                                             desc='defines ideal CT value for use in adjusting ke to yaw adjust CT if '
                                                  'keCorrCT>0.0'),
                        promotes=['*'])

    # flags
    openmdao_object.add('fp15', IndepVarComp('model_params:axialIndProvided',
                                             True if not use_rotor_components else False, pass_by_obj=True,
                                             desc='if axial induction is not provided, then it will be calculated based '
                                                  'on CT'),
                        promotes=['*'])
    openmdao_object.add('fp16', IndepVarComp('model_params:useaUbU', True, pass_by_obj=True,
                                             desc='if True then zone velocity decay rates (MU) will be adjusted based '
                                                  'on yaw'),
                        promotes=['*'])

    # ################   Visualization   ###########################
    # shear layer (only influences visualization)
    openmdao_object.add('fp17', IndepVarComp('model_params:shearCoefficientAlpha', 0.10805, pass_by_obj=True),
                        promotes=['*'])
    openmdao_object.add('fp18', IndepVarComp('model_params:shearZh', 90.0, pass_by_obj=True), promotes=['*'])

    # ##################   other   ##################
    # this is currently not used. Defaults to original if use_rotor_components=False
    openmdao_object.add('fp19', IndepVarComp('model_params:FLORISoriginal', False, pass_by_obj=True,
                                             desc='override all parameters and use FLORIS as original in Gebraad et al.'
                                                  '2014, Wind plant power optimization through yaw control using a '
                                                  'parametric model for wake effect-a CFD simulation study'),
                        promotes=['*'])

    # ###############    Wake Expansion Continuation (WEC) ##############
    openmdao_object.add('fp20', IndepVarComp('model_params:wec_factor', val=1.0, pass_by_obj=True,
                                             desc='relaxation factor as defined in Thomas 2018. doi:10.1088/1742-6596/1037/4/042012'), promotes=['*'])


class FLORISParameters(Component):
    """Container of FLORIS wake model parameters"""

    def __init__(self, use_rotor_components):

        super(FLORISParameters, self).__init__()

        # add params for every model parameter
        self.add_param('model_params:kd', 0.15 if not use_rotor_components else 0.17, pass_by_obj=True,
                                desc='model parameter that defines the sensitivity of the wake deflection to yaw')
        self.add_param('model_params:initialWakeDisplacement', -4.5, pass_by_obj=True,
                                desc='defines the wake at the rotor to be slightly offset from the rotor. This is'
                                     'necessary for tuning purposes')
        self.add_param('model_params:bd', -0.01, pass_by_obj=True,
                                desc='defines rate of wake displacement if initialWakeAngle is not used')
        self.add_param('model_params:initialWakeAngle', 1.5, pass_by_obj=True,
                                desc='sets how angled the wake flow should be at the rotor')
        self.add_param('model_params:useWakeAngle', False if not use_rotor_components else True, pass_by_obj=True,
                                desc='define whether an initial angle or initial offset should be used for wake center. '
                                     'if True, then bd will be ignored and initialWakeAngle will'
                                     'be used. The reverse is also true')
        self.add_param('model_params:ke', 0.065 if not use_rotor_components else 0.05, pass_by_obj=True,
                                desc='parameter defining overall wake expansion')
        self.add_param('model_params:me', np.array([-0.5, 0.22, 1.0]) if not use_rotor_components else np.array([-0.5, 0.3, 1.0]),
                                pass_by_obj=True,
                                desc='parameters defining relative zone expansion. Mixing zone (me[2]) must always be 1.0')
        self.add_param('model_params:adjustInitialWakeDiamToYaw', False if not use_rotor_components else True,
                                pass_by_obj=True,
                                desc='if True then initial wake diameter will be set to rotorDiameter*cos(yaw)')
        self.add_param('model_params:MU', np.array([0.5, 1.0, 5.5]), pass_by_obj=True,
                                desc='velocity deficit decay rates for each zone. Middle zone must always be 1.0')
        self.add_param('model_params:aU', 5.0 if not use_rotor_components else 12.0, units='deg', pass_by_obj=True,
                                desc='zone decay adjustment parameter independent of yaw')
        self.add_param('model_params:bU', 1.66 if not use_rotor_components else 1.3, pass_by_obj=True,
                                desc='zone decay adjustment parameter dependent yaw')
        self.add_param('model_params:cos_spread', 2.0, pass_by_obj=True,
                                desc='spread of cosine smoothing factor (multiple of sum of wake and rotor radii)')
        self.add_param('model_params:keCorrArray', 0.0, pass_by_obj=True,
                                desc='multiplies the ke value by 1+keCorrArray*(sum of rotors relative overlap with '
                                     'inner two zones for including array affects')
        self.add_param('model_params:keCorrCT', 0.0, pass_by_obj=True,
                                desc='adjust ke by adding a precentage of the difference of CT and ideal CT as defined in'
                                     'Region2CT')
        self.add_param('model_params:Region2CT', 4.0*(1.0/3.0)*(1.0-(1.0/3.0)), pass_by_obj=True,
                                desc='defines ideal CT value for use in adjusting ke to yaw adjust CT if keCorrCT>0.0')
        self.add_param('model_params:axialIndProvided', True if not use_rotor_components else False, pass_by_obj=True,
                                desc='if axial induction is not provided, then it will be calculated based on CT')
        self.add_param('model_params:useaUbU', True, pass_by_obj=True,
                                desc='if True then zone velocity decay rates (MU) will be adjusted based on yaw')
        # ################   Visualization   ###########################
        # shear layer (only influences visualization)
        self.add_param('model_params:shearCoefficientAlpha', 0.10805, pass_by_obj=True)
        self.add_param('model_params:shearZh', 90.0, pass_by_obj=True)
        # ##################   other   ##################
        self.add_param('model_params:FLORISoriginal', False, pass_by_obj=True,
                                desc='override all parameters and use FLORIS as original in first Wind Energy paper')
        self.add_param('model_params:shearExp', 0.15, pass_by_obj=True, desc='wind shear exponent')
        self.add_param('model_params:z_ref', 90., units='m', pass_by_obj=True, desc='height at which wind_speed is measured')
        self.add_param('model_params:z0', 0., units='m', pass_by_obj=True, desc='ground height')
        # ###############    Wake Expansion Continuation (WEC) ##############
        self.add_param('model_params:wec_factor', val=1.0, pass_by_obj=True, desc='relaxation factor as defined in Thomas 2018. doi:10.1088/1742-6596/1037/4/042012')

        # add corresponding unknowns
        self.add_output('floris_params:kd', 0.15 if not use_rotor_components else 0.17, pass_by_obj=True,
                                desc='model parameter that defines the sensitivity of the wake deflection to yaw')
        self.add_output('floris_params:initialWakeDisplacement', -4.5, pass_by_obj=True,
                                desc='defines the wake at the rotor to be slightly offset from the rotor. This is'
                                     'necessary for tuning purposes')
        self.add_output('floris_params:bd', -0.01, pass_by_obj=True,
                                desc='defines rate of wake displacement if initialWakeAngle is not used')
        self.add_output('floris_params:initialWakeAngle', 1.5, pass_by_obj=True,
                                desc='sets how angled the wake flow should be at the rotor')
        self.add_output('floris_params:useWakeAngle', False if not use_rotor_components else True, pass_by_obj=True,
                                desc='define whether an initial angle or initial offset should be used for wake center. '
                                     'if True, then bd will be ignored and initialWakeAngle will'
                                     'be used. The reverse is also true')
        self.add_output('floris_params:ke', 0.065 if not use_rotor_components else 0.05, pass_by_obj=True,
                                desc='parameter defining overall wake expansion')
        self.add_output('floris_params:me', np.array([-0.5, 0.22, 1.0]) if not use_rotor_components else np.array([-0.5, 0.3, 1.0]),
                                pass_by_obj=True,
                                desc='parameters defining relative zone expansion. Mixing zone (me[2]) must always be 1.0')
        self.add_output('floris_params:adjustInitialWakeDiamToYaw', False if not use_rotor_components else True,
                                pass_by_obj=True,
                                desc='if True then initial wake diameter will be set to rotorDiameter*cos(yaw)')
        self.add_output('floris_params:MU', np.array([0.5, 1.0, 5.5]), pass_by_obj=True,
                                desc='velocity deficit decay rates for each zone. Middle zone must always be 1.0')
        self.add_output('floris_params:aU', 5.0 if not use_rotor_components else 12.0, units='deg', pass_by_obj=True,
                                desc='zone decay adjustment parameter independent of yaw')
        self.add_output('floris_params:bU', 1.66 if not use_rotor_components else 1.3, pass_by_obj=True,
                                desc='zone decay adjustment parameter dependent yaw')
        self.add_output('floris_params:cos_spread', 2.0, pass_by_obj=True,
                                desc='spread of cosine smoothing factor (multiple of sum of wake and rotor radii)')
        self.add_output('floris_params:keCorrArray', 0.0, pass_by_obj=True,
                                desc='multiplies the ke value by 1+keCorrArray*(sum of rotors relative overlap with '
                                     'inner two zones for including array affects')
        self.add_output('floris_params:keCorrCT', 0.0, pass_by_obj=True,
                                desc='adjust ke by adding a precentage of the difference of CT and ideal CT as defined in'
                                     'Region2CT')
        self.add_output('floris_params:Region2CT', 4.0*(1.0/3.0)*(1.0-(1.0/3.0)), pass_by_obj=True,
                                desc='defines ideal CT value for use in adjusting ke to yaw adjust CT if keCorrCT>0.0')

        self.add_output('floris_params:axialIndProvided', True if not use_rotor_components else False, pass_by_obj=True,
                                desc='if axial induction is not provided, then it will be calculated based on CT')
        self.add_output('floris_params:useaUbU', True, pass_by_obj=True,
                                desc='if True then zone velocity decay rates (MU) will be adjusted based on yaw')
        self.add_output('floris_params:shearExp', 0.15, pass_by_obj=True, desc='wind shear exponent')
        self.add_output('floris_params:z_ref', 90., pass_by_obj=True, units='m', desc='height at which wind_speed is measured')
        self.add_output('floris_params:z0', 0., pass_by_obj=True, units='m', desc='ground height')
        # ################   Visualization   ###########################
        # shear layer (only influences visualization)
        self.add_output('floris_params:shearCoefficientAlpha', 0.10805, pass_by_obj=True)
        self.add_output('floris_params:shearZh', 90.0, pass_by_obj=True)

        # ##################   other   ##################
        self.add_output('floris_params:FLORISoriginal', False, pass_by_obj=True,
                                desc='override all parameters and use FLORIS as original in first Wind Energy paper')

        # ###############    Wake Expansion Continuation (WEC) ##############
        self.add_output('floris_params:wec_factor', val=1.0, pass_by_obj=True,
                         desc='relaxation factor as defined in Thomas 2018. doi:10.1088/1742-6596/1037/4/042012')

    def solve_nonlinear(self, params, unknowns, resids):

        for p in params:
            name = re.search('(?<=:)(.*)$', p).group(0)
            self.unknowns['floris_params:'+name] = params['model_params:'+name]


class floris_wrapper(Group):

    def __init__(self, nTurbines, direction_id=0, wake_model_options=None):

        super(floris_wrapper, self).__init__()

        if wake_model_options is None:
            wake_model_options = {'differentiable': True, 'use_rotor_components': False, 'nSamples': 0, 'verbose': False}

        self.direction_id = direction_id
        self.nTurbines = nTurbines
        nSamples = wake_model_options['nSamples']

        self.add('floris_params', FLORISParameters(wake_model_options['use_rotor_components']), promotes=['*'])

        self.add('floris_model', Floris(nTurbines, direction_id, wake_model_options['differentiable'],
                                        wake_model_options['use_rotor_components'], wake_model_options['nSamples']),
                 promotes=['turbineXw', 'turbineYw', 'yaw%i' % direction_id, 'hubHeight',
                           'rotorDiameter', 'Ct', 'wind_speed', 'axialInduction', 'wtVelocity%i' % direction_id,
                           'floris_params*'] if nSamples == 0 else ['turbineXw', 'turbineYw', 'yaw%i' % direction_id, 'hubHeight',
                           'rotorDiameter', 'Ct', 'wind_speed', 'axialInduction', 'wtVelocity%i' % direction_id,
                           'floris_params*', 'wsPositionXw', 'wsPositionYw', 'wsPositionZ', 'wsArray%i' % direction_id])


# Testing code for development only
if __name__ == "__main__":

    nTurbines = 2
    direction_id = 0

    prob = Problem()
    prob.root = Group()
    prob.root.add('ftest', floris_wrapper(nTurbines, direction_id), promotes=['*'])

    prob.setup(check=True)


