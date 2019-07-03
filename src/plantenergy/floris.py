"""
floris.py
Created by Jared J. Thomas, Jul. 2016.
Brigham Young University
"""
from __future__ import print_function, division, absolute_import
import re

import numpy as np

import openmdao.api as om

from florisse.floris import Floris


class FLORISParameters(om.ExplicitComponent):
    """Container of FLORIS wake model parameters"""

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('use_rotor_components', types=bool, default=False,
                             desc="Set to True to use rotor components.")

    def setup(self):
        use_rotor_components = self.options['use_rotor_components']

        # add params for every model parameter
        self.add_discrete_input('model_params:kd', 0.15 if not use_rotor_components else 0.17,
                                desc='model parameter that defines the sensitivity of the wake deflection to yaw')
        self.add_discrete_input('model_params:initialWakeDisplacement', -4.5,
                                desc='defines the wake at the rotor to be slightly offset from the rotor. This is'
                                     'necessary for tuning purposes')
        self.add_discrete_input('model_params:bd', -0.01,
                                desc='defines rate of wake displacement if initialWakeAngle is not used')
        self.add_discrete_input('model_params:initialWakeAngle', 1.5,
                                desc='sets how angled the wake flow should be at the rotor')
        self.add_discrete_input('model_params:useWakeAngle', False if not use_rotor_components else True,
                                desc='define whether an initial angle or initial offset should be used for wake center. '
                                     'if True, then bd will be ignored and initialWakeAngle will'
                                     'be used. The reverse is also true')
        self.add_discrete_input('model_params:ke', 0.065 if not use_rotor_components else 0.05,
                                desc='parameter defining overall wake expansion')
        self.add_discrete_input('model_params:me', np.array([-0.5, 0.22, 1.0]) if not use_rotor_components else np.array([-0.5, 0.3, 1.0]),

                                desc='parameters defining relative zone expansion. Mixing zone (me[2]) must always be 1.0')
        self.add_discrete_input('model_params:adjustInitialWakeDiamToYaw', False if not use_rotor_components else True,

                                desc='if True then initial wake diameter will be set to rotorDiameter*cos(yaw)')
        self.add_discrete_input('model_params:MU', np.array([0.5, 1.0, 5.5]),
                                desc='velocity deficit decay rates for each zone. Middle zone must always be 1.0')
        self.add_discrete_input('model_params:aU', 5.0 if not use_rotor_components else 12.0,
                                desc='zone decay adjustment parameter independent of yaw (deg)')
        self.add_discrete_input('model_params:bU', 1.66 if not use_rotor_components else 1.3,
                                desc='zone decay adjustment parameter dependent yaw')
        self.add_discrete_input('model_params:cos_spread', 2.0,
                                desc='spread of cosine smoothing factor (multiple of sum of wake and rotor radii)')
        self.add_discrete_input('model_params:keCorrArray', 0.0,
                                desc='multiplies the ke value by 1+keCorrArray*(sum of rotors relative overlap with '
                                     'inner two zones for including array affects')
        self.add_discrete_input('model_params:keCorrCT', 0.0,
                                desc='adjust ke by adding a precentage of the difference of CT and ideal CT as defined in'
                                     'Region2CT')
        self.add_discrete_input('model_params:Region2CT', 4.0*(1.0/3.0)*(1.0-(1.0/3.0)),
                                desc='defines ideal CT value for use in adjusting ke to yaw adjust CT if keCorrCT>0.0')
        self.add_discrete_input('model_params:axialIndProvided', True if not use_rotor_components else False,
                                desc='if axial induction is not provided, then it will be calculated based on CT')
        self.add_discrete_input('model_params:useaUbU', True,
                                desc='if True then zone velocity decay rates (MU) will be adjusted based on yaw')

        # ################   Visualization   ###########################
        # shear layer (only influences visualization)
        self.add_discrete_input('model_params:shearCoefficientAlpha', 0.10805)
        self.add_discrete_input('model_params:shearZh', 90.0)

        # ##################   other   ##################
        self.add_discrete_input('model_params:FLORISoriginal', False,
                                desc='override all parameters and use FLORIS as original in first Wind Energy paper')

        self.add_input('model_params:shearExp', 0.15, desc='wind shear exponent')
        self.add_input('model_params:z_ref', 90., units='m', desc='height at which wind_speed is measured (m)')
        self.add_input('model_params:z0', 0., units='m', desc='ground height (m)')

        # ###############    Wake Expansion Continuation (WEC) ##############
        self.add_discrete_input('model_params:wec_factor', val=1.0,
                                desc='relaxation factor as defined in Thomas 2018. doi:10.1088/1742-6596/1037/4/042012')

        # add corresponding unknowns
        self.add_discrete_output('floris_params:kd', 0.15 if not use_rotor_components else 0.17,
                                desc='model parameter that defines the sensitivity of the wake deflection to yaw')
        self.add_discrete_output('floris_params:initialWakeDisplacement', -4.5,
                                desc='defines the wake at the rotor to be slightly offset from the rotor. This is'
                                     'necessary for tuning purposes')
        self.add_discrete_output('floris_params:bd', -0.01,
                                desc='defines rate of wake displacement if initialWakeAngle is not used')
        self.add_discrete_output('floris_params:initialWakeAngle', 1.5,
                                desc='sets how angled the wake flow should be at the rotor')
        self.add_discrete_output('floris_params:useWakeAngle', False if not use_rotor_components else True,
                                desc='define whether an initial angle or initial offset should be used for wake center. '
                                     'if True, then bd will be ignored and initialWakeAngle will'
                                     'be used. The reverse is also true')
        self.add_discrete_output('floris_params:ke', 0.065 if not use_rotor_components else 0.05,
                                desc='parameter defining overall wake expansion')
        self.add_discrete_output('floris_params:me', np.array([-0.5, 0.22, 1.0]) if not use_rotor_components else np.array([-0.5, 0.3, 1.0]),

                                desc='parameters defining relative zone expansion. Mixing zone (me[2]) must always be 1.0')
        self.add_discrete_output('floris_params:adjustInitialWakeDiamToYaw', False if not use_rotor_components else True,

                                desc='if True then initial wake diameter will be set to rotorDiameter*cos(yaw)')
        self.add_discrete_output('floris_params:MU', np.array([0.5, 1.0, 5.5]),
                                desc='velocity deficit decay rates for each zone. Middle zone must always be 1.0')
        self.add_discrete_output('floris_params:aU', 5.0 if not use_rotor_components else 12.0,
                                desc='zone decay adjustment parameter independent of yaw (deg)')
        self.add_discrete_output('floris_params:bU', 1.66 if not use_rotor_components else 1.3,
                                desc='zone decay adjustment parameter dependent yaw')
        self.add_discrete_output('floris_params:cos_spread', 2.0,
                                desc='spread of cosine smoothing factor (multiple of sum of wake and rotor radii)')
        self.add_discrete_output('floris_params:keCorrArray', 0.0,
                                desc='multiplies the ke value by 1+keCorrArray*(sum of rotors relative overlap with '
                                     'inner two zones for including array affects')
        self.add_discrete_output('floris_params:keCorrCT', 0.0,
                                desc='adjust ke by adding a precentage of the difference of CT and ideal CT as defined in'
                                     'Region2CT')
        self.add_discrete_output('floris_params:Region2CT', 4.0*(1.0/3.0)*(1.0-(1.0/3.0)),
                                desc='defines ideal CT value for use in adjusting ke to yaw adjust CT if keCorrCT>0.0')

        self.add_discrete_output('floris_params:axialIndProvided', True if not use_rotor_components else False,
                                desc='if axial induction is not provided, then it will be calculated based on CT')
        self.add_discrete_output('floris_params:useaUbU', True,
                                desc='if True then zone velocity decay rates (MU) will be adjusted based on yaw')

        self.add_output('floris_params:shearExp', 0.15, desc='wind shear exponent')
        self.add_output('floris_params:z_ref', 90., units='m', desc='height at which wind_speed is measured')
        self.add_output('floris_params:z0', 0., units='m', desc='ground height')

        # ################   Visualization   ###########################
        # shear layer (only influences visualization)
        self.add_discrete_output('floris_params:shearCoefficientAlpha', 0.10805)
        self.add_discrete_output('floris_params:shearZh', 90.0)

        # ##################   other   ##################
        self.add_discrete_output('floris_params:FLORISoriginal', False,
                                desc='override all parameters and use FLORIS as original in first Wind Energy paper')

        # ###############    Wake Expansion Continuation (WEC) ##############
        self.add_discrete_output('floris_params:wec_factor', val=1.0,
                         desc='relaxation factor as defined in Thomas 2018. doi:10.1088/1742-6596/1037/4/042012')

        # Derivatives
        self.declare_partials(of='floris_params:shearExp', wrt='model_params:shearExp', val=1.0)
        self.declare_partials(of='floris_params:z_ref', wrt='model_params:z_ref', val=1.0)
        self.declare_partials(of='floris_params:z0', wrt='model_params:z0', val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        for p in inputs:
            name = re.search('(?<=:)(.*)$', p).group(0)
            outputs['floris_params:'+name] = inputs['model_params:'+name]

        for p in discrete_inputs._dict:
            name = re.search('(?<=:)(.*)$', p).group(0)
            discrete_outputs['floris_params:'+name] = discrete_inputs['model_params:'+name]


def add_floris_params_IndepVarComps(openmdao_group, use_rotor_components=False):
    ivc = openmdao_group.add_subsystem('model_params', om.IndepVarComp(), promotes_outputs=['*'])

    print(use_rotor_components)
    # permanently alter defaults here

    # ##################   wake deflection   ##################

    # ## parameters
    # original model
    ivc.add_discrete_output('model_params:kd', 0.15 if not use_rotor_components else 0.17,
                            desc='model parameter that defines the sensitivity of the wake deflection '
                            'to yaw')

    ivc.add_discrete_output('model_params:initialWakeDisplacement', -4.5,
                            desc='defines the wake at the rotor to be slightly offset from the rotor. '
                            'This is necessary for tuning purposes')

    ivc.add_discrete_output('model_params:bd', -0.01,
                            desc='defines rate of wake displacement if initialWakeAngle is not used')

    # added
    ivc.add_discrete_output('model_params:initialWakeAngle', 1.5,
                            desc='sets how angled the wake flow should be at the rotor')


    # ## flags
    ivc.add_discrete_output('model_params:useWakeAngle', False if not use_rotor_components else True,
                            desc='define whether an initial angle or initial offset should be used for'
                            'wake center. If True, then bd will be ignored and initialWakeAngle '
                            'will be used. The reverse is also true')


    # ##################   wake expansion   ##################

    # ## parameters
    # original model
    ivc.add_discrete_output('model_params:ke', 0.065 if not use_rotor_components else 0.05,
                            desc='parameter defining overall wake expansion')

    ivc.add_discrete_output('model_params:me', np.array([-0.5, 0.22, 1.0]) if not use_rotor_components else np.array([-0.5, 0.3, 1.0]),
                            desc='parameters defining relative zone expansion. Mixing zone (me[2]) '
                            'must always be 1.0')


    # ## flags
    ivc.add_discrete_output('model_params:adjustInitialWakeDiamToYaw', False,
                            desc='if True then initial wake diameter will be set to '
                            'rotorDiameter*cos(yaw)')


    # ##################   wake velocity   ##################

    # ## parameters
    # original model
    ivc.add_discrete_output('model_params:MU', np.array([0.5, 1.0, 5.5]),
                            desc='velocity deficit decay rates for each zone. Middle zone must always '
                            'be 1.0')

    ivc.add_discrete_output('model_params:aU', 5.0 if not use_rotor_components else 12.0,
                            desc='zone decay adjustment parameter independent of yaw (deg)')

    ivc.add_discrete_output('model_params:bU', 1.66 if not use_rotor_components else 1.3,
                            desc='zone decay adjustment parameter dependent yaw')

    # added
    ivc.add_discrete_output('model_params:cos_spread', 2.0,
                            desc='spread of cosine smoothing factor (multiple of sum of wake and '
                            'rotor radii)')

    ivc.add_discrete_output('model_params:keCorrArray', 0.0,
                            desc='multiplies the ke value by 1+keCorrArray*(sum of rotors relative '
                            'overlap with inner two zones for including array affects')

    ivc.add_discrete_output('model_params:keCorrCT', 0.0,
                            desc='adjust ke by adding a precentage of the difference of CT and ideal '
                            'CT as defined in Region2CT')

    ivc.add_discrete_output('model_params:Region2CT', 4.0*(1.0/3.0)*(1.0-(1.0/3.0)),
                            desc='defines ideal CT value for use in adjusting ke to yaw adjust CT if '
                            'keCorrCT>0.0')

    # flags
    ivc.add_discrete_output('model_params:axialIndProvided', True if not use_rotor_components else False,
                            desc='if axial induction is not provided, then it will be calculated based '
                            'on CT')

    ivc.add_discrete_output('model_params:useaUbU', True,
                            desc='if True then zone velocity decay rates (MU) will be adjusted based '
                            'on yaw')

    # ################   Visualization   ###########################
    # shear layer (only influences visualization)
    ivc.add_discrete_output('model_params:shearCoefficientAlpha', 0.10805)
    ivc.add_discrete_output('model_params:shearZh', 90.0)

    # ##################   other   ##################
    # this is currently not used. Defaults to original if use_rotor_components=False
    ivc.add_discrete_output('model_params:FLORISoriginal', False,
                            desc='override all parameters and use FLORIS as original in Gebraad et al.'
                            '2014, Wind plant power optimization through yaw control using a '
                            'parametric model for wake effect-a CFD simulation study')

    # ###############    Wake Expansion Continuation (WEC) ##############
    ivc.add_discrete_output('model_params:wec_factor', val=1.0,
                            desc='relaxation factor as defined in Thomas 2018. doi:10.1088/1742-6596/1037/4/042012')


class floris_wrapper(om.Group):

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

        if wake_model_options is None:
            wake_model_options = {'differentiable': True, 'use_rotor_components': False, 'nSamples': 0, 'verbose': False}

        nSamples = wake_model_options['nSamples']

        self.add_subsystem('floris_params', FLORISParameters(use_rotor_components=wake_model_options['use_rotor_components']),
                           promotes=['*'])

        if nSamples == 0:
            floris_promotes = ['turbineXw', 'turbineYw', 'yaw%i' % direction_id, 'hubHeight',
                               'rotorDiameter', 'Ct', 'wind_speed', 'axialInduction', 'wtVelocity%i' % direction_id,
                               'floris_params*']
        else:
            floris_promotes = ['turbineXw', 'turbineYw', 'yaw%i' % direction_id, 'hubHeight',
                               'rotorDiameter', 'Ct', 'wind_speed', 'axialInduction', 'wtVelocity%i' % direction_id,
                               'floris_params*', 'wsPositionXw', 'wsPositionYw', 'wsPositionZ', 'wsArray%i' % direction_id]

        self.add_subsystem('floris_model', Floris(nTurbines=nTurbines, direction_id=direction_id,
                                                  differentiable=wake_model_options['differentiable'],
                                                  use_rotor_components=wake_model_options['use_rotor_components'],
                                                  nSamples=wake_model_options['nSamples']),
                           promotes=floris_promotes)


# Testing code for development only
if __name__ == "__main__":

    nTurbines = 2
    direction_id = 0

    prob = om.Problem()
    prob.model.add_subsystem('ftest', floris_wrapper(nTurbines=nTurbines, direction_id=direction_id), promotes=['*'])

    prob.setup(check=True)


