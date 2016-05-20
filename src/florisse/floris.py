import numpy as np

from openmdao.api import Group, Component, Problem, IndepVarComp

import _floris
import _florisDiscontinuous

# import _florisHubSmooth as _floris


def add_floris_parameters(openmdao_comp, use_rotor_components=False):
    # altering the values in this function will have no effect during optimization. To change defaults permanently,
    # alter the values in add_floris_IndepVarComps().

    # ##################   wake deflection   ##################

    # ## parameters
    # original model
    openmdao_comp.add_param('model_params:kd', 0.15 if not use_rotor_components else 0.17, pass_by_obj=True,
                            desc='model parameter that defines the sensitivity of the wake deflection to yaw')
    openmdao_comp.add_param('model_params:initialWakeDisplacement', -4.5, pass_by_obj=True,
                            desc='defines the wake at the rotor to be slightly offset from the rotor. This is'
                                 'necessary for tuning purposes')
    openmdao_comp.add_param('model_params:bd', -0.01, pass_by_obj=True,
                            desc='defines rate of wake displacement if initialWakeAngle is not used')
    # added
    openmdao_comp.add_param('model_params:initialWakeAngle', 1.5, pass_by_obj=True,
                            desc='sets how angled the wake flow should be at the rotor')

    # ## flags
    openmdao_comp.add_param('model_params:useWakeAngle', False if not use_rotor_components else True, pass_by_obj=True,
                            desc='define whether an initial angle or initial offset should be used for wake center. '
                                 'if True, then bd will be ignored and initialWakeAngle will'
                                 'be used. The reverse is also true')

    # ##################   wake expansion   ##################

    # ## parameters
    # original model
    openmdao_comp.add_param('model_params:ke', 0.065 if not use_rotor_components else 0.05, pass_by_obj=True,
                            desc='parameter defining overall wake expansion')
    openmdao_comp.add_param('model_params:me', np.array([-0.5, 0.22, 1.0]) if not use_rotor_components else np.array([-0.5, 0.3, 1.0]),
                            pass_by_obj=True,
                            desc='parameters defining relative zone expansion. Mixing zone (me[2]) must always be 1.0')

    # ## flags
    openmdao_comp.add_param('model_params:adjustInitialWakeDiamToYaw', False if not use_rotor_components else True,
                            pass_by_obj=True,
                            desc='if True then initial wake diameter will be set to rotorDiameter*cos(yaw)')

    # ##################   wake velocity   ##################

    # ## parameters
    # original model
    openmdao_comp.add_param('model_params:MU', np.array([0.5, 1.0, 5.5]), pass_by_obj=True,
                            desc='velocity deficit decay rates for each zone. Middle zone must always be 1.0')
    openmdao_comp.add_param('model_params:aU', 5.0 if not use_rotor_components else 12.0, units='deg', pass_by_obj=True,
                            desc='zone decay adjustment parameter independent of yaw')
    openmdao_comp.add_param('model_params:bU', 1.66 if not use_rotor_components else 1.3, pass_by_obj=True,
                            desc='zone decay adjustment parameter dependent yaw')
    # added
    openmdao_comp.add_param('model_params:cos_spread', 2.0, pass_by_obj=True,
                            desc='spread of cosine smoothing factor (multiple of sum of wake and rotor radii)')
    openmdao_comp.add_param('model_params:keCorrArray', 0.0, pass_by_obj=True,
                            desc='multiplies the ke value by 1+keCorrArray*(sum of rotors relative overlap with '
                                 'inner two zones for including array affects')
    openmdao_comp.add_param('model_params:keCorrCT', 0.0, pass_by_obj=True,
                            desc='adjust ke by adding a precentage of the difference of CT and ideal CT as defined in'
                                 'Region2CT')
    openmdao_comp.add_param('model_params:Region2CT', 4.0*(1.0/3.0)*(1.0-(1.0/3.0)), pass_by_obj=True,
                            desc='defines ideal CT value for use in adjusting ke to yaw adjust CT if keCorrCT>0.0')

    # flags
    openmdao_comp.add_param('model_params:axialIndProvided', True if not use_rotor_components else False, pass_by_obj=True,
                            desc='if axial induction is not provided, then it will be calculated based on CT')
    openmdao_comp.add_param('model_params:useaUbU', True, pass_by_obj=True,
                            desc='if True then zone velocity decay rates (MU) will be adjusted based on yaw')

    # ################   Visualization   ###########################
    # shear layer (only influences visualization)
    openmdao_comp.add_param('model_params:shearCoefficientAlpha', 0.10805, pass_by_obj=True)
    openmdao_comp.add_param('model_params:shearZh', 90.0, pass_by_obj=True)

    # ##################   other   ##################
    openmdao_comp.add_param('model_params:FLORISoriginal', False, pass_by_obj=True,
                            desc='override all parameters and use FLORIS as original in first Wind Energy paper')


    ####### apply to things now external to the Floris model
    # openmdao_comp.add_param('model_params:CTcorrected', True, pass_by_obj=True,
    #                desc='CT factor already corrected by CCBlade calculation (approximately factor cos(yaw)^2)')
    #
    # openmdao_comp.add_param('model_params:CPcorrected', True, pass_by_obj=True,
    #                desc = 'CP factor already corrected by CCBlade calculation (assumed with approximately factor cos(yaw)^3)')
    # openmdao_comp.add_param('model_params:pP', 1.88, pass_by_obj=True)

    ####### unused
    # openmdao_comp.add_param('model_params:baselineCT', 4./3.*(1.-1./3.), pass_by_obj=True)
    # openmdao_comp.add_param('model_params:keCorrTI', 0.0, pass_by_obj=True)
    # openmdao_comp.add_param('model_params:baselineTI', 0.045, pass_by_obj=True)
    # openmdao_comp.add_param('model_params:keCorrHR', 0.0, pass_by_obj=True) # neutral, with heating rate 0, is baseline
    # openmdao_comp.add_param('model_params:keCorrHRTI', 0.0, pass_by_obj=True)
    # openmdao_comp.add_param('model_params:keSaturation', 0.0, pass_by_obj=True)
    # openmdao_comp.add_param('model_params:kdCorrYawDirection', 0.0, pass_by_obj=True)


def add_floris_params_IndepVarComps(openmdao_object, use_rotor_components=False):

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


class Floris(Component):

    def __init__(self, nTurbines, direction_id=0, model_options=None):

        super(Floris, self).__init__()

        if model_options is None:
            model_options = {'differentiable': True, 'use_rotor_components': False, 'nSamples': 0, 'verbose': False}

        self.direction_id = direction_id
        self.differentiable = model_options['differentiable']
        self.nTurbines = nTurbines
        self.verbose = model_options['verbose']
        self.nSamples = model_options['nSamples']
        use_rotor_components = model_options['use_rotor_components']

        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1.0e-6
        self.deriv_options['step_calc'] = 'relative'

        if not self.differentiable:
            self.deriv_options['type'] = 'fd'
            self.deriv_options['form'] = 'forward'

        # FLORIS parameters
        add_floris_parameters(self, use_rotor_components=use_rotor_components)

        # input arrays
        self.add_param('turbineXw', np.zeros(nTurbines), units='m',
                       desc='x coordinates of turbines in wind dir. ref. frame')
        self.add_param('turbineYw', np.zeros(nTurbines), units='m',
                       desc='y coordinates of turbines in wind dir. ref. frame')
        self.add_param('yaw%i' % direction_id, np.zeros(nTurbines), units='deg',
                       desc='yaw of each turbine wrt wind dir.')
        self.add_param('hubHeight', np.zeros(nTurbines), units='m',
                       desc='hub heights of all turbines')
        self.add_param('rotorDiameter', np.zeros(nTurbines) + 126.4, units='m', desc='rotor diameter of each turbine')
        self.add_param('Ct', np.zeros(nTurbines)+4.0*(1./3.)*(1.0-(1./3.)), desc='thrust coefficient of each turbine')
        self.add_param('wind_speed', 8.0, units='m/s', desc='free stream wind velocity')
        self.add_param('axialInduction', np.zeros(nTurbines)+1./3., desc='axial induction of all turbines')

        # output arrays
        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s',
                        desc='effective hub velocity for each turbine')
        self.add_output('wakeCentersYT', np.zeros(nTurbines*nTurbines), units='m',
                        desc='wake center y position at each turbine')
        self.add_output('wakeDiametersT', np.zeros(3*nTurbines*nTurbines), units='m',
                        desc='wake diameter of each zone of each wake at each turbine')
        self.add_output('wakeOverlapTRel', np.zeros(3*nTurbines*nTurbines),
                        desc='relative wake zone overlap to rotor area')

        # ############################ visualization arrays ##################################
        if self.nSamples > 0:
            # visualization input
            self.add_param('wsPositionXw', np.zeros(self.nSamples), units='m', pass_by_object=True,
                           desc='downwind position of desired measurements in wind ref. frame')
            self.add_param('wsPositionYw', np.zeros(self.nSamples), units='m', pass_by_object=True,
                           desc='crosswind position of desired measurements in wind ref. frame')
            self.add_param('wsPositionZ', np.zeros(self.nSamples), units='m', pass_by_object=True,
                           desc='position of desired measurements in wind ref. frame')

            # visualization output
            self.add_output('wsArray%i' % direction_id, np.zeros(self.nSamples), units='m/s', pass_by_object=True,
                            desc='wind speed at measurement locations')

    def solve_nonlinear(self, params, unknowns, resids):

        # x and y positions w.r.t. the wind direction (wind dir. = +x)
        turbineXw = params['turbineXw']
        turbineYw = params['turbineYw']

        # yaw wrt wind dir.
        yawDeg = params['yaw%i' % self.direction_id]

        # turbine specs
        rotorDiameter = params['rotorDiameter']
        hubHeight = params['hubHeight']

        # air flow
        Vinf = params['wind_speed']
        Ct = params['Ct']

        # wake deflection
        kd = params['model_params:kd']
        bd = params['model_params:bd']
        initialWakeDisplacement = params['model_params:initialWakeDisplacement']
        useWakeAngle = params['model_params:useWakeAngle']
        initialWakeAngle = params['model_params:initialWakeAngle']

        # wake expansion
        ke = params['model_params:ke']
        adjustInitialWakeDiamToYaw = params['model_params:adjustInitialWakeDiamToYaw']

        # velocity deficit
        MU = params['model_params:MU']
        useaUbU = params['model_params:useaUbU']
        aU = params['model_params:aU']
        bU = params['model_params:bU']
        me = params['model_params:me']
        cos_spread = params['model_params:cos_spread']

        # logicals
        Region2CT = params['model_params:Region2CT']
        axialInduction = params['axialInduction']
        keCorrCT = params['model_params:keCorrCT']
        keCorrArray = params['model_params:keCorrArray']
        axialIndProvided = params['model_params:axialIndProvided']

        # visualization
        # shear layer (only influences visualization)
        shearCoefficientAlpha = params['model_params:shearCoefficientAlpha']
        shearZh = params['model_params:shearZh']

        if self.nSamples > 0:
            wsPositionXYZw = np.array([params['wsPositionXw'], params['wsPositionYw'], params['wsPositionZ']])
        else:
            nSamples = 1
            wsPositionXYZw = np.zeros([3, nSamples])

        # print option
        if self.verbose:
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print "free-stream wind speed %s" % Vinf
            print "axial induction of turbines %s" % axialInduction
            print "C_T of turbines %s" % Ct
            print "yaw of turbines %s" % yawDeg

        if self.differentiable:
            # call to fortran code to obtain output values
            wtVelocity, wsArray, wakeCentersYT, wakeDiametersT, wakeOverlapTRel = \
                _floris.floris(turbineXw, turbineYw, yawDeg, rotorDiameter, hubHeight, Vinf,
                                               Ct, axialInduction, ke, kd, me, initialWakeDisplacement, bd,
                                               MU, aU, bU, initialWakeAngle, cos_spread, keCorrCT,
                                               Region2CT, keCorrArray, useWakeAngle,
                                               adjustInitialWakeDiamToYaw, axialIndProvided, useaUbU, wsPositionXYZw,
                                               shearCoefficientAlpha, shearZh)
        else:
            # call to fortran code to obtain output values
            # print rotorDiameter.shape
            wtVelocity, wakeCentersYT, wakeDiametersT, wakeOverlapTRel = \
                _florisDiscontinuous.floris(turbineXw, turbineYw, yawDeg, rotorDiameter, Vinf,
                                                           Ct, axialInduction, ke, kd, me, initialWakeDisplacement, bd,
                                                           MU, aU, bU, initialWakeAngle, keCorrCT,
                                                           Region2CT, keCorrArray, useWakeAngle,
                                                           adjustInitialWakeDiamToYaw, axialIndProvided, useaUbU)

        # pass outputs to self
        unknowns['wtVelocity%i' % self.direction_id] = wtVelocity
        unknowns['wakeCentersYT'] = wakeCentersYT
        unknowns['wakeDiametersT'] = wakeDiametersT
        unknowns['wakeOverlapTRel'] = wakeOverlapTRel
        if self.nSamples > 0:
            unknowns['wsArray%i' % self.direction_id] = wsArray

    def apply_nonlinear(self, params, unknowns, resids):

        # x and y positions w.r.t. the wind direction (wind dir. = +x)
        turbineXw = params['turbineXw']
        turbineYw = params['turbineYw']

        # yaw wrt wind dir.
        yawDeg = params['yaw%i' % self.direction_id]

        # turbine specs
        rotorDiameter = params['rotorDiameter']
        hubHeight = params['hubHeight']

        # air flow
        Vinf = params['wind_speed']
        Ct = params['Ct']

        # wake deflection
        kd = params['model_params:kd']
        bd = params['model_params:bd']
        initialWakeDisplacement = params['model_params:initialWakeDisplacement']
        useWakeAngle = params['model_params:useWakeAngle']
        initialWakeAngle = params['model_params:initialWakeAngle']

        # wake expansion
        ke = params['model_params:ke']
        adjustInitialWakeDiamToYaw = params['model_params:adjustInitialWakeDiamToYaw']

        # velocity deficit
        MU = params['model_params:MU']
        useaUbU = params['model_params:useaUbU']
        aU = params['model_params:aU']
        bU = params['model_params:bU']
        me = params['model_params:me']
        cos_spread = params['model_params:cos_spread']

        # logicals
        Region2CT = params['model_params:Region2CT']
        axialInduction = params['axialInduction']
        keCorrCT = params['model_params:keCorrCT']
        keCorrArray = params['model_params:keCorrArray']
        axialIndProvided = params['model_params:axialIndProvided']

        # visualization
        # shear layer (only influences visualization)
        shearCoefficientAlpha = params['model_params:shearCoefficientAlpha']
        shearZh = params['model_params:shearZh']
        if self.nSamples > 0:
            wsPositionXYZw = np.array([params['wsPositionXw'], params['wsPositionYw'], params['wsPositionZ']])
        else:
            nSamples = 1
            wsPositionXYZw = np.zeros([3, nSamples])

        if self.differentiable:
            # call to fortran code to obtain output values
            wtVelocity, wsArray, wakeCentersYT, wakeDiametersT, wakeOverlapTRel = \
                _floris.floris(turbineXw, turbineYw, yawDeg, rotorDiameter, hubHeight, Vinf,
                                               Ct, axialInduction, ke, kd, me, initialWakeDisplacement, bd,
                                               MU, aU, bU, initialWakeAngle, cos_spread, keCorrCT,
                                               Region2CT, keCorrArray, useWakeAngle,
                                               adjustInitialWakeDiamToYaw, axialIndProvided, useaUbU, wsPositionXYZw,
                                               shearCoefficientAlpha, shearZh)
        else:
             # call to fortran code to obtain output values
            wtVelocity, wakeCentersYT, wakeDiametersT, wakeOverlapTRel = \
                _florisDiscontinuous.floris(turbineXw, turbineYw, yawDeg, rotorDiameter, Vinf,
                                                           Ct, axialInduction, ke, kd, me, initialWakeDisplacement, bd,
                                                           MU, aU, bU, initialWakeAngle, cos_spread, keCorrCT,
                                                           Region2CT, keCorrArray, useWakeAngle,
                                                           adjustInitialWakeDiamToYaw, axialIndProvided, useaUbU)
        # pass outputs to self
        resids['wtVelocity%i' % self.direction_id] = \
            wtVelocity - unknowns['wtVelocity%i' % self.direction_id]

    def linearize(self, params, unknowns, resids):

        # obtain id for this wind direction
        direction_id = self.direction_id

        # x and y positions w.r.t. the wind dir. (wind dir. = +x)
        turbineXw = params['turbineXw']
        turbineYw = params['turbineYw']

        # yaw wrt wind dir. (wind dir. = +x)
        yawDeg = params['yaw%i' % self.direction_id]

        # turbine specs
        rotorDiameter = params['rotorDiameter']

        # air flow
        Vinf = params['wind_speed']
        Ct = params['Ct']

        # wake deflection
        kd = params['model_params:kd']
        bd = params['model_params:bd']
        initialWakeDisplacement = params['model_params:initialWakeDisplacement']
        useWakeAngle = params['model_params:useWakeAngle']
        initialWakeAngle = params['model_params:initialWakeAngle']

        # wake expansion
        ke = params['model_params:ke']
        adjustInitialWakeDiamToYaw = params['model_params:adjustInitialWakeDiamToYaw']

        # velocity deficit
        MU = params['model_params:MU']
        useaUbU = params['model_params:useaUbU']
        aU = params['model_params:aU']
        bU = params['model_params:bU']
        me = params['model_params:me']
        cos_spread = params['model_params:cos_spread']

        # logicals
        Region2CT = params['model_params:Region2CT']
        axialInduction = params['axialInduction']
        keCorrCT = params['model_params:keCorrCT']
        keCorrArray = params['model_params:keCorrArray']
        axialIndProvided = params['model_params:axialIndProvided']

        # define jacobian size
        nTurbines = len(turbineXw)
        nDirs = nTurbines

        # define input array to direct differentiation
        wtVelocityb = np.eye(nDirs, nTurbines)

        # call to fortran code to obtain output values
        turbineXwb, turbineYwb, yawDegb, rotorDiameterb, Ctb, axialInductionb = \
            _floris.floris_bv(turbineXw, turbineYw, yawDeg, rotorDiameter, Vinf,
                                             Ct, axialInduction, ke, kd, me, initialWakeDisplacement, bd,
                                             MU, aU, bU, initialWakeAngle, cos_spread, keCorrCT,
                                             Region2CT, keCorrArray, useWakeAngle,
                                             adjustInitialWakeDiamToYaw, axialIndProvided, useaUbU,
                                             wtVelocityb)

        # initialize Jacobian dict
        J = {}

        # collect values of the Jacobian
        J['wtVelocity%i' % direction_id, 'turbineXw'] = turbineXwb
        J['wtVelocity%i' % direction_id, 'turbineYw'] = turbineYwb
        J['wtVelocity%i' % direction_id, 'yaw%i' % direction_id] = yawDegb
        J['wtVelocity%i' % direction_id, 'rotorDiameter'] = rotorDiameterb
        J['wtVelocity%i' % direction_id, 'Ct'] = Ctb
        J['wtVelocity%i' % direction_id, 'axialInduction'] = axialInductionb

        return J


# Testing code for development only
if __name__ == "__main__":
    # top = Problem()
    #
    # root = top.root = Group()
    #
    # root.add('p1', IndepVarComp('x', np.array([3.0])))
    # root.add('p2', IndepVarComp('y', np.array([2.0])))
    # root.add('p3', IndepVarComp('z', np.array([10.0])))
    # root.add('p', AdjustCtCpYaw(nTurbines=np.array([1])))
    #
    # root.connect('p1.x', 'p.Ct_in')
    # root.connect('p2.y', 'p.Cp_in')
    # root.connect('p3.z', 'p.yaw')
    #
    # top.setup()
    # top.check_partial_derivatives()
    # top.run()
    #
    # print(root.p.unknowns['Ct_out'])
    # print(root.p.unknowns['Cp_out'])
    #
    # top = Problem()
    #
    # root = top.root = Group()
    #
    # root.add('p1', IndepVarComp('x', np.array([10.0])))
    # root.add('p2', IndepVarComp('y', np.array([10.0])))
    # root.add('p3', IndepVarComp('z', 90.))
    # root.add('p', WindFrame(nTurbines=np.array([1]), resolution=0))
    #
    # root.connect('p1.x', 'p.turbineX')
    # root.connect('p2.y', 'p.turbineY')
    # root.connect('p3.z', 'p.wind_direction')
    #
    # top.setup()
    # top.check_partial_derivatives()
    # top.run()
    #
    # print(root.p.unknowns['turbineXw'])
    # print(root.p.unknowns['turbineYw'])
    #
    # top = Problem()
    #
    # root = top.root = Group()
    #
    # root.add('p1', IndepVarComp('x', np.array([10.0])))
    # root.add('p2', IndepVarComp('y', np.array([10.0])))
    # root.add('p', floris_wcent_wdiam(nTurbines=np.array([1])))
    #
    # root.connect('p1.x', 'p.turbineXw')
    # root.connect('p2.y', 'p.turbineYw')
    # root.connect('p1.x', 'p.yaw')
    # root.connect('p1.x', 'p.rotorDiameter')
    # root.connect('p1.x', 'p.Ct')
    #
    # top.setup()
    # top.check_partial_derivatives()
    # #top.run()
    #
    # print(root.p.unknowns['wakeDiametersT'])
    # print(root.p.unknowns['wakeCentersYT'])
    #
    # top = Problem()
    #
    # root = top.root = Group()
    #
    # root.add('p1', IndepVarComp('x', np.array([10.0])))
    # root.add('p2', IndepVarComp('y', np.array([10.0, 10.0, 10.0])))
    # root.add('p', floris_overlap(nTurbines=np.array([1])))
    #
    # root.connect('p1.x', 'p.turbineXw')
    # root.connect('p1.x', 'p.turbineYw')
    # root.connect('p1.x', 'p.rotorDiameter')
    # root.connect('p1.x', 'p.wakeCentersYT')
    # root.connect('p2.y', 'p.wakeDiametersT')
    #
    # top.setup()
    # top.check_partial_derivatives()
    # #top.run()
    #
    # print(root.p.unknowns['wakeOverlapTRel'])

    top = Problem()

    root = top.root = Group()
    wakeOL = np.array([ 0.94882764,  0.,          0.,          0.,          0.005853,    0.,          0.,
    0. ,         0.00603356,  0.,          0.,          0.,          0.,
    0.94876119,  0. ,         0. ,         0.,          0.00585258 , 0. ,         0.,
    0.  ,        0.00603362,  0. ,         0. ,         0. ,         0.,
    0.94882764 , 0.    ,      0.  ,        0. ,         0.005853   , 0. ,         0.,
    0.        ,  0.00603356 , 0.  ,        0.   ,       0. ,         0.,
    0.94837338,  0.  ,        0.   ,       0.  ,        0.00585014 , 0. ,         0.,
    0.    ,      0.00603391])

    root.add('p1', IndepVarComp('x', np.array([10.0, 10.0, 20, 20])))
    root.add('p2', IndepVarComp('y', wakeOL), promotes=['*'])
    root.add('p', floris_velocity(nTurbines=np.array([4])))

    root.connect('y', 'p.wakeOverlapTRel')

    top.setup()

    top.run()
    top.check_partial_derivatives()
    print(root.p.unknowns['power'])
    print(root.p.unknowns['wt_power'])
    print(root.p.unknowns['wtVelocity'])
    d_wakeOL = np.zeros([wakeOL.size])
    step = 200
    for i in range(0, 2):
        top.run()
        shifthigh = np.zeros_like(wakeOL)
        shifthigh[i] = step
        shiftlow = np.zeros_like(wakeOL)
        shiftlow[i] = -step
        print shifthigh, shiftlow
        top['y'] = wakeOL+shifthigh
        print 'topy = ', top['y']
        top.setup()
        top.run()
        high = top.root.p.unknowns['power']
        top.root.p.params['wakeOverlapTRel'] = wakeOL+shiftlow
        print 'topy = ', top['y'], wakeOL+shiftlow
        top.setup()
        top.run()
        low = top.root.p.unknowns['power']
        print high, low
        d_wakeOL[i] = (high-low)/(2*step)
    print 'd_wakeOL: ', d_wakeOL



# class floris_verbosity(Component):
#
#     def __init__(self, nTurbines, verbose):
#         super(floris_verbosity, self).__init__()
#         # variables for verbosity
#         self.add_param('Ct', np.zeros(nTurbines))
#         self.add_param('Cp', np.zeros(nTurbines), desc='power coefficient for all turbines')
#         self.add_param('axialInduction', np.zeros(nTurbines), desc='axial induction of all turbines')
#         self.add_param('yaw', np.zeros(nTurbines), desc='yaw of each turbine')
#
#         self.verbose = verbose
#
#     def solve_nonlinear(self, params, unknowns, resids):
#          # variables to satisfy verbosity
#         axialInd = params['axialInduction']
#         Cp = params['Cp']
#         Ct = params['Ct']
#         yaw = params['yaw * np.pi / 180']
#         windDirection = params['windDirection']
#         Vinf = params['Vinf']
#         verbose = self.verbose
#
#         if verbose:
#             np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
#             print "wind direction %s deg" % (windDirection * 180.0 / np.pi)
#             print "free-stream wind speed %s" % Vinf
#             print "axial induction turbines %s" % axialInd
#             print "C_P turbines %s" % Cp
#             print "C_T turbines %s" % Ct
#             print "yaw turbines %s" % yaw
#
#
#         # optional print statements from power
#         if verbose:
#             print "wind speed at turbines %s [m/s]" % wtVelocity
#             print "rotor area %d" % (np.pi*rotorDiameter[0]*rotorDiameter[0]/4.0)
#             print "rho %s" % rho
#             print "generatorEfficiency %s" % generatorEfficiency
#             print "powers turbines %s [kW]" % wt_power
#
#     def linearize(self):
#         J = {}
#         return J
