import numpy as np

from openmdao.api import Group, IndepVarComp, ParallelGroup, ScipyGMRES, NLGaussSeidel

from openmdao.core.mpi_wrap import MPI
if MPI:
    from openmdao.api import PetscKSP

from floris import Floris, add_floris_params_IndepVarComps

from GeneralWindFarmComponents import WindFrame, AdjustCtCpYaw, MUX, WindFarmAEP, DeMUX, \
    CPCT_Interpolate_Gradients_Smooth, WindDirectionPower, add_gen_params_IdepVarComps, \
    CPCT_Interpolate_Gradients


class RotorSolveGroup(Group):

    def __init__(self, nTurbines, direction_id=0, datasize=0, differentiable=True,
                 use_rotor_components=False, nSamples=0, model=Floris,
                 model_options=None):

        super(RotorSolveGroup, self).__init__()

        if model_options is None:
            model_options = {'differentiable': differentiable, 'use_rotor_components': use_rotor_components,
                             'nSamples': nSamples}

        from openmdao.core.mpi_wrap import MPI

        # set up iterative solvers
        epsilon = 1E-6
        if MPI:
            self.ln_solver = PetscKSP()
        else:
            self.ln_solver = ScipyGMRES()
        self.nl_solver = NLGaussSeidel()
        self.ln_solver.options['atol'] = epsilon

        self.add('CtCp', CPCT_Interpolate_Gradients_Smooth(nTurbines, direction_id=direction_id, datasize=datasize),
                 promotes=['gen_params:*', 'yaw%i' % direction_id,
                           'wtVelocity%i' % direction_id, 'Cp_out'])

        # TODO refactor the model component instance
        self.add('floris', model(nTurbines, direction_id=direction_id, model_options=model_options),
                 promotes=(['model_params:*', 'wind_speed', 'axialInduction',
                            'turbineXw', 'turbineYw', 'rotorDiameter', 'yaw%i' % direction_id, 'hubHeight',
                            'wtVelocity%i' % direction_id]
                           if (nSamples == 0) else
                           ['model_params:*', 'wind_speed', 'axialInduction',
                            'turbineXw', 'turbineYw', 'rotorDiameter', 'yaw%i' % direction_id, 'hubHeight',
                            'wtVelocity%i' % direction_id, 'wsPositionX', 'wsPositionY', 'wsPositionZ',
                            'wsArray%i' % direction_id]))
        self.connect('CtCp.Ct_out', 'floris.Ct')


class DirectionGroup(Group):
    """
    Group containing all necessary components for wind plant calculations
    in a single direction
    """

    def __init__(self, nTurbines, direction_id=0, use_rotor_components=False, datasize=0,
                 differentiable=True, add_IdepVarComps=True, nSamples=0, model=Floris,
                 model_options=None):
        super(DirectionGroup, self).__init__()

        if add_IdepVarComps:
            add_floris_params_IndepVarComps(self, use_rotor_components=use_rotor_components)
            add_gen_params_IdepVarComps(self, datasize=datasize)

        self.add('directionConversion', WindFrame(nTurbines, differentiable=differentiable, nSamples=nSamples),
                 promotes=['*'])

        if use_rotor_components:
            self.add('rotorGroup', RotorSolveGroup(nTurbines, direction_id=direction_id,
                                                 datasize=datasize, differentiable=differentiable,
                                                 nSamples=nSamples, use_rotor_components=use_rotor_components,
                                                 model=Floris, model_options=model_options),
                     promotes=(['gen_params:*', 'yaw%i' % direction_id, 'wtVelocity%i' % direction_id,
                                'model_params:*', 'wind_speed', 'axialInduction',
                                'turbineXw', 'turbineYw', 'rotorDiameter', 'hubHeight']
                               if (nSamples == 0) else
                               ['gen_params:*', 'yaw%i' % direction_id, 'wtVelocity%i' % direction_id,
                                'model_params:*', 'wind_speed', 'axialInduction',
                                'turbineXw', 'turbineYw', 'rotorDiameter', 'hubHeight', 'wsPositionX', 'wsPositionY', 'wsPositionZ',
                                'wsArray%i' % direction_id]))
        else:
            self.add('CtCp', AdjustCtCpYaw(nTurbines, direction_id, differentiable),
                     promotes=['Ct_in', 'Cp_in', 'gen_params:*', 'yaw%i' % direction_id])

            self.add('myModel', model(nTurbines, direction_id=direction_id, model_options=model_options),
                     promotes=(['model_params:*', 'wind_speed', 'axialInduction',
                                'turbineXw', 'turbineYw', 'rotorDiameter', 'yaw%i' % direction_id, 'hubHeight',
                                'wtVelocity%i' % direction_id]
                               if (nSamples == 0) else
                               ['model_params:*', 'wind_speed', 'axialInduction',
                                'turbineXw', 'turbineYw', 'rotorDiameter', 'yaw%i' % direction_id, 'hubHeight',
                                'wtVelocity%i' % direction_id, 'wsPositionX', 'wsPositionY', 'wsPositionZ',
                                'wsArray%i' % direction_id]))

        self.add('powerComp', WindDirectionPower(nTurbines=nTurbines, direction_id=direction_id, differentiable=True,
                                                 use_rotor_components=use_rotor_components),
                 promotes=['air_density', 'generatorEfficiency', 'rotorDiameter',
                           'wtVelocity%i' % direction_id,
                           'wtPower%i' % direction_id, 'dir_power%i' % direction_id])

        if use_rotor_components:
            self.connect('rotorGroup.Cp_out', 'powerComp.Cp')
        else:
            self.connect('CtCp.Ct_out', 'myModel.Ct')
            self.connect('CtCp.Cp_out', 'powerComp.Cp')


class AEPGroup(Group):
    """
    Group containing all necessary components for wind plant AEP calculations using the FLORIS model
    """

    def __init__(self, nTurbines, nDirections=1, use_rotor_components=False, datasize=0,
                 differentiable=True, optimizingLayout=False, nSamples=0, model=Floris,
                 model_options=None):

        super(AEPGroup, self).__init__()

        if model_options is None:
            model_options = {'differentiable': differentiable, 'use_rotor_components': use_rotor_components,
                             'nSamples': nSamples, 'verbose': False}

        # providing default unit types for general MUX/DeMUX components
        power_units = 'kW'
        direction_units = 'deg'
        wind_speed_units = 'm/s'

        # print 'SAMPLES: ', nSamples

        # add necessary inputs for group
        self.add('dv0', IndepVarComp('windDirections', np.zeros(nDirections), units=direction_units), promotes=['*'])
        self.add('dv1', IndepVarComp('windSpeeds', np.zeros(nDirections), units=wind_speed_units), promotes=['*'])
        self.add('dv2', IndepVarComp('windFrequencies', np.ones(nDirections)), promotes=['*'])
        self.add('dv3', IndepVarComp('turbineX', np.zeros(nTurbines), units='m'), promotes=['*'])
        self.add('dv4', IndepVarComp('turbineY', np.zeros(nTurbines), units='m'), promotes=['*'])

        # add vars to be seen by MPI and gradient calculations
        self.add('dv5', IndepVarComp('rotorDiameter', np.zeros(nTurbines), units='m'), promotes=['*'])
        self.add('dv6', IndepVarComp('axialInduction', np.zeros(nTurbines)), promotes=['*'])
        self.add('dv7', IndepVarComp('generatorEfficiency', np.zeros(nTurbines)), promotes=['*'])
        self.add('dv8', IndepVarComp('air_density', val=1.1716, units='kg/(m*m*m)'), promotes=['*'])

        # add variable tree IndepVarComps
        add_floris_params_IndepVarComps(self, use_rotor_components=use_rotor_components)
        add_gen_params_IdepVarComps(self, datasize=datasize)
        self.add('jp0', IndepVarComp('model_params:alpha', 0.1, pass_by_obj=True,
                                     desc='parameter for jensen'),
                 promotes=['*'])

        # add variable tree and indep-var stuff for Larsen
        self.add('lp0', IndepVarComp('model_params:Ia', val=0.0, pass_by_object=True), promotes=['*']) # Ambient Turbulence Intensity
        self.add('lp1', IndepVarComp('model_params:air_density', val=0.0,  units='kg/m*m*m', pass_by_object=True), promotes=['*'])
        self.add('lp2', IndepVarComp('model_params:windSpeedToCPCT_wind_speed', np.zeros(datasize), units='m/s',
                                     desc='range of wind speeds', pass_by_obj=True), promotes=['*'])
        self.add('lp3', IndepVarComp('model_params:windSpeedToCPCT_CP', np.zeros(datasize),
                                     desc='power coefficients', pass_by_obj=True), promotes=['*'])
        self.add('lp4', IndepVarComp('model_params:windSpeedToCPCT_CT', np.zeros(datasize),
                                     desc='thrust coefficients', pass_by_obj=True), promotes=['*'])
        self.add('lp5', IndepVarComp('hubHeight', np.zeros(nTurbines)), promotes=['*'])

        if not use_rotor_components:
            self.add('dv9', IndepVarComp('Ct_in', np.zeros(nTurbines)), promotes=['*'])
            self.add('dv10', IndepVarComp('Cp_in', np.zeros(nTurbines)), promotes=['*'])

        # add components and groups
        self.add('windDirectionsDeMUX', DeMUX(nDirections, units=direction_units))
        self.add('windSpeedsDeMUX', DeMUX(nDirections, units=wind_speed_units))

        pg = self.add('all_directions', ParallelGroup(), promotes=['*'])
        if use_rotor_components:
            for direction_id in np.arange(0, nDirections):
                # print 'assigning direction group %i' % direction_id
                pg.add('direction_group%i' % direction_id,
                       DirectionGroup(nTurbines=nTurbines, direction_id=direction_id,
                                      use_rotor_components=use_rotor_components, datasize=datasize,
                                      differentiable=differentiable, add_IdepVarComps=False, nSamples=nSamples,
                                      model=model, model_options=model_options),
                       promotes=(['gen_params:*', 'model_params:*', 'air_density',
                                  'axialInduction', 'generatorEfficiency', 'turbineX', 'turbineY', 'hubHeight',
                                  'yaw%i' % direction_id, 'rotorDiameter', 'wtVelocity%i' % direction_id,
                                  'wtPower%i' % direction_id, 'dir_power%i' % direction_id]
                                 if (nSamples == 0) else
                                 ['gen_params:*', 'model_params:*', 'air_density',
                                  'axialInduction', 'generatorEfficiency', 'turbineX', 'turbineY', 'hubHeight',
                                  'yaw%i' % direction_id, 'rotorDiameter', 'wsPositionX', 'wsPositionY',
                                  'wsPositionZ', 'wtVelocity%i' % direction_id,
                                  'wtPower%i' % direction_id, 'dir_power%i' % direction_id, 'wsArray%i' % direction_id]))
        else:
            for direction_id in np.arange(0, nDirections):
                # print 'assigning direction group %i' % direction_id
                pg.add('direction_group%i' % direction_id,
                       DirectionGroup(nTurbines=nTurbines, direction_id=direction_id,
                                      use_rotor_components=use_rotor_components, datasize=datasize,
                                      differentiable=differentiable, add_IdepVarComps=False, nSamples=nSamples,
                                      model=model, model_options=model_options),
                       promotes=(['Ct_in', 'Cp_in', 'gen_params:*', 'model_params:*', 'air_density', 'axialInduction',
                                  'generatorEfficiency', 'turbineX', 'turbineY', 'yaw%i' % direction_id, 'rotorDiameter',
                                  'hubHeight', 'wtVelocity%i' % direction_id, 'wtPower%i' % direction_id,
                                  'dir_power%i' % direction_id]
                                 if (nSamples == 0) else
                                 ['Ct_in', 'Cp_in', 'gen_params:*', 'model_params:*', 'air_density', 'axialInduction',
                                  'generatorEfficiency', 'turbineX', 'turbineY', 'yaw%i' % direction_id, 'rotorDiameter',
                                  'hubHeight',  'wsPositionX', 'wsPositionY', 'wsPositionZ',
                                  'wtVelocity%i' % direction_id, 'wtPower%i' % direction_id,
                                  'dir_power%i' % direction_id, 'wsArray%i' % direction_id]))

        self.add('powerMUX', MUX(nDirections, units=power_units))
        self.add('AEPcomp', WindFarmAEP(nDirections), promotes=['*'])

        # connect components
        self.connect('windDirections', 'windDirectionsDeMUX.Array')
        self.connect('windSpeeds', 'windSpeedsDeMUX.Array')
        for direction_id in np.arange(0, nDirections):
            self.add('y%i' % direction_id, IndepVarComp('yaw%i' % direction_id, np.zeros(nTurbines), units='deg'), promotes=['*'])
            self.connect('windDirectionsDeMUX.output%i' % direction_id, 'direction_group%i.wind_direction' % direction_id)
            self.connect('windSpeedsDeMUX.output%i' % direction_id, 'direction_group%i.wind_speed' % direction_id)
            self.connect('dir_power%i' % direction_id, 'powerMUX.input%i' % direction_id)
        self.connect('powerMUX.Array', 'dirPowers')