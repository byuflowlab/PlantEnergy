import numpy as np

from openmdao.api import Group, IndepVarComp, ParallelGroup, ScipyGMRES, NLGaussSeidel

from openmdao.core.mpi_wrap import MPI
if MPI:
    from openmdao.api import PetscKSP

from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps
#from wakeexchange.gauss import add_gauss_params_IndepVarComps # not currently using gaussian model

from plantenergy.GeneralWindFarmComponents import WindFrame, AdjustCtCpYaw, MUX, WindFarmAEP, DeMUX, \
    CPCT_Interpolate_Gradients_Smooth, WindDirectionPower, add_gen_params_IdepVarComps, \
    CPCT_Interpolate_Gradients


class RotorSolveGroup(Group):

    def __init__(self, nTurbines, direction_id=0, datasize=0, differentiable=True,
                 use_rotor_components=False, nSamples=0, wake_model=floris_wrapper,
                 wake_model_options=None):

        super(RotorSolveGroup, self).__init__()

        if wake_model_options is None:
            wake_model_options = {'differentiable': differentiable, 'use_rotor_components': use_rotor_components,
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
        self.add('floris', wake_model(nTurbines, direction_id=direction_id, wake_model_options=wake_model_options),
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
                 differentiable=True, add_IdepVarComps=True, params_IdepVar_func=add_floris_params_IndepVarComps,
                 params_IndepVar_args=None, nSamples=0, wake_model=floris_wrapper, wake_model_options=None, cp_points=1,
                 cp_curve_spline=None):

        super(DirectionGroup, self).__init__()

        if add_IdepVarComps:
            if params_IdepVar_func is not None:
                if (params_IndepVar_args is None) and (wake_model is floris_wrapper):
                    params_IndepVar_args = {'use_rotor_components': False}
                elif params_IndepVar_args is None:
                    params_IndepVar_args = {}
                params_IdepVar_func(self, **params_IndepVar_args)
            add_gen_params_IdepVarComps(self, datasize=datasize)

        self.add('directionConversion', WindFrame(nTurbines, differentiable=differentiable, nSamples=nSamples),
                 promotes=['*'])

        if use_rotor_components:
            self.add('rotorGroup', RotorSolveGroup(nTurbines, direction_id=direction_id,
                                                 datasize=datasize, differentiable=differentiable,
                                                 nSamples=nSamples, use_rotor_components=use_rotor_components,
                                                 wake_model=wake_model, wake_model_options=wake_model_options),
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

            self.add('myModel', wake_model(nTurbines, direction_id=direction_id, wake_model_options=wake_model_options),
                     promotes=(['model_params:*', 'wind_speed', 'axialInduction',
                                'turbineXw', 'turbineYw', 'rotorDiameter', 'yaw%i' % direction_id, 'hubHeight',
                                'wtVelocity%i' % direction_id]
                               if (nSamples == 0) else
                               ['model_params:*', 'wind_speed', 'axialInduction',
                                'turbineXw', 'turbineYw', 'rotorDiameter', 'yaw%i' % direction_id, 'hubHeight',
                                'wtVelocity%i' % direction_id, 'wsPositionXw', 'wsPositionYw', 'wsPositionZ',
                                'wsArray%i' % direction_id]))

        self.add('powerComp', WindDirectionPower(nTurbines=nTurbines, direction_id=direction_id, differentiable=True,
                                                 use_rotor_components=use_rotor_components, cp_points=cp_points,
                                                 cp_curve_spline=cp_curve_spline),
                 promotes=['air_density', 'generatorEfficiency', 'rotorDiameter',
                           'wtVelocity%i' % direction_id, 'rated_power',
                           'wtPower%i' % direction_id, 'dir_power%i' % direction_id, 'cut_in_speed', 'cp_curve_cp',
                           'cp_curve_wind_speed','cut_out_speed', 'rated_wind_speed', 'use_power_curve_definition'])

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
                 differentiable=True, optimizingLayout=False, nSamples=0, wake_model=floris_wrapper,
                 wake_model_options=None, params_IdepVar_func=add_floris_params_IndepVarComps,
                 params_IndepVar_args=None, cp_points=1, cp_curve_spline=None, rec_func_calls=False):

        super(AEPGroup, self).__init__()

        if wake_model_options is None:
            wake_model_options = {'differentiable': differentiable, 'use_rotor_components': use_rotor_components,
                             'nSamples': nSamples, 'verbose': False}

        # providing default unit types for general MUX/DeMUX components
        power_units = 'kW'
        direction_units = 'deg'
        wind_speed_units = 'm/s'

        # add necessary inputs for group
        self.add('dv0', IndepVarComp('windDirections', np.zeros(nDirections), units=direction_units), promotes=['*'])
        self.add('dv1', IndepVarComp('windSpeeds', np.zeros(nDirections), units=wind_speed_units), promotes=['*'])
        self.add('dv2', IndepVarComp('windFrequencies', np.ones(nDirections)), promotes=['*'])
        self.add('dv3', IndepVarComp('turbineX', np.zeros(nTurbines), units='m'), promotes=['*'])
        self.add('dv4', IndepVarComp('turbineY', np.zeros(nTurbines), units='m'), promotes=['*'])
        self.add('dv4p5', IndepVarComp('hubHeight', np.zeros(nTurbines), units='m'), promotes=['*'])

        # add vars to be seen by MPI and gradient calculations
        self.add('dv5', IndepVarComp('rotorDiameter', np.zeros(nTurbines), units='m'), promotes=['*'])
        self.add('dv6', IndepVarComp('axialInduction', np.zeros(nTurbines)), promotes=['*'])
        self.add('dv7', IndepVarComp('generatorEfficiency', np.zeros(nTurbines)), promotes=['*'])
        self.add('dv8', IndepVarComp('air_density', val=1.1716, units='kg/(m*m*m)'), promotes=['*'])
        self.add('dv9', IndepVarComp('rated_power', np.ones(nTurbines)*5000., units='kW',
                       desc='rated power for each turbine', pass_by_obj=True), promotes=['*'])
        if not use_rotor_components:
            self.add('dv10', IndepVarComp('Ct_in', np.zeros(nTurbines)), promotes=['*'])
            self.add('dv11', IndepVarComp('Cp_in', np.zeros(nTurbines)), promotes=['*'])
            self.add('dv12', IndepVarComp('cut_out_speed', val=np.zeros(nTurbines), units='m/s',
                                                   pass_by_obj=True), promotes=['*'])
            self.add('dv14', IndepVarComp('rated_wind_speed', val=np.zeros(nTurbines), units='m/s',
                                                   pass_by_obj=True), promotes=['*'])
            self.add('dv15', IndepVarComp('use_power_curve_definition', val=False,
                                                   pass_by_obj=True), promotes=['*'])

        self.add('dv16', IndepVarComp('cp_curve_cp', np.zeros(datasize),
                                               desc='cp curve cp data', pass_by_obj=True), promotes=['*'])
        self.add('dv17', IndepVarComp('cp_curve_wind_speed', np.zeros(datasize), units='m/s',
                                               desc='cp curve velocity data', pass_by_obj=True), promotes=['*'])
        self.add('dv18', IndepVarComp('cut_in_speed', np.zeros(nTurbines), units='m/s',
                                               desc='cut-in speed of wind turbines', pass_by_obj=True), promotes=['*'])


        # add variable tree IndepVarComps
        add_gen_params_IdepVarComps(self, datasize=datasize)

        # indep variable components for wake model
        if params_IdepVar_func is not None:
            if (params_IndepVar_args is None) and (wake_model is floris_wrapper):
                params_IndepVar_args = {'use_rotor_components': False}
            elif params_IndepVar_args is None:
                params_IndepVar_args = {}
            params_IdepVar_func(self, **params_IndepVar_args)

        # add components and groups
        self.add('windDirectionsDeMUX', DeMUX(nDirections, units=direction_units))
        self.add('windSpeedsDeMUX', DeMUX(nDirections, units=wind_speed_units))

        # print("initializing parallel groups")
        # if use_parallel_group:
        #     direction_group = ParallelGroup()
        # else:
        #     direction_group = Group()

        pg = self.add('all_directions', ParallelGroup(), promotes=['*'])
        if use_rotor_components:
            for direction_id in np.arange(0, nDirections):
                # print('assigning direction group %i'.format(direction_id))
                pg.add('direction_group%i' % direction_id,
                       DirectionGroup(nTurbines=nTurbines, direction_id=direction_id,
                                      use_rotor_components=use_rotor_components, datasize=datasize,
                                      differentiable=differentiable, add_IdepVarComps=False, nSamples=nSamples,
                                      wake_model=wake_model, wake_model_options=wake_model_options, cp_points=cp_points),
                       promotes=(['gen_params:*', 'model_params:*', 'air_density',
                                  'axialInduction', 'generatorEfficiency', 'turbineX', 'turbineY', 'hubHeight',
                                  'yaw%i' % direction_id, 'rotorDiameter', 'rated_power', 'wtVelocity%i' % direction_id,
                                  'wtPower%i' % direction_id, 'dir_power%i' % direction_id]
                                 if (nSamples == 0) else
                                 ['gen_params:*', 'model_params:*', 'air_density',
                                  'axialInduction', 'generatorEfficiency', 'turbineX', 'turbineY', 'hubHeight',
                                  'yaw%i' % direction_id, 'rotorDiameter', 'rated_power', 'wsPositionX', 'wsPositionY',
                                  'wsPositionZ', 'wtVelocity%i' % direction_id,
                                  'wtPower%i' % direction_id, 'dir_power%i' % direction_id, 'wsArray%i' % direction_id]))
        else:
            for direction_id in np.arange(0, nDirections):
                # print('assigning direction group %i'.format(direction_id))
                pg.add('direction_group%i' % direction_id,
                       DirectionGroup(nTurbines=nTurbines, direction_id=direction_id,
                                      use_rotor_components=use_rotor_components, datasize=datasize,
                                      differentiable=differentiable, add_IdepVarComps=False, nSamples=nSamples,
                                      wake_model=wake_model, wake_model_options=wake_model_options, cp_points=cp_points,
                                      cp_curve_spline=cp_curve_spline),
                       promotes=(['Ct_in', 'Cp_in', 'gen_params:*', 'model_params:*', 'air_density', 'axialInduction',
                                  'generatorEfficiency', 'turbineX', 'turbineY', 'yaw%i' % direction_id, 'rotorDiameter',
                                  'hubHeight', 'rated_power', 'wtVelocity%i' % direction_id, 'wtPower%i' % direction_id,
                                  'dir_power%i' % direction_id, 'cut_in_speed', 'cp_curve_cp', 'cp_curve_wind_speed',
                                  'cut_out_speed', 'rated_wind_speed', 'use_power_curve_definition']
                                 if (nSamples == 0) else
                                 ['Ct_in', 'Cp_in', 'gen_params:*', 'model_params:*', 'air_density', 'axialInduction',
                                  'generatorEfficiency', 'turbineX', 'turbineY', 'yaw%i' % direction_id, 'rotorDiameter',
                                  'hubHeight',  'rated_power', 'cut_in_speed', 'wsPositionX', 'wsPositionY', 'wsPositionZ',
                                  'wtVelocity%i' % direction_id, 'wtPower%i' % direction_id,
                                  'dir_power%i' % direction_id, 'wsArray%i' % direction_id, 'cut_in_speed', 'cp_curve_cp',
                                  'cp_curve_wind_speed', 'cut_out_speed', 'rated_wind_speed', 'use_power_curve_definition']))

        # print("parallel groups initialized")
        self.add('powerMUX', MUX(nDirections, units=power_units))
        self.add('AEPcomp', WindFarmAEP(nDirections, rec_func_calls=rec_func_calls), promotes=['*'])

        # connect components
        self.connect('windDirections', 'windDirectionsDeMUX.Array')
        self.connect('windSpeeds', 'windSpeedsDeMUX.Array')
        for direction_id in np.arange(0, nDirections):
            self.add('y%i' % direction_id, IndepVarComp('yaw%i' % direction_id, np.zeros(nTurbines), units='deg'), promotes=['*'])
            self.connect('windDirectionsDeMUX.output%i' % direction_id, 'direction_group%i.wind_direction' % direction_id)
            self.connect('windSpeedsDeMUX.output%i' % direction_id, 'direction_group%i.wind_speed' % direction_id)
            self.connect('dir_power%i' % direction_id, 'powerMUX.input%i' % direction_id)
        self.connect('powerMUX.Array', 'dirPowers')