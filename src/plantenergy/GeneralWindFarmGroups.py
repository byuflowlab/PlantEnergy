from __future__ import print_function, division, absolute_import

import numpy as np

import openmdao.api as om
from openmdao.utils.mpi import MPI

from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps
#from wakeexchange.gauss import add_gauss_params_IndepVarComps # not currently using gaussian model

from plantenergy.GeneralWindFarmComponents import WindFrame, AdjustCtCpYaw, WindFarmAEP, \
    CPCT_Interpolate_Gradients_Smooth, WindDirectionPower, add_gen_params_IdepVarComps


class RotorSolveGroup(om.Group):

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")
        self.options.declare('direction_id', types=int, default=0,
                             desc="Direction index.")
        self.options.declare('datasize', types=int, default=0,
                             desc="Dimension of the coefficient arrays.")
        self.options.declare('differentiable', types=bool, default=True,
                             desc="Differentiation flag, set to False to force finite difference.")
        self.options.declare('use_rotor_components', types=bool, default=False,
                             desc="Set to True to use rotor components.")
        self.options.declare('nSamples', types=int, default=0,
                             desc="Number of samples for the visualization arrays.")
        self.options.declare('wake_model', default=floris_wrapper,
                             desc="Wake Model")
        self.options.declare('wake_model_options', types=dict, default=None, allow_none=True,
                             desc="Wake Model instantiation parameters.")

    def setup(self):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']
        datasize = opt['datasize']
        differentiable = opt['differentiable']
        use_rotor_components = opt['use_rotor_components']
        nSamples = opt['nSamples']
        wake_model = opt['wake_model']
        wake_model_options = opt['wake_model_options']

        if wake_model_options is None:
            wake_model_options = {'differentiable': differentiable, 'use_rotor_components': use_rotor_components,
                             'nSamples': nSamples}

        # set up iterative solvers
        epsilon = 1E-6
        if MPI:
            self.linear_solver = om.PetscKSP()
        else:
            self.linear_solver = om.ScipyKrylov()

        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options['atol'] = epsilon

        self.add_subsystem('CtCp', CPCT_Interpolate_Gradients_Smooth(nTurbines=nTurbines, direction_id=direction_id, datasize=datasize),
                           promotes=['gen_params:*', 'yaw%i' % direction_id,
                                     'wtVelocity%i' % direction_id, 'Cp_out'])

        if (nSamples == 0):
            floris_promotes = ['model_params:*', 'wind_speed', 'axialInduction',
                               'turbineXw', 'turbineYw', 'rotorDiameter', 'yaw%i' % direction_id, 'hubHeight',
                                      'wtVelocity%i' % direction_id]
        else:
            floris_promotes = ['model_params:*', 'wind_speed', 'axialInduction',
                               'turbineXw', 'turbineYw', 'rotorDiameter', 'yaw%i' % direction_id, 'hubHeight',
                               'wtVelocity%i' % direction_id, 'wsPositionX', 'wsPositionY', 'wsPositionZ',
                               'wsArray%i' % direction_id]

        # TODO refactor the model component instance
        self.add_subsystem('floris', wake_model(nTurbines=nTurbines, direction_id=direction_id, wake_model_options=wake_model_options),
                           promotes=floris_promotes)

        self.connect('CtCp.Ct_out', 'floris.Ct')



class DirectionGroup(om.Group):
    """
    Group containing all necessary components for wind plant calculations
    in a single direction
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")
        self.options.declare('direction_id', types=int, default=0,
                             desc="Direction index.")
        self.options.declare('use_rotor_components', types=bool, default=False,
                             desc="Set to True to use rotor components.")
        self.options.declare('datasize', types=int, default=0,
                             desc="Dimension of the coefficient arrays.")
        self.options.declare('differentiable', types=bool, default=True,
                             desc="Differentiation flag, set to False to force finite difference.")
        self.options.declare('add_IdepVarComps', types=bool, default=True,
                             desc="When True (default), add indepvarcomps.")
        self.options.declare('params_IdepVar_func', default=add_floris_params_IndepVarComps,
                             desc="Function to call to add indepvarcomps.")
        self.options.declare('params_IdepVar_args', types=dict, default=None, allow_none=True,
                             desc="Arguments for function that adds indepvarcomps.")
        self.options.declare('nSamples', types=int, default=0,
                             desc="Number of samples for the visualization arrays.")
        self.options.declare('wake_model', default=floris_wrapper,
                             desc="Wake Model")
        self.options.declare('wake_model_options', types=dict, default=None, allow_none=True,
                             desc="Wake Model")
        self.options.declare('cp_points', types=int, default=1,
                             desc="Number of spline control points.")
        self.options.declare('cp_curve_spline', default=None,
                             desc="Values for cp spline. When set to None (default), the component will make a spline using np.interp.")

    def setup(self):

        self.linear_solver = om.DirectSolver()

        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']
        datasize = opt['datasize']
        differentiable = opt['differentiable']
        add_IdepVarComps = opt['add_IdepVarComps']
        params_IdepVar_func = opt['params_IdepVar_func']
        params_IdepVar_args = opt['params_IdepVar_args']
        use_rotor_components = opt['use_rotor_components']
        nSamples = opt['nSamples']
        wake_model = opt['wake_model']
        wake_model_options = opt['wake_model_options']
        cp_points = opt['cp_points']
        cp_curve_spline = opt['cp_curve_spline']

        if add_IdepVarComps:
            if params_IdepVar_func is not None:
                if (params_IdepVar_args is None) and (wake_model is floris_wrapper):
                    params_IdepVar_args = {'use_rotor_components': False}
                elif params_IdepVar_args is None:
                    params_IdepVar_args = {}
                params_IdepVar_func(self, **params_IdepVar_args)
            add_gen_params_IdepVarComps(self, datasize=datasize)

        self.add_subsystem('directionConversion', WindFrame(nTurbines=nTurbines, differentiable=differentiable, nSamples=nSamples),
                           promotes=['*'])

        if use_rotor_components:

            if (nSamples == 0):
                rotor_promotes = ['gen_params:*', 'yaw%i' % direction_id, 'wtVelocity%i' % direction_id,
                                  'model_params:*', 'wind_speed', 'axialInduction',
                                  'turbineXw', 'turbineYw', 'rotorDiameter', 'hubHeight']
            else:
                rotor_promotes = ['gen_params:*', 'yaw%i' % direction_id, 'wtVelocity%i' % direction_id,
                                  'model_params:*', 'wind_speed', 'axialInduction',
                                  'turbineXw', 'turbineYw', 'rotorDiameter', 'hubHeight', 'wsPositionX', 'wsPositionY', 'wsPositionZ',
                                  'wsArray%i' % direction_id]

            self.add_subsystem('rotorGroup', RotorSolveGroup(nTurbines=nTurbines, direction_id=direction_id,
                                                             datasize=datasize, differentiable=differentiable,
                                                             nSamples=nSamples, use_rotor_components=use_rotor_components,
                                                             wake_model=wake_model, wake_model_options=wake_model_options),
                               promotes=rotor_promotes)

        else:
            self.add_subsystem('CtCp', AdjustCtCpYaw(nTurbines=nTurbines, direction_id=direction_id, differentiable=differentiable),
                               promotes=['Ct_in', 'Cp_in', 'gen_params:*', 'yaw%i' % direction_id])

            if (nSamples == 0):
                wake_promotes = ['model_params:*', 'wind_speed', 'axialInduction',
                                 'turbineXw', 'turbineYw', 'rotorDiameter', 'yaw%i' % direction_id, 'hubHeight',
                                 'wtVelocity%i' % direction_id]
            else:
                wake_promotes = ['model_params:*', 'wind_speed', 'axialInduction',
                                 'turbineXw', 'turbineYw', 'rotorDiameter', 'yaw%i' % direction_id, 'hubHeight',
                                 'wtVelocity%i' % direction_id, 'wsPositionXw', 'wsPositionYw', 'wsPositionZ',
                                 'wsArray%i' % direction_id]

            self.add_subsystem('myModel', wake_model(nTurbines=nTurbines, direction_id=direction_id, wake_model_options=wake_model_options),
                               promotes=wake_promotes)

        self.add_subsystem('powerComp', WindDirectionPower(nTurbines=nTurbines, direction_id=direction_id, differentiable=True,
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


class AEPGroup(om.Group):
    """
    Group containing all necessary components for wind plant AEP calculations using the FLORIS model
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")
        self.options.declare('nDirections', types=int, default=1,
                             desc="Number of directions.")
        self.options.declare('use_rotor_components', types=bool, default=False,
                             desc="Set to True to use rotor components.")
        self.options.declare('datasize', types=int, default=0,
                             desc="Dimension of the coefficient arrays.")
        self.options.declare('differentiable', types=bool, default=True,
                             desc="Differentiation flag, set to False to force finite difference.")
        self.options.declare('optimizingLayout', types=bool, default=False,
                             desc="This option does nothing.")
        self.options.declare('nSamples', types=int, default=0,
                             desc="Number of samples for the visualization arrays.")
        self.options.declare('wake_model', default=floris_wrapper,
                             desc="Wake Model")
        self.options.declare('wake_model_options', types=dict, default=None, allow_none=True,
                             desc="Wake Model")
        self.options.declare('params_IdepVar_func', default=add_floris_params_IndepVarComps,
                             desc="Function to call to add indepvarcomps.")
        self.options.declare('params_IdepVar_args', types=dict, default=None, allow_none=True,
                             desc="Arguments for function that adds indepvarcomps.")
        self.options.declare('cp_points', types=int, default=1,
                             desc="Number of spline control points.")
        self.options.declare('cp_curve_spline', default=None,
                             desc="Values for cp spline. When set to None (default), the component will make a spline using np.interp.")
        self.options.declare('record_function_calls', default=False,
                            desc="If true, than function calls and sensitiv ity function calls will be recorded at the top level")
        self.options.declare('runparallel', default=False,
                             desc="If true, then groups will be executed in parallel")

    def setup(self):
        opt = self.options
        nTurbines = opt['nTurbines']
        nDirections = opt['nDirections']
        use_rotor_components = opt['use_rotor_components']
        datasize = opt['datasize']
        differentiable = opt['differentiable']
        nSamples = opt['nSamples']
        wake_model = opt['wake_model']
        wake_model_options = opt['wake_model_options']
        params_IdepVar_func = opt['params_IdepVar_func']
        params_IdepVar_args = opt['params_IdepVar_args']
        cp_points = opt['cp_points']
        cp_curve_spline = opt['cp_curve_spline']
        record_function_calls = opt['record_function_calls']
        runparallel = opt['runparallel']

        if wake_model_options is None:
            wake_model_options = {'differentiable': differentiable, 'use_rotor_components': use_rotor_components,
                             'nSamples': nSamples, 'verbose': False}

        # providing default unit types for general MUX/DeMUX components
        power_units = 'kW'
        direction_units = 'deg'
        wind_speed_units = 'm/s'

        ivc = self.add_subsystem('desvars', om.IndepVarComp(), promotes=['*'])

        # add necessary inputs for group
        ivc.add_output('windDirections', np.zeros(nDirections), units=direction_units)
        ivc.add_output('windSpeeds', np.zeros(nDirections), units=wind_speed_units)
        ivc.add_output('windFrequencies', np.ones(nDirections))
        ivc.add_output('turbineX', np.zeros(nTurbines), units='m')
        ivc.add_output('turbineY', np.zeros(nTurbines), units='m')
        ivc.add_output('hubHeight', np.zeros(nTurbines), units='m')

        # add vars to be seen by MPI and gradient calculations
        ivc.add_output('rotorDiameter', np.zeros(nTurbines), units='m')
        ivc.add_output('axialInduction', np.zeros(nTurbines))
        ivc.add_output('generatorEfficiency', np.zeros(nTurbines))
        ivc.add_output('air_density', val=1.1716, units='kg/(m*m*m)')
        ivc.add_discrete_output('rated_power', np.ones(nTurbines)*5000.,
                                desc='rated power for each turbine (kW)')

        if not use_rotor_components:
            ivc.add_output('Ct_in', np.zeros(nTurbines))
            ivc.add_output('Cp_in', np.zeros(nTurbines))
            ivc.add_discrete_output('cut_out_speed', val=np.zeros(nTurbines),
                                    desc='in units of m/s')
            ivc.add_discrete_output('rated_wind_speed', val=np.zeros(nTurbines),
                                    desc='in units of m/s')
            ivc.add_discrete_output('use_power_curve_definition', val=False)

        ivc.add_discrete_output('cp_curve_cp', np.zeros(datasize),
                                desc='cp curve cp data')
        ivc.add_discrete_output('cp_curve_wind_speed', np.zeros(datasize),
                                desc='cp curve velocity data (m/s)')
        ivc.add_discrete_output('cut_in_speed', np.zeros(nTurbines),
                                desc='cut-in speed of wind turbines (m/s)')

                # add variable tree IndepVarComps
        add_gen_params_IdepVarComps(self, datasize=datasize)

        # indep variable components for wake model
        if params_IdepVar_func is not None:
            if (params_IdepVar_args is None) and (wake_model is floris_wrapper):
                params_IdepVar_args = {'use_rotor_components': False}
            elif params_IdepVar_args is None:
                params_IdepVar_args = {}
            params_IdepVar_func(self, **params_IdepVar_args)

        # indepvarcomp for yaw
        dir_ivc = self.add_subsystem('y_ivc', om.IndepVarComp(), promotes=['*'])

        # add components and groups
        wind_direct_demux = self.add_subsystem(name='windDirectionsDeMUX', subsys=om.DemuxComp(vec_size=nDirections))
        wind_direct_demux.add_var('r', shape=(nDirections, ), units=direction_units)
        wind_speed_demux = self.add_subsystem(name='windSpeedsDeMUX', subsys=om.DemuxComp(vec_size=nDirections))
        wind_speed_demux.add_var('r', shape=(nDirections, ), units=wind_speed_units)

        if runparallel:
            pg = self.add_subsystem('all_directions', om.ParallelGroup(), promotes=['*'])
        else:
            pg = self.add_subsystem('all_directions', om.Group(), promotes=['*'])


        if use_rotor_components:
            for direction_id in np.arange(0, nDirections, dtype=int):
                # print('assigning direction group %i'.format(direction_id))

                if (nSamples == 0):
                    dir_promotes = ['gen_params:*', 'model_params:*', 'air_density',
                                      'axialInduction', 'generatorEfficiency', 'turbineX', 'turbineY', 'hubHeight',
                                      'yaw%i' % direction_id, 'rotorDiameter', 'rated_power', 'wtVelocity%i' % direction_id,
                                      'wtPower%i' % direction_id, 'dir_power%i' % direction_id]
                else:
                    dir_promotes = ['gen_params:*', 'model_params:*', 'air_density',
                                    'axialInduction', 'generatorEfficiency', 'turbineX', 'turbineY', 'hubHeight',
                                    'yaw%i' % direction_id, 'rotorDiameter', 'rated_power', 'wsPositionX', 'wsPositionY',
                                    'wsPositionZ', 'wtVelocity%i' % direction_id,
                                    'wtPower%i' % direction_id, 'dir_power%i' % direction_id, 'wsArray%i' % direction_id]

                pg.add_subsystem('direction_group%i' % direction_id,
                                 DirectionGroup(nTurbines=nTurbines, direction_id=int(direction_id),
                                                use_rotor_components=use_rotor_components, datasize=datasize,
                                                differentiable=differentiable, add_IdepVarComps=False, nSamples=nSamples,
                                                wake_model=wake_model, wake_model_options=wake_model_options, cp_points=cp_points),
                                 promotes=dir_promotes)
        else:
            for direction_id in np.arange(0, nDirections, dtype=int):
                # print('assigning direction group %i'.format(direction_id))

                if (nSamples == 0):
                    dir_promotes = ['Ct_in', 'Cp_in', 'gen_params:*', 'model_params:*', 'air_density', 'axialInduction',
                                    'generatorEfficiency', 'turbineX', 'turbineY', 'yaw%i' % direction_id, 'rotorDiameter',
                                    'hubHeight', 'rated_power', 'wtVelocity%i' % direction_id, 'wtPower%i' % direction_id,
                                    'dir_power%i' % direction_id, 'cut_in_speed', 'cp_curve_cp', 'cp_curve_wind_speed',
                                    'cut_out_speed', 'rated_wind_speed', 'use_power_curve_definition']
                else:
                    dir_promotes = ['Ct_in', 'Cp_in', 'gen_params:*', 'model_params:*', 'air_density', 'axialInduction',
                                    'generatorEfficiency', 'turbineX', 'turbineY', 'yaw%i' % direction_id, 'rotorDiameter',
                                    'hubHeight',  'rated_power', 'cut_in_speed', 'wsPositionX', 'wsPositionY', 'wsPositionZ',
                                    'wtVelocity%i' % direction_id, 'wtPower%i' % direction_id,
                                    'dir_power%i' % direction_id, 'wsArray%i' % direction_id, 'cut_in_speed', 'cp_curve_cp',
                                    'cp_curve_wind_speed', 'cut_out_speed', 'rated_wind_speed', 'use_power_curve_definition']

                pg.add_subsystem('direction_group%i' % direction_id,
                                 DirectionGroup(nTurbines=nTurbines, direction_id=int(direction_id),
                                                use_rotor_components=use_rotor_components, datasize=datasize,
                                                differentiable=differentiable, add_IdepVarComps=False, nSamples=nSamples,
                                                wake_model=wake_model, wake_model_options=wake_model_options, cp_points=cp_points,
                                                cp_curve_spline=cp_curve_spline),
                                 promotes=dir_promotes)

        # print("parallel groups initialized")
        power_mux = self.add_subsystem(name='powerMUX', subsys=om.MuxComp(vec_size=nDirections))
        power_mux.add_var('r', shape=(1, ), units=power_units)

        self.add_subsystem('AEPcomp', WindFarmAEP(nDirections=nDirections, record_function_calls=record_function_calls), promotes=['*'])

        # connect components
        self.connect('windDirections', 'windDirectionsDeMUX.r')
        self.connect('windSpeeds', 'windSpeedsDeMUX.r')
        for direction_id in np.arange(0, nDirections, dtype=int):
            dir_ivc.add_output('yaw%i' % direction_id, np.zeros(nTurbines), units='deg')
            self.connect('windDirectionsDeMUX.r_%i' % direction_id, 'direction_group%i.wind_direction' % direction_id)
            self.connect('windSpeedsDeMUX.r_%i' % direction_id, 'direction_group%i.wind_speed' % direction_id)
            self.connect('dir_power%i' % direction_id, 'powerMUX.r_%i' % direction_id)

        self.connect('powerMUX.r', 'dirPowers')
