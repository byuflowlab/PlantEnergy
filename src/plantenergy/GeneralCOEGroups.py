import numpy as np

import openmdao.api as om

from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps

from plantenergy.GeneralWindFarmComponents import WindFarmAEP, add_gen_params_IdepVarComps
from plantenergy.GeneralWindFarmGroups import DirectionGroup, RotorSolveGroup


class COE(om.ExplicitComponent):

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                               desc="Number of wind turbines.")

    def setup(self):
        nTurbines = self.options['nTurbines']

        self.add_input('turbineHeight', np.zeros(nTurbines), units='m',
                       desc='tower height of each wind turbine in the park')

        self.add_input('aep', val=1.0, desc='annual energy production')

        self.add_output('coe', val=0.0, desc='plant cost of energy')
        self.add_output('capex', val=0.0, desc='capital expenditures')

        # Derivatives
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        turbineHeight = inputs['turbineHeight']
        aep = inputs['aep']
        capex = 0.0

        for i in range(turbineHeight.size):
            capex = capex + 4915000 + 3.*11.*(turbineHeight[i]**2.15)

        coe = (capex*0.08 + 52.*5000.*turbineHeight.size) / aep

        outputs['capex'] = capex
        outputs['coe'] = coe



class COEGroup(om.Group):
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
        self.options.declare('wake_model', types=om.Component, default=floris_wrapper,
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

        # add components and groups
        wind_direct_demux = self.add_subsystem(name='windDirectionsDeMUX', subsys=om.DemuxComp())
        wind_direct_demux.add_var('r', shape=(nDirections,), axis=1, units=direction_units)
        wind_speed_demux = self.add_subsystem(name='windSpeedsDeMUX', subsys=om.DemuxComp())
        wind_speed_demux.add_var('r', shape=(nDirections,), axis=1, units=wind_speed_units)

        pg = self.add('all_directions', om.ParallelGroup(), promotes=['*'])

        if use_rotor_components:
            for direction_id in np.arange(0, nDirections):
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
                                 DirectionGroup(nTurbines=nTurbines, direction_id=direction_id,
                                                use_rotor_components=use_rotor_components, datasize=datasize,
                                                differentiable=differentiable, add_IdepVarComps=False, nSamples=nSamples,
                                                wake_model=wake_model, wake_model_options=wake_model_options, cp_points=cp_points),
                                 promotes=dir_promotes)
        else:
            for direction_id in np.arange(0, nDirections):
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
                                 DirectionGroup(nTurbines=nTurbines, direction_id=direction_id,
                                                use_rotor_components=use_rotor_components, datasize=datasize,
                                                differentiable=differentiable, add_IdepVarComps=False, nSamples=nSamples,
                                                wake_model=wake_model, wake_model_options=wake_model_options, cp_points=cp_points,
                                                cp_curve_spline=cp_curve_spline),
                                 promotes=dir_promotes)

        # print("parallel groups initialized")
        power_mux = self.add_subsystem(name='powerMUX', subsys=om.MuxComp())
        wind_speed_demux.add_var('r', shape=(nDirections,), axis=1, units=power_units)

        self.add_subsystem('AEPcomp', WindFarmAEP(nDirections), promotes=['*'])

        dir_ivc = self.add_subsystem('y_ivc', om.IndepVarComp(), promotes=['*'])

        # connect components
        self.connect('windDirections', 'windDirectionsDeMUX.r')
        self.connect('windSpeeds', 'windSpeedsDeMUX.r')
        for direction_id in np.arange(0, nDirections):
            dir_ivc.add_output('yaw%i' % direction_id, np.zeros(nTurbines), units='deg')
            self.connect('windDirectionsDeMUX.r_%i' % direction_id, 'direction_group%i.wind_direction' % direction_id)
            self.connect('windSpeedsDeMUX.r_%i' % direction_id, 'direction_group%i.wind_speed' % direction_id)
            self.connect('dir_power%i' % direction_id, 'powerMUX.r_%i' % direction_id)

        self.connect('powerMUX.r', 'dirPowers')

        # add coe calculator
        self.add_subsystem('coecalc', COE(nTurbines=nTurbines), promotes=['*'])
        self.connect('AEP','aep')
        self.connect('hubHeight','turbineHeight')