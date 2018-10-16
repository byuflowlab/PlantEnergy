import numpy as np

from openmdao.api import Group, Component, IndepVarComp, ParallelGroup, ScipyGMRES, NLGaussSeidel

#from openmdao.core.mpi_wrap import MPI
#if MPI:
#    from openmdao.api import PetscKSP

from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps

from plantenergy.GeneralWindFarmComponents import MUX, WindFarmAEP, DeMUX, add_gen_params_IdepVarComps
from plantenergy.GeneralWindFarmGroups import DirectionGroup, RotorSolveGroup

class COE(Component):

   def __init__(self, nTurbines):
      
      super(COE, self).__init__()

      self.add_param('turbineHeight', np.zeros(nTurbines), units='m', desc='tower height of each wind turbine in the park') 
      
      self.add_param('aep', val=1.0, desc='annual energy production')
      
      self.add_output('coe', val=0.0, desc='plant cost of energy')
      self.add_output('capex', val=0.0, desc='capital expenditures')

   def solve_nonlinear(self, params, unknowns, resids):
      
      turbineHeight = params['turbineHeight']
      aep = params['aep']
      capex = 0.0
      
      for i in range(turbineHeight.size):
         
         capex = capex + 4915000 + 3.*11.*(turbineHeight[i]**2.15)
      
      coe = (capex*0.08 + 52.*5000.*turbineHeight.size) / aep
      
      unknowns['capex'] = capex
      unknowns['coe'] = coe

class COEGroup(Group):
    """
    Group containing all necessary components for wind plant AEP calculations using the FLORIS model
    """

    def __init__(self, nTurbines, nDirections=1, use_rotor_components=False, datasize=0,
                 differentiable=True, optimizingLayout=False, nSamples=0, wake_model=floris_wrapper,
                 wake_model_options=None, params_IdepVar_func=add_floris_params_IndepVarComps,
                 params_IndepVar_args=None, cp_points=1, cp_curve_spline=None, rec_func_calls=False):

        super(COEGroup, self).__init__()

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

        self.add('dv12', IndepVarComp('cp_curve_cp', np.zeros(datasize),
                                               desc='cp curve cp data', pass_by_obj=True), promotes=['*'])
        self.add('dv13', IndepVarComp('cp_curve_wind_speed', np.zeros(datasize), units='m/s',
                                               desc='cp curve velocity data', pass_by_obj=True), promotes=['*'])
        self.add('dv14', IndepVarComp('cut_in_speed', np.zeros(nTurbines), units='m/s',
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
                                  'dir_power%i' % direction_id, 'cut_in_speed', 'cp_curve_cp', 'cp_curve_wind_speed']
                                 if (nSamples == 0) else
                                 ['Ct_in', 'Cp_in', 'gen_params:*', 'model_params:*', 'air_density', 'axialInduction',
                                  'generatorEfficiency', 'turbineX', 'turbineY', 'yaw%i' % direction_id, 'rotorDiameter',
                                  'hubHeight',  'rated_power', 'cut_in_speed', 'wsPositionX', 'wsPositionY', 'wsPositionZ',
                                  'wtVelocity%i' % direction_id, 'wtPower%i' % direction_id,
                                  'dir_power%i' % direction_id, 'wsArray%i' % direction_id, 'cut_in_speed', 'cp_curve_cp',
                                  'cp_curve_wind_speed']))

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
        
        # add coe calculator
        self.add('coecalc', COE(nTurbines), promotes=['*'])
        self.connect('AEP','aep')
        self.connect('hubHeight','turbineHeight')