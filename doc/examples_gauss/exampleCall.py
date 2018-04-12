from openmdao.api import Problem
from plantenergy.GeneralWindFarmGroups import AEPGroup
from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps

import time
import numpy as np


if __name__ == "__main__":

    # define turbine locations in global reference frame
    turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
    turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])

    # initialize input variable arrays
    nTurbs = turbineX.size
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = 126.4            # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 1.0#0.944
        yaw[turbI] = 0.     # deg.

    # Define flow properties
    wind_speed = 8.0        # m/s
    air_density = 1.1716    # kg/m^3
    # wind_direction = 240    # deg (N = 0 deg., using direction FROM, as in met-mast data)
    wind_direction = 270.-0.523599*180./np.pi    # deg (N = 0 deg., using direction FROM, as in met-mast data)
    print(wind_direction)
    wind_frequency = 1.    # probability of wind in this direction at this speed

    # set up problem
    prob = Problem(root=AEPGroup(nTurbs, nDirections=1, use_rotor_components=False, datasize=0,
                 differentiable=False, optimizingLayout=False, nSamples=0, wake_model=gauss_wrapper,
                 wake_model_options=None, params_IdepVar_func=add_gauss_params_IndepVarComps,
                 params_IndepVar_args=None))

    # initialize problem
    prob.setup()

    # assign values to turbine states
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    prob['yaw0'] = yaw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['windSpeeds'] = np.array([wind_speed])
    prob['air_density'] = air_density
    prob['windDirections'] = np.array([wind_direction])
    prob['windFrequencies'] = np.array([wind_frequency])
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp
    prob['model_params:spread_mode'] = 'bastankhah'
    prob['model_params:yaw_mode'] = 'bastankhah'
    prob['model_params:ky'] = 0.7
    prob['model_params:kz'] = 0.7
    prob['model_params:alpha'] = 2.32
    prob['model_params:beta'] = 0.154
    prob['model_params:I'] = 0.1

    # run the problem
    print('start Bastankhah run')
    tic = time.time()
    prob.run()
    toc = time.time()

    print(prob['turbineX'])

    # print the results
    print('Bastankhah calculation took %.06f sec.'.format(toc-tic))
    print('turbine X positions in wind frame (m): %s'.format(prob['turbineX']))
    print('turbine Y positions in wind frame (m): %s'.format(prob['turbineY']))
    print('yaw (deg) = '.format(prob['yaw0']))
    print('Effective hub velocities (m/s) = '.format(prob['wtVelocity0']))
    print('Turbine powers (kW) = '.format((prob['wtPower0'])))
    print('wind farm power (kW): %s'.format((prob['dir_power0'])))
    print('wind farm AEP for this direction and speed (kWh): %s'.format(prob['AEP']))
