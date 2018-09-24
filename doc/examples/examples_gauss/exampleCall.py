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
    hubHeight = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = 126.4            # m
        hubHeight[turbI] = 90.
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
    turbulence_intensity = .063 # ambient turbulence intensity of the inflow to the farm
    z_ref = 90.0        # reference height of wind speeds
    z_0 = 0.001         # ground height
    shear_exp = 0.11

    # set up model
    sort_turbs = True
    wake_combination_method = 1  # can be [0:Linear freestreem superposition,
                                        #  1:Linear upstream velocity superposition,
                                        #  2:Sum of squares freestream superposition,
                                        #  3:Sum of squares upstream velocity superposition]

    ti_calculation_method = 2  # can be [0:No added TI calculations,
                                       # 2:TI by Niayifar and Porte Agel 2016
    calc_k_star = True
    ky = 0.3837 * turbulence_intensity + 0.003678
    kz = 0.3837 * turbulence_intensity + 0.003678


    filename = "../input_files/NREL5MWCPCT_dict.p"
    # filename = "../input_files/NREL5MWCPCT_smooth_dict.p"
    import cPickle as pickle

    data = pickle.load(open(filename, "rb"))
    ct_curve = np.zeros([data['wind_speed'].size, 2])
    ct_curve[:, 0] = data['wind_speed']
    ct_curve[:, 1] = data['CT']

    gauss_model_options = {'nSamples': 0,
                           'nRotorPoints': 1,
                           'use_ct_curve': True,
                           'ct_curve': ct_curve,
                           'interp_type': 1,
                           'use_rotor_components': False,
                           'verbose': False}

    # set up problem
    prob = Problem(root=AEPGroup(nTurbs, nDirections=1, use_rotor_components=False, datasize=0,
                 differentiable=False, optimizingLayout=False, nSamples=0, wake_model=gauss_wrapper,
                 wake_model_options=gauss_model_options, params_IdepVar_func=add_gauss_params_IndepVarComps,
                 params_IndepVar_args={'nRotorPoints': 1}))

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
    prob['hubHeight'] = hubHeight

    # assign values to model specific inputs
    prob['model_params:wake_combination_method'] = wake_combination_method
    prob['model_params:ti_calculation_method'] = ti_calculation_method
    prob['model_params:calc_k_star'] = calc_k_star
    prob['model_params:sort'] = sort_turbs
    prob['model_params:z_ref'] = z_ref
    prob['model_params:z_0'] = z_0
    prob['model_params:ky'] = ky
    prob['model_params:kz'] = kz
    prob['model_params:I'] = turbulence_intensity
    prob['model_params:shear_exp'] = shear_exp

    # run the problem
    print('start Bastankhah run')
    tic = time.time()
    prob.run()
    toc = time.time()

    # print the results
    print('Bastankhah calculation took %.06f sec.' % (toc-tic))
    print('turbine X positions in wind frame (m): %s' % format(prob['turbineX']))
    print('turbine Y positions in wind frame (m): %s' % format(prob['turbineY']))
    print('yaw (deg) = %s' % format(prob['yaw0']))
    print('Effective hub velocities (m/s) = %s' % format(prob['wtVelocity0']))
    print('Turbine powers (kW) = %s' % format((prob['wtPower0'])))
    print('wind farm power (kW): %s' % format((prob['dir_power0'])))
    print('wind farm AEP for this direction and speed (kWh): %s' % format(prob['AEP']))
