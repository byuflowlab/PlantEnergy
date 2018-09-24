from openmdao.api import Problem
from plantenergy.GeneralWindFarmGroups import AEPGroup
from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps

import time
import numpy as np


if __name__ == "__main__":

    # define turbine locations in global reference frame
    turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
    turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])
    turbineZ = np.array([90.0, 100.0, 90.0, 80.0, 70.0, 90.0])


    # define turbine size
    hub_height = 90.0

    z_ref = hub_height
    z_0 = 0.0

    # load performance characteristics
    cut_in_speed = 3.  # m/s
    rated_power = 5000.  # kW

    filename = "input_files/NREL5MWCPCT_dict.p"
    # filename = "../input_files/NREL5MWCPCT_smooth_dict.p"
    from numpy.core import multiarray
    import cPickle as pickle

    data = pickle.load(open(filename, "r"))
    ct_curve = np.zeros([data['wind_speed'].size, 2])
    ct_curve[:, 0] = data['wind_speed']
    ct_curve[:, 1] = data['CT']

    # cp_curve_cp = data['CP']
    # cp_curve_vel = data['wind_speed']

    loc0 = np.where(data['wind_speed'] < 11.55)
    loc1 = np.where(data['wind_speed'] > 11.7)

    from scipy.interpolate import UnivariateSpline
    cp_curve_cp = np.hstack([data['CP'][loc0], data['CP'][loc1]])
    cp_curve_vel = np.hstack([data['wind_speed'][loc0], data['wind_speed'][loc1]])
    cp_curve_spline = UnivariateSpline(cp_curve_vel, cp_curve_cp, ext='const')
    cp_curve_spline.set_smoothing_factor(.000001)

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
        hubHeight[turbI] = hub_height            # m
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

    wake_model_options = {'nSamples': 0,
                          'nRotorPoints': 1,
                          'use_ct_curve': True,
                          'ct_curve': ct_curve,
                          'interp_type': 1,
                          'differentiable': True,
                          'use_rotor_components': False}
    # set up problem
    prob = Problem(root=AEPGroup(nTurbs, differentiable=True, use_rotor_components=False, wake_model=floris_wrapper,
                                 params_IdepVar_func=add_floris_params_IndepVarComps,
                                 wake_model_options=wake_model_options))

    # initialize problem
    prob.setup()

    # assign values to turbine states
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    prob['hubHeight'] = turbineZ
    prob['yaw0'] = yaw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['hubHeight'] = hubHeight
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['windSpeeds'] = np.array([wind_speed])
    prob['air_density'] = air_density
    prob['windDirections'] = np.array([wind_direction])
    prob['windFrequencies'] = np.array([wind_frequency])
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp
    prob['model_params:cos_spread'] = 1E12         # turns off cosine spread (just needs to be very large)
    prob['model_params:shearExp'] = 0.25         # turns off cosine spread (just needs to be very large)
    prob['model_params:z_ref'] = 80.         # turns off cosine spread (just needs to be very large)
    prob['model_params:z0'] = 0.         # turns off cosine spread (just needs to be very large)
    # prob['floris_params:useWakeAngle'] = True
    # run the problem
    print('set speeds ', prob['windSpeeds'])
    print('start FLORIS run')
    # print prob['windSpeeds']
    # quit()
    tic = time.time()
    prob.run()
    toc = time.time()

    print(prob['turbineX'])

    # print the results
    print('FLORIS calculation took {0} sec.'.format(toc-tic))
    print('turbine X positions in wind frame (m): {0}'.format(prob['turbineX']))
    print('turbine Y positions in wind frame (m): {0}'.format(prob['turbineY']))
    print('yaw (deg) = ', prob['yaw0'])
    print('Effective hub velocities (m/s) = ', prob['wtVelocity0'])
    print('Turbine powers (kW) = ', prob['wtPower0'])
    print('wind farm power (kW) = ', prob['dir_power0'])
    print('wind farm AEP for this direction and speed (kWh) = ', prob['AEP'])
