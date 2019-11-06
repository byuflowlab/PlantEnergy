# Software example when an explicit connection to a rotor solver is made
# Currently not functional

from openmdao.api import Problem, Group
from plantenergy.GeneralWindFarmGroups import AEPGroup

import time
import numpy as np
import cPickle as pickle

if __name__ == "__main__":

    use_rotor_components = True

    if use_rotor_components:
        # NREL5MWCPCT = pickle.load(open('./input_files/NREL5MWCPCT_dict.p'))
        NREL5MWCPCT = pickle.load(open('./input_files/NREL5MWCPCT_smooth_dict.p'))
        datasize = NREL5MWCPCT['CP'].size
    else:
        datasize = 0

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
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 0.944
        yaw[turbI] = 0.     # deg.

    # Define flow properties
    wind_speed = 8.0        # m/s
    air_density = 1.1716    # kg/m^3
    wind_direction = 270.-0.523599*180./np.pi    # deg (N = 0 deg., using direction FROM, as in met-mast data)
    wind_frequency = 0.1    # probability of wind in this direction at this speed

    # set up problem
    prob = Problem(root=AEPGroup(nTurbs, use_rotor_components=use_rotor_components, datasize=datasize))

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
    prob['model_params:FLORISoriginal'] = False

    if use_rotor_components:
        prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
        prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
        prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
        # prob['floris_params:ke'] = 0.05
        # prob['floris_params:kd'] = 0.17
        # prob['floris_params:aU'] = 12.0
        # prob['floris_params:bU'] = 1.3
        # prob['floris_params:initialWakeAngle'] = 1.5
        # prob['floris_params:useaUbU'] = True
        # prob['floris_params:useWakeAngle'] = True
        # prob['floris_params:adjustInitialWakeDiamToYaw'] = False
    else:
        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp

    # run the problem
    print 'start FLORIS run'
    tic = time.time()
    prob.run()
    toc = time.time()

    # print the results
    print 'FLORIS calculation took %.06f sec.' % (toc-tic)
    print 'turbine X positions in wind frame (m): %s' % prob['turbineX']
    print 'turbine Y positions in wind frame (m): %s' % prob['turbineY']
    print 'yaw (deg) = ', prob['yaw0']
    print 'Effective hub velocities (m/s) = ', prob['wtVelocity0']
    print 'Turbine powers (kW) = ', prob['wtPower0']
    print 'wind farm power (kW): %s' % prob['dir_power0']
    print 'wind farm AEP for this direction and speed (kWh): %s' % prob['AEP']