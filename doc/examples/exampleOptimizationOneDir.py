# No longer functional - OptPowerOneDir is no longer used

from openmdao.api import Problem, ScipyOptimizer, pyOptSparseDriver
from plantenergy.OptimizationGroups import OptPowerOneDir

import time
import numpy as np


if __name__ == "__main__":

    # define turbine size
    rotor_diameter = 126.4  # (m)

    # define turbine locations in global reference frame
    nRows = 3       # number of rows and columns in grid
    spacing = 5     # turbine grid spacing in diameters
    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX = np.ndarray.flatten(xpoints)
    turbineY = np.ndarray.flatten(ypoints)

    # initialize input variable arrays
    nTurbs = turbineX.size
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency  = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter      # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 0.944
        yaw[turbI] = 10.     # deg.

    # Define flow properties
    wind_speed = 8.0        # m/s
    air_density = 1.1716    # kg/m^3
    wind_direction = 240    # deg (N = 0 deg., using direction FROM, as in met-mast data)
    wind_frequency = 0.1    # probability of wind in this direction at this speed

    # initialize problem
    prob = Problem(root=OptPowerOneDir(nTurbs, use_rotor_components=False))

    # set up optimizer
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'

    # set optimizer options
    prob.driver.opt_settings['Verify level'] = 3
    prob.driver.opt_settings['Print file'] = 'SNOPT_print_exampleOptOneDir.out'
    prob.driver.opt_settings['Summary file'] = 'SNOPT_summary_exampleOptOneDir.out'

    # prob.driver.options['tol'] = 1.0E-8
    prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX))
    prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY))
    prob.driver.add_desvar('yaw0', low=-30.0, high=30.0)
    prob.driver.add_objective('obj')

    # initialize problem
    prob.setup()

    # assign initial values to design variables
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
    prob['model_params:FLORISoriginal'] = True
    prob['gen_params:CPcorrected'] = False
    prob['gen_params:CTcorrected'] = False

    # run the problem
    print('start FLORIS run')
    tic = time.time()
    prob.run()
    toc = time.time()

    # print the results
    print('FLORIS Opt. calculation took {0} sec.'.format(toc-tic))
    print('turbine X positions in wind frame (m): {0}'.format(prob['turbineX']))
    print('turbine Y positions in wind frame (m): {0}'.format(prob['turbineY']))
    print('yaw (deg) = ', prob['yaw0'])
    print('Effective hub velocities (m/s) = ', prob['wtVelocity0'])
    print('Turbine powers (kW) = ', prob['wtPower0'])
    print('wind farm power (kW) = ', prob['dir_power0'])
    print('wind farm AEP for this direction and speed (kWh) = ', prob['AEP'])

    # data = prob.check_partial_derivatives()

