from __future__ import print_function

from openmdao.api import Problem, pyOptSparseDriver
from plantenergy.OptimizationGroups import OptAEP
from plantenergy.GeneralWindFarmComponents import calculate_boundary

import time
import numpy as np
import pylab as plt

import cProfile

import sys

if __name__ == "__main__":

    ######################### for MPI functionality #########################
    from openmdao.core.mpi_wrap import MPI

    if MPI: # pragma: no cover
        # if you called this script with 'mpirun', then use the petsc data passing
        from openmdao.core.petsc_impl import PetscImpl as impl

    else:
        # if you didn't use 'mpirun', then use the numpy data passing
        from openmdao.api import BasicImpl as impl

    def mpi_print(prob, *args):
        """ helper function to only print on rank 0 """
        if prob.root.comm.rank == 0:
            print(*args)

    prob = Problem(impl=impl)


    # load wind rose data
    windRose = np.loadtxt('./input_files/windrose_amalia_directionally_averaged_speeds.txt')
    indexes = np.where(windRose[:, 1] > 0.1)
    windDirections = windRose[indexes[0], 0]
    windSpeeds = windRose[indexes[0], 1]
    windFrequencies = windRose[indexes[0], 2]

    nDirections = len(windDirections)

    # load turbine positions
    locations = np.loadtxt('./input_files/layout_amalia.txt')
    turbineX = locations[:, 0]
    turbineY = locations[:, 1]

    # generate boundary constraint
    boundaryVertices, boundaryNormals = calculate_boundary(locations)
    nVertices = boundaryVertices.shape[0]

    # define turbine size
    rotor_diameter = 126.4  # (m)

    # initialize input variable arrays
    nTurbines = turbineX.size
    rotorDiameter = np.zeros(nTurbines)
    axialInduction = np.zeros(nTurbines)
    Ct = np.zeros(nTurbines)
    Cp = np.zeros(nTurbines)
    generatorEfficiency = np.zeros(nTurbines)
    yaw = np.zeros(nTurbines)
    minSpacing = 2.                         # number of rotor diameters

    # define initial values
    for turbI in range(0, nTurbines):
        rotorDiameter[turbI] = rotor_diameter      # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 0.944
        yaw[turbI] = 0.     # deg.

    # Define flow properties
    air_density = 1.1716    # kg/m^3

    # initialize problem
    prob = Problem(impl=impl, root=OptAEP(nTurbines=nTurbines, nDirections=nDirections,
                                          minSpacing=minSpacing, differentiable=True,
                                          use_rotor_components=False, nVertices=nVertices))

    # Tell the whole model to finite difference
    prob.root.deriv_options['type'] = 'fd'

    # set optimizer options (pyoptsparse)
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SLSQP' #NSGA2, CONMIN, SNOPT, SLSQP, COBYLA
    #SLSQP options
    prob.driver.opt_settings['MAXIT'] = 5
    #NSGA2 options
    #prob.driver.opt_settings['maxGen'] = 10
    #SNOPT options
    # prob.driver.opt_settings['Verify level'] = 3
    #prob.driver.opt_settings['Print file'] = 'SNOPT_print_exampleOptAEP_amalia.out'
    #prob.driver.opt_settings['Summary file'] = 'SNOPT_summary_exampleOptAEP_amalia.out'
    #prob.driver.opt_settings['Major iterations limit'] = 1000
    #prob.driver.opt_settings['Major optimality tolerance'] = 1E-5

    # set optimizer options (scipy)
    #prob.driver = ScipyOptimizer()
    #prob.driver.options['optimizer'] = 'SLSQP' #'COBYLA' 'BFGS' 'SLSQP'
    #prob.driver.options['tol'] = 1.0e-6
    #prob.driver.options['maxiter'] = 2000 #maximum number of solver iterations
    #prob.driver.options['disp'] = True

    # set objective function
    prob.driver.add_objective('obj', scaler=1E-5)

    # select design variables
    prob.driver.add_desvar('turbineX', scaler=1.0)
    prob.driver.add_desvar('turbineY', scaler=1.0)
    # for direction_id in range(0, nDirections):
    #     prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1.0)

    # add constraints
    prob.driver.add_constraint('sc', lower=np.zeros(int((nTurbines-1.)*nTurbines/2.)), scaler=1.0)
    prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbines), scaler=1.0)

    tic = time.time()
    prob.setup(check=False)
    toc = time.time()

    # print the results
    mpi_print(prob, ('FLORIS setup took %.03f sec.' % (toc-tic)))

    # assign initial values to design variables
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    for direction_id in range(0, nDirections):
        prob['yaw%i' % direction_id] = yaw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['windSpeeds'] = windSpeeds
    prob['air_density'] = air_density
    prob['windDirections'] = windDirections
    prob['windFrequencies'] = windFrequencies
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp

    # provide values for hull constraint
    prob['boundaryVertices'] = boundaryVertices
    prob['boundaryNormals'] = boundaryNormals

    # run the problem
    mpi_print(prob, 'start FLORIS run')
    tic = time.time()
    # cProfile.run('prob.run()')
    prob.run()
    toc = time.time()

    if prob.root.comm.rank == 0:
        # print the results
        mpi_print(prob, ('FLORIS Opt. calculation took %.03f sec.' % (toc-tic)))

        for direction_id in range(0, windDirections.size):
            mpi_print(prob,  'yaw%i (deg) = ' % direction_id, prob['yaw%i' % direction_id])
        # for direction_id in range(0, windDirections.size):
            # mpi_print(prob,  'velocitiesTurbines%i (m/s) = ' % direction_id, prob['velocitiesTurbines%i' % direction_id])
        # for direction_id in range(0, windDirections.size):
        #     mpi_print(prob,  'wt_power%i (kW) = ' % direction_id, prob['wt_power%i' % direction_id])

        mpi_print(prob,  'turbine X positions (m): %s' % prob['turbineX'])
        mpi_print(prob,  'turbine Y positions (m): %s' % prob['turbineY'])
        mpi_print(prob,  'wind farm power in each direction (kW): %s' % prob['dirPowers'])
        mpi_print(prob,  'AEP (kWh): %s' % prob['AEP'])

        xbounds = [min(turbineX), min(turbineX), max(turbineX), max(turbineX), min(turbineX)]
        ybounds = [min(turbineY), max(turbineY), max(turbineY), min(turbineY), min(turbineX)]

        plt.figure()
        plt.plot(turbineX, turbineY, 'ok', label='Original')
        plt.plot(prob['turbineX'], prob['turbineY'], 'or', label='Optimized')
        plt.plot(xbounds, ybounds, ':k')
        for i in range(0, nTurbines):
            plt.plot([turbineX[i], prob['turbineX'][i]], [turbineY[i], prob['turbineY'][i]], '--k')
        plt.legend()
        plt.xlabel('Turbine X Position (m)')
        plt.ylabel('Turbine Y Position (m)')

        np.savetxt('AmaliaOptimizedXY.txt', np.c_[prob['turbineX'], prob['turbineY']], header="turbineX, turbineY")

        plt.show()

        #
    #
    # windSpeeds = np.array([6.53163342, 6.11908394, 6.13415514, 6.0614625,  6.21344602,
    #                             5.87000793, 5.62161519, 5.96779107, 6.33589422, 6.4668016,
    #                             7.9854581,  7.6894432,  7.5089221,  7.48638098, 7.65764618,
    #                             6.82414044, 6.36728201, 5.95982999, 6.05942132, 6.1176321,
    #                             5.50987893, 4.18461796, 4.82863115, 0.,         0.,         0.,
    #                             5.94115843, 5.94914252, 5.59386528, 6.42332524, 7.67904937,
    #                             7.89618066, 8.84560463, 8.51601497, 8.40826823, 7.89479475,
    #                             7.86194762, 7.9242645,  8.56269962, 8.94563889, 9.82636368,
    #                            10.11153102, 9.71402212, 9.95233636,  10.35446959, 9.67156182,
    #                             9.62462527, 8.83545158, 8.18011771, 7.9372492,  7.68726143,
    #                             7.88134508, 7.31394723, 7.01839896, 6.82858346, 7.06213432,
    #                             7.01949894, 7.00575122, 7.78735165, 7.52836352, 7.21392201,
    #                             7.4356621,  7.54099962, 7.61335262, 7.90293531, 7.16021596,
    #                             7.19617087, 7.5593657,  7.03278586, 6.76105501, 6.48004694,
    #                             6.94716392])
    #
    # windFrequencies = np.array([1.17812570e-02, 1.09958570e-02, 9.60626600e-03, 1.21236860e-02,
    #                            1.04722450e-02, 1.00695140e-02, 9.68687400e-03, 1.00090550e-02,
    #                            1.03715390e-02, 1.12172280e-02, 1.52249700e-02, 1.56279300e-02,
    #                            1.57488780e-02, 1.70577560e-02, 1.93535770e-02, 1.41980570e-02,
    #                            1.20632100e-02, 1.20229000e-02, 1.32111160e-02, 1.74605400e-02,
    #                            1.72994400e-02, 1.43993790e-02, 7.87436000e-03, 0.00000000e+00,
    #                            2.01390000e-05, 0.00000000e+00, 3.42360000e-04, 3.56458900e-03,
    #                            7.18957000e-03, 8.80068000e-03, 1.13583200e-02, 1.41576700e-02,
    #                            1.66951900e-02, 1.63125500e-02, 1.31709000e-02, 1.09153300e-02,
    #                            9.48553000e-03, 1.01097900e-02, 1.18819700e-02, 1.26069900e-02,
    #                            1.58895900e-02, 1.77021600e-02, 2.04208100e-02, 2.27972500e-02,
    #                            2.95438600e-02, 3.02891700e-02, 2.69861000e-02, 2.21527500e-02,
    #                            2.12465500e-02, 1.82861400e-02, 1.66147400e-02, 1.90111800e-02,
    #                            1.90514500e-02, 1.63932050e-02, 1.76215200e-02, 1.65341460e-02,
    #                            1.44597600e-02, 1.40370300e-02, 1.65745000e-02, 1.56278200e-02,
    #                            1.53459200e-02, 1.75210100e-02, 1.59702700e-02, 1.51041500e-02,
    #                            1.45201100e-02, 1.34527800e-02, 1.47819600e-02, 1.33923300e-02,
    #                            1.10562900e-02, 1.04521380e-02, 1.16201970e-02, 1.10562700e-02])