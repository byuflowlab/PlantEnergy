from __future__ import print_function

from openmdao.api import Problem, pyOptSparseDriver
from plantenergy.OptimizationGroups import OptAEP
from plantenergy import config
from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps

import time
import numpy as np
import matplotlib.pyplot as plt

import cProfile

import sys

def round_farm(rotor_diameter, center, radius, min_spacing=2.):

    # normalize inputs
    radius /= rotor_diameter
    center /= rotor_diameter

    # calculate how many circles can be fit in the wind farm area
    nCircles = np.floor(radius/min_spacing)
    radii = np.linspace(radius/nCircles, radius, nCircles)
    alpha_mins = 2.*np.arcsin(min_spacing/(2.*radii))
    nTurbines_circles = np.floor(2. * np.pi / alpha_mins)

    nTurbines = int(np.sum(nTurbines_circles))+1

    alphas = 2.*np.pi/nTurbines_circles

    turbineX = np.zeros(nTurbines)
    turbineY = np.zeros(nTurbines)

    index = 0
    turbineX[index] = center[0]
    turbineY[index] = center[1]
    index += 1
    for circle in np.arange(0, int(nCircles)):
        for turb in np.arange(0, int(nTurbines_circles[circle])):
            angle = alphas[circle]*turb
            w = radii[circle]*np.cos(angle)
            h = radii[circle]*np.sin(angle)
            x = center[0] + w
            y = center[1] + h
            turbineX[index] = x
            turbineY[index] = y
            index += 1

    return turbineX*rotor_diameter, turbineY*rotor_diameter

if __name__ == "__main__":

    ######################### for MPI functionality #########################
    from openmdao.core.mpi_wrap import MPI

    if MPI:  # pragma: no cover
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

    size = 3  # number of processors (and number of wind directions to run)

    #########################################################################
    # define turbine size
    rotor_diameter = 126.4  # (m)

    # define turbine locations in global reference frame
    # original example case
    # turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])   # m
    # turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])   # m

    # # Scaling grid case
    # nRows = 3  # number of rows and columns in grid
    # spacing = 3.5  # turbine grid spacing in diameters
    #
    # # Set up position arrays
    # points = np.linspace(start=spacing * rotor_diameter, stop=nRows * spacing * rotor_diameter, num=nRows)
    # xpoints, ypoints = np.meshgrid(points, points)
    # turbineX = np.ndarray.flatten(xpoints)
    # turbineY = np.ndarray.flatten(ypoints)

    # scaling circle case
    center = np.array([2000., 2000.])
    radius = 5. * rotor_diameter
    min_spacing = 4.

    turbineX, turbineY = round_farm(rotor_diameter, np.copy(center), np.copy(radius), min_spacing=2.)

    # set values for circular boundary constraint
    nVertices = 1
    boundary_center_x = center[0]
    boundary_center_y = center[1]
    xmax = np.max(turbineX)
    ymax = np.max(turbineY)
    xmin = np.min(turbineX)
    ymin = np.min(turbineY)
    boundary_radius = 0.5 * (xmax - xmin)

    # initialize input variable arrays
    nTurbs = turbineX.size
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)
    minSpacing = 2.  # number of rotor diameters

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter  # m
        axialInduction[turbI] = 1.0 / 3.0
        Ct[turbI] = 4.0 * axialInduction[turbI] * (1.0 - axialInduction[turbI])
        Cp[turbI] = 0.7737 / 0.944 * 4.0 * 1.0 / 3.0 * np.power((1 - 1.0 / 3.0), 2)
        generatorEfficiency[turbI] = 0.944
        yaw[turbI] = 0.  # deg.

    # Define flow properties
    wind_speed = 8.0  # m/s
    air_density = 1.1716  # kg/m^3
    windDirections = np.linspace(0, 270, size)
    windSpeeds = np.ones(size) * wind_speed
    windFrequencies = np.ones(size) / size

    # initialize problem
    prob = Problem(impl=impl, root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=nVertices,
                                          minSpacing=minSpacing, differentiable=True, use_rotor_components=False,
                                          wake_model=gauss_wrapper, params_IdepVar_func=add_gauss_params_IndepVarComps,
                                          params_IndepVar_args={}))

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
    prob.driver.add_desvar('turbineX', scaler=1E3)
    prob.driver.add_desvar('turbineY', scaler=1E3)
    # for direction_id in range(0, windDirections.size):
    #     prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1)

    # add constraints
    prob.driver.add_constraint('sc', lower=np.zeros(int(((nTurbs - 1.) * nTurbs / 2.))), scaler=1E3)
    prob.driver.add_constraint('boundaryDistances', lower=(np.zeros(1 * turbineX.size)), scaler=1E3)

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

    tic = time.time()
    prob.setup(check=False)
    toc = time.time()

    # print the results
    mpi_print(prob, ('Problem setup took %.03f sec.' % (toc - tic)))

    # time.sleep(10)
    # assign initial values to design variables
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    for direction_id in range(0, windDirections.size):
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

    # assign boundary values
    prob['boundary_center'] = np.array([boundary_center_x, boundary_center_y])
    prob['boundary_radius'] = boundary_radius

    # set options
    # prob['floris_params:FLORISoriginal'] = True
    # prob['floris_params:CPcorrected'] = False
    # prob['floris_params:CTcorrected'] = False

    prob.run_once()
    AEP_init = prob['AEP']
    # pass results to self for use with unit test
    Jp = prob.check_partial_derivatives(out_stream=None)
    Jt = prob.check_total_derivatives(out_stream=None)

    # run the problem
    mpi_print(prob, 'start Bastankhah run')
    tic = time.time()
    # cProfile.run('prob.run()')
    prob.run()
    toc = time.time()

    # print the results
    mpi_print(prob, ('Opt. calculation took %.03f sec.' % (toc - tic)))

    for direction_id in range(0, windDirections.size):
        mpi_print(prob, 'yaw%i (deg) = ' % direction_id, prob['yaw%i' % direction_id])
        # for direction_id in range(0, windDirections.size):
        # mpi_print(prob,  'velocitiesTurbines%i (m/s) = ' % direction_id, prob['velocitiesTurbines%i' % direction_id])
    # for direction_id in range(0, windDirections.size):
    #     mpi_print(prob,  'wt_power%i (kW) = ' % direction_id, prob['wt_power%i' % direction_id])

    mpi_print(prob, 'turbine X positions in wind frame (m): %s' % prob['turbineX'])
    mpi_print(prob, 'turbine Y positions in wind frame (m): %s' % prob['turbineY'])
    mpi_print(prob, 'wind farm power in each direction (kW): %s' % prob['dirPowers'])
    mpi_print(prob, 'Initial AEP (kWh): %s' % prob['AEP'])
    mpi_print(prob, 'Final AEP (kWh): %s' % AEP_init)
    mpi_print(prob, 'AEP improvement: %s' % (prob['AEP'] / AEP_init))

    boundary_circle = plt.Circle((boundary_center_x/rotor_diameter, boundary_center_y/rotor_diameter),
                                 boundary_radius/rotor_diameter, facecolor='none', edgecolor='k', linestyle='--')

    fig, ax = plt.subplots()
    for x, y in zip(turbineX/rotor_diameter, turbineY/rotor_diameter):
        circle_start = plt.Circle((x,y), 0.5, facecolor='none', edgecolor='r', linestyle=':', label='Start')
        ax.add_artist(circle_start)
    for x, y in zip(prob['turbineX'] / rotor_diameter, prob['turbineY'] / rotor_diameter):
        circle_end = plt.Circle((x,y), 0.5, facecolor='none', edgecolor='g', linestyle='--', label='End')
        ax.add_artist(circle_end)
    # ax.plot(turbineX / rotor_diameter, turbineY / rotor_diameter, 'sk', label='Original', mfc=None)
    # ax.plot(prob['turbineX'] / rotor_diameter, prob['turbineY'] / rotor_diameter, '^g', label='Optimized', mfc=None)
    ax.add_patch(boundary_circle)
    for i in range(0, nTurbs):
        ax.plot([turbineX[i] / rotor_diameter, prob['turbineX'][i] / rotor_diameter],
                [turbineY[i] / rotor_diameter, prob['turbineY'][i] / rotor_diameter], '--k')
    ax.legend([circle_start, circle_end], ['Start', 'End'])
    ax.set_xlabel('Turbine X Position ($X/D_r$)')
    ax.set_ylabel('Turbine Y Position ($Y/D_r$)')
    ax.set_xlim([(boundary_center_x - boundary_radius)/rotor_diameter - 1., (boundary_center_x + boundary_radius)/rotor_diameter + 1.])
    ax.set_ylim([(boundary_center_y - boundary_radius)/rotor_diameter - 1., (boundary_center_y + boundary_radius)/rotor_diameter + 1.])
    plt.axis('equal')
    plt.show()


# from __future__ import print_function
#
# from openmdao.api import Problem, pyOptSparseDriver
# from wakeexchange.OptimizationGroups import OptAEP
# from wakeexchange import config
# from wakeexchange.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
#
# import time
# import numpy as np
# import pylab as plt
#
# import cProfile
#
#
# import sys
#
# if __name__ == "__main__":
#
#     ######################### for MPI functionality #########################
#     from openmdao.core.mpi_wrap import MPI
#
#     if MPI: # pragma: no cover
#         # if you called this script with 'mpirun', then use the petsc data passing
#         from openmdao.core.petsc_impl import PetscImpl as impl
#
#     else:
#         # if you didn't use 'mpirun', then use the numpy data passing
#         from openmdao.api import BasicImpl as impl
#
#     def mpi_print(prob, *args):
#         """ helper function to only print on rank 0 """
#         if prob.root.comm.rank == 0:
#             print(*args)
#
#     prob = Problem(impl=impl)
#
#     size = 10 # number of processors (and number of wind directions to run)
#
#     #########################################################################
#     # define turbine size
#     rotor_diameter = 126.4  # (m)
#
#     # define turbine locations in global reference frame
#     # original example case
#     # turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])   # m
#     # turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])   # m
#
#     # Scaling grid case
#     nRows = int(sys.argv[1])     # number of rows and columns in grid
#     spacing = 5     # turbine grid spacing in diameters
#
#     # Set up position arrays
#     points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
#     xpoints, ypoints = np.meshgrid(points, points)
#     turbineX = np.ndarray.flatten(xpoints)
#     turbineY = np.ndarray.flatten(ypoints)
#
#     # set values for circular boundary constraint
#     nVertices = 1
#     boundary_center_x = np.average(turbineX)
#     boundary_center_y = np.average(turbineY)
#     xmax = np.max(turbineX)
#     ymax = np.max(turbineY)
#     xmin = np.min(turbineX)
#     ymin = np.min(turbineY)
#     boundary_radius = 0.5 * np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)
#
#     # initialize input variable arrays
#     nTurbs = turbineX.size
#     rotorDiameter = np.zeros(nTurbs)
#     axialInduction = np.zeros(nTurbs)
#     Ct = np.zeros(nTurbs)
#     Cp = np.zeros(nTurbs)
#     generatorEfficiency = np.zeros(nTurbs)
#     yaw = np.zeros(nTurbs)
#     minSpacing = 2.                         # number of rotor diameters
#
#     # define initial values
#     for turbI in range(0, nTurbs):
#         rotorDiameter[turbI] = rotor_diameter      # m
#         axialInduction[turbI] = 1.0/3.0
#         Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
#         Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
#         generatorEfficiency[turbI] = 0.944
#         yaw[turbI] = 0.     # deg.
#
#     # Define flow properties
#     wind_speed = 8.0        # m/s
#     air_density = 1.1716    # kg/m^3
#     windDirections = np.linspace(0, 270, size)
#     windSpeeds = np.ones(size)*wind_speed
#     windFrequencies = np.ones(size)/size
#
#     # initialize problem
#     prob = Problem(impl=impl, root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=nVertices,
#                                           minSpacing=minSpacing, differentiable=True, use_rotor_components=False,
#                                           wake_model=gauss_wrapper, params_IdepVar_func=add_gauss_params_IndepVarComps,
#                                           params_IndepVar_args={}))
#
#     # prob.root.deriv_options['type'] = 'fd'
#     # prob.root.deriv_options['form'] = 'central'
#     # prob.root.deriv_options['step_size'] = 1.0e-8
#
#     # set up optimizer
#     prob.driver = pyOptSparseDriver()
#     prob.driver.options['optimizer'] = 'SNOPT'
#     prob.driver.add_objective('obj', scaler=1E-3)
#     prob.driver.options['gradient method'] = 'snopt_fd'
#
#     # set optimizer options
#     prob.driver.opt_settings['Verify level'] = 3
#     prob.driver.opt_settings['Print file'] = 'SNOPT_print_exampleOptAEP.out'
#     prob.driver.opt_settings['Summary file'] = 'SNOPT_summary_exampleOptAEP.out'
#     # prob.driver.opt_settings['Major iterations limit'] = 1000
#     bm = 1
#     # select design variables
#     # prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX)*bm, scaler=1)
#     # prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY)*bm, scaler=1)
#     prob.driver.add_desvar('turbineX', scaler=1)
#     prob.driver.add_desvar('turbineY', scaler=1)
#     for direction_id in range(0, windDirections.size):
#         prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1)
#
#     # add constraints
#     prob.driver.add_constraint('sc', lower=np.zeros(int(((nTurbs-1.)*nTurbs/2.))), scaler=1E-3)
#     prob.driver.add_constraint('boundaryDistances', lower=np.zeros(1 * turbineX.size), scaler=1E1)
#
#     tic = time.time()
#     prob.setup(check=False)
#     toc = time.time()
#
#     # print the results
#     mpi_print(prob, ('Problem setup took %.03f sec.' % (toc-tic)))
#
#     # time.sleep(10)
#     # assign initial values to design variables
#     prob['turbineX'] = turbineX
#     prob['turbineY'] = turbineY
#     for direction_id in range(0, windDirections.size):
#         prob['yaw%i' % direction_id] = yaw
#
#     # assign values to constant inputs (not design variables)
#     prob['rotorDiameter'] = rotorDiameter
#     prob['axialInduction'] = axialInduction
#     prob['generatorEfficiency'] = generatorEfficiency
#     prob['windSpeeds'] = windSpeeds
#     prob['air_density'] = air_density
#     prob['windDirections'] = windDirections
#     prob['windFrequencies'] = windFrequencies
#     prob['Ct_in'] = Ct
#     prob['Cp_in'] = Cp
#
#     # assign boundary values
#     prob['boundary_center'] = np.array([boundary_center_x, boundary_center_y])
#     prob['boundary_radius'] = boundary_radius
#
#     # set options
#     # prob['floris_params:FLORISoriginal'] = True
#     # prob['floris_params:CPcorrected'] = False
#     # prob['floris_params:CTcorrected'] = False
#
#     prob.run_once()
#     AEP_init = prob['AEP']
#     # pass results to self for use with unit test
#     Jp = prob.check_partial_derivatives(out_stream=None)
#     Jt = prob.check_total_derivatives(out_stream=None)
#
#     # run the problem
#     mpi_print(prob, 'start Bastankhah run')
#     tic = time.time()
#     # cProfile.run('prob.run()')
#     prob.run()
#     toc = time.time()
#
#     # print the results
#     mpi_print(prob, ('Opt. calculation took %.03f sec.' % (toc-tic)))
#
#     for direction_id in range(0, windDirections.size):
#         mpi_print(prob,  'yaw%i (deg) = ' % direction_id, prob['yaw%i' % direction_id])
#     # for direction_id in range(0, windDirections.size):
#         # mpi_print(prob,  'velocitiesTurbines%i (m/s) = ' % direction_id, prob['velocitiesTurbines%i' % direction_id])
#     # for direction_id in range(0, windDirections.size):
#     #     mpi_print(prob,  'wt_power%i (kW) = ' % direction_id, prob['wt_power%i' % direction_id])
#
#     mpi_print(prob,  'turbine X positions in wind frame (m): %s' % prob['turbineX'])
#     mpi_print(prob,  'turbine Y positions in wind frame (m): %s' % prob['turbineY'])
#     mpi_print(prob,  'wind farm power in each direction (kW): %s' % prob['dirPowers'])
#     mpi_print(prob,  'Initial AEP (kWh): %s' % prob['AEP'])
#     mpi_print(prob,  'Final AEP (kWh): %s' % AEP_init)
#     mpi_print(prob,  'AEP improvement: %s' % (prob['AEP']/AEP_init))
#
#     xbounds = [min(turbineX)/rotor_diameter, min(turbineX)/rotor_diameter, max(turbineX)/rotor_diameter, max(turbineX)/rotor_diameter, min(turbineX)/rotor_diameter]
#     ybounds = [min(turbineY)/rotor_diameter, max(turbineY)/rotor_diameter, max(turbineY)/rotor_diameter, min(turbineY)/rotor_diameter, min(turbineX)/rotor_diameter]
#
#     plt.figure()
#     plt.plot(turbineX/rotor_diameter, turbineY/rotor_diameter, 'sk', label='Original', mfc=None)
#     plt.plot(prob['turbineX']/rotor_diameter, prob['turbineY']/rotor_diameter, '^g', label='Optimized', mfc=None)
#     plt.plot(xbounds, ybounds, ':k')
#     for i in range(0, nTurbs):
#         plt.plot([turbineX[i]/rotor_diameter, prob['turbineX'][i]/rotor_diameter], [turbineY[i]/rotor_diameter, prob['turbineY'][i]/rotor_diameter], '--k')
#     plt.legend()
#     plt.xlabel('Turbine X Position ($X/D_r$)')
#     plt.ylabel('Turbine Y Position ($Y/D_r$)')
#     plt.xlim([np.min(prob['turbineX']/rotor_diameter)-1., np.max(prob['turbineX']/rotor_diameter)+1.])
#     plt.ylim([np.min(prob['turbineY']/rotor_diameter)-1., np.max(prob['turbineY']/rotor_diameter)+1.])
#     plt.show()