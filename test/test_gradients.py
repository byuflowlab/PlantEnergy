from __future__ import print_function
import unittest
from openmdao.api import pyOptSparseDriver, Problem

from wakeexchange.OptimizationGroups import *
from wakeexchange.GeneralWindFarmComponents import calculate_boundary
from wakeexchange.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from wakeexchange.floris import floris_wrapper, add_floris_params_IndepVarComps
from wakeexchange.larsen import larsen_wrapper, add_larsen_params_IndepVarComps
# from wakeexchange.jensen import jensen_wrapper, add_jensen_params_IndepVarComps

import cPickle as pickle


class TotalDerivTestsFlorisAEPOpt(unittest.TestCase):

    def setUp(self):

        nTurbines = 3
        self.rtol = 1E-6
        self.atol = 1E-6

        np.random.seed(seed=10)

        turbineX = np.random.rand(nTurbines)*3000.
        turbineY = np.random.rand(nTurbines)*3000.

        # generate boundary constraint
        locations = np.zeros([nTurbines, 2])
        for i in range(0, nTurbines):
            locations[i] = np.array([turbineX[i], turbineY[i]])

        # print locations
        boundaryVertices, boundaryNormals = calculate_boundary(locations)
        nVertices = len(boundaryNormals)

        minSpacing = 2.

        # initialize input variable arrays
        rotorDiameter = np.ones(nTurbines)*np.random.random()*150.
        axialInduction = np.ones(nTurbines)*np.random.random()*(1./3.)
        Ct = np.ones(nTurbines)*np.random.random()
        Cp = np.ones(nTurbines)*np.random.random()
        generatorEfficiency = np.ones(nTurbines)*np.random.random()
        yaw = np.random.rand(nTurbines)*60. - 30.

        # Define flow properties
        nDirections = 4
        windSpeeds = np.random.rand(nDirections)*20        # m/s
        air_density = 1.1716    # kg/m^3
        windDirections = np.random.rand(nDirections)*360.0
        windFrequencies = np.random.rand(nDirections)

        # set up problem
        # prob = Problem(root=OptAEP(nTurbines, nDirections=1))

        prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=windDirections.size, nVertices=nVertices,
                                          minSpacing=minSpacing, use_rotor_components=False))

        # set up optimizer
        # prob.driver = pyOptSparseDriver()
        # prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('obj', scaler=1E-5)

        # set optimizer options
        # prob.driver.opt_settings['Verify level'] = 3
        # prob.driver.opt_settings['Print file'] = 'SNOPT_print_exampleOptAEP.out'
        # prob.driver.opt_settings['Summary file'] = 'SNOPT_summary_exampleOptAEP.out'
        # prob.driver.opt_settings['Major iterations limit'] = 1

        # select design variables
        prob.driver.add_desvar('turbineX', lower=np.ones(nTurbines)*min(turbineX), upper=np.ones(nTurbines)*max(turbineX), scaler=1.0)
        prob.driver.add_desvar('turbineY', lower=np.ones(nTurbines)*min(turbineY), upper=np.ones(nTurbines)*max(turbineY), scaler=1.0)
        for direction_id in range(0, windDirections.size):
            prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1.0)

        # add constraints
        prob.driver.add_constraint('sc', lower=np.zeros(int(((nTurbines-1.)*nTurbines/2.))))
        prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbines), scaler=1.0)


        # initialize problem
        prob.setup()

        # assign values to constant inputs (not design variables)
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['yaw0'] = yaw
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp
        prob['generatorEfficiency'] = generatorEfficiency
        prob['windSpeeds'] = windSpeeds
        prob['air_density'] = air_density
        prob['windDirections'] = windDirections
        prob['windFrequencies'] = windFrequencies
        prob['model_params:FLORISoriginal'] = True

        # provide values for hull constraint
        prob['boundaryVertices'] = boundaryVertices
        prob['boundaryNormals'] = boundaryNormals

        # run problem
        prob.run()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)
        self.nDirections = nDirections

        # print self.J

    def testObj(self):

        np.testing.assert_allclose(self.J[('obj', 'turbineX')]['J_fwd'], self.J[('obj', 'turbineX')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('obj', 'turbineY')]['J_fwd'], self.J[('obj', 'turbineY')]['J_fd'], self.rtol, self.atol)
        for dir in np.arange(0, self.nDirections):
            np.testing.assert_allclose(self.J[('obj', 'yaw%i' % dir)]['J_fwd'], self.J[('obj', 'yaw%i' % dir)]['J_fd'], self.rtol, self.atol)


class TotalDerivTestsFlorisAEPOptRotor(unittest.TestCase):

    def setUp(self):

        nTurbines = 4
        self.rtol = 1E-6
        self.atol = 1E-6

        np.random.seed(seed=10)

        turbineX = np.random.rand(nTurbines)*3000.
        turbineY = np.random.rand(nTurbines)*3000.

        minSpacing = 2

        # initialize input variable arrays
        rotorDiameter = np.ones(nTurbines)*np.random.random()*150.
        axialInduction = np.ones(nTurbines)*np.random.random()*(1./3.)
        generatorEfficiency = np.ones(nTurbines)*np.random.random()
        yaw = np.random.rand(nTurbines)*60. - 30.

        # Define flow properties
        nDirections = 50
        windSpeeds = np.random.rand(nDirections)*20        # m/s
        air_density = 1.1716    # kg/m^3
        windDirections = np.random.rand(nDirections)*360.0
        windFrequencies = np.random.rand(nDirections)

        # set up problem
        # prob = Problem(root=OptAEP(nTurbines, nDirections=1))

        prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=windDirections.size,
                                   minSpacing=minSpacing, use_rotor_components=True))

        # set up optimizer
        # prob.driver = pyOptSparseDriver()
        # prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('obj', scaler=1E-8)

        # set optimizer options
        # prob.driver.opt_settings['Verify level'] = 3
        # prob.driver.opt_settings['Print file'] = 'SNOPT_print_exampleOptAEP.out'
        # prob.driver.opt_settings['Summary file'] = 'SNOPT_summary_exampleOptAEP.out'
        # prob.driver.opt_settings['Major iterations limit'] = 1

        # select design variables
        prob.driver.add_desvar('turbineX', lower=np.ones(nTurbines)*min(turbineX), upper=np.ones(nTurbines)*max(turbineX), scaler=1E-2)
        prob.driver.add_desvar('turbineY', lower=np.ones(nTurbines)*min(turbineY), upper=np.ones(nTurbines)*max(turbineY), scaler=1E-2)
        for direction_id in range(0, windDirections.size):
            prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1E-1)

        # add constraints
        prob.driver.add_constraint('sc', lower=np.zeros(int(((nTurbines-1.)*nTurbines/2.))))

        # initialize problem
        prob.setup()



        # assign values to constant inputs (not design variables)
        NREL5MWCPCT = pickle.load(open('./input_files/NREL5MWCPCT_smooth_dict.p'))
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['yaw0'] = yaw
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['windSpeeds'] = windSpeeds
        prob['air_density'] = air_density
        prob['windDirections'] = windDirections
        prob['windFrequencies'] = windFrequencies
        prob['model_params:FLORISoriginal'] = False
        prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
        prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
        prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
        prob['model_params:ke'] = 0.05
        prob['model_params:kd'] = 0.17
        prob['model_params:aU'] = 12.0
        prob['model_params:bU'] = 1.3
        prob['model_params:initialWakeAngle'] = 1.5
        prob['model_params:useaUbU'] = True
        prob['model_params:useWakeAngle'] = True
        prob['model_params:adjustInitialWakeDiamToYaw'] = False
        # run problem
        prob.run()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)
        self.nDirections = nDirections

        # print self.J

    def testObj(self):

        np.testing.assert_allclose(self.J[('obj', 'turbineX')]['rel error'], self.J[('obj', 'turbineX')]['rel error'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('obj', 'turbineY')]['rel error'], self.J[('obj', 'turbineY')]['rel error'], self.rtol, self.atol)
        for dir in np.arange(0, self.nDirections):
            np.testing.assert_allclose(self.J[('obj', 'yaw%i' % dir)]['rel error'], self.J[('obj', 'yaw%i' % dir)]['rel error'], self.rtol, self.atol)

    def testCon(self):

        np.testing.assert_allclose(self.J[('sc', 'turbineX')]['rel error'], self.J[('sc', 'turbineX')]['rel error'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('sc', 'turbineY')]['rel error'], self.J[('sc', 'turbineY')]['rel error'], self.rtol, self.atol)
        for dir in np.arange(0, self.nDirections):
            np.testing.assert_allclose(self.J[('sc', 'yaw%i' % dir)]['rel error'], self.J[('sc', 'yaw%i' % dir)]['rel error'], self.rtol, self.atol)


class TotalDerivTestsGaussAEPOpt(unittest.TestCase):

    def setUp(self):

        nTurbines = 3
        self.rtol = 1E-6
        self.atol = 1E-6

        np.random.seed(seed=10)

        turbineX = np.random.rand(nTurbines)*3000.
        turbineY = np.random.rand(nTurbines)*3000.

        # generate boundary constraint
        locations = np.zeros([nTurbines, 2])
        for i in range(0, nTurbines):
            locations[i] = np.array([turbineX[i], turbineY[i]])

        # print locations
        boundaryVertices, boundaryNormals = calculate_boundary(locations)
        nVertices = len(boundaryNormals)

        minSpacing = 2.

        # initialize input variable arrays
        rotorDiameter = np.ones(nTurbines)*np.random.random()*150.
        axialInduction = np.ones(nTurbines)*np.random.random()*(1./3.)
        Ct = np.ones(nTurbines)*np.random.random()
        Cp = np.ones(nTurbines)*np.random.random()
        generatorEfficiency = np.ones(nTurbines)*np.random.random()
        yaw = np.random.rand(nTurbines)*60. - 30.

        # Define flow properties
        nDirections = 50
        windSpeeds = np.random.rand(nDirections)*20        # m/s
        air_density = 1.1716    # kg/m^3
        windDirections = np.random.rand(nDirections)*360.0
        windFrequencies = np.random.rand(nDirections)

        # set up problem
        # prob = Problem(root=OptAEP(nTurbines, nDirections=1))

        wake_model_options = {'differentiable': True, 'nSamples': 0}

        prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=windDirections.size, nVertices=nVertices,
                                   minSpacing=minSpacing, use_rotor_components=False, wake_model=gauss_wrapper,
                                   params_IdepVar_func=add_gauss_params_IndepVarComps,
                                   wake_model_options=wake_model_options, params_IndepVar_args=None))

        # set up optimizer
        # prob.driver = pyOptSparseDriver()
        # prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('obj', scaler=1E-5)

        # set optimizer options
        # prob.driver.opt_settings['Verify level'] = 3
        # prob.driver.opt_settings['Print file'] = 'SNOPT_print_exampleOptAEP.out'
        # prob.driver.opt_settings['Summary file'] = 'SNOPT_summary_exampleOptAEP.out'
        # prob.driver.opt_settings['Major iterations limit'] = 1

        # select design variables
        prob.driver.add_desvar('turbineX', lower=np.ones(nTurbines)*min(turbineX), upper=np.ones(nTurbines)*max(turbineX), scaler=1.0)
        prob.driver.add_desvar('turbineY', lower=np.ones(nTurbines)*min(turbineY), upper=np.ones(nTurbines)*max(turbineY), scaler=1.0)
        # prob.driver.add_desvar('hubHeight', lower=np.ones(nTurbines)*0.0, upper=np.ones(nTurbines)*120., scaler=1.0)
        for direction_id in range(0, windDirections.size):
            prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1.0)

        # add constraints
        prob.driver.add_constraint('sc', lower=np.zeros(int(((nTurbines-1.)*nTurbines/2.))))
        prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbines), scaler=1.0)


        # initialize problem
        prob.setup()

        # assign values to constant inputs (not design variables)
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['hubHeight'] = np.zeros_like(turbineX)+90.
        prob['yaw0'] = yaw
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp
        prob['generatorEfficiency'] = generatorEfficiency
        prob['windSpeeds'] = windSpeeds
        prob['air_density'] = air_density
        prob['windDirections'] = windDirections
        prob['windFrequencies'] = windFrequencies

        # provide values for hull constraint
        prob['boundaryVertices'] = boundaryVertices
        prob['boundaryNormals'] = boundaryNormals

        # run problem
        prob.run()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)
        self.nDirections = nDirections

        # print self.J

    def testObj(self):

        np.testing.assert_allclose(self.J[('obj', 'turbineX')]['J_fwd'], self.J[('obj', 'turbineX')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('obj', 'turbineY')]['J_fwd'], self.J[('obj', 'turbineY')]['J_fd'], self.rtol, self.atol)
        for dir in np.arange(0, self.nDirections):
            np.testing.assert_allclose(self.J[('obj', 'yaw%i' % dir)]['J_fwd'], self.J[('obj', 'yaw%i' % dir)]['J_fd'], self.rtol, self.atol)



class GradientTestsGauss(unittest.TestCase):

    def setUp(self):

        nTurbines = 4
        self.rtol = 1E-6
        self.atol = 1E-6

        # np.random.seed(seed=10)
        #
        # turbineX = np.random.rand(nTurbines)*.001
        # turbineY = np.random.rand(nTurbines)*.001
        #
        # # initialize input variable arrays
        # rotorDiameter = np.ones(nTurbines)*np.random.random()*150.
        # axialInduction = np.ones(nTurbines)*np.random.random()*(1./3.)
        # Ct = np.ones(nTurbines)*np.random.random()
        # Cp = np.ones(nTurbines)*np.random.random()
        # generatorEfficiency = np.ones(nTurbines)*np.random.random()
        # yaw = np.random.rand(nTurbines)*60. - 30.
        #
        # # Define flow properties
        # wind_speed = np.random.random()*20        # m/s
        # air_density = 1.1716    # kg/m^3
        # wind_direction = np.random.random()*360    # deg (N = 0 deg., using direction FROM, as in met-mast data)
        # wind_frequency = np.random.random()    # probability of wind in given direction
        #
        #
        #
        # spacing = .1     # turbine grid spacing in diameters
        #
        # # set up problem
        # prob = Problem(root=AEPGroup(nTurbines, nDirections=1, use_rotor_components=False,
        #                              wake_model=gauss_wrapper, differentiable=True,
        #                              params_IdepVar_func=add_gauss_params_IndepVarComps,
        #                              wake_model_options={'nSamples': 0},
        #                              params_IndepVar_args=None))
        #
        # # initialize problem
        # prob.setup()
        #
        # # assign values to constant inputs (not design variables)
        # prob['turbineX'] = turbineX
        # prob['turbineY'] = turbineY
        # prob['hubHeight'] = np.zeros_like(turbineX)+90.
        # prob['yaw0'] = yaw
        # prob['rotorDiameter'] = rotorDiameter
        # prob['axialInduction'] = axialInduction
        # prob['Ct_in'] = Ct
        # prob['Cp_in'] = Cp
        # prob['generatorEfficiency'] = generatorEfficiency
        # prob['windSpeeds'] = np.array([wind_speed])
        # prob['air_density'] = air_density
        # prob['windDirections'] = np.array([wind_direction])
        # prob['windFrequencies'] = np.array([wind_frequency])
        #
        #

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

        size = 4  # number of processors (and number of wind directions to run)

        #########################################################################
        # define turbine size
        rotor_diameter = 126.4  # (m)

        # define turbine locations in global reference frame
        # original example case
        # turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])   # m
        # turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])   # m

        # Scaling grid case
        nRows = 2  # number of rows and columns in grid
        # spacing = 2.1436  # turbine grid spacing in diameters
        spacing = .001  # turbine grid spacing in diameters
        print(spacing*rotor_diameter)
        # Set up position arrays
        points = np.linspace(start=spacing * rotor_diameter, stop=nRows * spacing * rotor_diameter, num=nRows)
        xpoints, ypoints = np.meshgrid(points, points)
        turbineX = np.ndarray.flatten(xpoints)
        turbineY = np.ndarray.flatten(ypoints)

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
        prob = Problem(root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size,
                                              minSpacing=minSpacing, differentiable=True, use_rotor_components=False,
                                              wake_model=gauss_wrapper,
                                              params_IdepVar_func=add_gauss_params_IndepVarComps,
                                              params_IndepVar_args={}))

        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('obj', scaler=1E-5)

        # set optimizer options
        prob.driver.opt_settings['Verify level'] = 3
        prob.driver.opt_settings['Print file'] = 'SNOPT_print_exampleOptAEP.out'
        prob.driver.opt_settings['Summary file'] = 'SNOPT_summary_exampleOptAEP.out'
        # prob.driver.opt_settings['Major iterations limit'] = 1000
        bm = 4
        # select design variables
        # prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs) * min(turbineX) * 0,
        #                        upper=np.ones(nTurbs) * max(turbineX) * bm, scaler=1)
        prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs) * min(turbineY) * 0,
                               upper=np.ones(nTurbs) * max(turbineY) * bm, scaler=1)
        # for direction_id in range(0, windDirections.size):
        #     prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1)

        # add constraints
        # prob.driver.add_constraint('sc', lower=np.zeros(int(((nTurbs - 1.) * nTurbs / 2.))), scaler=1.0)

        # tic = time.time()
        prob.setup(check=False)
        # toc = time.time()

        # print the results
        # mpi_print(prob, ('Problem setup took %.03f sec.' % (toc - tic)))

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
        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_partial_derivatives(out_stream=None)

        # print self.J

    def testGaussGrads_wtVelocity0(self):

        # np.testing.assert_allclose(self.J['all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'turbineXw')]['J_fwd'], self.J['all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'turbineXw')]['J_fd'], self.rtol, self.atol)
        # np.testing.assert_allclose(self.J['all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'turbineYw')]['J_fwd'], self.J['all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'turbineYw')]['J_fd'], self.rtol, self.atol)
        # np.testing.assert_allclose(self.J['all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'rotorDiameter')]['J_fwd'], self.J['all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'rotorDiameter')]['J_fd'], self.rtol, self.atol)
        # np.testing.assert_allclose(self.J['all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'yaw0')]['J_fwd'], self.J['all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'yaw0')]['J_fd'], self.rtol, self.atol)
        # np.testing.assert_allclose(self.J['all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'hubHeight')]['J_fwd'], self.J['all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'hubHeight')]['J_fd'], self.rtol, self.atol)
        #
        # np.testing.assert_allclose(self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'turbineXw')]['J_fwd'], self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'turbineXw')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'turbineYw')]['J_fwd'], self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'turbineYw')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'rotorDiameter')]['J_fwd'], self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'rotorDiameter')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'yaw0')]['J_fwd'], self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'yaw0')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'hubHeight')]['J_fwd'], self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'hubHeight')]['J_fd'], self.rtol, self.atol)


class GradientTestsCtCp(unittest.TestCase):

    def setUp(self):

        nTurbines = 4
        self.rtol = 1E-6
        self.atol = 1E-6

        np.random.seed(seed=10)

        turbineX = np.random.rand(nTurbines)*3000.
        turbineY = np.random.rand(nTurbines)*3000.

        # initialize input variable arrays
        rotorDiameter = np.ones(nTurbines)*np.random.random()*150.
        axialInduction = np.ones(nTurbines)*np.random.random()*(1./3.)
        Ct = np.ones(nTurbines)*np.random.random()
        Cp = np.ones(nTurbines)*np.random.random()
        generatorEfficiency = np.ones(nTurbines)*np.random.random()
        yaw = np.random.rand(nTurbines)*60. - 30.

        # Define flow properties
        wind_speed = np.random.random()*20        # m/s
        air_density = 1.1716    # kg/m^3
        wind_direction = np.random.random()*360    # deg (N = 0 deg., using direction FROM, as in met-mast data)
        wind_frequency = np.random.random()    # probability of wind in given direction

        # set up problem
        prob = Problem(root=AEPGroup(nTurbines=nTurbines, use_rotor_components=False))

        # initialize problem
        prob.setup()

        # assign values to constant inputs (not design variables)
                # assign values to constant inputs (not design variables)
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['yaw0'] = yaw
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp
        prob['generatorEfficiency'] = generatorEfficiency
        prob['windSpeeds'] = np.array([wind_speed])
        prob['windFrequencies'] = np.array([wind_frequency])
        prob['air_density'] = air_density
        prob['windDirections'] = np.array([wind_direction])
        prob['model_params:FLORISoriginal'] = False

        # run problem
        prob.run()

        # pass gradient test results to self for use with unit tests
        self.J = prob.check_partial_derivatives(out_stream=None)

    def testCtCp_Ct_out(self):
        np.testing.assert_allclose(self.J['all_directions.direction_group0.CtCp'][('Ct_out', 'Ct_in')]['J_fwd'], self.J['all_directions.direction_group0.CtCp'][('Ct_out', 'Ct_in')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.CtCp'][('Ct_out', 'Cp_in')]['J_fwd'], self.J['all_directions.direction_group0.CtCp'][('Ct_out', 'Cp_in')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.CtCp'][('Ct_out', 'yaw0')]['J_fwd'], self.J['all_directions.direction_group0.CtCp'][('Ct_out', 'yaw0')]['J_fd'], self.rtol, self.atol)

    def testCtCp_Cp_out(self):
        np.testing.assert_allclose(self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'Ct_in')]['J_fwd'], self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'Ct_in')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'Cp_in')]['J_fwd'], self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'Cp_in')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'yaw0')]['J_fwd'], self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'yaw0')]['J_fd'], self.rtol, self.atol)


class GradientTestsCtCpRotor(unittest.TestCase):

    def setUp(self):

        nTurbines = 4
        self.rtol = 1E-6
        self.atol = 1E-6

        use_rotor_components = True

        np.random.seed(seed=10)

        turbineX = np.random.rand(nTurbines)*3000.
        turbineY = np.random.rand(nTurbines)*3000.

        # initialize input variable arrays
        rotorDiameter = np.ones(nTurbines)*np.random.random()*150.
        axialInduction = np.ones(nTurbines)*np.random.random()*(1./3.)
        generatorEfficiency = np.ones(nTurbines)*np.random.random()
        yaw = np.random.rand(nTurbines)*60. - 30.

        # Define flow properties
        wind_speed = np.random.random()*20        # m/s
        air_density = 1.1716    # kg/m^3
        wind_direction = np.random.random()*360    # deg (N = 0 deg., using direction FROM, as in met-mast data)
        wind_frequency = np.random.random()    # probability of wind in given direction



        NREL5MWCPCT = pickle.load(open('./input_files/NREL5MWCPCT_dict.p'))
        datasize = NREL5MWCPCT['CP'].size

        # set up problem
        prob = Problem(root=AEPGroup(nTurbines=nTurbines, use_rotor_components=use_rotor_components, datasize=datasize))

        # initialize problem
        prob.setup(check=False)

        # assign values to constant inputs (not design variables)
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['yaw0'] = yaw
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['windSpeeds'] = np.array([wind_speed])
        prob['windFrequencies'] = np.array([wind_frequency])
        prob['air_density'] = air_density
        prob['windDirections'] = np.array([wind_direction])
        prob['model_params:FLORISoriginal'] = True

        # values for rotor coupling
        prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
        prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
        prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
        prob['model_params:ke'] = 0.05
        prob['model_params:kd'] = 0.17
        prob['model_params:aU'] = 12.0
        prob['model_params:bU'] = 1.3
        prob['model_params:initialWakeAngle'] = 3.0
        prob['model_params:useaUbU'] = True
        prob['model_params:useWakeAngle'] = True
        prob['model_params:adjustInitialWakeDiamToYaw'] = False

        # run problem
        prob.run()

        # pass gradient test results to self for use with unit tests
        self.J = prob.check_partial_derivatives(out_stream=None)

    def testCtCpRotor_Cp_out(self):
        np.testing.assert_allclose(self.J['all_directions.direction_group0.rotorGroup.CtCp'][('Cp_out', 'yaw0')]['J_fwd'], self.J['all_directions.direction_group0.rotorGroup.CtCp'][('Cp_out', 'yaw0')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.rotorGroup.CtCp'][('Cp_out', 'wtVelocity0')]['J_fwd'], self.J['all_directions.direction_group0.rotorGroup.CtCp'][('Cp_out', 'wtVelocity0')]['J_fd'], self.rtol, self.atol)

    def testCtCpRotor_Ct_out(self):
        np.testing.assert_allclose(self.J['all_directions.direction_group0.rotorGroup.CtCp'][('Ct_out', 'yaw0')]['J_fwd'], self.J['all_directions.direction_group0.rotorGroup.CtCp'][('Ct_out', 'yaw0')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.rotorGroup.CtCp'][('Ct_out', 'wtVelocity0')]['J_fwd'], self.J['all_directions.direction_group0.rotorGroup.CtCp'][('Ct_out', 'wtVelocity0')]['J_fd'], self.rtol, self.atol)


class GradientTestsPower(unittest.TestCase):

    def setUp(self):

        nTurbines = 4
        self.rtol = 1E-6
        self.atol = 1E-6

        np.random.seed(seed=10)

        turbineX = np.random.rand(nTurbines)*3000.
        turbineY = np.random.rand(nTurbines)*3000.

        # initialize input variable arrays
        rotorDiameter = np.ones(nTurbines)*np.random.random()*150.
        axialInduction = np.ones(nTurbines)*np.random.random()*(1./3.)
        Ct = np.ones(nTurbines)*np.random.random()
        Cp = np.ones(nTurbines)*np.random.random()
        generatorEfficiency = np.ones(nTurbines)*np.random.random()
        yaw = np.random.rand(nTurbines)*60. - 30.

        # Define flow properties
        wind_speed = np.random.random()*20.       # m/s
        air_density = 1.1716    # kg/m^3
        wind_direction = np.random.random()*360    # deg (N = 0 deg., using direction FROM, as in met-mast data)
        wind_frequency = np.random.random()    # probability of wind in given direction

        # set up problem
        prob = Problem(root=AEPGroup(nTurbines=nTurbines, use_rotor_components=False))

        # initialize problem
        prob.setup()

        # assign values to constant inputs (not design variables)
                # assign values to constant inputs (not design variables)
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['yaw0'] = yaw
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp
        prob['generatorEfficiency'] = generatorEfficiency
        prob['windSpeeds'] = np.array([wind_speed])
        prob['windFrequencies'] = np.array([wind_frequency])
        prob['air_density'] = air_density
        prob['windDirections'] = np.array([wind_direction])
        prob['model_params:FLORISoriginal'] = False

        # run problem
        prob.run()

        # pass gradient test results to self for use with unit tests
        self.J = prob.check_partial_derivatives(out_stream=None)

    def testPower_wtPower(self):
        np.testing.assert_allclose(self.J['all_directions.direction_group0.powerComp'][('wtPower0', 'wtVelocity0')]['J_fwd'], self.J['all_directions.direction_group0.powerComp'][('wtPower0', 'wtVelocity0')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.powerComp'][('wtPower0', 'Cp')]['J_fwd'], self.J['all_directions.direction_group0.powerComp'][('wtPower0', 'Cp')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.powerComp'][('wtPower0', 'rotorDiameter')]['J_fwd'], self.J['all_directions.direction_group0.powerComp'][('wtPower0', 'rotorDiameter')]['J_fd'], self.rtol, self.atol)

    def testPower_totalpower(self):
        np.testing.assert_allclose(self.J['all_directions.direction_group0.powerComp'][('dir_power0', 'wtVelocity0')]['J_fwd'], self.J['all_directions.direction_group0.powerComp'][('dir_power0', 'wtVelocity0')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.powerComp'][('dir_power0', 'Cp')]['J_fwd'], self.J['all_directions.direction_group0.powerComp'][('dir_power0', 'Cp')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.powerComp'][('dir_power0', 'rotorDiameter')]['J_fwd'], self.J['all_directions.direction_group0.powerComp'][('dir_power0', 'rotorDiameter')]['J_fd'], self.rtol, self.atol)

class GradientTestsConstraintComponents(unittest.TestCase):

    def setUp(self):

        nTurbines = 3
        self.rtol = 1E-6
        self.atol = 1E-3

        np.random.seed(seed=10)

        turbineX = np.random.rand(nTurbines)*3000.
        turbineY = np.random.rand(nTurbines)*3000.

        # generate boundary constraint
        locations = np.zeros([nTurbines, 2])
        for i in range(0, nTurbines):
            locations[i] = np.array([turbineX[i], turbineY[i]])

        # print locations
        boundaryVertices, boundaryNormals = calculate_boundary(locations)
        nVertices = len(boundaryNormals)

        boundary_center_x = 1500.
        boundary_center_y = 1500.
        boundary_radius = 2122.

        minSpacing = 2.

        # initialize input variable arrays
        rotorDiameter = np.ones(nTurbines)*np.random.random()*150.
        axialInduction = np.ones(nTurbines)*np.random.random()*(1./3.)
        Ct = np.ones(nTurbines)*np.random.random()
        Cp = np.ones(nTurbines)*np.random.random()
        generatorEfficiency = np.ones(nTurbines)*np.random.random()
        yaw = np.random.rand(nTurbines)*60. - 30.

        # Define flow properties
        nDirections = 5
        windSpeeds = np.random.rand(nDirections)*20        # m/s
        air_density = 1.1716    # kg/m^3
        windDirections = np.random.rand(nDirections)*360.0
        windFrequencies = np.random.rand(nDirections)

        # set up problem
        # prob = Problem(root=OptAEP(nTurbines, nDirections=1))

        wake_model_options = {'differentiable': True, 'nSamples': 0}

        prob_polygon = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=windDirections.size, nVertices=nVertices,
                                   minSpacing=minSpacing, use_rotor_components=False, wake_model=gauss_wrapper,
                                   params_IdepVar_func=add_gauss_params_IndepVarComps,
                                   wake_model_options=wake_model_options, params_IndepVar_args=None))

        prob_circle = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=windDirections.size, nVertices=1,
                                           minSpacing=minSpacing, use_rotor_components=False, wake_model=gauss_wrapper,
                                           params_IdepVar_func=add_gauss_params_IndepVarComps,
                                           wake_model_options=wake_model_options, params_IndepVar_args=None))

        probs = [prob_polygon, prob_circle]
        count = 0
        for prob in probs:
            # set up optimizer
            # prob.driver = pyOptSparseDriver()
            # prob.driver.options['optimizer'] = 'SNOPT'
            prob.driver.add_objective('obj', scaler=1E-5)

            # select design variables
            prob.driver.add_desvar('turbineX', lower=np.ones(nTurbines)*min(turbineX), upper=np.ones(nTurbines)*max(turbineX), scaler=1.0)
            prob.driver.add_desvar('turbineY', lower=np.ones(nTurbines)*min(turbineY), upper=np.ones(nTurbines)*max(turbineY), scaler=1.0)
            # prob.driver.add_desvar('hubHeight', lower=np.ones(nTurbines)*0.0, upper=np.ones(nTurbines)*120., scaler=1.0)
            for direction_id in range(0, windDirections.size):
                prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1.0)

            # add constraints
            prob.driver.add_constraint('sc', lower=np.zeros(int(((nTurbines-1.)*nTurbines/2.))))
            if prob is prob_polygon:
                prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbines), scaler=1.0)
            elif prob is prob_circle:
                prob.driver.add_constraint('boundaryDistances', lower=np.zeros(1 * nTurbines), scaler=1.0)

            # initialize problem
            prob.setup()

            # assign values to constant inputs (not design variables)
            prob['turbineX'] = turbineX
            prob['turbineY'] = turbineY
            prob['hubHeight'] = np.zeros_like(turbineX)+90.
            prob['yaw0'] = yaw
            prob['rotorDiameter'] = rotorDiameter
            prob['axialInduction'] = axialInduction
            prob['Ct_in'] = Ct
            prob['Cp_in'] = Cp
            prob['generatorEfficiency'] = generatorEfficiency
            prob['windSpeeds'] = windSpeeds
            prob['air_density'] = air_density
            prob['windDirections'] = windDirections
            prob['windFrequencies'] = windFrequencies

            # provide values for hull constraint
            if prob is prob_polygon:
                prob['boundaryVertices'] = boundaryVertices
                prob['boundaryNormals'] = boundaryNormals
            elif prob is prob_circle:
                prob['boundary_center'] = np.array([boundary_center_x, boundary_center_y])
                prob['boundary_radius'] = boundary_radius

            # run problem
            prob.run_once()

        # pass results to self for use with unit test
        self.J = prob_polygon.check_total_derivatives(out_stream=None)
        self.J_circle = prob_circle.check_total_derivatives(out_stream=None)
        self.nDirections = nDirections

        # print self.J


    def testSpacingCon(self):

        np.testing.assert_allclose(self.J[('sc', 'turbineX')]['J_fwd'], self.J[('sc', 'turbineX')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('sc', 'turbineY')]['J_fwd'], self.J[('sc', 'turbineY')]['J_fd'], self.rtol, self.atol)

    def testBoundaryConPolygon(self):

        np.testing.assert_allclose(self.J[('boundaryDistances', 'turbineX')]['J_fwd'], self.J[('boundaryDistances', 'turbineX')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('boundaryDistances', 'turbineY')]['J_fwd'], self.J[('boundaryDistances', 'turbineY')]['J_fd'], self.rtol, self.atol)

    def testBoundaryConCircle(self):

        np.testing.assert_allclose(self.J_circle[('boundaryDistances', 'turbineX')]['J_fwd'], self.J_circle[('boundaryDistances', 'turbineX')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J_circle[('boundaryDistances', 'turbineY')]['J_fwd'], self.J_circle[('boundaryDistances', 'turbineY')]['J_fd'], self.rtol, self.atol)


# TODO create gradient tests for all components

if __name__ == "__main__":
    unittest.main()


# indep_list = ['turbineX', 'turbineY', 'yaw', 'rotorDiameter']
# unknown_list = ['dir_power0']
# self.J = prob.calc_gradient(indep_list, unknown_list, return_format='array')
# print self.J