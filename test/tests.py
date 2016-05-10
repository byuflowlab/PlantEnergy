import unittest
# from florisse.GeneralWindFarmComponents import *
from openmdao.api import pyOptSparseDriver, Problem

from florisse.floris import *
from florisse.OptimizationGroups import *
from florisse.GeneralWindFarmComponents import calculate_boundary

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

        print locations
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
        nDirections = 50.0
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
        prob.driver.add_constraint('sc', lower=np.zeros(((nTurbines-1.)*nTurbines/2.)))
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

        np.testing.assert_allclose(self.J[('obj', 'turbineX')]['rel error'], self.J[('obj', 'turbineX')]['rel error'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('obj', 'turbineY')]['rel error'], self.J[('obj', 'turbineY')]['rel error'], self.rtol, self.atol)
        for dir in np.arange(0, self.nDirections):
            np.testing.assert_allclose(self.J[('obj', 'yaw%i' % dir)]['rel error'], self.J[('obj', 'yaw%i' % dir)]['rel error'], self.rtol, self.atol)

    def testSpacingCon(self):

        np.testing.assert_allclose(self.J[('sc', 'turbineX')]['rel error'], self.J[('sc', 'turbineX')]['rel error'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('sc', 'turbineY')]['rel error'], self.J[('sc', 'turbineY')]['rel error'], self.rtol, self.atol)

    def testBoundaryCon(self):

        np.testing.assert_allclose(self.J[('boundaryDistances', 'turbineX')]['rel error'], self.J[('boundaryDistances', 'turbineX')]['rel error'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('boundaryDistances', 'turbineY')]['rel error'], self.J[('boundaryDistances', 'turbineY')]['rel error'], self.rtol, self.atol)


class TotalDerivTestsFlorisAEPOptRotor(unittest.TestCase):

    def setUp(self):

        nTurbines = 4
        self.rtol = 1E-6
        self.atol = 1E-6

        np.random.seed(seed=10)

        turbineX = np.random.rand(nTurbines)*3000.
        turbineY = np.random.rand(nTurbines)*3000.

        minSpacing = 2.

        # initialize input variable arrays
        rotorDiameter = np.ones(nTurbines)*np.random.random()*150.
        axialInduction = np.ones(nTurbines)*np.random.random()*(1./3.)
        generatorEfficiency = np.ones(nTurbines)*np.random.random()
        yaw = np.random.rand(nTurbines)*60. - 30.

        # Define flow properties
        nDirections = 50.0
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
        prob.driver.add_constraint('sc', lower=np.zeros(((nTurbines-1.)*nTurbines/2.)))

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


class GradientTestsFLORIS(unittest.TestCase):

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

        spacing = 5     # turbine grid spacing in diameters


        # rotor_diameter = 126.4
        # nRows = 2
        # nDirections = 4.
        # # Set up position arrays
        # points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
        # xpoints, ypoints = np.meshgrid(points, points)
        # turbineX = np.ndarray.flatten(xpoints)
        # turbineY = np.ndarray.flatten(ypoints)
        #
        # # initialize input variable arrays
        # nTurbs = turbineX.size
        # rotorDiameter = np.zeros(nTurbs)
        # axialInduction = np.zeros(nTurbs)
        # Ct = np.zeros(nTurbs)
        # Cp = np.zeros(nTurbs)
        # generatorEfficiency = np.zeros(nTurbs)
        # yaw = np.zeros(nTurbs)
        # minSpacing = 2.                         # number of rotor diameters
        #
        # # define initial values
        # for turbI in range(0, nTurbs):
        #     rotorDiameter[turbI] = rotor_diameter      # m
        #     axialInduction[turbI] = 1.0/3.0
        #     Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        #     Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        #     generatorEfficiency[turbI] = 0.944
        #     yaw[turbI] = 0.     # deg.
        #
        # # Define flow properties
        # wind_speed = 8.0        # m/s
        # air_density = 1.1716    # kg/m^3
        # windDirections = np.linspace(0, 270, nDirections)
        # windFrequencies = np.ones_like(windDirections)*1.0/nDirections

        # set up problem
        prob = Problem(root=AEPGroup(nTurbines, nDirections=1, use_rotor_components=False))

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
        prob['windSpeeds'] = np.array([wind_speed])
        prob['air_density'] = air_density
        prob['windDirections'] = np.array([wind_direction])
        prob['windFrequencies'] = np.array([wind_frequency])
        prob['model_params:FLORISoriginal'] = False

        # run problem
        prob.run()

        # pass results to self for use with unit test
        self.J = prob.check_partial_derivatives(out_stream=None)

        # print self.J

    def testWindFrameGrads_turbineXw(self):

        np.testing.assert_allclose(self.J['all_directions.direction_group0.directionConversion'][('turbineXw', 'turbineX')]['J_fwd'], self.J['all_directions.direction_group0.directionConversion'][('turbineXw', 'turbineX')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.directionConversion'][('turbineXw', 'turbineY')]['J_fwd'], self.J['all_directions.direction_group0.directionConversion'][('turbineXw', 'turbineY')]['J_fd'], self.rtol, self.atol)

    def testWindFrameGrads_turbineYw(self):

        np.testing.assert_allclose(self.J['all_directions.direction_group0.directionConversion'][('turbineYw', 'turbineX')]['J_fwd'], self.J['all_directions.direction_group0.directionConversion'][('turbineYw', 'turbineX')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.directionConversion'][('turbineYw', 'turbineY')]['J_fwd'], self.J['all_directions.direction_group0.directionConversion'][('turbineYw', 'turbineY')]['J_fd'], self.rtol, self.atol)


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
        np.testing.assert_allclose(self.J['all_directions.direction_group0.myFloris.CtCp'][('Cp_out', 'yaw0')]['J_fwd'], self.J['all_directions.direction_group0.myFloris.CtCp'][('Cp_out', 'yaw0')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.myFloris.CtCp'][('Cp_out', 'wtVelocity0')]['J_fwd'], self.J['all_directions.direction_group0.myFloris.CtCp'][('Cp_out', 'wtVelocity0')]['J_fd'], self.rtol, self.atol)

    def testCtCpRotor_Ct_out(self):
        np.testing.assert_allclose(self.J['all_directions.direction_group0.myFloris.CtCp'][('Ct_out', 'yaw0')]['J_fwd'], self.J['all_directions.direction_group0.myFloris.CtCp'][('Ct_out', 'yaw0')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.myFloris.CtCp'][('Ct_out', 'wtVelocity0')]['J_fwd'], self.J['all_directions.direction_group0.myFloris.CtCp'][('Ct_out', 'wtVelocity0')]['J_fd'], self.rtol, self.atol)


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


class GradientTestsFlorisUnifiedBV(unittest.TestCase):

    def setUp(self):

        config.floris_single_component = True
        config.BV = True

        nTurbines = 4
        self.rtol = 1E-10
        self.atol = 1E-10

        np.random.seed(seed=100)

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

        config.floris_single_component = False
        config.BV = True

    def testUnified_wtVelocity(self):
        np.testing.assert_allclose(self.J['all_directions.direction_group0.myFloris'][('wtVelocity0', 'turbineXw')]['J_fwd'], self.J['all_directions.direction_group0.myFloris'][('wtVelocity0', 'turbineXw')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.myFloris'][('wtVelocity0', 'turbineYw')]['J_fwd'], self.J['all_directions.direction_group0.myFloris'][('wtVelocity0', 'turbineYw')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.myFloris'][('wtVelocity0', 'yaw0')]['J_fwd'], self.J['all_directions.direction_group0.myFloris'][('wtVelocity0', 'yaw0')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.myFloris'][('wtVelocity0', 'rotorDiameter')]['J_fwd'], self.J['all_directions.direction_group0.myFloris'][('wtVelocity0', 'rotorDiameter')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.myFloris'][('wtVelocity0', 'Ct')]['J_fwd'], self.J['all_directions.direction_group0.myFloris'][('wtVelocity0', 'Ct')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.myFloris'][('wtVelocity0', 'axialInduction')]['J_fwd'], self.J['all_directions.direction_group0.myFloris'][('wtVelocity0', 'axialInduction')]['J_fd'], self.rtol, self.atol)


if __name__ == "__main__":
    unittest.main()


# indep_list = ['turbineX', 'turbineY', 'yaw', 'rotorDiameter']
# unknown_list = ['dir_power0']
# self.J = prob.calc_gradient(indep_list, unknown_list, return_format='array')
# print self.J