from __future__ import print_function
import unittest
from openmdao.api import pyOptSparseDriver, Problem

from plantenergy.OptimizationGroups import *
from plantenergy.GeneralWindFarmComponents import calculate_boundary
from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps
# from wakeexchange.larsen import larsen_wrapper, add_larsen_params_IndepVarComps
# from wakeexchange.jensen import jensen_wrapper, add_jensen_params_IndepVarComps

import cPickle as pickle

from scipy.interpolate import UnivariateSpline


class TotalDerivTestsFlorisAEPOpt(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(TotalDerivTestsFlorisAEPOpt, self).setUpClass()

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

        # print(self.J)

    def testObj(self):

        np.testing.assert_allclose(self.J[('obj', 'turbineX')]['J_fwd'], self.J[('obj', 'turbineX')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('obj', 'turbineY')]['J_fwd'], self.J[('obj', 'turbineY')]['J_fd'], self.rtol, self.atol)
        for dir in np.arange(0, self.nDirections):
            np.testing.assert_allclose(self.J[('obj', 'yaw%i' % dir)]['J_fwd'], self.J[('obj', 'yaw%i' % dir)]['J_fd'], self.rtol, self.atol)

class TotalDerivTestsFlorisAEPOptRotor(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(TotalDerivTestsFlorisAEPOptRotor, self).setUpClass()

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

        # print(self.J)

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

class TotalDerivTestsGaussAEPOpt_VestasV80(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(TotalDerivTestsGaussAEPOpt_VestasV80, self).setUpClass()
        nTurbines = 16
        nDirections = 50
        self.rtol = 1E-2
        self.atol = 1E5

        np.random.seed(seed=10)

        turbineX = np.random.rand(nTurbines)*3000.
        turbineY = np.random.rand(nTurbines)*3000.

        # generate boundary constraint
        locations = np.zeros([nTurbines, 2])
        for i in range(0, nTurbines):
            locations[i] = np.array([turbineX[i], turbineY[i]])


        # print(locations)
        boundaryVertices, boundaryNormals = calculate_boundary(locations)
        nVertices = len(boundaryNormals)

        minSpacing = 2.
        # minSpacing.flat()

        # initialize input variable arrays
        rotor_diameter = 126.4 #np.random.random()*150.
        rotorDiameter = np.ones(nTurbines)*rotor_diameter
        hubHeight = np.ones(nTurbines)*90.
        axialInduction = np.ones(nTurbines)*1./3. #*np.random.random()*(1./3.)
        Ct = np.ones(nTurbines)*np.random.random()
        Cp = np.ones(nTurbines)*np.random.random()
        generatorEfficiency = np.ones(nTurbines)*0.94 #*np.random.random()
        yaw = np.zeros(nTurbines) # np.random.rand(nTurbines)*60. - 30.

        # Define flow properties
        windSpeeds = np.random.rand(nDirections)*20        # m/s
        air_density = 1.1716    # kg/m^3
        windDirections = np.random.rand(nDirections)*360.0
        windFrequencies = np.random.rand(nDirections)

        # define turbine size
        rotor_diameter = 80.  # (m)
        hub_height = 90.  # (m)

        # define turbine locations in global reference frame
        # original example case
        # turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])   # m
        # turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])   # m

        # set up problem
        # prob = Problem(root=OptAEP(nTurbines, nDirections=1))

        ct_curve = np.loadtxt('./input_files/mfg_ct_vestas_v80_niayifar2016.txt', delimiter=",")

        # air_density = 1.1716  # kg/m^3
        Ar = 0.25 * np.pi * rotor_diameter ** 2
        # cp_curve_vel = ct_curve[:, 0]
        power_data = np.loadtxt('./input_files/niayifar_vestas_v80_power_curve_observed.txt', delimiter=',')
        # cp_curve_cp = niayifar_power_model(cp_curve_vel)/(0.5*air_density*cp_curve_vel**3*Ar)
        cp_curve_cp = power_data[:, 1] * (1E6) / (0.5 * air_density * power_data[:, 0] ** 3 * Ar)
        cp_curve_vel = power_data[:, 0]
        cp_curve_spline = UnivariateSpline(cp_curve_vel, cp_curve_cp, ext='const')
        cp_curve_spline.set_smoothing_factor(.0001)
        # cp_curve_spline = None
        # xs = np.linspace(0, 35, 1000)
        # plt.plot(xs, cp_curve_spline(xs))
        # plt.scatter(cp_curve_vel, cp_curve_cp)
        # plt.show()
        # quit()
        nRotorPoints = 1

        wake_model_options = {'nSamples': 0,
                              'nRotorPoints': nRotorPoints,
                              'use_ct_curve': True,
                              'ct_curve_ct': ct_curve[:, 1],
                              'ct_curve_wind_speed': ct_curve[:, 0],
                              'interp_type': 1,
                              'differentiable': True}

        z_ref = 70.0
        z_0 = 0.0002
        # z_0 = 0.000
        TI = 0.077

        # k_calc = 0.022
        k_calc = 0.3837 * TI + 0.003678

        wake_combination_method = 1
        ti_calculation_method = 0
        calc_k_star = True
        sort_turbs = True
        wake_model_version = 2014
        expansion_factor = 3.
        use_parallel_group=False

        # print("HERE 0")
        prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=windDirections.size, nVertices=nVertices,
                                   minSpacing=minSpacing, use_rotor_components=False, wake_model=gauss_wrapper,
                                   params_IdepVar_func=add_gauss_params_IndepVarComps, differentiable=True,
                                   wake_model_options=wake_model_options,
                                   params_IndepVar_args={'nRotorPoints': nRotorPoints},
                                   cp_curve_spline=cp_curve_spline, cp_points=cp_curve_cp.size))
        # print("HERE 1")
        # set up optimizer
        # prob.driver = pyOptSparseDriver()
        # prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('obj', scaler=1)

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

        # prob['AEP_method'] = 'none'
        # prob['AEP_method'] = 'log'
        # prob['AEP_method'] = 'inverse'

        prob['rotorDiameter'] = rotorDiameter
        prob['hubHeight'] = hubHeight
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['windSpeeds'] = windSpeeds
        prob['air_density'] = air_density
        prob['windDirections'] = windDirections
        prob['windFrequencies'] = windFrequencies
        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp
        prob['cut_in_speed'] = np.ones(nTurbines) * 4.
        # prob['cut_in_speed'] = np.ones(nTurbines)*7.
        prob['rated_power'] = np.ones(nTurbines) * 2000.
        prob['cp_curve_cp'] = cp_curve_cp
        prob['cp_curve_vel'] = cp_curve_vel

        prob['model_params:wake_combination_method'] = wake_combination_method
        prob['model_params:ti_calculation_method'] = ti_calculation_method
        prob['model_params:calc_k_star'] = calc_k_star
        prob['model_params:sort'] = sort_turbs
        prob['model_params:z_ref'] = z_ref
        prob['model_params:z_0'] = z_0
        prob['model_params:ky'] = k_calc
        prob['model_params:kz'] = k_calc
        prob['model_params:print_ti'] = False
        prob['model_params:wake_model_version'] = wake_model_version
        prob['model_params:opt_exp_fac'] = expansion_factor

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives()
        self.nDirections = nDirections

    def testObj_x_v80(self):
        np.testing.assert_allclose(self.J[('obj', 'turbineX')]['J_fwd'],
                                   self.J[('obj', 'turbineX')]['J_fd'],
                                   self.rtol, self.atol)

    def testObj_y(self):
        np.testing.assert_allclose(self.J[('obj', 'turbineY')]['J_fwd'],
                                   self.J[('obj', 'turbineY')]['J_fd'],
                                   self.rtol, self.atol)

    def testObj_yaw(self):
        for dir in np.arange(0, self.nDirections):
            np.testing.assert_allclose(self.J[('obj', 'yaw%i' % dir)]['J_fwd'],
                                       self.J[('obj', 'yaw%i' % dir)]['J_fd'],
                                       self.rtol, self.atol)

class TotalDerivTestsGaussAEPOpt_NREL5MW(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(TotalDerivTestsGaussAEPOpt_NREL5MW, self).setUpClass()
        nTurbines = 16
        nDirections = 50
        self.rtol = 1E-2
        self.atol = 1E5

        np.random.seed(seed=10)

        turbineX = np.random.rand(nTurbines)*3000.
        turbineY = np.random.rand(nTurbines)*3000.

        # generate boundary constraint
        locations = np.zeros([nTurbines, 2])
        for i in range(0, nTurbines):
            locations[i] = np.array([turbineX[i], turbineY[i]])


        # print(locations)
        boundaryVertices, boundaryNormals = calculate_boundary(locations)
        nVertices = len(boundaryNormals)

        minSpacing = 2.
        # minSpacing.flat()

        # initialize input variable arrays
        rotor_diameter = 126.4 #np.random.random()*150.
        rotorDiameter = np.ones(nTurbines)*rotor_diameter
        hubHeight = np.ones(nTurbines)*90.
        axialInduction = np.ones(nTurbines)*1./3. #*np.random.random()*(1./3.)
        Ct = np.ones(nTurbines)*np.random.random()
        Cp = np.ones(nTurbines)*np.random.random()
        generatorEfficiency = np.ones(nTurbines)*0.94 #*np.random.random()
        yaw = np.zeros(nTurbines) # np.random.rand(nTurbines)*60. - 30.

        # Define flow properties
        windSpeeds = np.random.rand(nDirections)*20        # m/s
        air_density = 1.1716    # kg/m^3
        windDirections = np.random.rand(nDirections)*360.0
        windFrequencies = np.random.rand(nDirections)

        # define turbine size
        rotor_diameter = 126.4  # (m)
        hub_height = 90.  # (m)

        filename = "./input_files/NREL5MWCPCT_dict.p"

        data = pickle.load(open(filename, "rb"))
        cpct_data = np.zeros([data['wind_speed'].size, 3])
        cpct_data[:, 0] = data['wind_speed']
        cpct_data[:, 1] = data['CP']
        cpct_data[:, 2] = data['CT']

        # air_density = 1.1716  # kg/m^3
        Ar = 0.25 * np.pi * rotor_diameter ** 2
        cp_curve_cp = cpct_data[:, 1]
        cp_curve_vel = cpct_data[:, 0]
        cp_curve_spline = UnivariateSpline(cp_curve_vel, cp_curve_cp, ext='const')
        cp_curve_spline.set_smoothing_factor(.0001)

        nRotorPoints = 1

        wake_model_options = {'nSamples': 0,
                              'nRotorPoints': nRotorPoints,
                              'use_ct_curve': True,
                              'ct_curve_ct': cpct_data[:, 2],
                              'ct_curve_wind_speed': cpct_data[:, 0],
                              'interp_type': 1,
                              'differentiable': True}

        z_ref = 70.0
        z_0 = 0.0002
        # z_0 = 0.000
        TI = 0.077

        # k_calc = 0.022
        k_calc = 0.3837 * TI + 0.003678

        wake_combination_method = 1
        ti_calculation_method = 0
        calc_k_star = True
        sort_turbs = True
        wake_model_version = 2014
        expansion_factor = 3.
        use_parallel_group=False

        # print("HERE 0")
        prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=windDirections.size, nVertices=nVertices,
                                   minSpacing=minSpacing, use_rotor_components=False, wake_model=gauss_wrapper,
                                   params_IdepVar_func=add_gauss_params_IndepVarComps, differentiable=True,
                                   wake_model_options=wake_model_options,
                                   params_IndepVar_args={'nRotorPoints': nRotorPoints},
                                   cp_curve_spline=cp_curve_spline, cp_points=cp_curve_cp.size))
        # print("HERE 1")
        # set up optimizer
        # prob.driver = pyOptSparseDriver()
        # prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('obj', scaler=1)

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

        # prob['AEP_method'] = 'none'
        # prob['AEP_method'] = 'log'
        # prob['AEP_method'] = 'inverse'

        prob['rotorDiameter'] = rotorDiameter
        prob['hubHeight'] = hubHeight
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['windSpeeds'] = windSpeeds
        prob['air_density'] = air_density
        prob['windDirections'] = windDirections
        prob['windFrequencies'] = windFrequencies
        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp
        prob['cut_in_speed'] = np.ones(nTurbines) * 4.
        # prob['cut_in_speed'] = np.ones(nTurbines)*7.
        prob['rated_power'] = np.ones(nTurbines) * 2000.
        prob['cp_curve_cp'] = cp_curve_cp
        prob['cp_curve_vel'] = cp_curve_vel

        prob['model_params:wake_combination_method'] = wake_combination_method
        prob['model_params:ti_calculation_method'] = ti_calculation_method
        prob['model_params:calc_k_star'] = calc_k_star
        prob['model_params:sort'] = sort_turbs
        prob['model_params:z_ref'] = z_ref
        prob['model_params:z_0'] = z_0
        prob['model_params:ky'] = k_calc
        prob['model_params:kz'] = k_calc
        prob['model_params:print_ti'] = False
        prob['model_params:wake_model_version'] = wake_model_version
        prob['model_params:opt_exp_fac'] = expansion_factor

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)
        self.nDirections = nDirections

    def testObj_x_v80(self):
        np.testing.assert_allclose(self.J[('obj', 'turbineX')]['J_fwd'],
                                   self.J[('obj', 'turbineX')]['J_fd'],
                                   self.rtol, self.atol)

    def testObj_y(self):
        np.testing.assert_allclose(self.J[('obj', 'turbineY')]['J_fwd'],
                                   self.J[('obj', 'turbineY')]['J_fd'],
                                   self.rtol, self.atol)

    def testObj_yaw(self):
        for dir in np.arange(0, self.nDirections):
            np.testing.assert_allclose(self.J[('obj', 'yaw%i' % dir)]['J_fwd'],
                                       self.J[('obj', 'yaw%i' % dir)]['J_fd'],
                                       self.rtol, self.atol)

class GradientTestsGauss(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(GradientTestsGauss, self).setUpClass()

        nTurbines = 6
        nDirections = 20
        self.nDirections = nDirections
        self.rtol_p = 1E-6
        self.atol_p = 1E-6
        self.rtol_t = 1E-4
        self.atol_t = 1E-2

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

        size = nDirections  # number of processors (and number of wind directions to run)

        #########################################################################
        # define turbine size
        rotor_diameter = 126.4  # (m)
        hub_height = 90.  # (m)

        # define turbine locations in global reference frame
        # original example case
        # turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])   # m
        # turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])   # m

        # Scaling grid case
        nRows = 2  # number of rows and columns in grid
        # spacing = 2.1436  # turbine grid spacing in diameters
        spacing = 5.  # turbine grid spacing in diameters
        # print(spacing*rotor_diameter)
        # Set up position arrays
        points = np.linspace(start=spacing * rotor_diameter, stop=nRows * spacing * rotor_diameter, num=nRows)
        xpoints, ypoints = np.meshgrid(points, points)
        turbineX = np.ndarray.flatten(xpoints)
        turbineY = np.ndarray.flatten(ypoints)

        # initialize input variable arrays
        nTurbs = turbineX.size
        rotorDiameter = np.zeros(nTurbs)
        hubHeight = np.zeros(nTurbs)
        axialInduction = np.zeros(nTurbs)
        Ct = np.zeros(nTurbs)
        Cp = np.zeros(nTurbs)
        generatorEfficiency = np.zeros(nTurbs)
        yaw = np.zeros(nTurbs)
        minSpacing = np.array([2.])  # number of rotor diameters

        # define initial values
        for turbI in range(0, nTurbs):
            rotorDiameter[turbI] = rotor_diameter  # m
            hubHeight[turbI] = hub_height  # m
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

        ct_curve = np.loadtxt('./input_files/mfg_ct_vestas_v80_niayifar2016.txt', delimiter=",")

        # air_density = 1.1716  # kg/m^3
        Ar = 0.25 * np.pi * rotor_diameter ** 2
        # cp_curve_vel = ct_curve[:, 0]
        power_data = np.loadtxt('./input_files/niayifar_vestas_v80_power_curve_observed.txt', delimiter=',')
        # cp_curve_cp = niayifar_power_model(cp_curve_vel)/(0.5*air_density*cp_curve_vel**3*Ar)
        cp_curve_cp = power_data[:, 1] * (1E6) / (0.5 * air_density * power_data[:, 0] ** 3 * Ar)
        cp_curve_vel = power_data[:, 0]
        cp_curve_spline = UnivariateSpline(cp_curve_vel, cp_curve_cp, ext='const')
        cp_curve_spline.set_smoothing_factor(.0001)
        # cp_curve_spline = None
        # xs = np.linspace(0, 35, 1000)
        # plt.plot(xs, cp_curve_spline(xs))
        # plt.scatter(cp_curve_vel, cp_curve_cp)
        # plt.show()
        # quit()
        nRotorPoints = 1

        wake_model_options = {'nSamples': 0,
                              'nRotorPoints': nRotorPoints,
                              'use_ct_curve': True,
                              'ct_curve': ct_curve,
                              'interp_type': 1,
                              'differentiable': True}

        z_ref = 70.0
        z_0 = 0.0002
        # z_0 = 0.000
        TI = 0.077

        rotor_diameter = 80.0  # (m)
        hub_height = 70.0

        # k_calc = 0.022
        k_calc = 0.3837 * TI + 0.003678

        wake_combination_method = 1
        ti_calculation_method = 5
        calc_k_star = True
        sort_turbs = True
        wake_model_version = 2016

        # initialize problem
        prob = Problem(root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=1,
                                              minSpacing=minSpacing, differentiable=True, use_rotor_components=False,
                                              wake_model=gauss_wrapper, wake_model_options=wake_model_options,
                                              params_IdepVar_func=add_gauss_params_IndepVarComps,
                                              params_IndepVar_args={'nRotorPoints': nRotorPoints},
                                              cp_curve_spline=cp_curve_spline, cp_points=cp_curve_vel.size))

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
        prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs) * min(turbineX) * 0,
                               upper=np.ones(nTurbs) * max(turbineX) * bm, scaler=1)
        prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs) * min(turbineY) * 0,
                               upper=np.ones(nTurbs) * max(turbineY) * bm, scaler=1)
        for direction_id in range(0, windDirections.size):
            prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1)

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
        prob['hubHeight'] = hubHeight
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

        prob['cut_in_speed'] = np.ones(nTurbines) * 4.
        # prob['cut_in_speed'] = np.ones(nTurbines)*7.
        prob['rated_power'] = np.ones(nTurbines) * 2000.
        prob['cp_curve_cp'] = cp_curve_cp
        prob['cp_curve_vel'] = cp_curve_vel

        prob['model_params:wake_combination_method'] = wake_combination_method
        prob['model_params:ti_calculation_method'] = ti_calculation_method
        prob['model_params:calc_k_star'] = calc_k_star
        prob['model_params:sort'] = sort_turbs
        prob['model_params:z_ref'] = z_ref
        prob['model_params:z_0'] = z_0
        prob['model_params:ky'] = k_calc
        prob['model_params:kz'] = k_calc
        prob['model_params:print_ti'] = False
        prob['model_params:wake_model_version'] = wake_model_version
        # prob['model_params:I'] = TI
        # prob['model_params:shear_exp'] = shear_exp
        if nRotorPoints > 1:
            if rotor_pnt_typ == 0:
                prob['model_params:RotorPointsY'], prob['model_params:RotorPointsZ'] = circumference_points(
                    nRotorPoints, location=location)
            if rotor_pnt_typ == 1:
                prob['model_params:RotorPointsY'], prob['model_params:RotorPointsZ'] = sunflower_points(nRotorPoints)

        # run problem
        prob.run_once()
        print("RAN")
        # pass results to self for use with unit test
        self.J = prob.check_partial_derivatives(out_stream=None)
        self.Jt = prob.check_total_derivatives(out_stream=None)

        # print("J = ", self.J)
        # print(self.J)

    def testGaussGrads_wtVelocity0_turbineXw(self):
        np.testing.assert_allclose(self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'turbineXw')]['J_fwd'], self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'turbineXw')]['J_fd'], self.rtol_p, self.atol_p)

    def testGaussGrads_wtVelocity0_turbineYw(self):
        np.testing.assert_allclose(self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'turbineYw')]['J_fwd'], self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'turbineYw')]['J_fd'], self.rtol_p, self.atol_p)

    def testGaussGrads_wtVelocity0_rotorDiameter(self):
        np.testing.assert_allclose(self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'rotorDiameter')]['J_fwd'], self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'rotorDiameter')]['J_fd'], self.rtol_p, self.atol_p)

    def testGaussGrads_wtVelocity0_yaw0(self):
        np.testing.assert_allclose(self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'yaw0')]['J_fwd'], self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'yaw0')]['J_fd'], self.rtol_p, self.atol_p)

    def testGaussGrads_wtVelocity0_hubHeight(self):
        np.testing.assert_allclose(self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'hubHeight')]['J_fwd'], self.J['AEPgroup.all_directions.direction_group0.myModel.f_1'][('wtVelocity0', 'hubHeight')]['J_fd'], self.rtol_p, self.atol_p)

    def testObj_x(self):
        np.testing.assert_allclose(self.Jt[('obj', 'turbineX')]['J_fwd'], self.Jt[('obj', 'turbineX')]['J_fd'], self.rtol_t, self.atol_t)

    def testObj_y(self):
        np.testing.assert_allclose(self.Jt[('obj', 'turbineY')]['J_fwd'], self.Jt[('obj', 'turbineY')]['J_fd'],
                                   self.rtol_t, self.atol_t)

    def testObj_yaw(self):
        for dir in np.arange(0, self.nDirections):
            np.testing.assert_allclose(self.Jt[('obj', 'yaw%i' % dir)]['J_fwd'],
                                       self.Jt[('obj', 'yaw%i' % dir)]['J_fd'], self.rtol_t, self.atol_t)

class GradientTestsCtCpSingleValue(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(GradientTestsCtCpSingleValue, self).setUpClass()

        nTurbines = 15
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

class GradientTestsCpArray(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(GradientTestsCpArray, self).setUpClass()

        nTurbines = 15
        self.rtol = 1E-6
        self.atol = 1E-6

        np.random.seed(seed=10)

        turbineX = np.random.rand(nTurbines)*1000.
        turbineY = np.random.rand(nTurbines)*1000.

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

        # load cp_curve
        import cPickle as pickle
        data = pickle.load(open("./input_files/NREL5MWCPCT_dict.p", "r"))

        cp_curve_vel = data["wind_speed"]
        cp_curve_cp = data["CP"]
        cp_points = cp_curve_cp.size

        # set up problem
        prob = Problem(root=AEPGroup(nTurbines=nTurbines, use_rotor_components=False, cp_points=cp_points))

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
        prob['cp_curve_cp'] = cp_curve_cp
        prob['cp_curve_vel'] = cp_curve_vel
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

    def testCtCp_Cp_out(self):
        np.testing.assert_allclose(self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'Ct_in')]['J_fwd'], self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'Ct_in')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'Cp_in')]['J_fwd'], self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'Cp_in')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'yaw0')]['J_fwd'], self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'yaw0')]['J_fd'], self.rtol, self.atol)

class GradientTestsCpSpline(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(GradientTestsCpSpline, self).setUpClass()

        nTurbines = 15
        self.rtol = 1E-6
        self.atol = 1E-6

        np.random.seed(seed=10)

        turbineX = np.random.rand(nTurbines)*1000.
        turbineY = np.random.rand(nTurbines)*1000.

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

        # load cp_curve
        import cPickle as pickle
        data = pickle.load(open("./input_files/NREL5MWCPCT_dict.p", "r"))

        cp_curve_vel = data["wind_speed"]
        cp_curve_cp = data["CP"]
        cp_points = cp_curve_cp.size

        cp_curve_spline = UnivariateSpline(cp_curve_vel, cp_curve_cp, ext='const', k=1)
        cp_curve_spline.set_smoothing_factor(.000001)

        # set up problem
        prob = Problem(root=AEPGroup(nTurbines=nTurbines, use_rotor_components=False,
                                     cp_curve_spline=cp_curve_spline))

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

    def testCtCp_Cp_out(self):
        np.testing.assert_allclose(self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'Ct_in')]['J_fwd'], self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'Ct_in')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'Cp_in')]['J_fwd'], self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'Cp_in')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'yaw0')]['J_fwd'], self.J['all_directions.direction_group0.CtCp'][('Cp_out', 'yaw0')]['J_fd'], self.rtol, self.atol)

class GradientTestsCtCpRotor(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(GradientTestsCtCpRotor, self).setUpClass()

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

    @classmethod
    def setUpClass(self):
        super(GradientTestsPower, self).setUpClass()

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

        # air_density = 1.1716  # kg/m^3
        Ar = 0.25 * np.pi * rotorDiameter[0] ** 2
        # cp_curve_vel = ct_curve[:, 0]
        power_data = np.loadtxt('./input_files/niayifar_vestas_v80_power_curve_observed.txt', delimiter=',')
        # cp_curve_cp = niayifar_power_model(cp_curve_vel)/(0.5*air_density*cp_curve_vel**3*Ar)
        cp_curve_cp = power_data[:, 1] * (1E6) / (0.5 * air_density * power_data[:, 0] ** 3 * Ar)
        cp_curve_vel = power_data[:, 0]
        cp_curve_spline = UnivariateSpline(cp_curve_vel, cp_curve_cp, ext='const')
        cp_curve_spline.set_smoothing_factor(.0001)

        # set up problem
        prob = Problem(root=AEPGroup(nTurbines=nTurbines, use_rotor_components=False, wake_model=gauss_wrapper,
                                     cp_points=cp_curve_vel.size, cp_curve_spline=cp_curve_spline))

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


        prob['cut_in_speed'] = np.ones(nTurbines) * 4.
        # prob['cut_in_speed'] = np.ones(nTurbines)*7.
        prob['rated_power'] = np.ones(nTurbines) * 2000.
        prob['cp_curve_cp'] = cp_curve_cp
        prob['cp_curve_vel'] = cp_curve_vel
        # prob['use_cp_spline'] = True

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

    @classmethod
    def setUpClass(self):
        super(GradientTestsConstraintComponents, self).setUpClass()

        nTurbines = 38
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
            prob.driver.add_objective('obj', scaler=1E-3)

            # select design variables
            prob.driver.add_desvar('turbineX', lower=np.ones(nTurbines)*min(turbineX), upper=np.ones(nTurbines)*max(turbineX), scaler=1E3)
            prob.driver.add_desvar('turbineY', lower=np.ones(nTurbines)*min(turbineY), upper=np.ones(nTurbines)*max(turbineY), scaler=1E3)
            # prob.driver.add_desvar('hubHeight', lower=np.ones(nTurbines)*0.0, upper=np.ones(nTurbines)*120., scaler=1.0)
            for direction_id in range(0, windDirections.size):
                prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1.0)

            # add constraints
            prob.driver.add_constraint('sc', lower=np.zeros(int(((nTurbines-1.)*nTurbines/2.))))
            if prob is prob_polygon:
                prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbines), scaler=1E3)
            elif prob is prob_circle:
                prob.driver.add_constraint('boundaryDistances', lower=np.zeros(1 * nTurbines), scaler=1E3)

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

        # print(self.J)


    def testSpacingCon(self):

        np.testing.assert_allclose(self.J[('sc', 'turbineX')]['J_fwd'], self.J[('sc', 'turbineX')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('sc', 'turbineY')]['J_fwd'], self.J[('sc', 'turbineY')]['J_fd'], self.rtol, self.atol)

    def testBoundaryConPolygon(self):

        np.testing.assert_allclose(self.J[('boundaryDistances', 'turbineX')]['J_fwd'], self.J[('boundaryDistances', 'turbineX')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('boundaryDistances', 'turbineY')]['J_fwd'], self.J[('boundaryDistances', 'turbineY')]['J_fd'], self.rtol, self.atol)

    def testBoundaryConCircle(self):

        np.testing.assert_allclose(self.J_circle[('boundaryDistances', 'turbineX')]['J_fwd'], self.J_circle[('boundaryDistances', 'turbineX')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J_circle[('boundaryDistances', 'turbineY')]['J_fwd'], self.J_circle[('boundaryDistances', 'turbineY')]['J_fd'], self.rtol, self.atol)

    def testSpacingConRev(self):

        np.testing.assert_allclose(self.J[('sc', 'turbineX')]['J_rev'], self.J[('sc', 'turbineX')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('sc', 'turbineY')]['J_rev'], self.J[('sc', 'turbineY')]['J_fd'], self.rtol, self.atol)

    def testBoundaryConPolygonRev(self):

        np.testing.assert_allclose(self.J[('boundaryDistances', 'turbineX')]['J_rev'], self.J[('boundaryDistances', 'turbineX')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('boundaryDistances', 'turbineY')]['J_rev'], self.J[('boundaryDistances', 'turbineY')]['J_fd'], self.rtol, self.atol)

    def testBoundaryConCircleRev(self):

        np.testing.assert_allclose(self.J_circle[('boundaryDistances', 'turbineX')]['J_rev'], self.J_circle[('boundaryDistances', 'turbineX')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J_circle[('boundaryDistances', 'turbineY')]['J_rev'], self.J_circle[('boundaryDistances', 'turbineY')]['J_fd'], self.rtol, self.atol)


# TODO create gradient tests for all components

if __name__ == "__main__":
    unittest.main(verbosity=2)


# indep_list = ['turbineX', 'turbineY', 'yaw', 'rotorDiameter']
# unknown_list = ['dir_power0']
# self.J = prob.calc_gradient(indep_list, unknown_list, return_format='array')
# print(self.J)