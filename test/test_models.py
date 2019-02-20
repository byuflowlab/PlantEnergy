import unittest
import numpy as np

from openmdao.api import Group
from plantenergy.OptimizationGroups import AEPGroup

# from fusedwake.WindTurbine import WindTurbine
# from fusedwake.WindFarm import WindFarm
# from windIO.Plant import WTLayout, yaml

from openmdao.api import Problem


class test_floris(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(test_floris, self).setUpClass()

        try:
            from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps
            self.working_import = True
        except:
            self.working_import = False

        # define turbine locations in global reference frame
        turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
        turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])
        hub_height = 90.
        hubHeight = np.ones_like(turbineX)*hub_height

        # initialize input variable arrays
        nTurbines = turbineX.size
        rotorDiameter = np.zeros(nTurbines)
        axialInduction = np.zeros(nTurbines)
        Ct = np.zeros(nTurbines)
        Cp = np.zeros(nTurbines)
        generatorEfficiency = np.zeros(nTurbines)
        yaw = np.zeros(nTurbines)

        # define initial values
        for turbI in range(0, nTurbines):
            rotorDiameter[turbI] = 126.4            # m
            axialInduction[turbI] = 1.0/3.0
            Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
            Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
            generatorEfficiency[turbI] = 0.944
            yaw[turbI] = 0.     # deg.

        # Define flow properties
        nDirections = 1
        wind_speed = 8.0                                # m/s
        air_density = 1.1716                            # kg/m^3
        wind_direction = 270.-0.523599*180./np.pi       # deg (N = 0 deg., using direction FROM, as in met-mast data)
        wind_frequency = 1.                             # probability of wind in this direction at this speed

        # set up problem

        model_options = {'differentiable': True,
                         'use_rotor_components': False,
                         'nSamples': 0,
                         'verbose': False,
                         'use_ct_curve': False,
                         'ct_curve': None,
                         'interp_type': 1,
                         'nRotorPoints': 1}

        # prob = Problem(root=AEPGroup(nTurbines, nDirections, wake_model=floris_wrapper, wake_model_options=model_options,
        #                              params_IdepVar_func=add_floris_params_IndepVarComps,
        #                              params_IndepVar_args={'use_rotor_components': False}))
        from plantenergy.OptimizationGroups import OptAEP
        prob = Problem(root=OptAEP(nTurbines, differentiable=True, use_rotor_components=False, wake_model=floris_wrapper,
                                     params_IdepVar_func=add_floris_params_IndepVarComps,
                                     wake_model_options=model_options))

        # initialize problem
        prob.setup(check=False)

        # assign values to turbine states
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['hubHeight'] = hubHeight
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
        prob['model_params:cos_spread'] = 1E12         # turns off cosine spread (just needs to be very large)
        prob['model_params:useWakeAngle'] = False

        # run the problem
        prob.run()
        print(prob['wtVelocity0'])
        self.prob = prob

    def testImport(self):
        self.assertEqual(self.working_import, True, "floris_wrapper Import Failed")

    def testRun(self):
        np.testing.assert_allclose(self.prob['wtVelocity0'],  np.array([8., 8., 6.81027708, 6.81027708, 6.80533179, 6.80533179]))

class test_jensen(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(test_jensen, self).setUpClass()

        try:
            from plantenergy.jensen import jensen_wrapper, add_jensen_params_IndepVarComps
            self.working_import = True
        except:
            self.working_import = False

        # define turbine locations in global reference frame
        turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
        turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])

        # initialize input variable arrays
        nTurbines = turbineX.size
        rotorDiameter = np.zeros(nTurbines)
        axialInduction = np.zeros(nTurbines)
        Ct = np.zeros(nTurbines)
        Cp = np.zeros(nTurbines)
        generatorEfficiency = np.zeros(nTurbines)
        yaw = np.zeros(nTurbines)

        # define initial values
        for turbI in range(0, nTurbines):
            rotorDiameter[turbI] = 126.4            # m
            axialInduction[turbI] = 1.0/3.0
            Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
            Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
            generatorEfficiency[turbI] = 0.944
            yaw[turbI] = 0.     # deg.

        # Define flow properties
        nDirections = 1
        wind_speed = 8.1                                # m/s
        air_density = 1.1716                            # kg/m^3
        wind_direction = 270.-0.523599*180./np.pi       # deg (N = 0 deg., using direction FROM, as in met-mast data)
        wind_frequency = 1.                             # probability of wind in this direction at this speed

        # set up problem

        wake_model_options = None
        # prob = Problem(root=AEPGroup(nTurbines, nDirections, wake_model=jensen_wrapper, wake_model_options=wake_model_options,
        #                              params_IdepVar_func=add_jensen_params_IndepVarComps,
        #                              params_IndepVar_args={'use_angle': False}))
        prob = Problem(
            root=AEPGroup(nTurbines, nDirections, wake_model=jensen_wrapper, wake_model_options=wake_model_options,
                          params_IdepVar_func=add_jensen_params_IndepVarComps,
                          params_IndepVar_args={'use_angle': False}))

        # initialize problem
        prob.setup(check=True)

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
        # prob['model_params:spread_angle'] = 20.0
        # prob['model_params:alpha'] = 0.1

        # run the problem
        prob.run()

        self.prob = prob

    def testImport(self):
        self.assertEqual(self.working_import, True, "jensen_wrapper Import Failed")

    def testRun(self):
        np.testing.assert_allclose(self.prob['wtVelocity0'],  np.array([8.1, 8.1, 6.74484, 6.74484, 6.616713, 6.616713]))

    def testPower(self):
        np.testing.assert_allclose(self.prob['wtPower0'], np.array([1791.08942 , 1791.08942 , 1034.134939, 1034.134939,  976.313113,
        976.313113]))

class test_guass(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(test_guass, self).setUpClass()

        try:
            from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
            self.working_import = True
        except:
            self.working_import = False

        # define turbine locations in global reference frame
        turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
        turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])
        hubHeight = np.zeros_like(turbineX)+90.
        # import matplotlib.pyplot as plt
        # plt.plot(turbineX, turbineY, 'o')
        # plt.plot(np.array([0.0, ]))
        # plt.show()

        # initialize input variable arrays
        nTurbines = turbineX.size
        rotorDiameter = np.zeros(nTurbines)
        axialInduction = np.zeros(nTurbines)
        Ct = np.zeros(nTurbines)
        Cp = np.zeros(nTurbines)
        generatorEfficiency = np.zeros(nTurbines)
        yaw = np.zeros(nTurbines)

        # define initial values
        for turbI in range(0, nTurbines):
            rotorDiameter[turbI] = 126.4            # m
            axialInduction[turbI] = 1.0/3.0
            Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
            Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
            generatorEfficiency[turbI] = 0.944
            yaw[turbI] = 0.     # deg.

        # Define flow properties
        nDirections = 1
        wind_speed = 8.0                                # m/s
        air_density = 1.1716                            # kg/m^3
        wind_direction = 270.-0.523599*180./np.pi       # deg (N = 0 deg., using direction FROM, as in met-mast data)
        wind_frequency = 1.                             # probability of wind in this direction at this speed

        # set up problem

        wake_model_options = {'nSamples': 0}
        prob = Problem(root=AEPGroup(nTurbines=nTurbines, nDirections=nDirections, wake_model=gauss_wrapper,
                                     wake_model_options=wake_model_options, datasize=0, use_rotor_components=False,
                                     params_IdepVar_func=add_gauss_params_IndepVarComps, differentiable=True,
                                     params_IndepVar_args={}))

        # initialize problem
        prob.setup(check=True)

        # assign values to turbine states
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['hubHeight'] = hubHeight
        prob['yaw0'] = yaw

        # assign values to constant inputs (not design variables)
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['windSpeeds'] = np.array([wind_speed])
        prob['model_params:z_ref'] = 90.
        prob['air_density'] = air_density
        prob['windDirections'] = np.array([wind_direction])
        prob['windFrequencies'] = np.array([wind_frequency])
        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp

        # run the problem
        prob.run()

        self.prob = prob

    def testImport(self):
        self.assertEqual(self.working_import, True, "gauss_wrapper Import Failed")

    def testRun(self):
        np.testing.assert_allclose(self.prob['wtVelocity0'], np.array([ 8., 8., 5.922961, 5.922961, 5.478532, 5.478241]))


# class TestLarsenWrapper(unittest.TestCase):
#
#     def setUp(self):
#         try:
#             from wakeexchange.larsen import larsen_wrapper, add_larsen_params_IndepVarComps
#             self.working_import = True
#         except:
#             self.working_import = False
#
#         # define turbine locations in global reference frame
#         turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
#         turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])
#
#         # initialize input variable arrays
#         nTurbines = turbineX.size
#         rotorDiameter = np.zeros(nTurbines)
#         axialInduction = np.zeros(nTurbines)
#         Ct = np.zeros(nTurbines)
#         Cp = np.zeros(nTurbines)
#         generatorEfficiency = np.zeros(nTurbines)
#         yaw = np.zeros(nTurbines)
#         hubHeight = np.zeros(nTurbines)
#
#         # define initial values
#         for turbI in range(0, nTurbines):
#             rotorDiameter[turbI] = 126.4            # m
#             axialInduction[turbI] = 1.0/3.0
#             Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
#             Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
#             generatorEfficiency[turbI] = 0.944
#             yaw[turbI] = 0.     # deg.
#             hubHeight[turbI] = 90.0
#
#         # Define flow properties
#         nDirections = 1
#         wind_speed = 8.0                                # m/s
#         air_density = 1.1716                            # kg/m^3
#         wind_direction = 270.-0.523599*180./np.pi       # deg (N = 0 deg., using direction FROM, as in met-mast data)
#         wind_frequency = 1.                             # probability of wind in this direction at this speed
#         Ia = 0.05                                       # ambient turbulence intensity
#
#         # load wind turbine data
#         NREL5MWCPCT = pickle.load(open('../data/turbine_data/NREL5MWCPCT_smooth_dict.p'))
#
#         # set up problem
#         model_options = {}
#         model_options['tuning'] = 'standard'
#         model_options['differentiable'] = False
#         model_options['verbose'] = False
#         model_options['nSamples'] = 0
#         model_options['use_rotor_components'] = False
#         model_options['datasize'] = NREL5MWCPCT['CP'].size
#
#         #calculate P based on Cp
#         Power = np.zeros(NREL5MWCPCT['CP'].size)
#         for i in range(0, NREL5MWCPCT['CP'].size):
#             Power[i] = 0.5*air_density*NREL5MWCPCT['CP'][i]*np.pi*(rotorDiameter[0]/2.0)**(2.0)*NREL5MWCPCT['wind_speed'][i]**(3.0)
#
#         B = np.array([NREL5MWCPCT['wind_speed'], Power, NREL5MWCPCT['CT']])
#         B = np.transpose(B)
#         b = np.copy(B)
#         WT = WindTurbine(name='test_turbine', refCurvesFile=b, H=hubHeight[0], R=rotorDiameter[0])
#
#         #Make WindFarm instance
#
#         xcoord = np.ones(nTurbines)
#         ycoord = np.ones(nTurbines)
#
#         a = np.array([xcoord, ycoord])
#
#         # ---- Modify test.yml file to match number of turbines ---- #
#         yml_coordinates = []
#         for i in range(0, nTurbines):
#             yml_coordinates.append({'position': [100.0*i, 100.0*i], 'turbine_type': 'NREL5MW', 'name': 'WT', 'row': 1})
#
#         yml = '../data/turbine_data/NREL5MW.yml'
#
#         test_yml = WTLayout(yml)
#
#         test_yml.data['layout'] = yml_coordinates
#
#         ctcurve = []
#         powcurve = []
#
#         #print(NREL5MWCPCT['CP'].size)
#         # plt.figure()
#         # plt.plot(NREL5MWCPCT['wind_speed'], NREL5MWCPCT['CT'])
#         # plt.show()
#         # #quit()
#
#         for i in range(0, NREL5MWCPCT['CP'].size):
#             ctcurve.append([NREL5MWCPCT['wind_speed'][i], NREL5MWCPCT['CT'][i]])
#             powcurve.append([NREL5MWCPCT['wind_speed'][i], Power[i]])
#
#         test_yml.data['turbine_types'][0]['c_t_curve'] = ctcurve
#         test_yml.data['turbine_types'][0]['power_curve'] = powcurve
#
#         # create temporary yml filename
#         temp_yml = '../data/turbine_data/temp.yml'
#
#         with open(temp_yml, 'w') as f:
#             yaml.dump(test_yml.data, f, default_flow_style=False)
#         # ----------------------------------------------------------- #
#
#         # create windfarm instance from temp_yml data
#         WF = WindFarm(name='Test_Farm', yml=temp_yml, array=a, WT=WT)
#
#         os.remove(temp_yml)
#
#         model_options['wf_instance'] = WF
#
#         prob = Problem(root=AEPGroup(nTurbines, nDirections, wake_model=larsen_wrapper, wake_model_options=model_options,
#                                      params_IdepVar_func=add_larsen_params_IndepVarComps,
#                                      params_IndepVar_args={'nTurbines': nTurbines, 'datasize': model_options['datasize']}))
#
#         # initialize problem
#         prob.setup(check=True)
#
#         # assign values to turbine states
#         prob['turbineX'] = turbineX
#         prob['turbineY'] = turbineY
#         prob['yaw0'] = yaw
#
#         # assign values to constant inputs (not design variables)
#         prob['rotorDiameter'] = rotorDiameter
#         prob['hubHeight'] = hubHeight
#         prob['axialInduction'] = axialInduction
#         prob['generatorEfficiency'] = generatorEfficiency
#         prob['windSpeeds'] = np.array([wind_speed])
#         prob['air_density'] = air_density
#         prob['windDirections'] = np.array([wind_direction])
#         prob['windFrequencies'] = np.array([wind_frequency])
#         prob['Ct_in'] = Ct
#         prob['Cp_in'] = Cp
#         prob['model_params:Ia'] = Ia
#
#         # run the problem
#         prob.run()
#
#         self.prob = prob
#
#     def testImport(self):
#         self.assertEqual(self.working_import, True, "larsen_wrapper Import Failed")
#
#     def testRun(self):
#         np.testing.assert_allclose(self.prob['wtVelocity0'], np.array([8., 8., 6.796097, 6.796097, 6.573742, 6.573742]))


if __name__ == "__main__":
    unittest.main()