import unittest
import numpy as np

from openmdao.api import Problem
from plantenergy.OptimizationGroups import AEPGroup

def get_nrel_5mw_cpct():

    filename = "./input_files/NREL5MWCPCT_dict.p"
    # filename = "../input_files/NREL5MWCPCT_smooth_dict.p"
    from numpy.core import multiarray
    import cPickle as pickle

    data = pickle.load(open(filename, "r"))

    ct = data['CT']
    cp = data['CP']
    ct_vel = cp_vel = data['wind_speed']

    return ct, ct_vel, cp, cp_vel

# class test_cpct_splines(unittest.TestCase):
#
#     def setUp(self):
#         try:
#             from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
#             self.working_import = True
#         except:
#             self.working_import = False
#
#         # define turbine locations in global reference frame
#         turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
#         turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])
#         hubHeight = np.zeros_like(turbineX)+90.
#
#         # initialize input variable arrays
#         nTurbines = turbineX.size
#         rotorDiameter = np.zeros(nTurbines)
#         axialInduction = np.zeros(nTurbines)
#         Ct = np.zeros(nTurbines)
#         Cp = np.zeros(nTurbines)
#         generatorEfficiency = np.zeros(nTurbines)
#         yaw = np.zeros(nTurbines)
#
#         # define initial values
#         for turbI in range(0, nTurbines):
#             rotorDiameter[turbI] = 126.4            # m
#             axialInduction[turbI] = 1.0/3.0
#             Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
#             Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
#             generatorEfficiency[turbI] = 0.944
#             yaw[turbI] = 0.     # deg.
#
#         # load cp and ct curves
#         ct, ct_vel, cp, cp_vel = get_nrel_5mw_cpct()
#
#         # Define flow properties
#         nDirections = 1
#         wind_speed = 8.0                                # m/s
#         air_density = 1.1716                            # kg/m^3
#         wind_direction = 270.-0.523599*180./np.pi       # deg (N = 0 deg., using direction FROM, as in met-mast data)
#         wind_frequency = 1.                             # probability of wind in this direction at this speed
#
#         # set up problem
#         wake_model_options = {'nSamples': 0,
#                               'use_ct_curve': True,
#                               'ct_curve': }
#
#         prob = Problem(root=AEPGroup(nTurbines=nTurbines, nDirections=nDirections, wake_model=gauss_wrapper,
#                                      wake_model_options=wake_model_options, datasize=0, use_rotor_components=False,
#                                      params_IdepVar_func=add_gauss_params_IndepVarComps, differentiable=True,
#                                      params_IndepVar_args={}))
#
#         # initialize problem
#         prob.setup(check=True)
#
#         # assign values to turbine states
#         prob['turbineX'] = turbineX
#         prob['turbineY'] = turbineY
#         prob['hubHeight'] = hubHeight
#         prob['yaw0'] = yaw
#
#         # assign values to constant inputs (not design variables)
#         prob['rotorDiameter'] = rotorDiameter
#         prob['axialInduction'] = axialInduction
#         prob['generatorEfficiency'] = generatorEfficiency
#         prob['windSpeeds'] = np.array([wind_speed])
#         prob['model_params:z_ref'] = 90.
#         prob['air_density'] = air_density
#         prob['windDirections'] = np.array([wind_direction])
#         prob['windFrequencies'] = np.array([wind_frequency])
#         prob['Ct_in'] = Ct
#         prob['Cp_in'] = Cp
#
#         # run the problem
#         prob.run()
#
#         self.prob = prob
#
#     def testImport(self):
#         self.assertEqual(self.working_import, True, "gauss_wrapper Import Failed")
#
#     def testRun(self):
#         np.testing.assert_allclose(self.prob['wtVelocity0'], np.array([ 8., 8., 5.922961, 5.922961, 5.478532, 5.478241]))
#
