import unittest

# from wakeexchange.GeneralWindFarmComponents import *
# from wakeexchange.GeneralWindFarmGroups import *
from wakeexchange.OptimizationGroups import AEPGroup

from openmdao.api import pyOptSparseDriver, Problem

import numpy as np


class TestFlorisWrapper(unittest.TestCase):

    def setUp(self):
        try:
            from wakeexchange.floris import floris_wrapper
            self.working_import = True
        except:
            self.working_import = False

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
        nDirections = 1
        wind_speed = 8.0                                # m/s
        air_density = 1.1716                            # kg/m^3
        wind_direction = 270.-0.523599*180./np.pi       # deg (N = 0 deg., using direction FROM, as in met-mast data)
        wind_frequency = 1.                             # probability of wind in this direction at this speed

        # set up problem

        model_options = {'differentiable': True, 'use_rotor_components': False, 'nSamples': 0, 'verbose': False}
        prob = Problem(root=AEPGroup(nTurbs, nDirections, model=floris_wrapper, model_options=model_options))

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
        prob['model_params:cos_spread'] = 1E12         # turns off cosine spread (just needs to be very large)
        prob['model_params:useWakeAngle'] = False

        # run the problem
        prob.run()

        self.prob = prob

    def testImport(self):
        self.assertEqual(self.working_import, True, "floris_wrapper Import Failed")

    def testRun(self):
        np.testing.assert_allclose(self.prob['wtVelocity0'],  np.array([8., 8., 6.81027708, 6.81027708, 6.80533179, 6.80533179]))


if __name__ == "__main__":
    unittest.main()