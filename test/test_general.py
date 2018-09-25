import unittest
import numpy as np

from openmdao.api import Problem, Group

class test_windframe(unittest.TestCase):

    def setUp(self):

        # define turbine locations in global reference frame
        turbineX = np.array([-10., 10.])
        turbineY = np.array([10., -10.])

        # define wind direction in meteorological coordinates
        wind_direction = 90.

        # import component
        from plantenergy.GeneralWindFarmComponents import WindFrame

        # define openmdao problem
        prob = Problem(root=Group())
        prob.root.add('windframe', WindFrame(nTurbines=2), promotes=['*'])

        # initialize openmdao problem
        prob.setup(check=False)

        # assign values to turbine states
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY

        # set wind direction
        prob['wind_direction'] = np.array([wind_direction])

        # run the problem
        prob.run()

        self.prob = prob

    def test_windframe_x_locations(self):
        np.testing.assert_allclose(self.prob['turbineXw'], np.array([10., -10.]))

    def test_windframe_y_locations(self):
        np.testing.assert_allclose(self.prob['turbineYw'], np.array([-10., 10.]))


if __name__ == "__main__":
    unittest.main(verbosity=2)