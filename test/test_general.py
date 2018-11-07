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

class test_power_curve(unittest.TestCase):
    def setUp(self):
        # define turbine locations in global reference frame
        turbineX = np.array([-10., 10.])
        turbineY = np.array([10., -10.])

        # define wind direction in meteorological coordinates
        wind_direction = 90.

        nTurbines = 100

        # import component
        from plantenergy.GeneralWindFarmComponents import WindDirectionPower

        # define openmdao problem
        prob = Problem(root=Group())
        prob.root.add('power', WindDirectionPower(nTurbines=nTurbines), promotes=['*'])

        # initialize openmdao problem
        prob.setup(check=False)

        # set inflow velocity
        prob['wtVelocity0'] = np.linspace(3, 30, nTurbines)

        cutin = 4.0
        cutout = 25.
        ratedws = 9.8
        ratedp = 3.35E6

        # assign values to turbine states
        prob['cut_in_speed'] = np.ones(nTurbines) * cutin
        prob['cut_out_speed'] = np.ones(nTurbines) * cutout
        prob['rated_power'] = np.ones(nTurbines) * ratedp
        prob['rated_wind_speed'] = np.ones(nTurbines) * ratedws
        prob['use_power_curve_definition'] = True

        # run the problem
        prob.run()

        def power_curve(velocity, cutin, cutout, ratedws, ratedp):

            power = np.zeros_like(velocity)

            for i in np.arange(0, velocity.size):
                if velocity[i] > cutin and velocity[i] < ratedws:
                    power[i] = ratedp*((velocity[i]-cutin)/(ratedws-cutin))**3
                elif velocity[i] > ratedws and velocity[i] < cutout:
                    power[i] = ratedp

            return power

        self.power_func = power_curve(prob['wtVelocity0'], cutin, cutout, ratedws, ratedp)
        self.prob = prob

        # from matplotlib import pylab as plt
        # plt.plot(prob['wtVelocity0'], self.power_func)
        # plt.plot(prob['wtVelocity0'], self.prob['wtPower0'], '--r')
        #
        # plt.show()
    def test_power_curve_output(self):
        np.testing.assert_equal(self.prob['wtPower0'], self.power_func)
    def test_power_summation(self):
        np.testing.assert_equal(self.prob['dir_power0'], np.sum(self.power_func))

if __name__ == "__main__":
    unittest.main(verbosity=2)