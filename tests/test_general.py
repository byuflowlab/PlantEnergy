import unittest

import numpy as np

import openmdao.api as om

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
        prob = om.Problem()
        prob.model.add_subsystem('windframe', WindFrame(nTurbines=2), promotes=['*'])

        # initialize openmdao problem
        prob.setup(check=False)

        # assign values to turbine states
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY

        # set wind direction
        prob['wind_direction'] = np.array([wind_direction])

        # run the problem
        prob.run_model()

        self.prob = prob

    def test_windframe_x_locations(self):
        np.testing.assert_allclose(self.prob['turbineXw'], np.array([10., -10.]))

    def test_windframe_y_locations(self):
        np.testing.assert_allclose(self.prob['turbineYw'], np.array([-10., 10.]))

class test_power_curve(unittest.TestCase):
    def setUp(self):

        nTurbines = 100

        # import component
        from plantenergy.GeneralWindFarmComponents import WindDirectionPower

        # define openmdao problem
        prob = om.Problem()
        prob.model.add_subsystem('power', WindDirectionPower(nTurbines=nTurbines), promotes=['*'])

        # initialize openmdao problem
        prob.setup(check=False)

        # set inflow velocity
        prob['wtVelocity0'] = np.linspace(5, 30, nTurbines)

        cutin = 4.0
        cutout = 25.
        ratedws = 9.8
        ratedp = 3.35E3 # in MW

        # assign values to turbine states
        prob['cut_in_speed'] = np.ones(nTurbines) * cutin
        prob['cut_out_speed'] = np.ones(nTurbines) * cutout
        prob['rated_power'] = np.ones(nTurbines) * ratedp
        prob['rated_wind_speed'] = np.ones(nTurbines) * ratedws
        prob['use_power_curve_definition'] = True

        # run the problem
        prob.run_model()

        def power_curve(velocity, cutin, cutout, ratedws, ratedp):

            power = np.zeros_like(velocity)
            for i in np.arange(0, velocity.size):
                if velocity[i] > cutin and velocity[i] < ratedws:
                    power[i] = ratedp * ((velocity[i]) / (ratedws - cutin)) ** 3
                elif velocity[i] > ratedws and velocity[i] < cutout:
                    power[i] = ratedp

                if power[i] > ratedp:
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

class test_separation_constraint_calculations(unittest.TestCase):

    def setUp(self):

        # set tolerance of test
        self.tol = 1E-6

        # define turbine locations in global reference frame
        turbineX = np.array([0., 100.])
        turbineY = np.array([0., 100.])

        nTurbines = 2

        # import component
        from plantenergy.GeneralWindFarmComponents import SpacingComp

        # define openmdao problem
        prob = om.Problem()
        prob.model.add_subsystem('power', SpacingComp(nTurbines=nTurbines), promotes=['*'])

        # initialize openmdao problem
        prob.setup(check=False)

        # set inflow velocity
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY

        # run the problem
        prob.run_model()
        self.prob = prob

    def test_spacing_calc(self):
        np.testing.assert_almost_equal(self.prob['wtSeparationSquared'], 20000.)


class test_boundary_distance_constraint_calculations_polygon(unittest.TestCase):

    def setUp(self):
        # import component
        from plantenergy.GeneralWindFarmComponents import BoundaryComp
        from plantenergy.GeneralWindFarmComponents import calculate_boundary

        # set tolerance of test
        self.tol = 1E-6

        # define turbine locations in global reference frame
        turbineX = np.array([0., 100.])
        turbineY = np.array([0., 100.])
        nTurbines = turbineX.size

        verticy_locations = np.array([[-1.0, -1.0],
                                      [-1., 101],
                                      [101, 101],
                                      [101, -1.],
                                      ])

        boundaryVertices, boundaryNormals = calculate_boundary(verticy_locations)
        nVertices = len(boundaryNormals)

        # define openmdao problem
        prob = om.Problem()
        prob.model.add_subsystem('power', BoundaryComp(nTurbines=nTurbines, nVertices=nVertices), promotes=['*'])

        # initialize openmdao problem
        prob.setup(check=False)

        # set inflow velocity
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['boundaryVertices'] = boundaryVertices
        prob['boundaryNormals'] = boundaryNormals
        prob['turbineY'] = turbineY


        # run the problem
        prob.run_model()
        self.prob = prob

    def test_spacing_calc(self):
        np.testing.assert_almost_equal(self.prob['boundaryDistances'], np.array([[1, 1, 101, 101],[101, 101, 1, 1]]))

class test_boundary_distance_constraint_calculations_circle(unittest.TestCase):

    def setUp(self):
        # import component
        from plantenergy.GeneralWindFarmComponents import BoundaryComp
        from plantenergy.GeneralWindFarmComponents import calculate_boundary

        # set tolerance of test
        self.tol = 1E-6

        # define turbine locations in global reference frame
        turbineX = np.array([0., 100., 0.])
        turbineY = np.array([0., 0., 101.])
        nTurbines = turbineX.size

        boundaryCenter = np.array([0.,0.])
        boundaryRadius = np.array([100.])

        nVertices = 1

        # define openmdao problem
        prob = om.Problem()
        prob.model.add_subsystem('power', BoundaryComp(nTurbines=nTurbines, nVertices=nVertices), promotes=['*'])

        # initialize openmdao problem
        prob.setup(check=False)

        # set inflow velocity
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['boundary_center'] = boundaryCenter
        prob['boundary_radius'] = boundaryRadius
        prob['turbineY'] = turbineY


        # run the problem
        prob.run_model()
        self.prob = prob

    def test_spacing_calc(self):
        np.testing.assert_almost_equal(self.prob['boundaryDistances'], np.array([[10000],[0],[-201]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)