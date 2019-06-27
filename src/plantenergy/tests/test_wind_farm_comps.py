from __future__ import print_function, division, absolute_import
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from plantenergy.GeneralWindFarmComponents import WindFrame, AdjustCtCpYaw, WindFarmAEP, \
     WindDirectionPower, SpacingComp, BoundaryComp


class TestWindFarmComponentPartials(unittest.TestCase):

    def test_WindFrame(self):
        prob = om.Problem()
        model = prob.model

        nTurbine = 5
        turbineX = 10.0 * np.random.random(nTurbine)
        turbineY = 10.0 * np.random.random(nTurbine)

        wind_direction = 53.

        model.add_subsystem('comp', WindFrame(nTurbines=nTurbine),
                            promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['wind_direction'] = np.array([wind_direction])

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, atol=1e2, rtol=1e-9)

    def test_AdjustCtCpYaw(self):
        prob = om.Problem()
        model = prob.model

        nTurbine = 5

        model.add_subsystem('comp', AdjustCtCpYaw(nTurbines=nTurbine),
                            promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob['Ct_in'] = np.random.random(nTurbine)
        prob['Cp_in'] = np.random.random(nTurbine)
        prob['yaw0'] = np.random.random(nTurbine)

        prob['gen_params:CTcorrected'] = False
        prob['gen_params:CPcorrected'] = False

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, atol=1e2, rtol=1e-9)

        # Verify derivs for corrected option turned on

        prob['gen_params:CTcorrected'] = True
        prob['gen_params:CPcorrected'] = True

        # run the problem
        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, atol=1e2, rtol=1e-9)

    def test_WindFarmAEP(self):
        prob = om.Problem()
        model = prob.model

        nDirections = 5

        model.add_subsystem('comp', WindFarmAEP(nDirections=nDirections),
                            promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob['dirPowers'] = .01 * np.random.random(nDirections) + .01
        prob['windFrequencies'] = .01 * np.random.random(nDirections) + .01

        prob['gen_params:AEP_method'] = 'none'

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, atol=1e2, rtol=1e-9)

        prob['gen_params:AEP_method'] = 'log'

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, atol=1e2, rtol=1e-9)

        prob['gen_params:AEP_method'] = 'inverse'

        # run the problem
        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, atol=1e2, rtol=1e-9)

    def test_WindDirectionPower(self):
        prob = om.Problem()
        model = prob.model

        nTurbines = 3
        ncp = 5

        model.add_subsystem('comp', WindDirectionPower(nTurbines=nTurbines),
                            promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob['wtVelocity0'] = np.random.random(nTurbines)
        prob['Cp'] = np.random.random(nTurbines)
        prob['rotorDiameter'] = np.random.random(nTurbines)

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, atol=1e2, rtol=1e-9)

    def test_SpacingComp(self):
        prob = om.Problem()
        model = prob.model

        nTurbines = 5

        model.add_subsystem('comp', SpacingComp(nTurbines=nTurbines),
                            promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob['turbineX'] = 10.0 * np.random.random(nTurbines)
        prob['turbineY'] = 10.0 * np.random.random(nTurbines)

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, atol=1e2, rtol=1e-9)

    def test_BoundaryComp(self):
        # Circle
        prob = om.Problem()
        model = prob.model

        nTurbines = 5
        nVertices = 1

        model.add_subsystem('comp', BoundaryComp(nTurbines=nTurbines, nVertices=nVertices),
                            promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob['turbineX'] = 10.0 * np.random.random(nTurbines)
        prob['turbineY'] = 10.0 * np.random.random(nTurbines)

        prob['boundary_radius'] = 7.0
        prob['boundary_center'] = np.array([5.1, 4.9])

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, atol=1e2, rtol=1e-9)

        # Polygon
        prob = om.Problem()
        model = prob.model

        nTurbines = 5
        nVertices = 3

        model.add_subsystem('comp', BoundaryComp(nTurbines=nTurbines, nVertices=nVertices),
                            promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob['turbineX'] = 10.0 * np.random.random(nTurbines)
        prob['turbineY'] = 10.0 * np.random.random(nTurbines)

        prob['boundaryVertices'] = np.array([[5.0, 11.0], [11.0, -0.5], [-0.5, -0.5]])
        prob['boundaryNormals'] = np.array([[0.0, 1.0], [0.7, -0.3], [-0.7, -0.3]])

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, atol=1e2, rtol=1e-9)

if __name__ == '__main__':
    unittest.main()
