from __future__ import print_function, division, absolute_import
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from plantenergy.GeneralWindFarmComponents import WindFrame, AdjustCtCpYaw, WindFarmAEP, \
     WindDirectionPower, SpacingComp


class TestWindFarmComponentPartials(unittest.TestCase):

    def test_WindFrame(self):
        prob = om.Problem()
        model = prob.model

        nTurbine = 5
        turbineX = 10.0 * np.random.random(nTurbine)
        turbineY = 10.0 * np.random.random(nTurbine)

        wind_direction = 53.

        model.add_subsystem('windframe', WindFrame(nTurbines=nTurbine),
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

        model.add_subsystem('windframe', AdjustCtCpYaw(nTurbines=nTurbine),
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

        model.add_subsystem('windframe', WindFarmAEP(nDirections=nDirections),
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

        model.add_subsystem('windframe', WindDirectionPower(nTurbines=nTurbines),
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

        model.add_subsystem('windframe', SpacingComp(nTurbines=nTurbines),
                            promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob['turbineX'] = 10.0 * np.random.random(nTurbines)
        prob['turbineY'] = 10.0 * np.random.random(nTurbines)

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, atol=1e2, rtol=1e-9)

if __name__ == '__main__':
    unittest.main()
