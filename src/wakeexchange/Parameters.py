from openmdao.api import Component
import numpy as np


class FLORISParameters(Component):
    """Container of FLORIS wake parameters"""

    def __init__(self):

        super(FLORISParameters, self).__init__()

        # original tuning parameters
        self.add_param('floris_params:pP', 1.88, pass_by_obj=True)
        self.add_param('floris_params:ke', 0.065, pass_by_obj=True)
        self.add_param('floris_params:keCorrArray', 0.0, pass_by_obj=True)
        self.add_param('floris_params:keCorrCT', 0.0, pass_by_obj=True)
        self.add_param('floris_params:Region2CT', 4.0*(1.0/3.0)*(1.0-(1.0/3.0)), pass_by_obj=True)
        self.add_param('floris_params:kd', 0.15)
        self.add_param('floris_params:me', np.array([-0.5, 0.22, 1.0]), pass_by_obj=True)

        self.add_param('floris_params:initialWakeDisplacement', -4.5, pass_by_obj=True)
        self.add_param('floris_params:initialWakeAngle', 3.0, pass_by_obj=True)

        self.add_param('floris_params:baselineCT', 4./3.*(1.-1./3.), pass_by_obj=True)

        self.add_param('floris_params:keCorrTI', 0.0, pass_by_obj=True)

        self.add_param('floris_params:baselineTI', 0.045, pass_by_obj=True)

        self.add_param('floris_params:keCorrHR', 0.0, pass_by_obj=True) # neutral, with heating rate 0, is baseline

        self.add_param('floris_params:keCorrHRTI', 0.0, pass_by_obj=True)

        self.add_param('floris_params:keSaturation', 0.0, pass_by_obj=True)

        self.add_param('floris_params:kdCorrYawDirection', 0.0, pass_by_obj=True)

        self.add_param('floris_params:MU', np.array([0.5, 1.0, 10]), pass_by_obj=True)

        self.add_param('floris_params:CTcorrected', True, pass_by_obj=True,
                       desc='CT factor already corrected by CCBlade calculation (approximately factor cos(yaw)^2)')

        self.add_param('floris_params:CPcorrected', True, pass_by_obj=True,
                       desc = 'CP factor already corrected by CCBlade calculation (assumed with approximately factor cos(yaw)^3)')

        self.add_param('floris_params:axialIndProvided', False, pass_by_obj=True,
                       desc='CT factor already corrected by CCBlade calculation (approximately factor cos(yaw)^2)')

        self.add_param('floris_params:useWakeAngle', True, pass_by_obj=True)

        self.add_param('floris_params:bd', -0.01, pass_by_obj=True)

        self.add_param('floris_params:useaUbU', False, pass_by_obj=True)
        self.add_param('floris_params:aU', 5.0, units='deg', pass_by_obj=True)
        self.add_param('floris_params:bU', 1.66, pass_by_obj=True)

        self.add_param('floris_params:adjustInitialWakeDiamToYaw', True, pass_by_obj=True)

        self.add_param('floris_params:FLORISoriginal', False, pass_by_obj=True,
                       desc='override all parameters and use FLORIS as original in first Wind Energy paper')

        self.add_param('floris_params:cos_spread', val=3.0, pass_by_obj=True,
                       desc='spread of cosine smoothing factor (percent of sum of wake and rotor radii)')
    #
    # def solve_nonlinear(self, params, unknowns, resids):
    #
    #     return
    #
    # def linearize(self, params, unknowns, resids):
    #
    #     return