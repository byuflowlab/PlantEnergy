"""
larsen.py
Created by Jared J. Thomas, Jul. 2016.
Brigham Young University
"""

from openmdao.api import IndepVarComp, Component, Group
import numpy as np

from fusedwake.gcl.python.gcl import GCLarsen #use w/GCL.f


def add_larsen_params_IndepVarComps(openmdao_object, nTurbines, datasize):

    # add variable tree and indep-var stuff for Larsen
    openmdao_object.add('lp0', IndepVarComp('model_params:Ia', val=0.0, pass_by_object=True,
                                            desc='ambient turbulence intensity'), promotes=['*'])
    openmdao_object.add('lp1', IndepVarComp('model_params:air_density', val=0.0,  units='kg/m*m*m',
                                            pass_by_object=True), promotes=['*'])
    openmdao_object.add('lp2', IndepVarComp('model_params:windSpeedToCPCT_wind_speed', np.zeros(datasize), units='m/s',
                                            desc='range of wind speeds', pass_by_obj=True), promotes=['*'])
    openmdao_object.add('lp3', IndepVarComp('model_params:windSpeedToCPCT_CP', np.zeros(datasize),
                                            desc='power coefficients', pass_by_obj=True), promotes=['*'])
    openmdao_object.add('lp4', IndepVarComp('model_params:windSpeedToCPCT_CT', np.zeros(datasize),
                                            desc='thrust coefficients', pass_by_obj=True), promotes=['*'])
    # TODO make hubHeight a standard connection
    openmdao_object.add('lp5', IndepVarComp('hubHeight', np.zeros(nTurbines)), promotes=['*'])


class GC_Larsen(Component):
    """
    This component was written by Bryce Ingersoll
    """

    def __init__(self, nTurbines, direction_id=0, model_options=None):
        super(GC_Larsen, self).__init__()

        self.add_param('wind_speed', val=8.0, units='m/s')
        self.add_param('wind_direction', val=0.0)
        self.direction_id = direction_id
        self.datasize = model_options['datasize']

        self.wf_instance = model_options['wf_instance']

        # coordinates
        self.add_param('turbineXw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineYw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineZ', val=np.zeros(nTurbines), units='m')

        self.add_param('rotorDiameter', val=np.zeros(nTurbines), units='m')
        self.add_param('hubHeight', np.zeros(nTurbines), units='m')
        self.add_param('model_params:Ia', val=0.0)   # Ambient Turbulence Intensity
        self.add_param('model_params:air_density', val=0.0,  units='kg/m*m*m')

        self.add_param('Ct_length', val = 0.0)
        self.add_param('model_params:windSpeedToCPCT_wind_speed', np.zeros(self.datasize), units='m/s',
                       desc='range of wind speeds', pass_by_obj=True)
        self.add_param('model_params:windSpeedToCPCT_CP', np.zeros(self.datasize),
                       desc='power coefficients', pass_by_obj=True)
        self.add_param('model_params:windSpeedToCPCT_CT', np.zeros(self.datasize),
                       desc='thrust coefficients', pass_by_obj=True)

        #outputs
        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')

        self.add_param('axialInduction', np.zeros(nTurbines))
        self.add_param('yaw%i' % direction_id, np.zeros(nTurbines), units='deg')
        self.add_param('Ct', np.zeros(nTurbines))
        # TODO: get rid of above params

    def solve_nonlinear(self, params, unknowns, resids):

        #Define WindTurbine
        WS = params['wind_speed']
        WD = params['wind_direction']

        xcoord = params['turbineXw']
        ycoord = params['turbineYw']

        nTurbines = xcoord.size

        # Does opposite translation that is done in WindFarm -#
        angle = np.radians(270.+WD)
        ROT = np.array([[np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]])
        [xcoord, ycoord] = np.dot(ROT, [xcoord, ycoord])
        a = np.array([xcoord, ycoord])

        #np.savetxt('test_coord.dat',a.T)
        #-----------------------------------------------------#

        H = params['hubHeight']


        Ia = params['model_params:Ia']


        # -- to run using GCL.F, create gcl instance in wrapper -- #
        self.wf_instance.xyz = np.vstack([a, H])
        self.wf_instance.nWT = nTurbines

        #print(self.wf_instance.WT.refCurvesArray)
        #quit()

        gcl = GCLarsen(WF=self.wf_instance, TI=Ia, z0=0.0001, NG=8, sup='lin', inflow='log')
        gcl(WS=WS, WD=WD, version='fort_gcl')
        hubVelocity = gcl(WS=WS, WD=WD).u_wt
        # -------------------------------------------------------- #

        unknowns['wtVelocity%i' % self.direction_id] = hubVelocity


class larsen_wrapper(Group):
    """
    This Group was written by Bryce Ingersoll
    """

    def __init__(self, nTurbs, direction_id=0, wake_model_options=None):
        super(larsen_wrapper, self).__init__()

        self.add('larsen_model', GC_Larsen(nTurbs, direction_id=direction_id, model_options=wake_model_options),
                 promotes=['*'])


# # Testing code for development only
# if __name__ == "__main__":
#
#     nTurbines = 2
#     direction_id = 0
#
#     prob = Problem()
#     prob.root = Group()
#     prob.root.add('ftest', floris_wrapper(nTurbines, direction_id), promotes=['*'])
#
#     prob.setup(check=True)
