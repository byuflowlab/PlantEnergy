#!/usr/bin/env python
# encoding: utf-8

"""
OptimizationGroups.py
Created by Jared J. Thomas, Nov. 2015.
Brigham Young University
"""

import numpy as np

from openmdao.api import Group, IndepVarComp, ExecComp

from plantenergy.GeneralWindFarmGroups import DirectionGroup, AEPGroup
from plantenergy.GeneralCOEGroups import COEGroup
from plantenergy.GeneralWindFarmComponents import SpacingComp, BoundaryComp # cost model functions are defunct: calcICC, calcFCR, calcLLC, calcLRC, calcOandM
from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps

import warnings

class OptAEP(Group):
    """
        Group adding optimization parameters to an AEPGroup


        ----------------
        Design Variables
        ----------------
        turbineX:   1D numpy array containing the x coordinates of each turbine in the global reference frame
        turbineY:   1D numpy array containing the x coordinates of each turbine in the global reference frame
        yaw_i:      1D numpy array containing the yaw angle of each turbine in the wind direction reference frame for
                    direction i

        ---------------
        Constant Inputs
        ---------------
        rotorDiameter:                          1D numpy array containing the rotor diameter of each turbine

        axialInduction:                         1D numpy array containing the axial induction of each turbine. These
                                                values are not actually used unless the appropriate floris_param is set.

        generator_efficiency:                   1D numpy array containing the efficiency of each turbine generator

        wind_speed:                             scalar containing a generally applied inflow wind speed

        air_density:                            scalar containing the inflow air density

        windDirections:                         1D numpy array containing the angle from N CW to the inflow direction

        windrose_frequencies:                   1D numpy array containing the probability of each wind direction

        Ct:                                     1D numpy array containing the thrust coefficient of each turbine

        Cp:                                     1D numpy array containing the power coefficient of each turbine

        floris_params:FLORISoriginal(False):    boolean specifying which formulation of the FLORIS model to use. (True
                                                specfies to use the model as originally formulated and published).

        floris_params:CPcorrected(True):        boolean specifying whether the Cp values provided have been adjusted
                                                for yaw

        floris_params:CTcorrected(True):        boolean specifying whether the Ct values provided have been adjusted
                                                for yaw

        -------
        Returns
        -------
        AEP:                scalar containing the final AEP for the wind farm

        power_directions:   1D numpy array containing the power production for each wind direction (unweighted)

        velocitiesTurbines: 1D numpy array of velocity at each turbine in each direction. Currently only accessible by
                            *.AEPgroup.dir%i.unknowns['velocitiesTurbines']

        wt_powers: 1D numpy array of power production at each turbine in each direction. Currently only accessible by
                            *.AEPgroup.dir%i.unknowns['velocitiesTurbines']

    """

    def __init__(self, nTurbines, nDirections=1, minSpacing=2., use_rotor_components=True,
                 datasize=0, differentiable=True, force_fd=False, nVertices=0, wake_model=floris_wrapper,
                 wake_model_options=None, params_IdepVar_func=add_floris_params_IndepVarComps,
                 params_IndepVar_args={'use_rotor_components': False}, cp_points=1, cp_curve_spline=None,
                 rec_func_calls=False):

        # print("initializing OptAEP Group")
        super(OptAEP, self).__init__()

        if wake_model_options is None:
            wake_model_options = {'differentiable': differentiable, 'use_rotor_components': use_rotor_components,
                             'nSamples': 0, 'verbose': False}

        try:
            nSamples = wake_model_options['nSamples']
        except:
            nSamples = 0

        if force_fd:
            self.deriv_options['type'] = 'fd'
            self.deriv_options['form'] = 'forward'

        # ##### add major components and groups
        # add group that calculates AEP
        self.add('AEPgroup', AEPGroup(nTurbines=nTurbines, nDirections=nDirections,
                                            use_rotor_components=use_rotor_components,
                                            datasize=datasize, differentiable=differentiable, wake_model=wake_model,
                                            wake_model_options=wake_model_options,
                                            params_IdepVar_func=params_IdepVar_func,
                                            params_IndepVar_args=params_IndepVar_args, nSamples=nSamples,
                                            cp_points=cp_points, cp_curve_spline=cp_curve_spline,
                                      rec_func_calls=rec_func_calls),
                 promotes=['*'])


        # add component that calculates spacing between each pair of turbines
        self.add('spacing_comp', SpacingComp(nTurbines=nTurbines), promotes=['*'])

        if nVertices > 0:
            # add component that enforces a convex hull wind farm boundary
            self.add('boundary_con', BoundaryComp(nVertices=nVertices, nTurbines=nTurbines), promotes=['*'])
            self.add('bv0', IndepVarComp('boundary_radius', val=1000., units='m',
                                         pass_by_obj=True, desc='radius of wind farm boundary'), promotes=['*'])
            self.add('bv1', IndepVarComp('boundary_center', val=np.array([0., 0.]), units='m', pass_by_obj=True,
                                         desc='x and y positions of circular wind farm boundary center'), promotes=['*'])
        else:
            warnings.warn("nVertices has been set to zero. No boundary constraints can be used unless nVertices > 0",
                                RuntimeWarning)
        # ##### add constraint definitions

        # self.add('s0', IndepVarComp('minSpacing', np.array([minSpacing]), units='m',
        #          pass_by_obj=True, desc='minimum allowable spacing between wind turbines'), promotes=['*'])

        self.add('spacing_con', ExecComp('sc = wtSeparationSquared-(minSpacing*rotorDiameter[0])**2',
                                         minSpacing=np.array([minSpacing]), rotorDiameter=np.zeros(nTurbines),
                                         sc=np.zeros(int(((nTurbines-1.)*nTurbines/2.))),
                                         wtSeparationSquared=np.zeros(int(((nTurbines-1.)*nTurbines/2.)))),
                 promotes=['*'])


        # add objective component
        self.add('obj_comp', ExecComp('obj = -1.*AEP', AEP=0.0), promotes=['*'])

class OptCOE(Group):

    def __init__(self, nTurbines, nDirections=1, minSpacing=2., use_rotor_components=True,
                 datasize=0, differentiable=True, force_fd=False, nVertices=0, wake_model=floris_wrapper,
                 wake_model_options=None, params_IdepVar_func=add_floris_params_IndepVarComps,
                 params_IndepVar_args={'use_rotor_components': False}, cp_points=1, cp_curve_spline=None,
                 rec_func_calls=False):

        # print("initializing OptCOE Group")
        super(OptCOE, self).__init__()

        if wake_model_options is None:
            wake_model_options = {'differentiable': differentiable, 'use_rotor_components': use_rotor_components,
                             'nSamples': 0, 'verbose': False}

        try:
            nSamples = wake_model_options['nSamples']
        except:
            nSamples = 0

        if force_fd:
            self.deriv_options['type'] = 'fd'
            self.deriv_options['form'] = 'forward'

        # ##### add major components and groups
        # add group that calculates AEP
        self.add('COEgroup', COEGroup(nTurbines=nTurbines, nDirections=nDirections,
                                            use_rotor_components=use_rotor_components,
                                            datasize=datasize, differentiable=differentiable, wake_model=wake_model,
                                            wake_model_options=wake_model_options,
                                            params_IdepVar_func=params_IdepVar_func,
                                            params_IndepVar_args=params_IndepVar_args, nSamples=nSamples,
                                            cp_points=cp_points, cp_curve_spline=cp_curve_spline,
                                      rec_func_calls=rec_func_calls),
                 promotes=['*'])


        # add component that calculates spacing between each pair of turbines
        self.add('spacing_comp', SpacingComp(nTurbines=nTurbines), promotes=['*'])

        if nVertices > 0:
            # add component that enforces a convex hull wind farm boundary
            self.add('boundary_con', BoundaryComp(nVertices=nVertices, nTurbines=nTurbines), promotes=['*'])
            self.add('bv0', IndepVarComp('boundary_radius', val=1000., units='m',
                                         pass_by_obj=True, desc='radius of wind farm boundary'), promotes=['*'])
            self.add('bv1', IndepVarComp('boundary_center', val=np.array([0., 0.]), units='m', pass_by_obj=True,
                                         desc='x and y positions of circular wind farm boundary center'), promotes=['*'])
        else:
            warnings.warn("nVertices has been set to zero. No boundary constraints can be used unless nVertices > 0",
                                RuntimeWarning)
        # ##### add constraint definitions

        # self.add('s0', IndepVarComp('minSpacing', np.array([minSpacing]), units='m',
        #          pass_by_obj=True, desc='minimum allowable spacing between wind turbines'), promotes=['*'])

        self.add('spacing_con', ExecComp('sc = wtSeparationSquared-(minSpacing*rotorDiameter[0])**2',
                                         minSpacing=np.array([minSpacing]), rotorDiameter=np.zeros(nTurbines),
                                         sc=np.zeros(int(((nTurbines-1.)*nTurbines/2.))),
                                         wtSeparationSquared=np.zeros(int(((nTurbines-1.)*nTurbines/2.)))),
                 promotes=['*'])


        # add objective component
        #self.add('obj_comp', ExecComp('obj = -1.*AEP', AEP=0.0), promotes=['*'])
        self.add('obj_comp', ExecComp('obj = -1000000./coe', coe=0.0), promotes=['*'])


# Currently unused code
'''
class OptPowerOneDir(Group):
    """ Group connecting the floris model for optimization with one wind direction"""

    def __init__(self, nTurbines, resolution=0, minSpacing=2., differentiable=True, use_rotor_components=True):

        super(OptPowerOneDir, self).__init__()

        # add major components
        self.add('dirComp', AEPGroup(nTurbines, differentiable=differentiable,
                                           use_rotor_components=use_rotor_components), promotes=['*'])
        self.add('spacing_comp', SpacingComp(nTurbines=nTurbines), promotes=['*'])

        # add constraint definitions
        self.add('spacing_con', ExecComp('sc = wtSeparationSquared-(minSpacing*rotorDiameter[0])**2',
                                         minSpacing=minSpacing, rotorDiameter=np.zeros(nTurbines),
                                         sc=np.zeros(((nTurbines-1.)*nTurbines/2.)),
                                         wtSeparationSquared=np.zeros(((nTurbines-1.)*nTurbines/2.))),
                 promotes=['*'])

        # add objective component
        self.add('obj_comp', ExecComp('obj = -1.*dir_power0', dir_power0=0.0), promotes=['*'])

        # initialize design variables for optimization
        # self.add('p1', IndepVarComp('turbineX', np.zeros(nTurbines)), promotes=['*'])
        # self.add('p2', IndepVarComp('turbineY', np.zeros(nTurbines)), promotes=['*'])
        # self.add('p3', IndepVarComp('yaw', np.zeros(nTurbines)), promotes=['*'])



class OptCOE(Group):
    """
        Group adding optimization parameters to an AEPGroup


        ----------------
        Design Variables
        ----------------
        turbineX:   1D numpy array containing the x coordinates of each turbine in the global reference frame
        turbineY:   1D numpy array containing the x coordinates of each turbine in the global reference frame
        yaw_i:      1D numpy array containing the yaw angle of each turbine in the wind direction reference frame for
                    direction i

        ---------------
        Constant Inputs
        ---------------
        rotorDiameter:                          1D numpy array containing the rotor diameter of each turbine

        axialInduction:                         1D numpy array containing the axial induction of each turbine. These
                                                values are not actually used unless the appropriate floris_param is set.

        generator_efficiency:                   1D numpy array containing the efficiency of each turbine generator

        wind_speed:                             scalar containing a generally applied inflow wind speed

        air_density:                            scalar containing the inflow air density

        windDirections:                         1D numpy array containing the angle from N CW to the inflow direction

        windrose_frequencies:                   1D numpy array containing the probability of each wind direction

        Ct:                                     1D numpy array containing the thrust coefficient of each turbine

        Cp:                                     1D numpy array containing the power coefficient of each turbine

        floris_params:FLORISoriginal(False):    boolean specifying which formulation of the FLORIS model to use. (True
                                                specfies to use the model as originally formulated and published).

        floris_params:CPcorrected(True):        boolean specifying whether the Cp values provided have been adjusted
                                                for yaw

        floris_params:CTcorrected(True):        boolean specifying whether the Ct values provided have been adjusted
                                                for yaw

        -------
        Returns
        -------
        COE:                scalar containing the final COE for the wind farm

        power_directions:   1D numpy array containing the power production for each wind direction (unweighted)

        velocitiesTurbines: 1D numpy array of velocity at each turbine in each direction. Currently only accessible by
                            *.AEPgroup.dir%i.unknowns['velocitiesTurbines']

        wt_powers: 1D numpy array of power production at each turbine in each direction. Currently only accessible by
                            *.AEPgroup.dir%i.unknowns['velocitiesTurbines']

    """

    def __init__(self, nTurbines, nDirections=1, minSpacing=2., use_rotor_components=True,
                 datasize=0, differentiable=True, force_fd=False, nVertices=0, wake_model=floris_wrapper,
                 wake_model_options=None, params_IdepVar_func=add_floris_params_IndepVarComps,
                 params_IndepVar_args={'use_rotor_components': False}, nTopologyPoints=0):


        super(OptCOE, self).__init__()

        if wake_model_options is None:
            wake_model_options = {'differentiable': differentiable, 'use_rotor_components': use_rotor_components,
                             'nSamples': 0, 'verbose': False}

        try:
            nSamples = wake_model_options['nSamples']
        except:
            nSamples = 0


        if force_fd:
            self.fd_options['force_fd'] = True
            self.fd_options['form'] = 'forward'

        # ##### add major components and groups
        # add group that calculates AEP
        self.add('AEPgroup', AEPGroup(nTurbines=nTurbines, nDirections=nDirections,
                                            use_rotor_components=use_rotor_components,
                                            datasize=datasize, differentiable=differentiable, wake_model=wake_model,
                                            wake_model_options=wake_model_options,
                                            params_IdepVar_func=params_IdepVar_func,
                                            params_IndepVar_args=params_IndepVar_args, nSamples=nSamples),
                 promotes=['*'])

        # add component that calculates ICC
        self.add('ICCcomp', calcICC(nTurbines=nTurbines, nTopologyPoints=nTopologyPoints), promotes=['*'])

        # add component that calculates FCR
        self.add('FCRcomp', calcFCR(nTurbines=nTurbines), promotes=['*'])

        # add component that calculates LLC
        self.add('LLCcomp', calcLLC(nTurbines=nTurbines), promotes=['*'])

        # add component that calculates O&M
        self.add('OandMcomp', calcOandM(nTurbines=nTurbines), promotes=['*'])

        # add component that calculates LRC
        self.add('LCRcomp', calcLRC(nTurbines=nTurbines), promotes=['*'])

        # add component that calculates spacing between each pair of turbines
        self.add('spacing_comp', SpacingComp(nTurbines=nTurbines), promotes=['*'])

        if nVertices > 0:
            # add component that enforces a convex hull wind farm boundary
            self.add('boundary_con', BoundaryComp(nVertices=nVertices, nTurbines=nTurbines), promotes=['*'])

        # ##### add constraint definitions
        self.add('spacing_con', ExecComp('sc = wtSeparationSquared-(minSpacing*rotorDiameter[0])**2',
                                         minSpacing=minSpacing, rotorDiameter=np.zeros(nTurbines),
                                         sc=np.zeros(int(((nTurbines-1.)*nTurbines/2.))),
                                         wtSeparationSquared=np.zeros(int(((nTurbines-1.)*nTurbines/2.)))),
                 promotes=['*'])

        # add objective component
        self.add('obj_comp', ExecComp('obj = (FCR+ICC)/AEP+LLC+(OandM+LRC)/AEP', ICC=0.0, AEP=0.0, FCR=0.0, LLC=0.0, OandM=0.0, LRC=0.0), promotes=['*'])

'''