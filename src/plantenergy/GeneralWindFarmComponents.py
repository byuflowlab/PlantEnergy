from __future__ import print_function, division, absolute_import
import position_constraints

import matplotlib.pylab as plt
import numpy as np
from scipy import interp
from scipy.io import loadmat
from scipy.spatial import ConvexHull
from scipy.interpolate import UnivariateSpline

import openmdao.api as om

from akima import Akima, akima_interp
from plantenergy.utilities import hermite_spline


def add_gen_params_IdepVarComps(openmdao_group, datasize):
    ivc = om.IndepVarComp()

    ivc.add_discrete_output('gen_params:pP', 1.88)
    ivc.add_discrete_output('gen_params:windSpeedToCPCT_wind_speed', np.zeros(datasize),
                            desc='range of wind speeds')
    ivc.add_discrete_output('gen_params:windSpeedToCPCT_CP', np.zeros(datasize),
                            desc='power coefficients')
    ivc.add_discrete_output('gen_params:windSpeedToCPCT_CT', np.zeros(datasize),
                            desc='thrust coefficients')
    ivc.add_discrete_output('gen_params:CPcorrected', False)
    ivc.add_discrete_output('gen_params:CTcorrected', False)
    ivc.add_discrete_output('gen_params:AEP_method', 'none')

    openmdao_group.add_subsystem('gen_params', ivc, promotes_outputs=['*'])


class WindFrame(om.ExplicitComponent):
    """ Calculates the locations of each turbine in the wind direction reference frame """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")
        self.options.declare('differentiable', types=bool, default=True,
                             desc="Differentiation flag, set to False to force finite difference.")
        self.options.declare('nSamples', types=int, default=0,
                             desc="Number of samples for the visualization arrays.")

    def setup(self):
        opt = self.options
        differentiable = opt['differentiable']
        nTurbines = opt['nTurbines']
        nSamples = opt['nSamples']

        # flow property variables
        self.add_input('wind_direction', val=270.0, units='deg',
                       desc='wind direction using direction from, in deg. cw from north as in meteorological data')

        # Explicitly size input arrays
        self.add_input('turbineX', val=np.zeros(nTurbines), units='m', desc='x positions of turbines in original ref. frame')
        self.add_input('turbineY', val=np.zeros(nTurbines), units='m', desc='y positions of turbines in original ref. frame')

        # add output
        self.add_output('turbineXw', val=np.zeros(nTurbines), units='m', desc='downwind coordinates of turbines')
        self.add_output('turbineYw', val=np.zeros(nTurbines), units='m', desc='crosswind coordinates of turbines')

        # ############################ visualization arrays ##################################

        if nSamples > 0:

            # visualization input Note: always adding the inputs so that we
            # don't have to chnage the signature on compute.
            self.add_discrete_input('wsPositionX', np.zeros(nSamples),
                                    desc='X position of desired measurements in original ref. frame')
            self.add_discrete_input('wsPositionY', np.zeros(nSamples),
                                    desc='Y position of desired measurements in original ref. frame')
            self.add_discrete_input('wPositionZ', np.zeros(nSamples),
                                desc='Z position of desired measurements in original ref. frame')

            # visualization output
            self.add_discrete_output('wsPositionXw', np.zeros(nSamples),
                                     desc='position of desired measurements in wind ref. frame')
            self.add_discrete_output('wsPositionYw', np.zeros(nSamples),
                                     desc='position of desired measurements in wind ref. frame')

        # Derivatives
        if differentiable:
            self.declare_partials(of='*', wrt='wind_direction')

            row_col = np.arange(nTurbines)
            self.declare_partials(of='*', wrt=['turbineX', 'turbineY'], rows=row_col, cols=row_col)

        else:
            # finite difference used for testing only
            self.declare_partials(of='*', wrt='*', method='fd', form='forward', step=1.0e-5,
                                  step_calc='rel')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nSamples = self.options['nSamples']

        windDirectionDeg = inputs['wind_direction']

        # get turbine positions and velocity sampling positions
        turbineX = inputs['turbineX']
        turbineY = inputs['turbineY']

        # convert from meteorological polar system (CW, 0 deg.=N) to standard polar system (CCW, 0 deg.=E)
        windDirectionDeg = 270. - windDirectionDeg
        # windDirectionDeg = 90. - windDirectionDeg # how this was done in SusTech conference paper (oops!)
        if windDirectionDeg < 0.:
            windDirectionDeg += 360.
        windDirectionRad = np.pi*windDirectionDeg/180.0    # inflow wind direction in radians

        cos_wdr = np.cos(-windDirectionRad)
        sin_wdr = np.sin(-windDirectionRad)

        # convert to downwind(x)-crosswind(y) coordinates
        outputs['turbineXw'] = turbineX*cos_wdr - turbineY*sin_wdr
        outputs['turbineYw'] = turbineX*sin_wdr + turbineY*cos_wdr

        if nSamples > 0:
            velX = discrete_inputs['wsPositionX']
            velY = discrete_inputs['wsPositionY']

            discrete_outputs['wsPositionXw'] = velX*cos_wdr - velY*sin_wdr
            discrete_outputs['wsPositionYw'] = velX*sin_wdr + velY*cos_wdr

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        # obtain necessary inputs
        windDirectionDeg = inputs['wind_direction']
        turbineX = inputs['turbineX']
        turbineY = inputs['turbineY']

        # convert from meteorological polar system (CW, 0 deg.=N) to standard polar system (CCW, 0 deg.=E)
        windDirectionDeg = 270. - windDirectionDeg
        if windDirectionDeg < 0.:
            windDirectionDeg += 360.

        # convert inflow wind direction to radians
        windDirectionRad = np.pi*windDirectionDeg/180.0

        cos_wdr = np.cos(-windDirectionRad)
        sin_wdr = np.sin(-windDirectionRad)

        # calculate gradients of conversion to wind direction reference frame
        dturbineXw_dturbineX = cos_wdr
        dturbineXw_dturbineY = -sin_wdr
        dturbineYw_dturbineX = sin_wdr
        dturbineYw_dturbineY = cos_wdr

        # populate Jacobian dict
        partials['turbineXw', 'turbineX'] = dturbineXw_dturbineX
        partials['turbineXw', 'turbineY'] = dturbineXw_dturbineY
        partials['turbineYw', 'turbineX'] = dturbineYw_dturbineX
        partials['turbineYw', 'turbineY'] = dturbineYw_dturbineY

        deg2rad = np.pi / 180.0
        partials['turbineXw', 'wind_direction'] = -deg2rad * (turbineX * sin_wdr + turbineY * cos_wdr)
        partials['turbineYw', 'wind_direction'] = deg2rad * (turbineX * cos_wdr - turbineY * sin_wdr)


class AdjustCtCpYaw(om.ExplicitComponent):
    """ Adjust Cp and Ct to yaw if they are not already adjusted """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")
        self.options.declare('direction_id', types=int, default=0,
                             desc="Direction index.")
        self.options.declare('differentiable', types=bool, default=True,
                             desc="Differentiation flag, set to False to force finite difference.")

    def setup(self):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']
        differentiable = opt['differentiable']

        # Explicitly size input arrays
        self.add_input('Ct_in', val=np.zeros(nTurbines), desc='Thrust coefficient for all turbines')
        self.add_input('Cp_in', val=np.zeros(nTurbines)+(0.7737/0.944) * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2),
                       desc='power coefficient for all turbines')
        self.add_input('yaw%i' % direction_id, val=np.zeros(nTurbines), units='deg', desc='yaw of each turbine')

        # Explicitly size output arrays
        self.add_output('Ct_out', val=np.zeros(nTurbines), desc='Thrust coefficient for all turbines')
        self.add_output('Cp_out', val=np.zeros(nTurbines), desc='power coefficient for all turbines')

        # parameters since var trees are not supports
        self.add_discrete_input('gen_params:pP', 1.88)
        self.add_discrete_input('gen_params:CTcorrected', False,
                                desc='CT factor already corrected by CCBlade calculation (approximately factor cos(yaw)^2)')
        self.add_discrete_input('gen_params:CPcorrected', False,
                                desc='CP factor already corrected by CCBlade calculation (assumed with approximately factor cos(yaw)^3)')

        # Derivatives
        if differentiable:
            row_col = np.arange(nTurbines)
            self.declare_partials(of='Ct_out', wrt=['Ct_in', 'yaw%i' % direction_id],
                                  rows=row_col, cols=row_col)
            self.declare_partials(of='Cp_out', wrt=['Cp_in', 'yaw%i' % direction_id],
                                  rows=row_col, cols=row_col)

        else:
            self.declare_partials(of='*', wrt='*', method='fd', form='forward', step=1.0e-5,
                                  step_calc='rel')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        direction_id = self.options['direction_id']

        # collect inputs
        Ct = inputs['Ct_in']
        Cp = inputs['Cp_in']
        yaw = inputs['yaw%i' % direction_id] * np.pi / 180.
        # print('in Ct correction, Ct_in: '.format(Ct))

        pP = discrete_inputs['gen_params:pP']
        CTcorrected = discrete_inputs['gen_params:CTcorrected']
        CPcorrected = discrete_inputs['gen_params:CPcorrected']

        # calculate new CT values, if desired
        if not CTcorrected:
            # print("ct not corrected")
            outputs['Ct_out'] = np.cos(yaw)*np.cos(yaw)*Ct
            # print('in ct correction Ct_out: '.format(outputs['Ct_out']))
        else:
            outputs['Ct_out'] = Ct

        # calculate new CP values, if desired
        if not CPcorrected:
            outputs['Cp_out'] = Cp * np.cos(yaw) ** pP
        else:
            outputs['Cp_out'] = Cp

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        direction_id = self.options['direction_id']
        yaw_name = 'yaw%i' % direction_id

        # collect inputs
        Ct = inputs['Ct_in']
        Cp = inputs['Cp_in']
        deg2rad = np.pi / 180.0
        yaw = inputs[yaw_name] * deg2rad

        pP = discrete_inputs['gen_params:pP']

        CTcorrected = discrete_inputs['gen_params:CTcorrected']
        CPcorrected = discrete_inputs['gen_params:CPcorrected']

        # calculate gradients and populate Jacobian dict
        if not CTcorrected:
            partials['Ct_out', 'Ct_in'] = np.cos(yaw) * np.cos(yaw)
            partials['Ct_out', yaw_name] = Ct * (-2. * np.sin(yaw) * np.cos(yaw)) * deg2rad
        else:
            partials['Ct_out', 'Ct_in'][:] = 1.0
            partials['Ct_out', yaw_name][:] = 0.0

        if not CPcorrected:
            partials['Cp_out', 'Cp_in'] = np.cos(yaw) ** pP
            partials['Cp_out', yaw_name] = (-Cp * pP * np.sin(yaw) * np.cos(yaw) ** (pP - 1.0)) * deg2rad
        else:
            partials['Cp_out', 'Cp_in'][:] = 1.0
            partials['Cp_out', yaw_name][:] = 0.0


class WindFarmAEP(om.ExplicitComponent):
    """ Estimate the AEP based on power production for each direction and weighted by wind direction frequency  """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nDirections', types=int, default=1,
                             desc="Number of directions for inputs dirPowers and windFrequencies.")

    def setup(self):
        nDirections = self.options['nDirections']

        # define inputs
        self.add_input('dirPowers', np.zeros(nDirections), units='kW',
                       desc='vector containing the power production at each wind direction ccw from north')
        self.add_input('windFrequencies', np.zeros(nDirections),
                       desc='vector containing the weighted frequency of wind at each direction ccw from east using '
                            'direction too')

        self.add_discrete_input('gen_params:AEP_method', val='none',
                                desc='select method with which aep is adjusted for optimization')

        # define output
        self.add_output('AEP', val=0.0, units='kW*h', desc='total annual energy output of wind farm')

        # Derivatives
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # locally name input values
        dirPowers = inputs['dirPowers']
        windFrequencies = inputs['windFrequencies']
        AEP_method = discrete_inputs['gen_params:AEP_method']

        # number of hours in a year
        hours = 8760.0

        # calculate approximate AEP
        AEP = sum(dirPowers*windFrequencies)*hours

        # promote AEP result to class attribute
        if AEP_method == 'none':
            outputs['AEP'] = AEP
        elif AEP_method == 'log':
            outputs['AEP'] = np.log(AEP)
        elif AEP_method == 'inverse':
            outputs['AEP'] = 1.0 / AEP
        else:
            raise ValueError('AEP_method must be one of ["none", "log", "inverse"]')
        # print(AEP)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        # # print('entering AEP - provideJ')
        AEP_method = discrete_inputs['gen_params:AEP_method']

        # assign params to local variables
        dirPowers = inputs['dirPowers']
        windFrequencies = inputs['windFrequencies']

        # number of hours in a year
        hours = 8760.0

        # calculate the derivative of outputs w.r.t. the power in each wind direction
        if AEP_method == 'none':
            partials['AEP', 'dirPowers'][:] = windFrequencies * hours
            partials['AEP', 'windFrequencies'][:] = dirPowers * hours

        elif AEP_method == 'log':
            AEP = sum(dirPowers * windFrequencies) * hours
            partials['AEP', 'dirPowers'] = (1.0 / AEP) * hours * windFrequencies
            partials['AEP', 'windFrequencies'] = (1.0 / AEP) * hours * dirPowers

        elif AEP_method == 'inverse':
            AEP = sum(dirPowers * windFrequencies) * hours
            partials['AEP', 'dirPowers'] = -(1.0 / AEP**2) * hours * windFrequencies
            partials['AEP', 'windFrequencies'] = -(1.0 / AEP**2) * hours * dirPowers

        else:
            raise ValueError('AEP_method must be one of ["none", "log", "inverse"]')


class WindDirectionPower(om.ExplicitComponent):

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")
        self.options.declare('direction_id', types=int, default=0,
                             desc="Direction index.")
        self.options.declare('differentiable', types=bool, default=True,
                             desc="Differentiation flag, set to False to force finite difference.")
        self.options.declare('use_rotor_components', types=bool, default=False,
                             desc="Set to True to use rotor components.")
        self.options.declare('cp_points', types=int, default=1,
                             desc="Number of spline control points.")
        self.options.declare('cp_curve_spline', default=None,
                             desc="Values for cp spline. When set to None (default), the component will make a spline using np.interp.")

    def setup(self):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']
        differentiable = opt['differentiable']
        cp_points = opt['cp_points']

        self.add_input('air_density', 1.1716, units='kg/(m*m*m)', desc='air density in free stream')
        self.add_input('rotorDiameter', np.zeros(nTurbines) + 126.4, units='m', desc='rotor diameters of all turbine')
        self.add_input('Cp', np.zeros(nTurbines)+(0.7737/0.944) * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2),
                       desc='power coefficient for all turbines')
        self.add_input('generatorEfficiency', np.zeros(nTurbines)+0.944, desc='generator efficiency of all turbines')
        self.add_input('wtVelocity%i' % direction_id, np.zeros(nTurbines), units='m/s',
                       desc='effective hub velocity for each turbine')

        self.add_discrete_input('rated_power', np.ones(nTurbines)*5000.,
                                desc='rated power for each turbine (kW)')
        self.add_discrete_input('cut_in_speed', np.ones(nTurbines) * 3.0,
                                desc='cut-in speed for each turbine (m/s)')
        self.add_discrete_input('cp_curve_cp', np.zeros(cp_points),
                                desc='cp as a function of wind speed')
        self.add_discrete_input('cp_curve_wind_speed', np.ones(cp_points),
                                desc='wind speeds corresponding to cp curve cp points (m/s)')

        # for power curve calculation
        self.add_discrete_input('use_power_curve_definition', val=False)
        self.add_discrete_input('rated_wind_speed', np.ones(nTurbines)*11.4,
                                desc='rated wind speed for each turbine (m/s)')
        self.add_discrete_input('cut_out_speed', np.ones(nTurbines) * 25.0,
                                desc='cut-out speed for each turbine (m/s)')

        # outputs
        self.add_output('wtPower%i' % direction_id, np.zeros(nTurbines), units='kW',
                        desc='power output of each turbine')
        self.add_output('dir_power%i' % direction_id, 0.0, units='kW',
                        desc='total power output of the wind farm')

        # Derivatives
        wrt = ['wtVelocity%i' % direction_id, 'rotorDiameter', 'Cp']
        if differentiable:
            self.declare_partials(of='*', wrt=wrt)

        else:
            # finite difference used for testing only
            self.declare_partials(of='*', wrt=wrt, method='fd', form='forward', step=1.0e-6,
                                  step_calc='rel')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']
        cp_points = opt['cp_points']
        cp_curve_spline = opt['cp_curve_spline']
        use_rotor_components = opt['use_rotor_components']

        # obtain necessary inputs
        wtVelocity = inputs['wtVelocity%i' % direction_id]
        rated_power = discrete_inputs['rated_power']
        cut_in_speed = discrete_inputs['cut_in_speed']
        air_density = inputs['air_density']
        rotorArea = 0.25 * np.pi * np.power(inputs['rotorDiameter'], 2)
        Cp = inputs['Cp']
        generatorEfficiency = inputs['generatorEfficiency']

        cp_curve_cp = discrete_inputs['cp_curve_cp']
        cp_curve_wind_speed = discrete_inputs['cp_curve_wind_speed']

        if discrete_inputs['use_power_curve_definition']:
            # obtain necessary inputs
            rated_wind_speed = discrete_inputs['rated_wind_speed']
            cut_out_speed = discrete_inputs['cut_out_speed']

            wtPower = np.zeros(nTurbines)

            # Check to see if turbine produces power for experienced wind speed
            for n in np.arange(0, nTurbines):
                # If we're between the cut-in and rated wind speeds
                if ((cut_in_speed[n] <= wtVelocity[n])
                        and (wtVelocity[n] < rated_wind_speed[n])):
                    # Calculate the curve's power
                    wtPower[n] = rated_power[n] * ((wtVelocity[n] - cut_in_speed[n])
                                                    / (rated_wind_speed[n] - cut_in_speed[n])) ** 3
                # If we're between the rated and cut-out wind speeds
                elif ((rated_wind_speed[n] <= wtVelocity[n])
                      and (wtVelocity[n] < cut_out_speed[n])):
                    # Produce the rated power
                    wtPower[n] = rated_power[n]

            # calculate total power for this direction
            dir_power = np.sum(wtPower)

        else:
            if cp_points > 1.:
                # print('entered Cp')
                if cp_curve_spline is None:
                    for i in np.arange(0, nTurbines):
                        Cp[i] = np.interp(wtVelocity[i], cp_curve_wind_speed, cp_curve_cp)
                        # Cp[i] = spl(wtVelocity[i])
                else:
                    # print('using spline')
                    Cp = cp_curve_spline(wtVelocity)

            # calculate initial values for wtPower (W)
            wtPower = generatorEfficiency*(0.5*air_density*rotorArea*Cp*np.power(wtVelocity, 3))

            # adjust units from W to kW
            wtPower /= 1000.0

            # rated_velocity = np.power(1000.*rated_power/(generator_efficiency*(0.5*air_density*rotorArea*Cp)), 1./3.)
            #
            # dwt_power_dvelocitiesTurbines = np.eye(nTurbines)*generator_efficiency*(1.5*air_density*rotorArea*Cp *
            #                                                                         np.power(wtVelocity, 2))
            # dwt_power_dvelocitiesTurbines /= 1000.

            # adjust wt power based on rated power
            if not use_rotor_components:
                for i in range(0, nTurbines):
                    if wtPower[i] >= rated_power[i]:
                        wtPower[i] = rated_power[i]

            for i in range(0, nTurbines):
                if wtVelocity[i] < cut_in_speed[i]:
                    wtPower[i] = 0.0

            # if np.any(rated_velocity+1.) >= np.any(wtVelocity) >= np.any(rated_velocity-1.) and not \
            #         use_rotor_components:
            #     for i in range(0, nTurbines):
            #         if wtVelocity[i] >= rated_velocity[i]+1.:
            #             spline_start_power = generator_efficiency[i]*(0.5*air_density*rotorArea[i]*Cp[i]*np.power(rated_velocity[i]-1., 3))
            #             deriv_spline_start_power = 3.*generator_efficiency[i]*(0.5*air_density*rotorArea[i]*Cp[i]*np.power(rated_velocity[i]-1., 2))
            #             spline_end_power = generator_efficiency[i]*(0.5*air_density*rotorArea[i]*Cp[i]*np.power(rated_velocity[i]+1., 3))
            #             wtPower[i], deriv = hermite_spline(wtVelocity[i], rated_velocity[i]-1.,
            #                                                                      rated_velocity[i]+1., spline_start_power,
            #                                                                      deriv_spline_start_power, spline_end_power, 0.0)
            #             dwt_power_dvelocitiesTurbines[i][i] = deriv/1000.
            #
            # if np.any(wtVelocity) >= np.any(rated_velocity+1.) and not use_rotor_components:
            #     for i in range(0, nTurbines):
            #         if wtVelocity[i] >= rated_velocity[i]+1.:
            #             wtPower = rated_power
            #             dwt_power_dvelocitiesTurbines[i][i] = 0.0

            # self.dwt_power_dvelocitiesTurbines = dwt_power_dvelocitiesTurbines

            # calculate total power for this direction
            self.wtPower = wtPower
            dir_power = np.sum(wtPower)

        # pass out results
        outputs['wtPower%i' % direction_id] = wtPower
        outputs['dir_power%i' % direction_id] = dir_power

        # print(wtPower)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']
        cp_points = opt['cp_points']
        cp_curve_spline = opt['cp_curve_spline']
        use_rotor_components = opt['use_rotor_components']

        # obtain necessary inputs
        wtVelocity = inputs['wtVelocity%i' % direction_id]
        air_density = inputs['air_density']
        rotorDiameter = inputs['rotorDiameter']
        rotorArea = 0.25*np.pi*np.power(rotorDiameter, 2)
        Cp = inputs['Cp']
        generatorEfficiency = inputs['generatorEfficiency']
        rated_power = discrete_inputs['rated_power']
        cut_in_speed = discrete_inputs['cut_in_speed']

        cp_curve_cp = discrete_inputs['cp_curve_cp']
        cp_curve_wind_speed = discrete_inputs['cp_curve_wind_speed']

        if discrete_inputs['use_power_curve_definition']:
            # obtain necessary inputs
            rated_wind_speed = discrete_inputs['rated_wind_speed']
            cut_out_speed = discrete_inputs['cut_out_speed']

            dwtPower_dwtVelocity = np.zeros([nTurbines, nTurbines])

            # Check to see if turbine produces power for experienced wind speed
            for n in np.arange(0, nTurbines):
                # If we're between the cut-in and rated wind speeds
                if ((cut_in_speed[n] <= wtVelocity[n])
                        and (wtVelocity[n] < rated_wind_speed[n])):
                    # Calculate the derivative of the power curve
                    dwtPower_dwtVelocity[n, n] = (3. * rated_power[n] * ((wtVelocity[n] - cut_in_speed[n])
                                                                         / (rated_wind_speed[n] - cut_in_speed[
                                n])) ** 2) * (1. / (rated_wind_speed[n] - cut_in_speed[n]))
                # If we're between the rated and cut-out wind speeds
                elif ((rated_wind_speed[n] <= wtVelocity[n])
                      and (wtVelocity[n] < cut_out_speed[n])):
                    # Produce the rated power
                    dwtPower_dwtVelocity[n, n] = 0.0

            # calculate total power for this direction
            ddir_power_dwtVelocity = np.matmul(dwtPower_dwtVelocity, np.ones(nTurbines))

            # populate Jacobian dict
            partials['wtPower%i' % direction_id, 'wtVelocity%i' % direction_id] = dwtPower_dwtVelocity
            partials['wtPower%i' % direction_id, 'rotorDiameter'] = np.zeros([nTurbines, nTurbines])
            partials['wtPower%i' % direction_id, 'Cp'] = np.zeros([nTurbines, nTurbines])

            partials['dir_power%i' % direction_id, 'wtVelocity%i' % direction_id] = np.reshape(ddir_power_dwtVelocity,
                                                                                        [1, nTurbines])
            partials['dir_power%i' % direction_id, 'rotorDiameter'] = np.zeros([1, nTurbines])
            partials['dir_power%i' % direction_id, 'Cp'] = np.zeros([1, nTurbines])

        else:
            dCpdV = np.zeros_like(Cp)

            if cp_points > 1. and cp_curve_spline is None:

                for i in np.arange(0, nTurbines):
                    Cp[i] = np.interp(wtVelocity[i], cp_curve_wind_speed, cp_curve_cp)
                    # Cp[i] = spl(wtVelocity[i])
                    dv = 1E-6
                    dCpdV[i] = (np.interp(wtVelocity[i]+dv, cp_curve_wind_speed, cp_curve_cp) -
                             np.interp(wtVelocity[i]- dv, cp_curve_wind_speed, cp_curve_cp))/(2.*dv)

            elif cp_curve_spline is not None:
                # get Cp from the spline

                dCpdV_spline = cp_curve_spline.derivative()

                Cp = np.zeros_like(wtVelocity)
                dCpdV = np.zeros_like(wtVelocity)
                for i in np.arange(0, len(wtVelocity)):
                    Cp[i] = cp_curve_spline(wtVelocity[i])
                    dCpdV[i] = dCpdV_spline(wtVelocity[i])

            # calcuate initial gradient values
            dwtPower_dwtVelocity = np.eye(nTurbines)*0.5*generatorEfficiency*air_density*rotorArea*\
                                   (3.*Cp*np.power(wtVelocity, 2) + np.power(wtVelocity,3)*dCpdV)
            dwtPower_dCp = np.eye(nTurbines)*generatorEfficiency*(0.5*air_density*rotorArea*np.power(wtVelocity, 3))
            dwtPower_drotorDiameter = np.eye(nTurbines)*generatorEfficiency*(0.5*air_density*(0.5*np.pi*rotorDiameter)*Cp *
                                                                               np.power(wtVelocity, 3))
            # dwt_power_dvelocitiesTurbines = self.dwt_power_dvelocitiesTurbines

            # adjust gradients for unit conversion from W to kW
            dwtPower_dwtVelocity /= 1000.
            dwtPower_dCp /= 1000.
            dwtPower_drotorDiameter /= 1000.

            # rated_velocity = np.power(1000.*rated_power/(generator_efficiency*(0.5*air_density*rotorArea*Cp)), 1./3.)

            # if np.any(rated_velocity+1.) >= np.any(wtVelocity) >= np.any(rated_velocity-1.) and not \
            #         use_rotor_components:
            #
            #     spline_start_power = generator_efficiency*(0.5*air_density*rotorArea*Cp*np.power(rated_velocity-1., 3))
            #     deriv_spline_start_power = 3.*generator_efficiency*(0.5*air_density*rotorArea*Cp*np.power(rated_velocity-1., 2))
            #     spline_end_power = generator_efficiency*(0.5*air_density*rotorArea*Cp*np.power(rated_velocity+1., 3))
            #     wtPower, dwt_power_dvelocitiesTurbines = hermite_spline(wtVelocity, rated_velocity-1.,
            #                                                              rated_velocity+1., spline_start_power,
            #                                                              deriv_spline_start_power, spline_end_power, 0.0)

            # set gradients for turbines above rated power to zero
            wtPower = self.wtPower #unknowns['wtPower%i' % direction_id]
            for i in range(0, nTurbines):
                if wtPower[i] >= rated_power[i]:
                    dwtPower_dwtVelocity[i][i] = 0.0
                    dwtPower_dCp[i][i] = 0.0
                    dwtPower_drotorDiameter[i][i] = 0.0

            # set gradients for turbines above rated power to zero
            for i in range(0, nTurbines):
                if wtVelocity[i] < cut_in_speed[i]:
                    dwtPower_dwtVelocity[i][i] = 0.0
                    dwtPower_dCp[i][i] = 0.0
                    dwtPower_drotorDiameter[i][i] = 0.0

            # compile elements of Jacobian
            ddir_power_dwtVelocity = np.array([np.sum(dwtPower_dwtVelocity, 0)])
            ddir_power_dCp = np.array([np.sum(dwtPower_dCp, 0)])
            ddir_power_drotorDiameter = np.array([np.sum(dwtPower_drotorDiameter, 0)])

            # populate Jacobian dict
            partials['wtPower%i' % direction_id, 'wtVelocity%i' % direction_id] = dwtPower_dwtVelocity
            partials['wtPower%i' % direction_id, 'Cp'] = dwtPower_dCp
            partials['wtPower%i' % direction_id, 'rotorDiameter'] = dwtPower_drotorDiameter

            partials['dir_power%i' % direction_id, 'wtVelocity%i' % direction_id] = ddir_power_dwtVelocity
            partials['dir_power%i' % direction_id, 'Cp'] = ddir_power_dCp
            partials['dir_power%i' % direction_id, 'rotorDiameter'] = ddir_power_drotorDiameter

class PositionConstraintComp(Component):
    """ Calculates spacing and boundary constraints
        Written by PJ Stanley, 2019
    """

    def __init__(self, nTurbines, nBoundaries):

        super(PositionConstraintComp, self).__init__()

        self.nTurbines = nTurbines
        # Explicitly size input arrays
        self.add_param('turbineX', val=np.zeros(nTurbines))
        self.add_param('turbineY', val=np.zeros(nTurbines))
        self.add_param('rotorDiameter', val=np.zeros(nTurbines))

        self.add_param('boundaryVertices', val=np.zeros((nBoundaries,2)))
        self.add_param('boundaryNormals', val=np.zeros((nBoundaries,2)))

        self.add_output('spacing_constraint', val=np.zeros((nTurbines-1)*nTurbines/2), pass_by_object=True)
        self.add_output('boundary_constraint', val=np.zeros(nTurbines), pass_by_object=True)


    def solve_nonlinear(self, params, unknowns, resids):

        global nCalls_con
        nCalls_con += 1

        turbineX = params['turbineX']
        turbineY = params['turbineY']
        rotorDiameter = params['rotorDiameter']
        nTurbines = turbineX.size()

        boundaryVertices = params['boundaryVertices']
        boundaryNormals = params['boundaryNormals']


        dx = np.eye(self.nTurbines)
        dy = np.zeros((self.nTurbines,self.nTurbines))
        _,ss_dx,_,bd_dx = position_constraints.constraints_position_dv(turbineX,dx,turbineY,dy,
                                boundaryVertices,boundaryNormals)

        dx = np.zeros((self.nTurbines,self.nTurbines))
        dy = np.eye(self.nTurbines)
        ss,ss_dy,bd,bd_dy = position_constraints.constraints_position_dv(turbineX,dx,turbineY,dy,
                                boundaryVertices,boundaryNormals)

        bounds = np.zeros(nTurbines)
        index = np.zeros(nTurbines)
        for i in range(nTurbines):
            bounds[i] = np.min(bd[i])
            index[i] = np.argmin(bd[i])

        self.index = index
        self.ss_dx = ss_dx
        self.ss_dy = ss_dy
        self.bd_dx = bd_dx
        self.bd_dy = bd_dy

        unknowns['spacing_constraint'] = ss-(2.*rotorDiameter[0])**2
        unknowns['boundary_constraint'] = bounds

    def linearize(self, params, unknowns, resids):

        nTurbines = params['turbineX'].size()

        # initialize Jacobian dict
        J = {}

        # populate Jacobian dict
        J[('spacing_constraint', 'turbineX')] = self.ss_dx.T
        J[('spacing_constraint', 'turbineY')] = self.ss_dy.T

        db_dx = np.zeros((self.nTurbines,self.nTurbines))
        db_dy = np.zeros((self.nTurbines,self.nTurbines))
        for i in range(nTurbines):
            db_dx[i][i] = self.bd_dx[i][i][self.index[i]]
            db_dy[i][i] = self.bd_dy[i][i][self.index[i]]
        J[('boundary_constraint','turbineX')] = db_dx
        J[('boundary_constraint','turbineY')] = db_dy

        return J

class SpacingComp(om.ExplicitComponent):
    """
    Calculates inter-turbine spacing for all turbine pairs
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")

    def setup(self):
        nTurbines = self.options['nTurbines']

        # Explicitly size input arrays
        self.add_input('turbineX', val=np.zeros(nTurbines), units='m',
                       desc='x coordinates of turbines in wind dir. ref. frame')
        self.add_input('turbineY', val=np.zeros(nTurbines), units='m',
                       desc='y coordinates of turbines in wind dir. ref. frame')

        # Explicitly size output array
        self.add_output('wtSeparationSquared', val=np.zeros(int(nTurbines*(nTurbines-1)/2)),
                        desc='spacing of all turbines in the wind farm')

        # Derivatives
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        nTurbines = self.options['nTurbines']

        turbineX = inputs['turbineX']
        turbineY = inputs['turbineY']
        separation_squared = position_constraints.turbine_spacing_squared(turbineX, turbineY)

        outputs['wtSeparationSquared'] = separation_squared

    def compute_partials(self, inputs, partials):
        nTurbines = self.options['nTurbines']

        # obtain necessary inputs
        turbineX = inputs['turbineX']
        turbineY = inputs['turbineY']

        # get number of turbines
        nTurbines = turbineX.size

        turbineXd = np.eye(nTurbines)
        turbineYd = np.zeros((nTurbines, nTurbines))

        _, separation_squareddx = \
            position_constraints.turbine_spacing_squared_dv(turbineX, turbineXd, turbineY, turbineYd)

        turbineXd = np.zeros((nTurbines, nTurbines))
        turbineYd = np.eye(nTurbines)

        _, separation_squareddy = \
            position_constraints.turbine_spacing_squared_dv(turbineX, turbineXd, turbineY, turbineYd)

        # populate Jacobian dict

        partials['wtSeparationSquared', 'turbineX'] = np.transpose(separation_squareddx)
        partials['wtSeparationSquared', 'turbineY'] = np.transpose(separation_squareddy)

class BoundaryComp(om.ExplicitComponent):

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")
        self.options.declare('nVertices', types=int, default=0,
                             desc="Number of vertices.")

    def setup(self):
        opt = self.options
        nTurbines = opt['nTurbines']
        nVertices = opt['nVertices']

        if nVertices > 1:
            self.type = poly_type = 'polygon'
        elif nVertices == 1:
            self.type = poly_type = 'circle'
        else:
            ValueError('nVertices in BoundaryComp must be greater than 0')

        if poly_type == 'polygon':
            #     Explicitly size input arrays
            self.add_discrete_input('boundaryVertices', np.zeros([nVertices, 2]),
                                    desc="vertices of the convex hull CCW in order s.t. boundaryVertices[i] -> first point of face"
                                    "for unit_normals[i] (m)")
            self.add_discrete_input('boundaryNormals', np.zeros([nVertices, 2]),
                                    desc="unit normal vector for each boundary face CCW where boundaryVertices[i] is "
                                    "the first point of the corresponding face")
        else:
            self.add_discrete_input('boundary_radius', val=1000., desc='radius of wind farm boundary (m)')
            self.add_discrete_input('boundary_center', val=np.array([0., 0.]),
                                    desc='x and y positions of circular wind farm boundary center (m)')

        self.add_input('turbineX', np.zeros(nTurbines), units='m',
                       desc='x coordinates of turbines in global ref. frame')
        self.add_input('turbineY', np.zeros(nTurbines), units='m',
                       desc='y coordinates of turbines in global ref. frame')

        # Explicitly size output array
        # (vector with positive elements if turbines outside of hull)
        self.add_output('boundaryDistances', np.zeros([nTurbines, nVertices]),
                        desc="signed perpendicular distance from each turbine to each face CCW; + is inside")

        # Derivatives
        if poly_type == 'polygon':
            self.declare_partials(of='*', wrt='*')
        else:
            row_col = np.arange(nTurbines)
            self.declare_partials(of='*', wrt='*', rows=row_col, cols=row_col)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nTurbines = self.options['nTurbines']

        turbineX = inputs['turbineX']
        turbineY = inputs['turbineY']

        if self.type == 'polygon':
            # put locations in correct arrangement for calculations
            locations = np.zeros([nTurbines, 2], dtype=inputs._data.dtype)
            for i in range(0, nTurbines):
                locations[i] = np.array([turbineX[i], turbineY[i]], dtype=inputs._data.dtype)

            # print("in comp, locs are: ".format(locations))

            # calculate distance from each point to each face
            outputs['boundaryDistances'] = position_constraints.boundary_distances(turbineX, turbineY,
                                                               params['boundaryVertices'], params['boundaryNormals'])
        else:
            xc = discrete_inputs['boundary_center'][0]
            yc = discrete_inputs['boundary_center'][1]
            r = discrete_inputs['boundary_radius']
            outputs['boundaryDistances'][:, 0] = r**2 - (np.power((turbineX - xc), 2) + np.power((turbineY - yc), 2))

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        opt = self.options
        nTurbines = opt['nTurbines']
        nVertices = opt['nVertices']

        if self.type == 'polygon':
            unit_normals = discrete_inputs['boundaryNormals']

            # initialize array to hold distances from each point to each face
            dfaceDistance_dx = np.zeros([int(nTurbines*nVertices), nTurbines], dtype=inputs._data.dtype)
            dfaceDistance_dy = np.zeros([int(nTurbines*nVertices), nTurbines], dtype=inputs._data.dtype)

            for i in range(0, nTurbines):
                # determine if point is inside or outside of each face, and distance from each face
                for j in range(0, nVertices):

                    # define the derivative vectors from the point of interest to the first point of the face
                    dpa_dx = np.array([-1.0, 0.0], dtype=inputs._data.dtype)
                    dpa_dy = np.array([0.0, -1.0], dtype=inputs._data.dtype)

                    # find perpendicular distance derivatives from point to current surface (vector projection)
                    ddistanceVec_dx = np.vdot(dpa_dx, unit_normals[j])*unit_normals[j]
                    ddistanceVec_dy = np.vdot(dpa_dy, unit_normals[j])*unit_normals[j]

                    # calculate derivatives for the sign of perpendicular distance from point to current face
                    dfaceDistance_dx[i*nVertices+j, i] = np.vdot(ddistanceVec_dx, unit_normals[j])
                    dfaceDistance_dy[i*nVertices+j, i] = np.vdot(ddistanceVec_dy, unit_normals[j])

        else:
            turbineX = inputs['turbineX']
            turbineY = inputs['turbineY']
            xc = discrete_inputs['boundary_center'][0]
            yc = discrete_inputs['boundary_center'][1]

            dfaceDistance_dx = - 2. * (turbineX - xc)
            dfaceDistance_dy = - 2. * (turbineY - yc)

        # return Jacobian dict
        partials['boundaryDistances', 'turbineX'] = dfaceDistance_dx
        partials['boundaryDistances', 'turbineY'] = dfaceDistance_dy


def calculate_boundary(vertices):

    # find the points that actually comprise a convex hull
    hull = ConvexHull(list(vertices))

    # keep only vertices that actually comprise a convex hull and arrange in CCW order
    vertices = vertices[hull.vertices]

    # get the real number of vertices
    nVertices = vertices.shape[0]

    # initialize normals array
    unit_normals = np.zeros([nVertices, 2])

    # determine if point is inside or outside of each face, and distance from each face
    for j in range(0, nVertices):

        # calculate the unit normal vector of the current face (taking points CCW)
        if j < nVertices - 1:  # all but the set of point that close the shape
            normal = np.array([vertices[j+1, 1]-vertices[j, 1],
                               -(vertices[j+1, 0]-vertices[j, 0])])
            unit_normals[j] = normal/np.linalg.norm(normal)
        else:   # the set of points that close the shape
            normal = np.array([vertices[0, 1]-vertices[j, 1],
                               -(vertices[0, 0]-vertices[j, 0])])
            unit_normals[j] = normal/np.linalg.norm(normal)

    return vertices, unit_normals


def calculate_distance(points, vertices, unit_normals, return_bool=False, dtype=None):

    """
    :param points: points that you want to calculate the distance from to the faces of the convex hull
    :param vertices: vertices of the convex hull CCW in order s.t. vertices[i] -> first point of face for
           unit_normals[i]
    :param unit_normals: unit normal vector for each face CCW where vertices[i] is first point of face
    :param return_bool: set to True to return an array of bools where True means the corresponding point
           is inside the hull
    :param dtype: Numpy dtype for new arrays
    :return face_distace: signed perpendicular distance from each point to each face; + is inside
    :return [inside]: (optional) an array of zeros and ones where 1.0 means the corresponding point is inside the hull
    """

    # print points.shape, vertices.shape, unit_normals.shape

    nPoints = points.shape[0]
    nVertices = vertices.shape[0]

    # initialize array to hold distances from each point to each face
    face_distance = np.zeros([nPoints, nVertices], dtype=dtype)

    if not return_bool:
        # loop through points and find distance to each face
        for i in range(0, nPoints):

            # determine if point is inside or outside of each face, and distance from each face
            for j in range(0, nVertices):

                # define the vector from the point of interest to the first point of the face
                pa = np.array([vertices[j, 0]-points[i, 0], vertices[j, 1]-points[i, 1]], dtype=dtype)

                # find perpendicular distance from point to current surface (vector projection)
                d_vec = np.vdot(pa, unit_normals[j])*unit_normals[j]

                # calculate the sign of perpendicular distance from point to current face (+ is inside, - is outside)
                face_distance[i, j] = np.vdot(d_vec, unit_normals[j])

        return face_distance

    else:
        # initialize array to hold boolean indicating whether a point is inside the hull or not
        inside = np.zeros(nPoints)

        # loop through points and find distance to each face
        for i in range(0, nPoints):

            # determine if point is inside or outside of each face, and distance from each face
            for j in range(0, nVertices):

                # define the vector from the point of interest to the first point of the face
                pa = np.array([vertices[j, 0]-points[i, 0], vertices[j, 1]-points[i, 1]], dtype=dtype)

                # find perpendicular distance from point to current surface (vector projection)
                d_vec = np.vdot(pa, unit_normals[j])*unit_normals[j]

                # calculate the sign of perpendicular distance from point to current face (+ is inside, - is outside)
                face_distance[i, j] = np.vdot(d_vec, unit_normals[j])

            # check if the point is inside the convex hull by checking the sign of the distance
            if np.all(face_distance[i] >= 0):
                inside[i] = 1.0

        return face_distance, inside


# Note, this version performs manual fd. Not converted to openmdao 2.0.
## ---- if you know wind speed to power and thrust, you can use these tools ----------------
#class CPCT_Interpolate_Gradients(om.ExplicitComponent):

    #def initialize(self):
        #"""
        #Declare options.
        #"""
        #self.options.declare('nTurbines', types=int, default=0,
                             #desc="Number of wind turbines.")
        #self.options.declare('direction_id', types=int, default=0,
                             #desc="Direction index.")
        #self.options.declare('datasize', types=int, default=0,
                             #desc="Dimension of the coefficient arrays.")

    #def setup(self):
        #opt = self.options
        #nTurbines = opt['nTurbines']
        #direction_id = opt['direction_id']
        #datasize = opt['datasize']

        ## add inputs and outputs
        #self.add_input('yaw%i' % direction_id, np.zeros(nTurbines), desc='yaw error', units='deg')
        #self.add_input('wtVelocity%i' % direction_id, np.zeros(nTurbines), units='m/s',
                       #desc='hub height wind speed') # Uhub
        #self.add_output('Cp_out', np.zeros(nTurbines))
        #self.add_output('Ct_out', np.zeros(nTurbines))

        ## add variable trees
        #self.add_discrete_input('gen_params:pP', 1.88)
        #self.add_discrete_input('gen_params:windSpeedToCPCT_wind_speed', np.zeros(datasize),
                                #desc='range of wind speeds (m/s)')
        #self.add_discrete_input('gen_params:windSpeedToCPCT_CP', np.zeros(datasize),
                                #desc='power coefficients')
        #self.add_discrete_input('gen_params:windSpeedToCPCT_CT', np.zeros(datasize),
                                #desc='thrust coefficients')

        ## Derivatives
        #self.declare_partials(of='*', wrt='*')

    #def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        #direction_id = self.options['direction_id']

        ## obtain necessary inputs
        #pP = discrete_inputs['gen_params:pP']
        #yaw = self.params['yaw%i' % direction_id]*np.pi/180.0
        #cos_yaw = np.cos(yaw)

        #wind_speed_ax = np.cos(cos_yaw)**(pP/3.0)*self.params['wtVelocity%i' % direction_id]
        ## use interpolation on precalculated CP-CT curve
        #wind_speed_ax = np.maximum(wind_speed_ax, self.params['gen_params:windSpeedToCPCT_wind_speed'][0])
        #wind_speed_ax = np.minimum(wind_speed_ax, self.params['gen_params:windSpeedToCPCT_wind_speed'][-1])
        #self.unknowns['Cp_out'] = interp(wind_speed_ax, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CP'])
        #self.unknowns['Ct_out'] = interp(wind_speed_ax, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CT'])

        ## for i in range(0, len(self.unknowns['Ct_out'])):
        ##     self.unknowns['Ct_out'] = max(max(self.unknowns['Ct_out']), self.unknowns['Ct_out'][i])
        ## normalize on incoming wind speed to correct coefficients for yaw
        #outputs['Cp_out'] = self.unknowns['Cp_out'] * np.cos(cos_yaw)**pP
        #outputs['Ct_out'] = self.unknowns['Ct_out'] * np.cos(cos_yaw)**2

    #def linearize(self, params, unknowns, resids):  # standard central differencing
        ## set step size for finite differencing
        #h = 1e-6
        #direction_id = self.direction_id

        ## calculate upper and lower function values
        #wind_speed_ax_high_yaw = np.cos((self.params['yaw%i' % direction_id]+h)*np.pi/180.0)**(self.params['gen_params:pP']/3.0)*self.params['wtVelocity%i' % direction_id]
        #wind_speed_ax_low_yaw = np.cos((self.params['yaw%i' % direction_id]-h)*np.pi/180.0)**(self.params['gen_params:pP']/3.0)*self.params['wtVelocity%i' % direction_id]
        #wind_speed_ax_high_wind = np.cos(cos_yaw)**(self.params['gen_params:pP']/3.0)*(self.params['wtVelocity%i' % direction_id]+h)
        #wind_speed_ax_low_wind = np.cos(cos_yaw)**(self.params['gen_params:pP']/3.0)*(self.params['wtVelocity%i' % direction_id]-h)

        ## use interpolation on precalculated CP-CT curve
        #wind_speed_ax_high_yaw = np.maximum(wind_speed_ax_high_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'][0])
        #wind_speed_ax_low_yaw = np.maximum(wind_speed_ax_low_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'][0])
        #wind_speed_ax_high_wind = np.maximum(wind_speed_ax_high_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'][0])
        #wind_speed_ax_low_wind = np.maximum(wind_speed_ax_low_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'][0])

        #wind_speed_ax_high_yaw = np.minimum(wind_speed_ax_high_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'][-1])
        #wind_speed_ax_low_yaw = np.minimum(wind_speed_ax_low_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'][-1])
        #wind_speed_ax_high_wind = np.minimum(wind_speed_ax_high_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'][-1])
        #wind_speed_ax_low_wind = np.minimum(wind_speed_ax_low_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'][-1])

        #CP_high_yaw = interp(wind_speed_ax_high_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CP'])
        #CP_low_yaw = interp(wind_speed_ax_low_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CP'])
        #CP_high_wind = interp(wind_speed_ax_high_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CP'])
        #CP_low_wind = interp(wind_speed_ax_low_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CP'])

        #CT_high_yaw = interp(wind_speed_ax_high_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CT'])
        #CT_low_yaw = interp(wind_speed_ax_low_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CT'])
        #CT_high_wind = interp(wind_speed_ax_high_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CT'])
        #CT_low_wind = interp(wind_speed_ax_low_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CT'])

        ## normalize on incoming wind speed to correct coefficients for yaw
        #CP_high_yaw = CP_high_yaw * np.cos((self.params['yaw%i' % direction_id]+h)*np.pi/180.0)**self.params['gen_params:pP']
        #CP_low_yaw = CP_low_yaw * np.cos((self.params['yaw%i' % direction_id]-h)*np.pi/180.0)**self.params['gen_params:pP']
        #CP_high_wind = CP_high_wind * np.cos((self.params['yaw%i' % direction_id])*np.pi/180.0)**self.params['gen_params:pP']
        #CP_low_wind = CP_low_wind * np.cos((self.params['yaw%i' % direction_id])*np.pi/180.0)**self.params['gen_params:pP']

        #CT_high_yaw = CT_high_yaw * np.cos((self.params['yaw%i' % direction_id]+h)*np.pi/180.0)**2
        #CT_low_yaw = CT_low_yaw * np.cos((self.params['yaw%i' % direction_id]-h)*np.pi/180.0)**2
        #CT_high_wind = CT_high_wind * np.cos((self.params['yaw%i' % direction_id])*np.pi/180.0)**2
        #CT_low_wind = CT_low_wind * np.cos((self.params['yaw%i' % direction_id])*np.pi/180.0)**2

        ## compute derivative via central differencing and arrange in sub-matrices of the Jacobian
        #dCP_dyaw = np.eye(self.nTurbines)*(CP_high_yaw-CP_low_yaw)/(2.0*h)
        #dCP_dwind = np.eye(self.nTurbines)*(CP_high_wind-CP_low_wind)/(2.0*h)
        #dCT_dyaw = np.eye(self.nTurbines)*(CT_high_yaw-CT_low_yaw)/(2.0*h)
        #dCT_dwind = np.eye(self.nTurbines)*(CT_high_wind-CT_low_wind)/(2.0*h)

        ## compile Jacobian dict from sub-matrices
        #J = {}
        #J['Cp_out', 'yaw%i' % direction_id] = dCP_dyaw
        #J['Cp_out', 'wtVelocity%i' % direction_id] = dCP_dwind
        #J['Ct_out', 'yaw%i' % direction_id] = dCT_dyaw
        #J['Ct_out', 'wtVelocity%i' % direction_id] = dCT_dwind

        #return J


class CPCT_Interpolate_Gradients_Smooth(om.ExplicitComponent):

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")
        self.options.declare('direction_id', types=int, default=0,
                             desc="Direction index.")
        self.options.declare('datasize', types=int, default=0,
                             desc="Dimension of the coefficient arrays.")

    def setup(self):
        opt = self.options
        nTurbines = opt['nTurbines']
        direction_id = opt['direction_id']
        datasize = opt['datasize']

        # add inputs and outputs
        self.add_input('yaw%i' % direction_id, np.zeros(nTurbines), desc='yaw error', units='deg')
        self.add_input('wtVelocity%i' % direction_id, np.zeros(nTurbines), units='m/s',
                       desc='hub height wind speed') # Uhub
        self.add_output('Cp_out', np.zeros(nTurbines))
        self.add_output('Ct_out', np.zeros(nTurbines))

        # add variable trees
        self.add_discrete_input('gen_params:pP', 3.0)
        self.add_discrete_input('gen_params:windSpeedToCPCT_wind_speed', np.zeros(datasize),
                                desc='range of wind speeds (m/s)')
        self.add_discrete_input('gen_params:windSpeedToCPCT_CP', np.zeros(datasize),
                                desc='power coefficients')
        self.add_discrete_input('gen_params:windSpeedToCPCT_CT', np.zeros(datasize),
                                desc='thrust coefficients')

        # Derivatives
        row_col = np.arange(nTurbines)
        self.declare_partials(of='*', wrt='*', rows=row_col, cols=row_col)

        # Can't complex step across Akima
        self.set_check_partial_options(wrt='*', method='fd')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        direction_id = self.options['direction_id']
        pP = discrete_inputs['gen_params:pP']

        wtVelocity = inputs['wtVelocity%i' % direction_id]
        yaw = inputs['yaw%i' % direction_id] * np.pi / 180.
        cos_yaw = np.cos(yaw)

        start = 5
        skip = 8
        # Cp = discrete_inputs['gen_params:windSpeedToCPCT_CP'][start::skip]
        Cp = discrete_inputs['gen_params:windSpeedToCPCT_CP']
        # Ct = discrete_inputs['gen_params:windSpeedToCPCT_CT'][start::skip]
        Ct = discrete_inputs['gen_params:windSpeedToCPCT_CT']
        # windspeeds = discrete_inputs['gen_params:windSpeedToCPCT_wind_speed'][start::skip]
        windspeeds = discrete_inputs['gen_params:windSpeedToCPCT_wind_speed']
        #
        # Cp = np.insert(Cp, 0, Cp[0]/2.0)
        # Cp = np.insert(Cp, 0, 0.0)
        # Ct = np.insert(Ct, 0, np.max(discrete_inputs['gen_params:windSpeedToCPCT_CP'])*0.99)
        # Ct = np.insert(Ct, 0, np.max(discrete_inputs['gen_params:windSpeedToCPCT_CT']))
        # windspeeds = np.insert(windspeeds, 0, 2.5)
        # windspeeds = np.insert(windspeeds, 0, 0.0)
        #
        # Cp = np.append(Cp, 0.0)
        # Ct = np.append(Ct, 0.0)
        # windspeeds = np.append(windspeeds, 30.0)

        CPspline = Akima(windspeeds, Cp)
        CTspline = Akima(windspeeds, Ct)

        CP, dCPdvel, _, _ = CPspline.interp(wtVelocity)
        CT, dCTdvel, _, _ = CTspline.interp(wtVelocity)

        # print('in solve_nonlinear', dCPdvel, dCTdvel)
        Cp_out = CP * cos_yaw**pP
        Ct_out = CT * cos_yaw**2.

        # print("in rotor, Cp = [%f. %f], Ct = [%f, %f]".format(Cp_out[0], Cp_out[1], Ct_out[0], Ct_out[1]))

        self.dCp_out_dyaw = (-np.sin(yaw))*(np.pi/180.)*pP*CP*cos_yaw**(pP-1.)
        self.dCp_out_dvel = dCPdvel * cos_yaw**pP

        # print('in solve_nonlinear', self.dCp_out_dyaw, self.dCp_out_dvel)

        self.dCt_out_dyaw = (-np.sin(yaw)) * (np.pi/180.)*2.*CT*cos_yaw
        self.dCt_out_dvel = dCTdvel*cos_yaw**2.

        # normalize on incoming wind speed to correct coefficients for yaw
        outputs['Cp_out'] = Cp_out
        outputs['Ct_out'] = Ct_out

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        direction_id = self.options['direction_id']

        # compile Jacobian dict
        partials['Cp_out', 'yaw%i' % direction_id] = self.dCp_out_dyaw
        partials['Cp_out', 'wtVelocity%i' % direction_id] = self.dCp_out_dvel
        partials['Ct_out', 'yaw%i' % direction_id] = self.dCt_out_dyaw
        partials['Ct_out', 'wtVelocity%i' % direction_id] = self.dCt_out_dvel


## legacy code for simple COE calculations - should be done more formally
#'''
#class calcICC(Component):
    #"""
    #Calculates ICC (initial capital cost) for given windfarm layout
    #The initial capital cost is the sum of the turbine system cost and the balance of station cost.
    #Neither cost includes construction financing or financing fees,
    #because these are calculated and added separately through the fixed charge rate.
    #The costs also do not include a debt service reserve fund, which is assumed to be zero for balance sheet financing.
    #"""

    #def __init__(self, nTurbines, nTopologyPoints):

        #super(calcICC, self).__init__()

        ## Add inputs
        #self.add_param('turbineX', val=np.zeros(nTurbines),
                       #desc='x coordinates of turbines in wind dir. ref. frame')
        #self.add_param('turbineY', val=np.zeros(nTurbines),
                       #desc='y coordinates of turbines in wind dir. ref. frame')

        #self.add_param('hubHeight', val=np.zeros(nTurbines), units='m')

        #self.add_param('rotorDiameter', val=np.zeros(nTurbines), units='m')

        #self.add_param('topologyX', val=np.zeros(nTopologyPoints),
                       #desc = 'x coordiantes of topology')
        #self.add_param('topologyY', val=np.zeros(nTopologyPoints),
                       #desc = 'y coordiantes of topology')
        #self.add_param('topologyZ', val=np.zeros(nTopologyPoints),
                       #desc = 'z coordiantes of topology')

        ## import topology information

        ## define output
        #self.add_output('ICC', val=0.0, units='$', desc='Initial Capital Cost')

    #def solve_nonlinear(self, params, unknowns, resids):


        #turbineX = params['turbineX']
        #turbineY = params['turbineY']
        #nTurbines = turbineX.size

        #topologyX = params['topologyX']
        #topologyY = params['topologyY']
        #topologyZ = params['topologyZ']

        ##calculate ICC
        #ICCpartsx = np.zeros([nTurbines,1])
        #ICCpartsy = np.zeros([nTurbines,1])

        ##need to come up with good way to interpolate between points
        ##right now, using linear interpolation
        #mx = (topologyZ[2]-topologyZ[0])/(topologyX[2]-topologyX[0])

        #my = (topologyZ[2]-topologyZ[0])/(topologyY[2]-topologyY[0])

        #for i in range(0, nTurbines):
            #ICCpartsx[i] = mx*(turbineX[i]-topologyX[2])+topologyZ[2]
            #ICCpartsy[i] = mx*(turbineY[i]-topologyY[2])+topologyZ[2]

        #unknowns['ICC'] = sum(ICCpartsx) +  sum(ICCpartsy)

#class calcFCR(Component):
    #"""
    #Calculates FCR (fixed charge rate) for given windfarm layout
    #"""

    #def __init__(self, nTurbines):

        #super(calcFCR, self).__init__()

        ## Add inputs
        #self.add_param('turbineX', val=np.zeros(nTurbines),
                       #desc='x coordinates of turbines in wind dir. ref. frame')
        #self.add_param('turbineY', val=np.zeros(nTurbines),
                       #desc='y coordinates of turbines in wind dir. ref. frame')

        ## define output
        #self.add_output('FCR', val=0.0, desc='Fixed Charge Rate')

    #def solve_nonlinear(self, params, unknowns, resids):


        #turbineX = params['turbineX']
        #turbineY = params['turbineY']
        #nTurbines = turbineX.size

        ##calculate FCR
        #unknowns['FCR'] = 10000.0

#class calcLLC(Component):
    #"""
    #Calculates LLC (landlease cost) for given windfarm layout
    #Annual operating expenses (AOE) include land or ocean bottom lease cost, levelized O&M cost,
    #and levelized replacement/overhaul cost (LRC). Land lease costs (LLC) are the rental or lease fees
    #charged for the turbine installation. LLC is expressed in units of $/kWh.
    #"""

    #def __init__(self, nTurbines):

        #super(calcLLC, self).__init__()

        ## Add inputs
        #self.add_param('turbineX', val=np.zeros(nTurbines),
                       #desc='x coordinates of turbines in wind dir. ref. frame')
        #self.add_param('turbineY', val=np.zeros(nTurbines),
                       #desc='y coordinates of turbines in wind dir. ref. frame')

        ## define output
        #self.add_output('LLC', val=0.0, units='$/kWh', desc='Landlease Cost')

    #def solve_nonlinear(self, params, unknowns, resids):


        #turbineX = params['turbineX']
        #turbineY = params['turbineY']
        #nTurbines = turbineX.size

        ##calculate LLC
        #unknowns['LLC'] = 10000.0

#class calcOandM(Component):
    #"""
    #Calculates O&M (levelized operation & maintenance cost) for given windfarm layout
    #A component of AOE that is larger than the LLC is O&M cost. O&M is expressed in units of $/kWh.
    #The O&M cost normally includes
        #- labor, parts, and supplies for scheduled turbine maintenance
        #- labor, parts, and supplies for unscheduled turbine maintenance
        #- parts and supplies for equipment and facilities maintenance
        #- labor for administration and support.
    #"""

    #def __init__(self, nTurbines):

        #super(calcOandM, self).__init__()

        ## Add inputs
        #self.add_param('turbineX', val=np.zeros(nTurbines),
                       #desc='x coordinates of turbines in wind dir. ref. frame')
        #self.add_param('turbineY', val=np.zeros(nTurbines),
                       #desc='y coordinates of turbines in wind dir. ref. frame')

        ## define output
        #self.add_output('OandM', val=0.0, units='$', desc='levelized O&M cost')

    #def solve_nonlinear(self, params, unknowns, resids):


        #turbineX = params['turbineX']
        #turbineY = params['turbineY']
        #nTurbines = turbineX.size

        ##calculate LLC

        ##need to know area of boundary?

        #unknowns['OandM'] = 10000.0

#class calcLRC(Component):
    #"""
    #Calculates LRC (levelized replacement/overhaul cost) for given windfarm layout
    #LRC distributes the cost of major replacements and overhauls over the life of the wind turbine and is expressed in $/kW machine rating.
    #"""

    #def __init__(self, nTurbines):

        #super(calcLRC, self).__init__()

        ## Add inputs
        #self.add_param('turbineX', val=np.zeros(nTurbines),
                       #desc='x coordinates of turbines in wind dir. ref. frame')
        #self.add_param('turbineY', val=np.zeros(nTurbines),
                       #desc='y coordinates of turbines in wind dir. ref. frame')

        #self.add_param('hubHeight', val=np.zeros(nTurbines), units='m')

        #self.add_param('rotorDiameter', val=np.zeros(nTurbines), units='m')


        ## define output
        #self.add_output('LRC', val=0.0, units='$', desc='Levelized Replacement Cost')

    #def solve_nonlinear(self, params, unknowns, resids):


        #turbineX = params['turbineX']
        #turbineY = params['turbineY']
        #nTurbines = turbineX.size

        ##calculate LLC
        #unknowns['LRC'] = 10000.0
#'''



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import os

    AmaliaLocationsAndHull = loadmat(os.path.join('..','..','doc','examples','input_files','Amalia_locAndHull.mat'))
    print(AmaliaLocationsAndHull.keys())
    turbineX = AmaliaLocationsAndHull['turbineX'].flatten()
    turbineY = AmaliaLocationsAndHull['turbineY'].flatten()

    print(turbineX.size)

    nTurbines = len(turbineX)
    locations = np.zeros([nTurbines, 2])
    for i in range(0, nTurbines):
        locations[i] = np.array([turbineX[i], turbineY[i]])

    # get boundary information
    vertices, unit_normals = calculate_boundary(locations)

    print(vertices, unit_normals)

    # define point of interest
    resolution = 100
    x = np.linspace(min(turbineX), max(turbineX), resolution)
    y = np.linspace(min(turbineY), max(turbineY), resolution)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    nPoints = len(xx)
    p = np.zeros([nPoints, 2])
    for i in range(0, nPoints):
        p[i] = np.array([xx[i], yy[i]])

    # calculate distance from each point to each face
    face_distance, inside = calculate_distance(p, vertices, unit_normals, return_bool=True)

    print(inside.shape)
    # reshape arrays for plotting
    xx = np.reshape(xx, (resolution, resolution))
    yy = np.reshape(yy, (resolution, resolution))
    inside = np.reshape(inside, (resolution, resolution))

    # plot points colored based on inside/outside of hull
    plt.figure()
    plt.pcolor(xx, yy, inside)
    plt.plot(turbineX, turbineY, 'ow')
    plt.show()

    import time

    def spacing_2loops(x, y):
        n = x.size
        separation_squared = np.zeros(int((n - 1) * n / 2))
        k = 0
        for i in range(0, n):
            for j in range(i + 1, n):
                separation_squared[k] = (x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2
                k += 1


        return separation_squared

    def spacing_1loop(x, y):
        n = x.size
        separation_squared = np.zeros(int((n - 1) * n / 2))
        indx_start = 0
        indx_end = n-1
        for i in range(0, n):
                # print i, indx_start, indx_end
                separation_squared[indx_start:indx_end] = (x[i+1:] - x[i]) ** 2 + (y[i+1:] - y[i]) ** 2
                # print separation_squared
                indx_start = np.copy(indx_end)
                indx_end = indx_start + n-(i+2)

        return separation_squared


    # def spacing_min(x, y):
    #     n = x.size
    #     min_separation_squared = np.zeros(n)
    #     for i in np.arange()

    x = np.arange(0, 10)
    y = np.arange(0, 10)

    tic = time.time()
    s2 = spacing_2loops(x, y)
    toc = time.time()
    print 'spacing for two loops:', s2
    print 'time for two loops:', toc-tic

    tic = time.time()
    s1 = spacing_1loop(x, y)
    toc = time.time()
    print 'spacing for one loops:', s1
    print 'time for one loops:', toc - tic

    print 'diff: ', s2-s1

    # import matplotlib.pyplot as plt
    # import os
    #
    # AmaliaLocationsAndHull = loadmat(os.path.join('..','..','doc','examples','input_files','Amalia_locAndHull.mat'))
    # print(AmaliaLocationsAndHull.keys())
    # turbineX = AmaliaLocationsAndHull['turbineX'].flatten()
    # turbineY = AmaliaLocationsAndHull['turbineY'].flatten()
    #
    # print(turbineX.size)
    #
    # nTurbines = len(turbineX)
    # locations = np.zeros([nTurbines, 2])
    # for i in range(0, nTurbines):
    #     locations[i] = np.array([turbineX[i], turbineY[i]])
    #
    # # get boundary information
    # vertices, unit_normals = calculate_boundary(locations)
    #
    # print(vertices, unit_normals)
    #
    # # define point of interest
    # resolution = 100
    # x = np.linspace(min(turbineX), max(turbineX), resolution)
    # y = np.linspace(min(turbineY), max(turbineY), resolution)
    # xx, yy = np.meshgrid(x, y)
    # xx = xx.flatten()
    # yy = yy.flatten()
    # nPoints = len(xx)
    # p = np.zeros([nPoints, 2])
    # for i in range(0, nPoints):
    #     p[i] = np.array([xx[i], yy[i]])
    #
    # # calculate distance from each point to each face
    # face_distance, inside = calculate_distance(p, vertices, unit_normals, return_bool=True)
    #
    # print(inside.shape)
    # # reshape arrays for plotting
    # xx = np.reshape(xx, (resolution, resolution))
    # yy = np.reshape(yy, (resolution, resolution))
    # inside = np.reshape(inside, (resolution, resolution))
    #
    # # plot points colored based on inside/outside of hull
    # plt.figure()
    # plt.pcolor(xx, yy, inside)
    # plt.plot(turbineX, turbineY, 'ow')
    # plt.show()

    # top = Problem()
    #
    # root = top.root = Group()
    #
    # root.add('p1', IndepVarComp('x', np.array([1.0, 1.0])))
    # root.add('p2', IndepVarComp('y', np.array([0.75, 0.25])))
    # root.add('p', WindFarmAEP(nDirections=2))
    #
    # root.connect('p1.x', 'p.power_directions')
    # root.connect('p2.y', 'p.windrose_frequencies')
    #
    # top.setup()
    # top.run()
    #
    # # should return 8760.0
    # print(root.p.unknowns['AEP'])
    # top.check_partial_derivatives()

    # top = Problem()
    #
    # root = top.root = Group()
    #
    # root.add('p1', IndepVarComp('x', 1.0))
    # root.add('p2', IndepVarComp('y', 2.0))
    # root.add('p', MUX(nElements=2))
    #
    # root.connect('p1.x', 'p.input0')
    # root.connect('p2.y', 'p.input1')
    #
    # top.setup()
    # top.run()
    #
    # # should return 8760.0
    # print(root.p.unknowns['Array'])
    # top.check_partial_derivatives()

    # top = Problem()
    #
    # root = top.root = Group()
    #
    # root.add('p1', IndepVarComp('x', np.zeros(2)))
    # root.add('p', DeMUX(nElements=2))
    #
    # root.connect('p1.x', 'p.Array')
    #
    # top.setup()
    # top.run()
    #
    # # should return 8760.0
    # print(root.p.unknowns['output0'])
    # print(root.p.unknowns['output1'])
    # top.check_partial_derivatives()

    # top = Problem()
    #
    # root = top.root = Group()
    #
    # root.add('p1', IndepVarComp('x', np.array([0, 3])))
    # root.add('p2', IndepVarComp('y', np.array([1, 0])))
    # root.add('p', SpacingComp(nTurbines=2))
    #
    # root.connect('p1.x', 'p.turbineX')
    # root.connect('p2.y', 'p.turbineY')
    #
    # top.setup()
    # top.run()
    #
    # # print(root.p.unknowns['output0'])
    # # print(root.p.unknowns['output1'])
    # top.check_partial_derivatives()

