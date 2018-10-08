from openmdao.api import Component, Group, Problem, IndepVarComp
from akima import Akima, akima_interp
from plantenergy.utilities import hermite_spline
import plantenergy.config as config

import numpy as np
from scipy import interp
from scipy.io import loadmat
from scipy.spatial import ConvexHull
from scipy.interpolate import UnivariateSpline

import matplotlib.pylab as plt

def add_gen_params_IdepVarComps(openmdao_group, datasize):
    openmdao_group.add('gp0', IndepVarComp('gen_params:pP', 1.88, pass_by_obj=True), promotes=['*'])
    openmdao_group.add('gp1', IndepVarComp('gen_params:windSpeedToCPCT_wind_speed', np.zeros(datasize), units='m/s',
                                  desc='range of wind speeds', pass_by_obj=True), promotes=['*'])
    openmdao_group.add('gp2', IndepVarComp('gen_params:windSpeedToCPCT_CP', np.zeros(datasize),
                                  desc='power coefficients', pass_by_obj=True), promotes=['*'])
    openmdao_group.add('gp3', IndepVarComp('gen_params:windSpeedToCPCT_CT', np.zeros(datasize),
                                  desc='thrust coefficients', pass_by_obj=True), promotes=['*'])
    openmdao_group.add('gp4', IndepVarComp('gen_params:CPcorrected', False,
                                  pass_by_obj=True), promotes=['*'])
    openmdao_group.add('gp5', IndepVarComp('gen_params:CTcorrected', False,
                                  pass_by_obj=True), promotes=['*'])
    openmdao_group.add('gp6', IndepVarComp('gen_params:AEP_method', 'none',
                                           pass_by_obj=True), promotes=['*'])


class WindFrame(Component):
    """ Calculates the locations of each turbine in the wind direction reference frame """

    def __init__(self, nTurbines, resolution=0, differentiable=True, nSamples=0):

        # print('entering windframe __init__ - analytic')

        super(WindFrame, self).__init__()

        # set finite difference options (fd used for testing only)
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['check_step_size'] = 1.0e-5
        self.deriv_options['check_step_calc'] = 'relative'

        if not differentiable:
            self.deriv_options['type'] = 'fd'
            self.deriv_options['form'] = 'forward'

        self.nTurbines = nTurbines
        self.nSamples = nSamples

        # flow property variables
        self.add_param('wind_speed', val=8.0, units='m/s', desc='free stream wind velocity')
        self.add_param('wind_direction', val=270.0, units='deg',
                       desc='wind direction using direction from, in deg. cw from north as in meteorological data')

        # Explicitly size input arrays
        self.add_param('turbineX', val=np.zeros(nTurbines), units='m', desc='x positions of turbines in original ref. frame')
        self.add_param('turbineY', val=np.zeros(nTurbines), units='m', desc='y positions of turbines in original ref. frame')

        # add output
        self.add_output('turbineXw', val=np.zeros(nTurbines), units='m', desc='downwind coordinates of turbines')
        self.add_output('turbineYw', val=np.zeros(nTurbines), units='m', desc='crosswind coordinates of turbines')

        # ############################ visualization arrays ##################################
        if nSamples > 0:
            # visualization input
            self.add_param('wsPositionX', np.zeros(nSamples), units='m', pass_by_object=True,
                           desc='X position of desired measurements in original ref. frame')
            self.add_param('wsPositionY', np.zeros(nSamples), units='m', pass_by_object=True,
                           desc='Y position of desired measurements in original ref. frame')
            self.add_param('wPositionZ', np.zeros(nSamples), units='m', pass_by_object=True,
                           desc='Z position of desired measurements in original ref. frame')

            # visualization output
            self.add_output('wsPositionXw', np.zeros(nSamples), units='m', pass_by_object=True,
                            desc='position of desired measurements in wind ref. frame')
            self.add_output('wsPositionYw', np.zeros(nSamples), units='m', pass_by_object=True,
                            desc='position of desired measurements in wind ref. frame')

    def solve_nonlinear(self, params, unknowns, resids):

        windDirectionDeg = params['wind_direction']

        # get turbine positions and velocity sampling positions
        turbineX = params['turbineX']
        turbineY = params['turbineY']

        if self.nSamples > 0:
            velX = params['wsPositionX']
            velY = params['wsPositionY']

        # adjust directions
        windDirectionDeg = 270. - windDirectionDeg
        # windDirectionDeg = 90. - windDirectionDeg # how this was done in SusTech conference paper (oops!)
        if windDirectionDeg < 0.:
            windDirectionDeg += 360.
        windDirectionRad = np.pi*windDirectionDeg/180.0    # inflow wind direction in radians

        # convert to downwind(x)-crosswind(y) coordinates
        unknowns['turbineXw'] = turbineX*np.cos(-windDirectionRad)-turbineY*np.sin(-windDirectionRad)
        unknowns['turbineYw'] = turbineX*np.sin(-windDirectionRad)+turbineY*np.cos(-windDirectionRad)

        if self.nSamples > 0:
            unknowns['wsPositionXw'] = velX*np.cos(-windDirectionRad)-velY*np.sin(-windDirectionRad)
            unknowns['wsPositionYw'] = velX*np.sin(-windDirectionRad)+velY*np.cos(-windDirectionRad)

    def linearize(self, params, unknowns, resids):

        # obtain necessary inputs
        nTurbines = self.nTurbines
        windDirectionDeg = params['wind_direction']

        # convert from meteorological polar system (CW, 0 deg.=N) to standard polar system (CCW, 0 deg.=E)
        windDirectionDeg = 270. - windDirectionDeg
        if windDirectionDeg < 0.:
            windDirectionDeg += 360.

        # convert inflow wind direction to radians
        windDirectionRad = np.pi*windDirectionDeg/180.0

        # calculate gradients of conversion to wind direction reference frame
        dturbineXw_dturbineX = np.eye(nTurbines, nTurbines)*np.cos(-windDirectionRad)
        dturbineXw_dturbineY = np.eye(nTurbines, nTurbines)*(-np.sin(-windDirectionRad))
        dturbineYw_dturbineX = np.eye(nTurbines, nTurbines)*np.sin(-windDirectionRad)
        dturbineYw_dturbineY = np.eye(nTurbines, nTurbines)*np.cos(-windDirectionRad)

        # initialize Jacobian dict
        J = {}

        # populate Jacobian dict
        J[('turbineXw', 'turbineX')] = dturbineXw_dturbineX
        J[('turbineXw', 'turbineY')] = dturbineXw_dturbineY
        J[('turbineYw', 'turbineX')] = dturbineYw_dturbineX
        J[('turbineYw', 'turbineY')] = dturbineYw_dturbineY

        return J


class AdjustCtCpYaw(Component):
    """ Adjust Cp and Ct to yaw if they are not already adjusted """

    def __init__(self, nTurbines, direction_id=0, differentiable=True):

        # print('entering adjustCtCp __init__ - analytic')
        super(AdjustCtCpYaw, self).__init__()

        self. direction_id = direction_id

        # set finite difference options (fd used for testing only)
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['check_step_size'] = 1.0e-5
        self.deriv_options['check_step_calc'] = 'relative'

        if not differentiable:
            self.deriv_options['type'] = 'fd'
            self.deriv_options['form'] = 'forward'

        # Explicitly size input arrays
        self.add_param('Ct_in', val=np.zeros(nTurbines), desc='Thrust coefficient for all turbines')
        self.add_param('Cp_in', val=np.zeros(nTurbines)+(0.7737/0.944) * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2),
                       desc='power coefficient for all turbines')
        self.add_param('yaw%i' % direction_id, val=np.zeros(nTurbines), units='deg', desc='yaw of each turbine')

        # Explicitly size output arrays
        self.add_output('Ct_out', val=np.zeros(nTurbines), desc='Thrust coefficient for all turbines')
        self.add_output('Cp_out', val=np.zeros(nTurbines), desc='power coefficient for all turbines')

        # parameters since var trees are not supports
        self.add_param('gen_params:pP', 1.88, pass_by_obj=True)
        self.add_param('gen_params:CTcorrected', False,
                       desc='CT factor already corrected by CCBlade calculation (approximately factor cos(yaw)^2)', pass_by_obj=True)
        self.add_param('gen_params:CPcorrected', False,
                       desc='CP factor already corrected by CCBlade calculation (assumed with approximately factor cos(yaw)^3)', pass_by_obj=True)
        # self.add_param('floris_params:FLORISoriginal', True,
        #                desc='override all parameters and use FLORIS as original in first Wind Energy paper', pass_by_obj=True)

    def solve_nonlinear(self, params, unknowns, resids):

        direction_id = self.direction_id

        # print('entering adjustCtCP - analytic')

        # collect inputs
        Ct = params['Ct_in']
        Cp = params['Cp_in']
        yaw = params['yaw%i' % direction_id] * np.pi / 180.
        # print('in Ct correction, Ct_in: '.format(Ct))

        pP = params['gen_params:pP']

        CTcorrected = params['gen_params:CTcorrected']
        CPcorrected = params['gen_params:CPcorrected']

        # calculate new CT values, if desired
        if not CTcorrected:
            # print("ct not corrected")
            unknowns['Ct_out'] = np.cos(yaw)*np.cos(yaw)*Ct
            # print('in ct correction Ct_out: '.format(unknowns['Ct_out']))
        else:
            unknowns['Ct_out'] = Ct

        # calculate new CP values, if desired
        if not CPcorrected:
            unknowns['Cp_out'] = Cp * np.cos(yaw) ** pP
        else:
            unknowns['Cp_out'] = Cp

    def linearize(self, params, unknowns, resids):

        direction_id = self.direction_id

        # collect inputs
        Ct = params['Ct_in']
        Cp = params['Cp_in']
        nTurbines = np.size(Ct)
        yaw = params['yaw%i' % direction_id] * np.pi / 180.

        pP = params['gen_params:pP']

        CTcorrected = params['gen_params:CTcorrected']
        CPcorrected = params['gen_params:CPcorrected']

        # initialize Jacobian dict
        J = {}

        # calculate gradients and populate Jacobian dict
        if not CTcorrected:
            J[('Ct_out', 'Ct_in')] = np.eye(nTurbines) * np.cos(yaw) * np.cos(yaw)
            J[('Ct_out', 'Cp_in')] = np.zeros((nTurbines, nTurbines))
            J[('Ct_out', 'yaw%i' % direction_id)] = np.eye(nTurbines) * Ct * (
                -2. * np.sin(yaw) * np.cos(yaw)) * np.pi / 180.
        else:
            J[('Ct_out', 'Ct_in')] = np.eye(nTurbines, nTurbines)
            J[('Ct_out', 'Cp_in')] = np.zeros((nTurbines, nTurbines))
            J[('Ct_out', 'yaw%i' % direction_id)] = np.zeros((nTurbines, nTurbines))

        if not CPcorrected:
            J[('Cp_out', 'Cp_in')] = np.eye(nTurbines, nTurbines) * np.cos(yaw) ** pP
            J[('Cp_out', 'Ct_in')] = np.zeros((nTurbines, nTurbines))
            J[('Cp_out', 'yaw%i' % direction_id)] = np.eye(nTurbines, nTurbines) * (
                -Cp * pP * np.sin(yaw) * np.cos(yaw) ** (pP - 1.0)) * np.pi / 180.
        else:
            J[('Cp_out', 'Cp_in')] = np.eye(nTurbines, nTurbines)
            J[('Cp_out', 'Ct_in')] = np.zeros((nTurbines, nTurbines))
            J[('Cp_out', 'yaw%i' % direction_id)] = np.zeros((nTurbines, nTurbines))

        return J


class WindFarmAEP(Component):
    """ Estimate the AEP based on power production for each direction and weighted by wind direction frequency  """

    def __init__(self, nDirections, rec_func_calls=False):

        super(WindFarmAEP, self).__init__()

        # set finite difference options (fd used for testing only)
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['check_step_size'] = 1.0e-6
        # self.deriv_options['check_step_calc'] = 'relative'

        # define inputs
        self.add_param('dirPowers', np.zeros(nDirections), units='kW',
                       desc='vector containing the power production at each wind direction ccw from north')
        self.add_param('windFrequencies', np.zeros(nDirections),
                       desc='vector containing the weighted frequency of wind at each direction ccw from east using '
                            'direction too')
        self.add_param('gen_params:AEP_method', val='none', pass_by_object=True,
                       desc='select method with which aep is adjusted for optimization')

        # define output
        self.add_output('AEP', val=0.0, units='kWh', desc='total annual energy output of wind farm')

        # pass bool for function call recording
        self.rec_func_calls = rec_func_calls

    def solve_nonlinear(self, params, unknowns, resids):

        # locally name input values
        dirPowers = params['dirPowers']
        windFrequencies = params['windFrequencies']
        AEP_method = params['gen_params:AEP_method']

        # number of hours in a year
        hours = 8760.0

        # calculate approximate AEP
        AEP = sum(dirPowers*windFrequencies)*hours

        # promote AEP result to class attribute
        if AEP_method == 'none':
            unknowns['AEP'] = AEP
        elif AEP_method =='log':
            unknowns['AEP'] = np.log(AEP)
        elif AEP_method == 'inverse':
            unknowns['AEP'] = (AEP)**(-1)
        else:
            raise ValueError('AEP_method must be one of ["none","log","inverse"]')
        # print(AEP)

        # increase objective function call count

        if self.rec_func_calls:
            comm = self.comm
            rank = comm.rank
            config.obj_func_calls_array[rank] += 1

    def linearize(self, params, unknowns, resids):

        # # print('entering AEP - provideJ')
        AEP_method = params['gen_params:AEP_method']

        # assign params to local variables
        windFrequencies = params['windFrequencies']
        dirPowers = params['dirPowers']
        nDirs = np.size(windFrequencies)

        # number of hours in a year
        hours = 8760.0

        # calculate the derivative of outputs w.r.t. the power in each wind direction
        if AEP_method == 'none':
            dAEP_dpower = np.ones(nDirs) * windFrequencies * hours
        elif AEP_method =='log':
            dAEP_dpower = dirPowers**(-1.)
        elif AEP_method == 'inverse':
            dAEP_dpower = -(hours*windFrequencies*dirPowers**2)**(-1)
        else:
            raise ValueError('AEP_method must be one of ["none","log","inverse"]')


        # initialize Jacobian dict
        J = {}

        # populate Jacobian dict
        J['AEP', 'dirPowers'] = np.array([dAEP_dpower])

        # increase gradient function call count
        if self.rec_func_calls:
            comm = self.comm
            rank = comm.rank
            config.sens_func_calls_array[rank] += 1
            # print(np.sum(config.sens_func_calls_array))
        # print "in linearize AEP"
        return J


class WindDirectionPower(Component):

    def __init__(self, nTurbines, direction_id=0, differentiable=True, use_rotor_components=False, cp_points=1.,
                 cp_curve_spline=None):

        super(WindDirectionPower, self).__init__()

        # define class attributes
        self.differentiable = differentiable
        self.nTurbines = nTurbines
        self.direction_id = direction_id
        self.use_rotor_components = use_rotor_components
        self.cp_points = cp_points
        self.cp_curve_spline = cp_curve_spline

        # set finite difference options (only used for testing)
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['check_step_size'] = 1.0e-6
        self.deriv_options['check_step_calc'] = 'relative'

        if not differentiable:
            self.deriv_options['type'] = 'fd'
            self.deriv_options['form'] = 'forward'

        self.add_param('air_density', 1.1716, units='kg/(m*m*m)', desc='air density in free stream')
        self.add_param('rotorDiameter', np.zeros(nTurbines) + 126.4, units='m', desc='rotor diameters of all turbine')
        self.add_param('Cp', np.zeros(nTurbines)+(0.7737/0.944) * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2), desc='power coefficient for all turbines')
        self.add_param('generatorEfficiency', np.zeros(nTurbines)+0.944, desc='generator efficiency of all turbines')
        self.add_param('wtVelocity%i' % direction_id, np.zeros(nTurbines), units='m/s',
                       desc='effective hub velocity for each turbine')

        self.add_param('rated_power', np.ones(nTurbines)*5000., units='kW',
                       desc='rated power for each turbine', pass_by_obj=True)
        self.add_param('cut_in_speed', np.ones(nTurbines) * 3.0, units='m/s',
                       desc='cut-in speed for each turbine', pass_by_obj=True)
        self.add_param('cp_curve_cp', np.zeros(cp_points),
                       desc='cp as a function of wind speed', pass_by_obj=True)
        self.add_param('cp_curve_vel', np.ones(cp_points), units='m/s',
                       desc='vel corresponding to cp curve points', pass_by_obj=True)
        # self.add_param('cp_curve_spline', None, units='m/s',
        #                desc='spline corresponding to cp curve', pass_by_obj=True)

        # outputs
        self.add_output('wtPower%i' % direction_id, np.zeros(nTurbines), units='kW', desc='power output of each turbine')
        self.add_output('dir_power%i' % direction_id, 0.0, units='kW', desc='total power output of the wind farm')

    def solve_nonlinear(self, params, unknowns, resids):

        # obtain necessary inputs
        use_rotor_components = self.use_rotor_components
        direction_id = self.direction_id
        nTurbines = self.nTurbines
        wtVelocity = self.params['wtVelocity%i' % direction_id]
        rated_power = params['rated_power']
        cut_in_speed = params['cut_in_speed']
        air_density = params['air_density']
        rotorArea = 0.25*np.pi*np.power(params['rotorDiameter'], 2)
        Cp = params['Cp']
        generatorEfficiency = params['generatorEfficiency']

        cp_curve_cp = params['cp_curve_cp']
        cp_curve_vel = params['cp_curve_vel']
        # cp_curve_spline = params['cp_curve_spline']
        cp_curve_spline = self.cp_curve_spline

        if self.cp_points > 1.:
            # print('entered Cp')
            if cp_curve_spline is None:
                for i in np.arange(0, nTurbines):
                    Cp[i] = np.interp(wtVelocity[i], cp_curve_vel, cp_curve_cp)
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
        dir_power = np.sum(wtPower)

        # pass out results
        unknowns['wtPower%i' % direction_id] = wtPower
        unknowns['dir_power%i' % direction_id] = dir_power

        # print(wtPower)

    def linearize(self, params, unknowns, resids):

        # obtain necessary inputs
        direction_id = self.direction_id
        use_rotor_components = self.use_rotor_components
        nTurbines = self.nTurbines
        wtVelocity = self.params['wtVelocity%i' % direction_id]
        air_density = params['air_density']
        rotorDiameter = params['rotorDiameter']
        rotorArea = 0.25*np.pi*np.power(rotorDiameter, 2)
        Cp = params['Cp']
        generatorEfficiency = params['generatorEfficiency']
        rated_power = params['rated_power']
        cut_in_speed = params['cut_in_speed']
        wtPower = unknowns['wtPower%i' % direction_id]

        cp_curve_cp = params['cp_curve_cp']
        cp_curve_vel = params['cp_curve_vel']

        cp_curve_spline = self.cp_curve_spline

        dCpdV = np.zeros_like(Cp)

        if self.cp_points > 1. and self.cp_curve_spline is None:

            for i in np.arange(0, nTurbines):
                Cp[i] = np.interp(wtVelocity[i], cp_curve_vel, cp_curve_cp)
                # Cp[i] = spl(wtVelocity[i])
                dv = 1E-6
                dCpdV[i] = (np.interp(wtVelocity[i]+dv, cp_curve_vel, cp_curve_cp) -
                         np.interp(wtVelocity[i]- dv, cp_curve_vel, cp_curve_cp))/(2.*dv)

        elif self.cp_curve_spline is not None:
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

        # initialize Jacobian dict
        J = {}

        # populate Jacobian dict
        J['wtPower%i' % direction_id, 'wtVelocity%i' % direction_id] = dwtPower_dwtVelocity
        J['wtPower%i' % direction_id, 'Cp'] = dwtPower_dCp
        J['wtPower%i' % direction_id, 'rotorDiameter'] = dwtPower_drotorDiameter

        J['dir_power%i' % direction_id, 'wtVelocity%i' % direction_id] = ddir_power_dwtVelocity
        J['dir_power%i' % direction_id, 'Cp'] = ddir_power_dCp
        J['dir_power%i' % direction_id, 'rotorDiameter'] = ddir_power_drotorDiameter

        return J


class SpacingComp(Component):
    """
    Calculates inter-turbine spacing for all turbine pairs
    """

    def __init__(self, nTurbines):

        super(SpacingComp, self).__init__()

        # set finite difference options (fd used for testing only)
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['check_step_size'] = 1.0e-5
        self.deriv_options['check_step_calc'] = 'relative'

        # Explicitly size input arrays
        self.add_param('turbineX', val=np.zeros(nTurbines),
                       desc='x coordinates of turbines in wind dir. ref. frame')
        self.add_param('turbineY', val=np.zeros(nTurbines),
                       desc='y coordinates of turbines in wind dir. ref. frame')

        # Explicitly size output array
        self.add_output('wtSeparationSquared', val=np.zeros(int((nTurbines-1)*nTurbines/2)),
                        desc='spacing of all turbines in the wind farm')

    def solve_nonlinear(self, params, unknowns, resids):
        # print('in dist const')

        turbineX = params['turbineX']
        turbineY = params['turbineY']
        nTurbines = turbineX.size
        separation_squared = np.zeros(int((nTurbines-1)*nTurbines/2))

        k = 0
        for i in range(0, nTurbines):
            for j in range(i+1, nTurbines):
                separation_squared[k] = (turbineX[j]-turbineX[i])**2+(turbineY[j]-turbineY[i])**2
                k += 1
        unknowns['wtSeparationSquared'] = separation_squared

    def linearize(self, params, unknowns, resids):

        # obtain necessary inputs
        turbineX = params['turbineX']
        turbineY = params['turbineY']

        # get number of turbines
        nTurbines = turbineX.size

        # initialize gradient calculation array
        dS = np.zeros((int((nTurbines-1.)*nTurbines/2.), 2*nTurbines))

        # set turbine pair counter to zero
        k = 0

        # calculate the gradient of the distance between each pair of turbines w.r.t. turbineX and turbineY
        for i in range(0, nTurbines):
            for j in range(i+1, nTurbines):
                # separation wrt Xj
                dS[k, j] = 2*(turbineX[j]-turbineX[i])
                # separation wrt Xi
                dS[k, i] = -2*(turbineX[j]-turbineX[i])
                # separation wrt Yj
                dS[k, j+nTurbines] = 2*(turbineY[j]-turbineY[i])
                # separation wrt Yi
                dS[k, i+nTurbines] = -2*(turbineY[j]-turbineY[i])
                # increment turbine pair counter
                k += 1

        # initialize Jacobian dict
        J = {}

        # populate Jacobian dict
        J['wtSeparationSquared', 'turbineX'] = dS[:, :nTurbines]
        J['wtSeparationSquared', 'turbineY'] = dS[:, nTurbines:]

        return J


class BoundaryComp(Component):

    def __init__(self, nTurbines, nVertices):

        super(BoundaryComp, self).__init__()

        self.nTurbines = nTurbines
        self.nVertices = nVertices
        if nVertices > 1:
            self.type = type = 'polygon'
        elif nVertices == 1:
            self.type = type = 'circle'
        else:
            ValueError('nVertices in BoundaryComp must be greater than 0')

        if type == 'polygon':
            #     Explicitly size input arrays
            self.add_param('boundaryVertices', np.zeros([nVertices, 2]), units='m', pass_by_obj=True,
                           desc="vertices of the convex hull CCW in order s.t. boundaryVertices[i] -> first point of face"
                                "for unit_normals[i]")
            self.add_param('boundaryNormals', np.zeros([nVertices, 2]), pass_by_obj=True,
                           desc="unit normal vector for each boundary face CCW where boundaryVertices[i] is "
                                "the first point of the corresponding face")
        elif type == 'circle':
            self.add_param('boundary_radius', val=1000., units='m', pass_by_obj=True, desc='radius of wind farm boundary')
            self.add_param('boundary_center', val=np.array([0., 0.]), units='m', pass_by_obj=True,
                           desc='x and y positions of circular wind farm boundary center')
        else:
            ValueError('Invalid value (%s) encountered in BoundaryComp input -type-. Must be one of [polygon, circle]'
                        %(type))

        self.add_param('turbineX', np.zeros(nTurbines), units='m',
                       desc='x coordinates of turbines in global ref. frame')
        self.add_param('turbineY', np.zeros(nTurbines), units='m',
                       desc='y coordinates of turbines in global ref. frame')

        # Explicitly size output array
        # (vector with positive elements if turbines outside of hull)
        self.add_output('boundaryDistances', np.zeros([nTurbines, nVertices]),
                        desc="signed perpendicular distance from each turbine to each face CCW; + is inside")

    def solve_nonlinear(self, params, unknowns, resids):

        turbineX = params['turbineX']
        turbineY = params['turbineY']

        if self.type == 'polygon':
            # put locations in correct arrangement for calculations
            locations = np.zeros([self.nTurbines, 2])
            for i in range(0, self.nTurbines):
                locations[i] = np.array([turbineX[i], turbineY[i]])

            # print("in comp, locs are: ".format(locations))

            # calculate distance from each point to each face
            unknowns['boundaryDistances'] = calculate_distance(locations,
                                                               params['boundaryVertices'], params['boundaryNormals'])

        elif self.type == 'circle':
            xc = params['boundary_center'][0]
            yc = params['boundary_center'][1]
            r = params['boundary_radius']
            unknowns['boundaryDistances'] = r**2 - (np.power((turbineX - xc), 2) + np.power((turbineY - yc), 2))

        else:
            ValueError('Invalid value (%s) encountered in BoundaryComp input -type-. Must be one of [polygon, circle]'
                        %(type))

    def linearize(self, params, unknowns, resids):

        if self.type == 'polygon':
            unit_normals = params['boundaryNormals']

            # initialize array to hold distances from each point to each face
            dfaceDistance_dx = np.zeros([int(self.nTurbines*self.nVertices), self.nTurbines])
            dfaceDistance_dy = np.zeros([int(self.nTurbines*self.nVertices), self.nTurbines])

            for i in range(0, self.nTurbines):
                # determine if point is inside or outside of each face, and distance from each face
                for j in range(0, self.nVertices):

                    # define the derivative vectors from the point of interest to the first point of the face
                    dpa_dx = np.array([-1.0, 0.0])
                    dpa_dy = np.array([0.0, -1.0])

                    # find perpendicular distance derivatives from point to current surface (vector projection)
                    ddistanceVec_dx = np.vdot(dpa_dx, unit_normals[j])*unit_normals[j]
                    ddistanceVec_dy = np.vdot(dpa_dy, unit_normals[j])*unit_normals[j]

                    # calculate derivatives for the sign of perpendicular distance from point to current face
                    dfaceDistance_dx[i*self.nVertices+j, i] = np.vdot(ddistanceVec_dx, unit_normals[j])
                    dfaceDistance_dy[i*self.nVertices+j, i] = np.vdot(ddistanceVec_dy, unit_normals[j])

        elif self.type == 'circle':
            turbineX = params['turbineX']
            turbineY = params['turbineY']
            xc = params['boundary_center'][0]
            yc = params['boundary_center'][1]

            A = np.eye(self.nTurbines, self.nTurbines)
            B =  - 2. * (turbineX - xc)
            C =  - 2. * (turbineY - yc)

            dfaceDistance_dx = A*B
            dfaceDistance_dy = A*C
        else:
            ValueError('Invalid value (%s) encountered in BoundaryComp input -type-. Must be one of [polygon, circle]'
                       % (type))


        # initialize Jacobian dict
        J = {}

        # return Jacobian dict
        J['boundaryDistances', 'turbineX'] = dfaceDistance_dx
        J['boundaryDistances', 'turbineY'] = dfaceDistance_dy

        return J


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


def calculate_distance(points, vertices, unit_normals, return_bool=False):

    """
    :param points: points that you want to calculate the distance from to the faces of the convex hull
    :param vertices: vertices of the convex hull CCW in order s.t. vertices[i] -> first point of face for
           unit_normals[i]
    :param unit_normals: unit normal vector for each face CCW where vertices[i] is first point of face
    :param return_bool: set to True to return an array of bools where True means the corresponding point
           is inside the hull
    :return face_distace: signed perpendicular distance from each point to each face; + is inside
    :return [inside]: (optional) an array of zeros and ones where 1.0 means the corresponding point is inside the hull
    """

    # print points.shape, vertices.shape, unit_normals.shape

    nPoints = points.shape[0]
    nVertices = vertices.shape[0]

    # initialize array to hold distances from each point to each face
    face_distance = np.zeros([nPoints, nVertices])

    if not return_bool:
        # loop through points and find distance to each face
        for i in range(0, nPoints):

            # determine if point is inside or outside of each face, and distance from each face
            for j in range(0, nVertices):

                # define the vector from the point of interest to the first point of the face
                pa = np.array([vertices[j, 0]-points[i, 0], vertices[j, 1]-points[i, 1]])

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
                pa = np.array([vertices[j, 0]-points[i, 0], vertices[j, 1]-points[i, 1]])

                # find perpendicular distance from point to current surface (vector projection)
                d_vec = np.vdot(pa, unit_normals[j])*unit_normals[j]

                # calculate the sign of perpendicular distance from point to current face (+ is inside, - is outside)
                face_distance[i, j] = np.vdot(d_vec, unit_normals[j])

            # check if the point is inside the convex hull by checking the sign of the distance
            if np.all(face_distance[i] >= 0):
                inside[i] = 1.0

        return face_distance, inside


class MUX(Component):
    """ Connect input elements into a single array  """

    def __init__(self, nElements, units=None):

        super(MUX, self).__init__()

        # set finite difference options (fd used for testing only)
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['check_step_size'] = 1.0e-5
        self.deriv_options['check_step_calc'] = 'relative'

        # define necessary class attributes
        self.nElements = nElements

        # define inputs
        if units is None:
            for i in range(0, nElements):
                self.add_param('input%i' % i, val=0.0, desc='scalar input')
        else:
            for i in range(0, nElements):
                self.add_param('input%i' % i, val=0.0, units=units, desc='scalar input')

        # define output array
        if units is None:
            self.add_output('Array', np.zeros(nElements), desc='ndArray of all the scalar inputs')
        else:
            self.add_output('Array', np.zeros(nElements), units=units, desc='ndArray of all the scalar inputs')

    def solve_nonlinear(self, params, unknowns, resids):

        # assign input values to elements of the output array
        for i in range(0, self.nElements):
            exec("unknowns['Array'][%i] = params['input%i']" % (i, i))

    def linearize(self, params, unknowns, resids):

        # initialize gradient calculation array
        dArray_dInput = np.zeros(self.nElements)

        # initialize Jacobian dict
        J = {}

        # calculate gradient and populate Jacobian dict
        for i in range(0, self.nElements):
            dArray_dInput[i] = 1.0
            J['Array', 'input%i' % i] = np.array(dArray_dInput)
            dArray_dInput[i] = 0.0

        return J


class DeMUX(Component):
    """ split a given array into separate elements """

    def __init__(self, nElements, units=None):

        super(DeMUX, self).__init__()

        # set finite difference options (fd used for testing only)
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['check_step_size'] = 1.0e-5
        self.deriv_options['check_step_calc'] = 'relative'

        # initialize necessary class attributes
        self.nElements = nElements

        # define input
        if units is None:
            self.add_param('Array', np.zeros(nElements), desc='ndArray of scalars')
        else:
            self.add_param('Array', np.zeros(nElements), units=units, desc='ndArray of scalars')

        # define outputs
        if units is None:
            for i in range(0, nElements):
                self.add_output('output%i' % i, val=0.0, desc='scalar output')
        else:
            for i in range(0, nElements):
                self.add_output('output%i' % i, val=0.0, units=units, desc='scalar output')

    def solve_nonlinear(self, params, unknowns, resids):

        # assign elements of the input array to outputs
        for i in range(0, self.nElements):
            exec("unknowns['output%i'] = params['Array'][%i]" % (i, i))

    def linearize(self, params, unknowns, resids):

        # initialize gradient calculation array
        doutput_dArray = np.eye(self.nElements)

        # intialize Jacobian dict
        J = {}

        # calculate the gradients and populate the Jacobian dict
        for i in range(0, self.nElements):
            J['output%i' % i, 'Array'] = np.reshape(doutput_dArray[i, :], (1, self.nElements))

        return J


# ---- if you know wind speed to power and thrust, you can use these tools ----------------
class CPCT_Interpolate_Gradients(Component):

    def __init__(self, nTurbines, direction_id=0, datasize=0):

        super(CPCT_Interpolate_Gradients, self).__init__()

        # set finite difference options (fd used for testing only)
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['check_step_size'] = 1.0e-5
        self.deriv_options['check_step_calc'] = 'relative'

        # define class attributes
        self.nTurbines = nTurbines
        self.direction_id = direction_id
        self.datasize = datasize

        # add inputs and outputs
        self.add_param('yaw%i' % direction_id, np.zeros(nTurbines), desc='yaw error', units='deg')
        self.add_param('wtVelocity%i' % direction_id, np.zeros(nTurbines), units='m/s', desc='hub height wind speed') # Uhub
        self.add_output('Cp_out', np.zeros(nTurbines))
        self.add_output('Ct_out', np.zeros(nTurbines))

        # add variable trees
        self.add_param('gen_params:pP', 1.88, pass_by_obj=True)
        self.add_param('gen_params:windSpeedToCPCT_wind_speed', np.zeros(datasize), units='m/s',
                       desc='range of wind speeds', pass_by_obj=True)
        self.add_param('gen_params:windSpeedToCPCT_CP', np.zeros(datasize), iotype='out',
                       desc='power coefficients', pass_by_obj=True)
        self.add_param('gen_params:windSpeedToCPCT_CT', np.zeros(datasize), iotype='out',
                       desc='thrust coefficients', pass_by_obj=True)

    def solve_nonlinear(self, params, unknowns, resids):

        # obtain necessary inputs
        direction_id = self.direction_id
        pP = self.params['gen_params:pP']

        wind_speed_ax = np.cos(self.params['yaw%i' % direction_id]*np.pi/180.0)**(pP/3.0)*self.params['wtVelocity%i' % direction_id]
        # use interpolation on precalculated CP-CT curve
        wind_speed_ax = np.maximum(wind_speed_ax, self.params['gen_params:windSpeedToCPCT_wind_speed'][0])
        wind_speed_ax = np.minimum(wind_speed_ax, self.params['gen_params:windSpeedToCPCT_wind_speed'][-1])
        self.unknowns['Cp_out'] = interp(wind_speed_ax, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CP'])
        self.unknowns['Ct_out'] = interp(wind_speed_ax, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CT'])

        # for i in range(0, len(self.unknowns['Ct_out'])):
        #     self.unknowns['Ct_out'] = max(max(self.unknowns['Ct_out']), self.unknowns['Ct_out'][i])
        # normalize on incoming wind speed to correct coefficients for yaw
        self.unknowns['Cp_out'] = self.unknowns['Cp_out'] * np.cos(self.params['yaw%i' % direction_id]*np.pi/180.0)**pP
        self.unknowns['Ct_out'] = self.unknowns['Ct_out'] * np.cos(self.params['yaw%i' % direction_id]*np.pi/180.0)**2

    def linearize(self, params, unknowns, resids):  # standard central differencing
        # set step size for finite differencing
        h = 1e-6
        direction_id = self.direction_id

        # calculate upper and lower function values
        wind_speed_ax_high_yaw = np.cos((self.params['yaw%i' % direction_id]+h)*np.pi/180.0)**(self.params['gen_params:pP']/3.0)*self.params['wtVelocity%i' % direction_id]
        wind_speed_ax_low_yaw = np.cos((self.params['yaw%i' % direction_id]-h)*np.pi/180.0)**(self.params['gen_params:pP']/3.0)*self.params['wtVelocity%i' % direction_id]
        wind_speed_ax_high_wind = np.cos(self.params['yaw%i' % direction_id]*np.pi/180.0)**(self.params['gen_params:pP']/3.0)*(self.params['wtVelocity%i' % direction_id]+h)
        wind_speed_ax_low_wind = np.cos(self.params['yaw%i' % direction_id]*np.pi/180.0)**(self.params['gen_params:pP']/3.0)*(self.params['wtVelocity%i' % direction_id]-h)

        # use interpolation on precalculated CP-CT curve
        wind_speed_ax_high_yaw = np.maximum(wind_speed_ax_high_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'][0])
        wind_speed_ax_low_yaw = np.maximum(wind_speed_ax_low_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'][0])
        wind_speed_ax_high_wind = np.maximum(wind_speed_ax_high_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'][0])
        wind_speed_ax_low_wind = np.maximum(wind_speed_ax_low_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'][0])

        wind_speed_ax_high_yaw = np.minimum(wind_speed_ax_high_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'][-1])
        wind_speed_ax_low_yaw = np.minimum(wind_speed_ax_low_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'][-1])
        wind_speed_ax_high_wind = np.minimum(wind_speed_ax_high_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'][-1])
        wind_speed_ax_low_wind = np.minimum(wind_speed_ax_low_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'][-1])

        CP_high_yaw = interp(wind_speed_ax_high_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CP'])
        CP_low_yaw = interp(wind_speed_ax_low_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CP'])
        CP_high_wind = interp(wind_speed_ax_high_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CP'])
        CP_low_wind = interp(wind_speed_ax_low_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CP'])

        CT_high_yaw = interp(wind_speed_ax_high_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CT'])
        CT_low_yaw = interp(wind_speed_ax_low_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CT'])
        CT_high_wind = interp(wind_speed_ax_high_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CT'])
        CT_low_wind = interp(wind_speed_ax_low_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CT'])

        # normalize on incoming wind speed to correct coefficients for yaw
        CP_high_yaw = CP_high_yaw * np.cos((self.params['yaw%i' % direction_id]+h)*np.pi/180.0)**self.params['gen_params:pP']
        CP_low_yaw = CP_low_yaw * np.cos((self.params['yaw%i' % direction_id]-h)*np.pi/180.0)**self.params['gen_params:pP']
        CP_high_wind = CP_high_wind * np.cos((self.params['yaw%i' % direction_id])*np.pi/180.0)**self.params['gen_params:pP']
        CP_low_wind = CP_low_wind * np.cos((self.params['yaw%i' % direction_id])*np.pi/180.0)**self.params['gen_params:pP']

        CT_high_yaw = CT_high_yaw * np.cos((self.params['yaw%i' % direction_id]+h)*np.pi/180.0)**2
        CT_low_yaw = CT_low_yaw * np.cos((self.params['yaw%i' % direction_id]-h)*np.pi/180.0)**2
        CT_high_wind = CT_high_wind * np.cos((self.params['yaw%i' % direction_id])*np.pi/180.0)**2
        CT_low_wind = CT_low_wind * np.cos((self.params['yaw%i' % direction_id])*np.pi/180.0)**2

        # compute derivative via central differencing and arrange in sub-matrices of the Jacobian
        dCP_dyaw = np.eye(self.nTurbines)*(CP_high_yaw-CP_low_yaw)/(2.0*h)
        dCP_dwind = np.eye(self.nTurbines)*(CP_high_wind-CP_low_wind)/(2.0*h)
        dCT_dyaw = np.eye(self.nTurbines)*(CT_high_yaw-CT_low_yaw)/(2.0*h)
        dCT_dwind = np.eye(self.nTurbines)*(CT_high_wind-CT_low_wind)/(2.0*h)

        # compile Jacobian dict from sub-matrices
        J = {}
        J['Cp_out', 'yaw%i' % direction_id] = dCP_dyaw
        J['Cp_out', 'wtVelocity%i' % direction_id] = dCP_dwind
        J['Ct_out', 'yaw%i' % direction_id] = dCT_dyaw
        J['Ct_out', 'wtVelocity%i' % direction_id] = dCT_dwind

        return J


class CPCT_Interpolate_Gradients_Smooth(Component):

    def __init__(self, nTurbines, direction_id=0, datasize=0):

        super(CPCT_Interpolate_Gradients_Smooth, self).__init__()

        # set finite difference options (fd used for testing only)
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['check_step_size'] = 1.0e-6
        self.deriv_options['check_step_calc'] = 'relative'

        # define class attributes
        self.nTurbines = nTurbines
        self.direction_id = direction_id
        self.datasize = datasize

        # add inputs and outputs
        self.add_param('yaw%i' % direction_id, np.zeros(nTurbines), desc='yaw error', units='deg')
        self.add_param('wtVelocity%i' % direction_id, np.zeros(nTurbines), units='m/s', desc='hub height wind speed') # Uhub
        self.add_output('Cp_out', np.zeros(nTurbines))
        self.add_output('Ct_out', np.zeros(nTurbines))

        # add variable trees
        self.add_param('gen_params:pP', 3.0, pass_by_obj=True)
        self.add_param('gen_params:windSpeedToCPCT_wind_speed', np.zeros(datasize), units='m/s',
                       desc='range of wind speeds', pass_by_obj=True)
        self.add_param('gen_params:windSpeedToCPCT_CP', np.zeros(datasize),
                       desc='power coefficients', pass_by_obj=True)
        self.add_param('gen_params:windSpeedToCPCT_CT', np.zeros(datasize),
                       desc='thrust coefficients', pass_by_obj=True)

    def solve_nonlinear(self, params, unknowns, resids):
        direction_id = self.direction_id
        pP = self.params['gen_params:pP']
        yaw = self.params['yaw%i' % direction_id]
        start = 5
        skip = 8
        # Cp = params['gen_params:windSpeedToCPCT_CP'][start::skip]
        Cp = params['gen_params:windSpeedToCPCT_CP']
        # Ct = params['gen_params:windSpeedToCPCT_CT'][start::skip]
        Ct = params['gen_params:windSpeedToCPCT_CT']
        # windspeeds = params['gen_params:windSpeedToCPCT_wind_speed'][start::skip]
        windspeeds = params['gen_params:windSpeedToCPCT_wind_speed']
        #
        # Cp = np.insert(Cp, 0, Cp[0]/2.0)
        # Cp = np.insert(Cp, 0, 0.0)
        # Ct = np.insert(Ct, 0, np.max(params['gen_params:windSpeedToCPCT_CP'])*0.99)
        # Ct = np.insert(Ct, 0, np.max(params['gen_params:windSpeedToCPCT_CT']))
        # windspeeds = np.insert(windspeeds, 0, 2.5)
        # windspeeds = np.insert(windspeeds, 0, 0.0)
        #
        # Cp = np.append(Cp, 0.0)
        # Ct = np.append(Ct, 0.0)
        # windspeeds = np.append(windspeeds, 30.0)

        CPspline = Akima(windspeeds, Cp)
        CTspline = Akima(windspeeds, Ct)

        # n = 500
        # x = np.linspace(0.0, 30., n)
        CP, dCPdvel, _, _ = CPspline.interp(params['wtVelocity%i' % direction_id])
        CT, dCTdvel, _, _ = CTspline.interp(params['wtVelocity%i' % direction_id])

        # print('in solve_nonlinear', dCPdvel, dCTdvel)
        # pP = 3.0
        # print("in rotor, pP = ", pP)
        Cp_out = CP*np.cos(yaw*np.pi/180.)**pP
        Ct_out = CT*np.cos(yaw*np.pi/180.)**2.

        # print("in rotor, Cp = [%f. %f], Ct = [%f, %f]".format(Cp_out[0], Cp_out[1], Ct_out[0], Ct_out[1]))

        self.dCp_out_dyaw = (-np.sin(yaw*np.pi/180.))*(np.pi/180.)*pP*CP*np.cos(yaw*np.pi/180.)**(pP-1.)
        self.dCp_out_dvel = dCPdvel*np.cos(yaw*np.pi/180.)**pP

        # print('in solve_nonlinear', self.dCp_out_dyaw, self.dCp_out_dvel)

        self.dCt_out_dyaw = (-np.sin(yaw*np.pi/180.))*(np.pi/180.)*2.*CT*np.cos(yaw*np.pi/180.)
        self.dCt_out_dvel = dCTdvel*np.cos(yaw*np.pi/180.)**2.

        # normalize on incoming wind speed to correct coefficients for yaw
        self.unknowns['Cp_out'] = Cp_out
        self.unknowns['Ct_out'] = Ct_out

    def linearize(self, params, unknowns, resids):  # standard central differencing

        # obtain necessary inputs
        direction_id = self.direction_id

        # compile Jacobian dict
        J = {}
        J['Cp_out', 'yaw%i' % direction_id] = np.eye(self.nTurbines)*self.dCp_out_dyaw
        J['Cp_out', 'wtVelocity%i' % direction_id] = np.eye(self.nTurbines)*self.dCp_out_dvel
        J['Ct_out', 'yaw%i' % direction_id] = np.eye(self.nTurbines)*self.dCt_out_dyaw
        J['Ct_out', 'wtVelocity%i' % direction_id] = np.eye(self.nTurbines)*self.dCt_out_dvel

        return J



# legacy code for simple COE calculations - should be done more formally
'''
class calcICC(Component):
    """
    Calculates ICC (initial capital cost) for given windfarm layout
    The initial capital cost is the sum of the turbine system cost and the balance of station cost.
    Neither cost includes construction financing or financing fees,
    because these are calculated and added separately through the fixed charge rate.
    The costs also do not include a debt service reserve fund, which is assumed to be zero for balance sheet financing.
    """

    def __init__(self, nTurbines, nTopologyPoints):

        super(calcICC, self).__init__()

        # Add inputs
        self.add_param('turbineX', val=np.zeros(nTurbines),
                       desc='x coordinates of turbines in wind dir. ref. frame')
        self.add_param('turbineY', val=np.zeros(nTurbines),
                       desc='y coordinates of turbines in wind dir. ref. frame')

        self.add_param('hubHeight', val=np.zeros(nTurbines), units='m')

        self.add_param('rotorDiameter', val=np.zeros(nTurbines), units='m')

        self.add_param('topologyX', val=np.zeros(nTopologyPoints),
                       desc = 'x coordiantes of topology')
        self.add_param('topologyY', val=np.zeros(nTopologyPoints),
                       desc = 'y coordiantes of topology')
        self.add_param('topologyZ', val=np.zeros(nTopologyPoints),
                       desc = 'z coordiantes of topology')

        # import topology information

        # define output
        self.add_output('ICC', val=0.0, units='$', desc='Initial Capital Cost')

    def solve_nonlinear(self, params, unknowns, resids):


        turbineX = params['turbineX']
        turbineY = params['turbineY']
        nTurbines = turbineX.size

        topologyX = params['topologyX']
        topologyY = params['topologyY']
        topologyZ = params['topologyZ']

        #calculate ICC
        ICCpartsx = np.zeros([nTurbines,1])
        ICCpartsy = np.zeros([nTurbines,1])

        #need to come up with good way to interpolate between points
        #right now, using linear interpolation
        mx = (topologyZ[2]-topologyZ[0])/(topologyX[2]-topologyX[0])

        my = (topologyZ[2]-topologyZ[0])/(topologyY[2]-topologyY[0])

        for i in range(0, nTurbines):
            ICCpartsx[i] = mx*(turbineX[i]-topologyX[2])+topologyZ[2]
            ICCpartsy[i] = mx*(turbineY[i]-topologyY[2])+topologyZ[2]

        unknowns['ICC'] = sum(ICCpartsx) +  sum(ICCpartsy)

class calcFCR(Component):
    """
    Calculates FCR (fixed charge rate) for given windfarm layout
    """

    def __init__(self, nTurbines):

        super(calcFCR, self).__init__()

        # Add inputs
        self.add_param('turbineX', val=np.zeros(nTurbines),
                       desc='x coordinates of turbines in wind dir. ref. frame')
        self.add_param('turbineY', val=np.zeros(nTurbines),
                       desc='y coordinates of turbines in wind dir. ref. frame')

        # define output
        self.add_output('FCR', val=0.0, desc='Fixed Charge Rate')

    def solve_nonlinear(self, params, unknowns, resids):


        turbineX = params['turbineX']
        turbineY = params['turbineY']
        nTurbines = turbineX.size

        #calculate FCR
        unknowns['FCR'] = 10000.0

class calcLLC(Component):
    """
    Calculates LLC (landlease cost) for given windfarm layout
    Annual operating expenses (AOE) include land or ocean bottom lease cost, levelized O&M cost,
    and levelized replacement/overhaul cost (LRC). Land lease costs (LLC) are the rental or lease fees
    charged for the turbine installation. LLC is expressed in units of $/kWh.
    """

    def __init__(self, nTurbines):

        super(calcLLC, self).__init__()

        # Add inputs
        self.add_param('turbineX', val=np.zeros(nTurbines),
                       desc='x coordinates of turbines in wind dir. ref. frame')
        self.add_param('turbineY', val=np.zeros(nTurbines),
                       desc='y coordinates of turbines in wind dir. ref. frame')

        # define output
        self.add_output('LLC', val=0.0, units='$/kWh', desc='Landlease Cost')

    def solve_nonlinear(self, params, unknowns, resids):


        turbineX = params['turbineX']
        turbineY = params['turbineY']
        nTurbines = turbineX.size

        #calculate LLC
        unknowns['LLC'] = 10000.0

class calcOandM(Component):
    """
    Calculates O&M (levelized operation & maintenance cost) for given windfarm layout
    A component of AOE that is larger than the LLC is O&M cost. O&M is expressed in units of $/kWh.
    The O&M cost normally includes
        - labor, parts, and supplies for scheduled turbine maintenance
        - labor, parts, and supplies for unscheduled turbine maintenance
        - parts and supplies for equipment and facilities maintenance
        - labor for administration and support.
    """

    def __init__(self, nTurbines):

        super(calcOandM, self).__init__()

        # Add inputs
        self.add_param('turbineX', val=np.zeros(nTurbines),
                       desc='x coordinates of turbines in wind dir. ref. frame')
        self.add_param('turbineY', val=np.zeros(nTurbines),
                       desc='y coordinates of turbines in wind dir. ref. frame')

        # define output
        self.add_output('OandM', val=0.0, units='$', desc='levelized O&M cost')

    def solve_nonlinear(self, params, unknowns, resids):


        turbineX = params['turbineX']
        turbineY = params['turbineY']
        nTurbines = turbineX.size

        #calculate LLC

        #need to know area of boundary?

        unknowns['OandM'] = 10000.0

class calcLRC(Component):
    """
    Calculates LRC (levelized replacement/overhaul cost) for given windfarm layout
    LRC distributes the cost of major replacements and overhauls over the life of the wind turbine and is expressed in $/kW machine rating.
    """

    def __init__(self, nTurbines):

        super(calcLRC, self).__init__()

        # Add inputs
        self.add_param('turbineX', val=np.zeros(nTurbines),
                       desc='x coordinates of turbines in wind dir. ref. frame')
        self.add_param('turbineY', val=np.zeros(nTurbines),
                       desc='y coordinates of turbines in wind dir. ref. frame')

        self.add_param('hubHeight', val=np.zeros(nTurbines), units='m')

        self.add_param('rotorDiameter', val=np.zeros(nTurbines), units='m')


        # define output
        self.add_output('LRC', val=0.0, units='$', desc='Levelized Replacement Cost')

    def solve_nonlinear(self, params, unknowns, resids):


        turbineX = params['turbineX']
        turbineY = params['turbineY']
        nTurbines = turbineX.size

        #calculate LLC
        unknowns['LRC'] = 10000.0
'''



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

