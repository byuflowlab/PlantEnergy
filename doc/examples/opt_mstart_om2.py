from __future__ import print_function

import openmdao.api as om
from plantenergy.OptimizationGroups import OptAEP
from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps
from plantenergy import config
# from plantenergy.jensen import jensen_wrapper, add_jensen_params_IdepVarComps
from plantenergy.utilities import sunflower_points
from plantenergy.GeneralWindFarmComponents import calculate_distance

from scipy.interpolate import UnivariateSpline

import time
import numpy as np
import matplotlib.pyplot as plt

from openmdao.devtools import iprofile


def plot_round_farm(turbineX, turbineY, rotor_diameter, boundary_center, boundary_radius, min_spacing=2.,
                    save_start=False, show_start=False, save_file=None):
    boundary_center_x = boundary_center[0]
    boundary_center_y = boundary_center[1]

    boundary_circle = plt.Circle((boundary_center_x / rotor_diameter, boundary_center_y / rotor_diameter),
                                 boundary_radius / rotor_diameter, facecolor='none', edgecolor='k', linestyle='--')

    fig, ax = plt.subplots()
    for x, y in zip(turbineX / rotor_diameter, turbineY / rotor_diameter):
        circle_start = plt.Circle((x, y), 0.5, facecolor='none', edgecolor='r', linestyle='-', label='Start')
        ax.add_artist(circle_start)
    # ax.plot(turbineX / rotor_diameter, turbineY / rotor_diameter, 'sk', label='Original', mfc=None)
    # ax.plot(prob['turbineX'] / rotor_diameter, prob['turbineY'] / rotor_diameter, '^g', label='Optimized', mfc=None)
    ax.add_patch(boundary_circle)
    plt.axis('equal')
    ax.legend([circle_start], ['turbines'])
    ax.set_xlabel('Turbine X Position ($X/D_r$)')
    ax.set_ylabel('Turbine Y Position ($Y/D_r$)')
    ax.set_xlim([(boundary_center_x - boundary_radius) / rotor_diameter - 1.,
                 (boundary_center_x + boundary_radius) / rotor_diameter + 1.])
    ax.set_ylim([(boundary_center_y - boundary_radius) / rotor_diameter - 1.,
                 (boundary_center_y + boundary_radius) / rotor_diameter + 1.])

    if save_start:
        if save_file is None:
            plt.savefig('round_farm_%iTurbines_%0.2fDSpacing.pdf' % (turbineX.size, min_spacing))
        else:
            plt.savefig(save_file)
    if show_start:
        plt.show()


def plot_square_farm(turbineX, turbineY, rotor_diameter, boundary_x, boundary_y, boundary_width, min_spacing=2,
                     save_start=False, show_start=False, save_file=None):
    full_bound_x = np.array([boundary_x[0], boundary_x[0], boundary_x[1], boundary_x[1], boundary_x[0]])
    full_bound_y = np.array([boundary_y[0], boundary_y[1], boundary_y[1], boundary_y[0], boundary_y[0]])

    real_bound_x = np.array(
        [boundary_x[0] + rotor_diameter / 2., boundary_x[0] + rotor_diameter / 2., boundary_x[1] - rotor_diameter / 2.,
         boundary_x[1] - rotor_diameter / 2., boundary_x[0] + rotor_diameter / 2.])
    real_bound_y = np.array(
        [boundary_y[0] + rotor_diameter / 2., boundary_y[1] - rotor_diameter / 2., boundary_y[1] - rotor_diameter / 2.,
         boundary_y[0] + rotor_diameter / 2., boundary_y[0] + rotor_diameter / 2.])

    fig, ax = plt.subplots()
    for x, y in zip(turbineX / rotor_diameter, turbineY / rotor_diameter):
        # print("here")
        circle_start = plt.Circle((x, y), 0.5, facecolor='none', edgecolor='r', linestyle='-', label='Start')
        ax.add_artist(circle_start)
    ax.plot(full_bound_x / rotor_diameter, full_bound_y / rotor_diameter)
    ax.plot(real_bound_x / rotor_diameter, real_bound_y / rotor_diameter, '--')
    # ax.plot(turbineX / rotor_diameter, turbineY / rotor_diameter, 'sk', label='Original', mfc=None)
    # ax.plot(prob['turbineX'] / rotor_diameter, prob['turbineY'] / rotor_diameter, '^g', label='Optimized', mfc=None)
    # ax.add_patch(boundary_circle)
    plt.axis('equal')
    # ax.legend([circle_start], ['turbines'])
    ax.set_xlabel('Turbine X Position ($X/D_r$)')
    ax.set_ylabel('Turbine Y Position ($Y/D_r$)')

    if save_start:
        if save_file is None:
            plt.savefig('round_farm_%iTurbines_%0.2fDSpacing.pdf' % (turbineX.size, min_spacing))
        else:
            plt.savefig(save_file)
    if show_start:
        plt.show()


if __name__ == "__main__":
    iprofile.setup()
    ######################### for MPI functionality #########################
    from openmdao.utils import mpi as MPI

    # if MPI:  # pragma: no cover
    #     # if you called this script with 'mpirun', then use the petsc data passing
    #     from openmdao import PetscImpl as impl
    #
    #     print("In MPI, impl = ", impl)
    #
    # else:
    #     # if you didn't use 'mpirun', then use the numpy data passing
    #     from openmdao import BasicImpl as impl
    #
    #
    # def print(prob, *args):
    #     """ helper function to only print on rank 0 """
    #     if prob.model.comm.rank == 0:
    #         print(*args)


    prob = om.Problem()

    #########################################################################

    # set up this run

    # specify which starting layout should be used
    # layout_number = int(sys.argv[1])
    layout_number = 0
    # relax_method = int(sys.argv[2])
    wec_method_number = 0
    # model_number = int(sys.argv[3])
    model = 1
    # opt_alg_number = int(sys.argv[4])
    opt_alg_number = 0


    run_number = layout_number

    # set model
    MODELS = ['FLORIS', 'BPA', 'JENSEN', 'LARSEN']
    print(MODELS[model])

    # select optimization approach/method
    opt_algs = ['snopt', 'ga', 'ps']
    opt_algorithm = opt_algs[opt_alg_number]

    # select wec method
    wec_methods = ['none', 'diam', 'angle']
    wec_method = wec_methods[wec_method_number]

    # pop_size = 760

    # save and show options
    show_start = True
    show_end = True
    save_start = False
    save_end = False

    save_locations = True
    save_aep = True
    save_time = True
    rec_func_calls = True

    input_directory = "./input_files/"

    # set options for BPA
    print_ti = False
    sort_turbs = True

    turbine_type = 'V80'  # can be 'V80' or 'NREL5MW'

    wake_model_version = 2016

    if wec_method == 'diam':
        output_directory = "./output_files/om2/%s_wec_diam/" % opt_algorithm
        relax = True
    elif wec_method == 'angle':
        output_directory = "./output_files/om2/%s_wec_angle/" % opt_algorithm
        relax = True
    elif wec_method == 'none':
        relax = False
        output_directory = "./output_files/om2/%s/" % opt_algorithm
    else:
        raise ValueError('wec_method must be diam, angle, or none')

    # create output directory if it does not exist yet
    import distutils.dir_util
    distutils.dir_util.mkpath(output_directory)

    differentiable = True

    if wec_method == 'diam':
        expansion_factors = np.array([3, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 1.0])
    elif wec_method == 'angle':
        # expansion_factors = 10*np.array([3, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 1.0])
        expansion_factors = np.array([60., 50., 40., 30., 20., 10., 0., 0.])

    # for expansion_factor in np.array([5., 4., 3., 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0]):
    # for expansion_factor in np.array([20., 15., 10., 5., 4., 3., 2.5, 1.25, 1.0]):
    # expansion_factors = np.array([20., 10., 5., 2.5, 1.25, 1.0])

    wake_combination_method = 1  # can be [0:Linear freestreem superposition,
    #  1:Linear upstream velocity superposition,
    #  2:Sum of squares freestream superposition,
    #  3:Sum of squares upstream velocity superposition]

    ti_calculation_method = 4  # can be [0:No added TI calculations,
    # 1:TI by Niayifar and Porte Agel altered by Annoni and Thomas,
    # 2:TI by Niayifar and Porte Agel 2016,
    # 3:TI by Niayifar and Porte Agel 2016 with added soft max function,
    # 4:TI by Niayifar and Porte Agel 2016 using area overlap ratio,
    # 5:TI by Niayifar and Porte Agel 2016 using area overlap ratio and SM function]

    ti_opt_method = 0  # can be [0:No added TI calculations,
    # 1:TI by Niayifar and Porte Agel altered by Annoni and Thomas,
    # 2:TI by Niayifar and Porte Agel 2016,
    # 3:TI by Niayifar and Porte Agel 2016 with added soft max function,
    # 4:TI by Niayifar and Porte Agel 2016 using area overlap ratio,
    # 5:TI by Niayifar and Porte Agel 2016 using area overlap ratio and SM function]

    final_ti_opt_method = 5

    sm_smoothing = 700.

    if ti_calculation_method == 0:
        calc_k_star_calc = False
    else:
        calc_k_star_calc = True

    if ti_opt_method == 0:
        calc_k_star_opt = False
    else:
        calc_k_star_opt = True

    nRotorPoints = 1

    wind_rose_file = 'nantucket'  # can be one of: 'amalia', 'nantucket', 'directional

    TI = 0.108
    k_calc = 0.3837 * TI + 0.003678
    # k_calc = 0.022
    # k_opt = 0.04

    shear_exp = 0.31

    # air_density = 1.1716  # kg/m^3
    air_density = 1.225  # kg/m^3 (from Jen)

    if turbine_type == 'V80':

        # define turbine size
        rotor_diameter = 80.  # (m)
        hub_height = 70.0

        z_ref = 80.0 #m
        z_0 = 0.0

        # load performance characteristics
        cut_in_speed = 4.  # m/s
        rated_power = 2000.  # kW
        generator_efficiency = 0.944

        ct_curve_data = np.loadtxt(input_directory + 'mfg_ct_vestas_v80_niayifar2016.txt', delimiter=",")
        ct_curve_wind_speed = ct_curve_data[:, 0]
        ct_curve_ct = ct_curve_data[:, 1]

        # air_density = 1.1716  # kg/m^3
        Ar = 0.25 * np.pi * rotor_diameter ** 2
        # cp_curve_wind_speed = ct_curve[:, 0]
        power_data = np.loadtxt(input_directory + 'niayifar_vestas_v80_power_curve_observed.txt', delimiter=',')
        # cp_curve_cp = niayifar_power_model(cp_curve_wind_speed)/(0.5*air_density*cp_curve_wind_speed**3*Ar)
        cp_curve_cp = power_data[:, 1] * (1E6) / (0.5 * air_density * power_data[:, 0] ** 3 * Ar)
        cp_curve_wind_speed = power_data[:, 0]
        cp_curve_spline = UnivariateSpline(cp_curve_wind_speed, cp_curve_cp, ext='const')
        cp_curve_spline.set_smoothing_factor(.0001)

    elif turbine_type == 'NREL5MW':

        # define turbine size
        rotor_diameter = 126.4  # (m)
        hub_height = 90.0

        z_ref = 80.0 # m
        z_0 = 0.0

        # load performance characteristics
        cut_in_speed = 3.  # m/s
        rated_power = 5000.  # kW
        generator_efficiency = 0.944

        filename = input_directory + "NREL5MWCPCT_dict.p"
        # filename = "../input_files/NREL5MWCPCT_smooth_dict.p"
        import cPickle as pickle

        data = pickle.load(open(filename, "rb"))
        ct_curve = np.zeros([data['wind_speed'].size, 2])
        ct_curve_wind_speed = data['wind_speed']
        ct_curve_ct = data['CT']

        # cp_curve_cp = data['CP']
        # cp_curve_wind_speed = data['wind_speed']

        loc0 = np.where(data['wind_speed'] < 11.55)
        loc1 = np.where(data['wind_speed'] > 11.7)

        cp_curve_cp = np.hstack([data['CP'][loc0], data['CP'][loc1]])
        cp_curve_wind_speed = np.hstack([data['wind_speed'][loc0], data['wind_speed'][loc1]])
        cp_curve_spline = UnivariateSpline(cp_curve_wind_speed, cp_curve_cp, ext='const')
        cp_curve_spline.set_smoothing_factor(.000001)
    else:
        raise ValueError("Turbine type is undefined.")

    # load starting locations
    layout_directory = input_directory

    layout_data = np.loadtxt(layout_directory + "nTurbs38_spacing5_layout_0.txt")
    # layout_data = np.loadtxt(layout_directory + "layouts/grid_16turbs/nTurbs16_spacing5_layout_%i.txt" % layout_number)
    # layout_data = np.loadtxt(layout_directory+"layouts/nTurbs9_spacing5_layout_%i.txt" % layout_number)

    turbineX = layout_data[:, 0] * rotor_diameter + rotor_diameter/2.
    turbineY = layout_data[:, 1] * rotor_diameter + rotor_diameter/2.

    turbineX_init = np.copy(turbineX)
    turbineY_init = np.copy(turbineY)

    nTurbines = turbineX.size

    # create boundary specifications
    boundary_radius = 0.5 * (rotor_diameter * 4000. / 126.4 - rotor_diameter)  # 1936.8
    center = np.array([boundary_radius, boundary_radius]) + rotor_diameter / 2.
    start_min_spacing = 5.
    nVertices = 1
    boundary_center_x = center[0]
    boundary_center_y = center[1]
    xmax = np.max(turbineX)
    ymax = np.max(turbineY)
    xmin = np.min(turbineX)
    ymin = np.min(turbineY)
    boundary_radius_plot = boundary_radius + 0.5 * rotor_diameter

    plot_round_farm(turbineX, turbineY, rotor_diameter, [boundary_center_x, boundary_center_y], boundary_radius,
                    show_start=show_start)
    # quit()
    # initialize input variable arrays
    nTurbs = nTurbines
    rotorDiameter = np.zeros(nTurbs)
    hubHeight = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)
    minSpacing = 2.  # number of rotor diameters

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter  # m
        hubHeight[turbI] = hub_height  # m
        axialInduction[turbI] = 1.0 / 3.0
        Ct[turbI] = 4.0 * axialInduction[turbI] * (1.0 - axialInduction[turbI])
        # print(Ct)
        Cp[turbI] = 4.0 * 1.0 / 3.0 * np.power((1 - 1.0 / 3.0), 2)
        generatorEfficiency[turbI] = generator_efficiency
        yaw[turbI] = 0.  # deg.

    # Define flow properties
    if wind_rose_file is 'nantucket':
        # windRose = np.loadtxt(input_directory + 'nantucket_windrose_ave_speeds.txt')
        windRose = np.loadtxt(input_directory + 'nantucket_wind_rose_for_LES.txt')
        windDirections = windRose[:, 0]
        windSpeeds = windRose[:, 1]
        windFrequencies = windRose[:, 2]
        size = np.size(windDirections)
    elif wind_rose_file is 'amalia':
        windRose = np.loadtxt(input_directory + 'amalia.txt')
        windDirections = windRose[:, 0]
        windSpeeds = windRose[:, 1]
        windFrequencies = windRose[:, 2]
        size = np.size(windDirections)
    elif wind_rose_file is 'directional':
        windRose = np.loadtxt(input_directory + 'directional_windrose.txt')
        windDirections = windRose[:, 0]
        windSpeeds = windRose[:, 1]
        windFrequencies = windRose[:, 2]
        size = np.size(windDirections)
    else:
        size = 20
        windDirections = np.linspace(0, 270, size)
        windFrequencies = np.ones(size) / size

    wake_model_options = {'nSamples': 0,
                          'nRotorPoints': nRotorPoints,
                          'use_ct_curve': True,
                          'ct_curve_ct': ct_curve_ct,
                          'ct_curve_wind_speed': ct_curve_wind_speed,
                          'interp_type': 1,
                          'use_rotor_components': False,
                          'differentiable': differentiable,
                          'verbose': False}

    if MODELS[model] == 'BPA':
        # initialize problem
        prob = om.Problem(model=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=nVertices,
                                              minSpacing=minSpacing, differentiable=differentiable,
                                              use_rotor_components=False,
                                              wake_model=gauss_wrapper,
                                              params_IdepVar_func=add_gauss_params_IndepVarComps,
                                              params_IdepVar_args={'nRotorPoints': nRotorPoints},
                                              wake_model_options=wake_model_options,
                                              cp_points=cp_curve_cp.size, cp_curve_spline=cp_curve_spline))
    elif MODELS[model] == 'FLORIS':
        # initialize problem
        prob = om.Problem(model=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=nVertices,
                                              minSpacing=minSpacing, differentiable=differentiable,
                                              use_rotor_components=False,
                                              wake_model=floris_wrapper,
                                              params_IdepVar_func=add_floris_params_IndepVarComps,
                                              params_IdepVar_args={}))
    # elif MODELS[model] == 'JENSEN':
    #     initialize problem
    # prob = Problem(model=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=nVertices,
    #                                       minSpacing=minSpacing, differentiable=False, use_rotor_components=False,
    #                                       wake_model=jensen_wrapper,
    #                                       params_IdepVar_func=add_jensen_params_IndepVarComps,
    #                                       params_IdepVar_args={}))
    else:
        ValueError('The %s model is not currently available. Please select BPA or FLORIS' % (MODELS[model]))
    # prob.model.deriv_options['type'] = 'fd'
    # prob.model.deriv_options['form'] = 'central'
    # prob.model.deriv_options['step_size'] = 1.0e-8

    # set to use more efficient solver
    prob.model.linear_solver = om.DirectSolver()

    prob.driver = om.pyOptSparseDriver()

    if opt_algorithm == 'snopt':
        # set up optimizer
        prob.driver.options['optimizer'] = 'SNOPT'
        # prob.driver.options['gradient method'] = 'snopt_fd'

        # set optimizer options
        prob.driver.opt_settings['Verify level'] = 1
        prob.driver.opt_settings['Major optimality tolerance'] = 1e-2
        prob.driver.opt_settings[
            'Print file'] = output_directory + 'SNOPT_print_multistart_%iturbs_%sWindRose_%idirs_%sModel_RunID%i.out' % (
            nTurbs, wind_rose_file, size, MODELS[model], run_number)
        prob.driver.opt_settings[
            'Summary file'] = output_directory + 'SNOPT_summary_multistart_%iturbs_%sWindRose_%idirs_%sModel_RunID%i.out' % (
            nTurbs, wind_rose_file, size, MODELS[model], run_number)

        prob.model.add_constraint('sc', lower=np.zeros(int(((nTurbs - 1.) * nTurbs / 2.))), scaler=1E-2)
        prob.model.add_constraint('boundaryDistances', lower=(np.zeros(1 * turbineX.size)), scaler=1E-2)

        prob.driver.options['dynamic_derivs_sparsity'] = True

    elif opt_algorithm == 'ga':

        prob.driver.options['optimizer'] = 'NSGA2'

        prob.driver.opt_settings['PrintOut'] = 1

        prob.driver.opt_settings['maxGen'] = 50000

        prob.driver.opt_settings['PopSize'] = 10 * nTurbines * 2

        # prob.driver.opt_settings['pMut_real'] = 0.001

        prob.driver.opt_settings['xinit'] = 1

        prob.driver.opt_settings['rtol'] = 1E-4

        prob.driver.opt_settings['atol'] = 1E-4

        prob.driver.opt_settings['min_tol_gens'] = 200

        prob.driver.opt_settings['file_number'] = run_number

        prob.model.add_constraint('sc', lower=np.zeros(int(((nTurbs - 1.) * nTurbs / 2.))), scaler=1E-2)
        prob.model.add_constraint('boundaryDistances', lower=(np.zeros(1 * turbineX.size)), scaler=1E-2)



    elif opt_algorithm == 'ps':

        prob.driver.options['optimizer'] = 'ALPSO'

        prob.driver.opt_settings['fileout'] = 1

        prob.driver.opt_settings[

            'filename'] = output_directory + 'ALPSO_summary_multistart_%iturbs_%sWindRose_%idirs_%sModel_RunID%i.out' % (

            nTurbs, wind_rose_file, size, MODELS[model], run_number)

        prob.driver.opt_settings['maxOuterIter'] = 10000

        prob.driver.opt_settings['SwarmSize'] = 10 * nTurbines * 2

        prob.driver.opt_settings['xinit'] = 1  # Initial Position Flag (0 - no position, 1 - position given)

        prob.driver.opt_settings[
            'Scaling'] = 1  # Design Variables Scaling Flag (0 - no scaling, 1 - scaling between [-1, 1])

        prob.driver.opt_settings['rtol'] = 1E-3  # Relative Tolerance for Lagrange Multipliers

        prob.driver.opt_settings['atol'] = 1E-2  # Absolute Tolerance for Lagrange Function

        prob.driver.opt_settings['dtol'] = 1E-1  # Relative Tolerance in Distance of All Particles to Terminate (GCPSO)

        prob.driver.opt_settings['itol'] = 1E-3  # Absolute Tolerance for Inequality constraints

        prob.driver.opt_settings['dynInnerIter'] = 1  # Dynamic Number of Inner Iterations Flag

        prob.model.add_constraint('sc', lower=np.zeros(int(((nTurbs - 1.) * nTurbs / 2.))), scaler=1E-2)
        prob.model.add_constraint('boundaryDistances', lower=(np.zeros(1 * turbineX.size)), scaler=1E-2)

        # prob.model.add_objective('obj', scaler=1E0)
    prob.model.add_objective('obj', scaler=1E-3)

    # select design variables
    prob.model.add_design_var('turbineX', scaler=1E1, lower=np.zeros(nTurbines),
                           upper=np.ones(nTurbines) * 3. * boundary_radius)
    prob.model.add_design_var('turbineY', scaler=1E1, lower=np.zeros(nTurbines),
                           upper=np.ones(nTurbines) * 3. * boundary_radius)

    # prob.model.ln_solver.options['single_voi_relevance_reduction'] = True
    # prob.model.ln_solver.options['mode'] = 'rev'

    if show_start:
        boundary_circle = plt.Circle((boundary_center_x / rotor_diameter, boundary_center_y / rotor_diameter),
                                     boundary_radius_plot / rotor_diameter, facecolor='none', edgecolor='k', linestyle='-')
        constraint_circle = plt.Circle((boundary_center_x / rotor_diameter, boundary_center_y / rotor_diameter),
                                     boundary_radius / rotor_diameter, facecolor='none', edgecolor='k', linestyle='--')

        fig, ax = plt.subplots()
        for x, y in zip(turbineX / rotor_diameter, turbineY / rotor_diameter):
            circle_start = plt.Circle((x, y), 0.5, facecolor='none', edgecolor='r', linestyle=':', label='Start')
            ax.add_artist(circle_start)
        # for x, y in zip(turbineX / rotor_diameter, turbineY / rotor_diameter):
        #     circle_end = plt.Circle((x, y), 0.5, facecolor='none', edgecolor='g', linestyle='--', label='End')
        #     ax.add_artist(circle_end)
        # ax.plot(turbineX / rotor_diameter, turbineY / rotor_diameter, 'sk', label='Original', mfc=None)
        # ax.plot(prob['turbineX'] / rotor_diameter, prob['turbineY'] / rotor_diameter, '^g', label='Optimized', mfc=None)
        ax.add_patch(boundary_circle)
        ax.add_patch(constraint_circle)
        # for i in range(0, nTurbs):
        #     ax.plot([turbineX[i] / rotor_diameter, prob['turbineX'][i] / rotor_diameter],
        #             [turbineY[i] / rotor_diameter, prob['turbineY'][i] / rotor_diameter], '--k')
        ax.legend([circle_start], ['Start'])
        # ax.legend([circle_start, circle_end], ['Start', 'End'])
        ax.set_xlabel('Turbine X Position ($X/D_r$)')
        ax.set_ylabel('Turbine Y Position ($Y/D_r$)')
        ax.set_xlim([(boundary_center_x - boundary_radius_plot) / rotor_diameter - 1.,
                     (boundary_center_x + boundary_radius_plot) / rotor_diameter + 1.])
        ax.set_ylim([(boundary_center_y - boundary_radius_plot) / rotor_diameter - 1.,
                     (boundary_center_y + boundary_radius_plot) / rotor_diameter + 1.])
        plt.axis('equal')
        plt.show()
    # if run_number == 0:
    #     # set up recorder
    #     recorder = SqliteRecorder(output_directory+'recorder_database_run%i' % run_number)
    #     recorder.options['record_params'] = True
    #     recorder.options['record_metadata'] = False
    #     recorder.options['record_unknowns'] = True
    #     recorder.options['record_derivs'] = False
    #     recorder.options['includes'] = ['turbineX', 'turbineY', 'AEP']
    #     prob.model.add_recorder(recorder)

    print("almost time for setup")
    tic = time.time()
    print("entering setup at time = ", tic)
    prob.setup(check=True)
    toc = time.time()
    print(prob, "setup complete at time = ", toc)

    # print the results
    print(prob, ('Problem setup took %.03f sec.' % (toc - tic)))

    # assign initial values to design variables
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    for direction_id in range(0, windDirections.size):
        prob['yaw%i' % direction_id] = yaw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['hubHeight'] = hubHeight
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['windSpeeds'] = windSpeeds
    prob['air_density'] = air_density
    prob['windDirections'] = windDirections
    prob['windFrequencies'] = windFrequencies
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp
    prob['cp_curve_cp'] = cp_curve_cp
    prob['cp_curve_wind_speed'] = cp_curve_wind_speed
    cutInSpeeds = np.ones(nTurbines) * cut_in_speed
    prob['cut_in_speed'] = cutInSpeeds
    ratedPowers = np.ones(nTurbines) * rated_power
    prob['rated_power'] = ratedPowers

    # assign boundary values
    prob['boundary_center'] = np.array([boundary_center_x, boundary_center_y])
    prob['boundary_radius'] = boundary_radius

    if MODELS[model] is 'BPA':
        prob['model_params:wake_combination_method'] = wake_combination_method
        prob['model_params:ti_calculation_method'] = ti_calculation_method
        prob['model_params:wake_model_version'] = wake_model_version
        prob['model_params:wec_factor'] = 1.0
        prob['model_params:calc_k_star'] = calc_k_star_calc
        prob['model_params:sort'] = sort_turbs
        prob['model_params:z_ref'] = z_ref
        prob['model_params:z_0'] = z_0
        prob['model_params:ky'] = k_calc
        prob['model_params:kz'] = k_calc
        prob['model_params:print_ti'] = print_ti
        prob['model_params:shear_exp'] = shear_exp
        prob['model_params:I'] = TI
        prob['model_params:sm_smoothing'] = sm_smoothing
        if nRotorPoints > 1:
            prob['model_params:RotorPointsY'], prob['model_params:RotorPointsZ'] = sunflower_points(nRotorPoints)

    prob.run_model()
    AEP_init_calc = np.copy(prob['AEP'])
    print(prob, AEP_init_calc * 1E-6)

    if MODELS[model] is 'BPA':
        prob['model_params:ti_calculation_method'] = ti_opt_method
        prob['model_params:calc_k_star'] = calc_k_star_opt

    prob.run_model()
    AEP_init_opt = np.copy(prob['AEP'])
    AEP_run_opt = np.copy(AEP_init_opt)
    print(prob, AEP_init_opt * 1E-6)

    config.obj_func_calls_array[:] = 0.0
    config.sens_func_calls_array[:] = 0.0

    expansion_factor_last = 0.0
    # Jt = prob.check_totals(out_stream=None)
    tick = time.time()
    if relax:
        for expansion_factor, i in zip(expansion_factors, np.arange(0, expansion_factors.size)):  # best so far
            # print("func calls: ", config.obj_func_calls_array, np.sum(config.obj_func_calls_array))
            # print("grad func calls: ", config.sens_func_calls_array, np.sum(config.sens_func_calls_array))
            # AEP_init_run_opt = prob['AEP']

            if expansion_factor_last == expansion_factor:
                ti_opt_method = final_ti_opt_method

            print(prob, "starting run with exp. fac = ", expansion_factor)

            if opt_algorithm == 'snopt':
                prob.driver.opt_settings['Print file'] = output_directory + \
                                                         'SNOPT_print_multistart_%iturbs_%sWindRose_%idirs_%sModel_RunID%i_EF%.3f_TItype%i.out' % (
                                                             nTurbs, wind_rose_file, size, MODELS[model], run_number,
                                                             expansion_factor, ti_opt_method)

                prob.driver.opt_settings['Summary file'] = output_directory + \
                                                           'SNOPT_summary_multistart_%iturbs_%sWindRose_%idirs_%sModel_RunID%i_EF%.3f_TItype%i.out' % (
                                                               nTurbs, wind_rose_file, size, MODELS[model], run_number,
                                                               expansion_factor, ti_opt_method)
            elif opt_algorithm == 'ps':
                prob.driver.opt_settings[
                    'filename'] = output_directory + 'ALPSO_summary_multistart_%iturbs_%sWindRose_%idirs_%sModel_RunID%i.out' % (
                    nTurbs, wind_rose_file, size, MODELS[model], run_number)

            turbineX = prob['turbineX']
            turbineY = prob['turbineY']
            prob['turbineX'] = turbineX
            prob['turbineY'] = turbineY

            if MODELS[model] is 'BPA':
                prob['model_params:ti_calculation_method'] = ti_opt_method
                prob['model_params:calc_k_star'] = calc_k_star_opt
                if wec_method == 'diam':
                    prob['model_params:wec_factor'] = expansion_factor
                elif wec_method == 'angle':
                    prob['model_params:wec_spreading_angle'] = expansion_factor

            # run the problem
            print(prob, 'start %s run' % (MODELS[model]))
            tic = time.time()
            prob.run_driver()
            toc = time.time()
            # print(np.sum(config.obj_func_calls_array))
            # print(np.sum(config.sens_func_calls_array))
            print(prob, 'end %s run' % (MODELS[model]))

            run_time = toc - tic
            # print(run_time, expansion_factor)

            AEP_run_opt = prob['AEP']
            # print(prob, "AEP improvement = ", AEP_run_opt / AEP_init_opt)

            if MODELS[model] is 'BPA':
                prob['model_params:wec_factor'] = 1.0
                prob['model_params:wec_spreading_angle'] = 0.0
                prob['model_params:ti_calculation_method'] = ti_calculation_method
                prob['model_params:calc_k_star'] = calc_k_star_calc

            prob.run_model()
            AEP_run_calc = prob['AEP']
            # print("compare: ", aep_run, prob['AEP'])
            print(prob, "AEP calc improvement = ", AEP_run_calc / AEP_init_calc)

            if prob.model.comm.rank == 0:
                # if save_aep:
                #     np.savetxt(output_directory + '%s_multistart_aep_results_%iturbs_%sWindRose_%idirs_%sModel_RunID%i_EF%.3f.txt' % (
                #         opt_algorithm, nTurbs, wind_rose_file, size, MODELS[model], run_number, expansion_factor),
                #                np.c_[AEP_init, prob['AEP']],
                #                header="Initial AEP, Final AEP")
                if save_locations:
                    np.savetxt(
                        output_directory + '%s_multistart_locations_%iturbs_%sWindRose_%idirs_%s_run%i_EF%.3f_TItype%i.txt' % (
                            opt_algorithm, nTurbs, wind_rose_file, size, MODELS[model], run_number, expansion_factor, ti_opt_method),
                        np.c_[turbineX_init, turbineY_init, prob['turbineX'], prob['turbineY']],
                        header="initial turbineX, initial turbineY, final turbineX, final turbineY")
                # if save_time:
                #     np.savetxt(output_directory + '%s_multistart_time_%iturbs_%sWindRose_%idirs_%s_run%i_EF%.3f.txt' % (
                #         opt_algorithm, nTurbs, wind_rose_file, size, MODELS[model], run_number, expansion_factor),
                #                np.c_[run_time],
                #                header="run time")
                if save_time and save_aep and rec_func_calls:
                    output_file = output_directory + '%s_multistart_rundata_%iturbs_%sWindRose_%idirs_%s_run%i.txt' \
                                  % (opt_algorithm, nTurbs, wind_rose_file, size, MODELS[model], run_number)
                    f = open(output_file, "a")

                    if i == 0:
                        header = "run number, exp fac, ti calc, ti opt, aep init calc (kW), aep init opt (kW), " \
                                 "aep run calc (kW), aep run opt (kW), run time (s), obj func calls, sens func calls"
                    else:
                        header = ''

                    np.savetxt(f, np.c_[run_number, expansion_factor, ti_calculation_method, ti_opt_method,
                                        AEP_init_calc, AEP_init_opt, AEP_run_calc, AEP_run_opt, run_time,
                                        config.obj_func_calls_array[0], config.sens_func_calls_array[0]],
                               header=header)
                    f.close()
            expansion_factor_last = expansion_factor
    else:
        # run the problem
        print(prob, 'start %s run' % (MODELS[model]))
        # cProfile.run('prob.run_driver()')
        if MODELS[model] is 'BPA':
            # prob['model_params:wec_factor'] = 1.
            prob['model_params:ti_calculation_method'] = ti_opt_method
            prob['model_params:calc_k_star'] = calc_k_star_opt
        tic = time.time()
        # cProfile.run('prob.run_driver()')

        iprofile.start()
        prob.run_driver()
        iprofile.stop()

        # quit()
        toc = time.time()

        run_time = toc - tic

        AEP_run_opt = prob['AEP']
        print(prob, "AEP improvement = ", AEP_run_opt / AEP_init_opt)

        if MODELS[model] is 'BPA':
            prob['model_params:wec_factor'] = 1.0
            prob['model_params:wec_spreading_angle'] = 0.0
            prob['model_params:ti_calculation_method'] = ti_calculation_method
            prob['model_params:calc_k_star'] = calc_k_star_calc

        prob.run_model()
        AEP_run_calc = prob['AEP']

        if prob.model.comm.rank == 0:

            if save_locations:
                np.savetxt(output_directory + '%s_multistart_locations_%iturbs_%sWindRose_%idirs_%s_run%i.txt' % (
                    opt_algorithm, nTurbs, wind_rose_file, size, MODELS[model], run_number),
                           np.c_[turbineX_init, turbineY_init, prob['turbineX'], prob['turbineY']],
                           header="initial turbineX, initial turbineY, final turbineX, final turbineY")

            if save_time and save_aep and rec_func_calls:
                output_file = output_directory + '%s_multistart_rundata_%iturbs_%sWindRose_%idirs_%s_run%i.txt' \
                              % (opt_algorithm, nTurbs, wind_rose_file, size, MODELS[model], run_number)
                f = open(output_file, "a")

                header = "run number, ti calc, ti opt, aep init calc (kW), aep init opt (kW), " \
                         "aep run calc (kW), aep run opt (kW), run time (s), obj func calls, sens func calls"

                np.savetxt(f, np.c_[run_number, ti_calculation_method, ti_opt_method,
                                    AEP_init_calc, AEP_init_opt, AEP_run_calc, AEP_run_opt, run_time,
                                    config.obj_func_calls_array[0], config.sens_func_calls_array[0]],
                           header=header)
                f.close()

    toct = time.time()
    total_time = toct - tict
    turbineX_end = np.copy(prob['turbineX'])
    turbineY_end = np.copy(prob['turbineY'])
    if prob.model.comm.rank == 0:

        # print the results
        print(prob, ('Opt. calculation took %.03f sec.' % (toct - tict)))

        for direction_id in range(0, windDirections.size):
            print(prob, 'yaw%i (deg) = ' % direction_id, prob['yaw%i' % direction_id])

        print(prob, 'turbine X positions in wind frame (m): %s' % prob['turbineX'])
        print(prob, 'turbine Y positions in wind frame (m): %s' % prob['turbineY'])
        print(prob, 'wind farm power in each direction (kW): %s' % prob['dirPowers'])
        print(prob, 'Initial AEP (kWh): %s' % AEP_init_opt)
        print(prob, 'Final AEP (kWh): %s' % AEP_run_opt)
        print(prob, 'AEP improvement: %s' % (AEP_run_opt / AEP_init_opt))

    plot_round_farm(turbineX_end, turbineY_end, rotor_diameter, [boundary_center_x, boundary_center_y], boundary_radius,
                    show_start=show_start)
