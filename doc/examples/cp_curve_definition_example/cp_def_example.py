from __future__ import print_function

from openmdao.api import Problem, pyOptSparseDriver
from plantenergy.OptimizationGroups import OptAEP
from plantenergy import config

import time
import numpy as np

if __name__ == "__main__":

    ######################### for MPI functionality #########################
    from openmdao.core.mpi_wrap import MPI

    if MPI:  # pragma: no cover
        # if you called this script with 'mpirun', then use the petsc data passing
        from openmdao.core.petsc_impl import PetscImpl as impl

        print("In MPI, impl = ", impl)

    else:
        # if you didn't use 'mpirun', then use the numpy data passing
        from openmdao.api import BasicImpl as impl


    def mpi_print(prob, *args):
        """ helper function to only print on rank 0 """
        if prob.root.comm.rank == 0:
            print(*args)


    prob = Problem(impl=impl)

    #########################################################################

    save_locations = True
    save_run_data = True

    input_directory = "./input_files/"

    # select model
    MODELS = ['FLORIS', 'JENSEN']
    model = 0
    print(MODELS[model])

    turbine_type = 'V80'  # can be 'V80' or 'NREL5MW'

    relax = False

    output_directory = "./output_files/"

    # create output directory if it does not exist yet
    import distutils.dir_util
    distutils.dir_util.mkpath(output_directory)

    differentiable = True

    wind_rose_file = 'nantucket'  # can be one of: 'amalia', 'nantucket', 'directional

    air_density = 1.225  # kg/m^3

    if turbine_type == 'V80':

        # define turbine size
        rotor_diameter = 80.  # (m)
        hub_height = 70.0

        z_ref = hub_height
        z_0 = 0.0

        # load performance characteristics
        cut_in_speed = 4.  # m/s
        rated_wind_speed = 15. # m/s
        cut_out_speed = 25. # m/s
        rated_power = 2000.  # kW
        use_power_curve_definition = True

        Ar = 0.25 * np.pi * rotor_diameter ** 2

    elif turbine_type == 'NREL5MW':

        # define turbine size
        rotor_diameter = 126.4  # (m)
        hub_height = 90.0

        z_ref = hub_height
        z_0 = 0.0

        # load performance characteristics
        cut_in_speed = 3.  # m/s
        rated_wind_speed = 11.4  # m/s
        cut_out_speed = 25.  # m/s
        rated_power = 5000.  # kW
        use_power_curve_definition = True

        filename = input_directory + "NREL5MWCPCT_dict.p"
        # filename = "../input_files/NREL5MWCPCT_smooth_dict.p"
        import cPickle as pickle

        data = pickle.load(open(filename, "rb"))
        ct_curve = np.zeros([data['wind_speed'].size, 2])
        ct_curve_wind_speed = data['wind_speed']
        ct_curve_ct = data['CT']
    else:
        raise ValueError("Turbine type is undefined.")

    layout_data = np.loadtxt(input_directory + "/nTurbs9_spacing5_layout_0.txt")

    turbineX = layout_data[:, 0] * rotor_diameter
    turbineY = layout_data[:, 1] * rotor_diameter

    turbineX_init = np.copy(turbineX)
    turbineY_init = np.copy(turbineY)

    nTurbines = turbineX.size

    nVertices = 0

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
        generatorEfficiency[turbI] = 1.
        yaw[turbI] = 0.  # deg.

    # Define flow properties
    if wind_rose_file is 'nantucket':
        # windRose = np.loadtxt(input_directory + 'nantucket_windrose_ave_speeds.txt')
        windRose = np.loadtxt(input_directory + 'nantucket_wind_rose.txt')
        windDirections = windRose[:, 0]
        windSpeeds = windRose[:, 1]*0.0 + 10.0
        windFrequencies = windRose[:, 2]
        size = np.size(windDirections)
    else:
        size = 20
        windDirections = np.linspace(0, 270, size)
        windFrequencies = np.ones(size) / size

    ############################    initialize system with given wake model     ########################################

    if MODELS[model] == 'FLORIS':
        from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps

        # initialize problem
        prob = Problem(impl=impl, root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=nVertices,
                                              minSpacing=minSpacing, differentiable=differentiable,
                                              use_rotor_components=False,
                                              wake_model=floris_wrapper,
                                              params_IdepVar_func=add_floris_params_IndepVarComps,
                                              params_IndepVar_args={}))
    elif MODELS[model] == 'JENSEN':
        from plantenergy.jensen import jensen_wrapper, add_jensen_params_IndepVarComps

        # initialize problem
        prob = Problem(impl=impl, root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=nVertices,
                                          minSpacing=minSpacing, differentiable=False, use_rotor_components=False,
                                          wake_model=jensen_wrapper,
                                          params_IdepVar_func=add_jensen_params_IndepVarComps,
                                          params_IndepVar_args={}))
    else:
        ValueError('The %s model is not currently available. Please select JENSEN or FLORIS' % (MODELS[model]))

    ####################################################################################################################


    ###############################    set up optimizer and optimization problem     ###################################

    prob.driver = pyOptSparseDriver()


    # set up optimizer
    prob.driver.options['optimizer'] = 'SNOPT'
    # prob.driver.options['gradient method'] = 'snopt_fd'

    # set optimizer options
    prob.driver.opt_settings['Verify level'] = 1
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-4
    prob.driver.opt_settings[
        'Print file'] = output_directory + 'SNOPT_print_multistart_%iturbs_%sWindRose_%idirs_%sModel.out' % (
        nTurbs, wind_rose_file, size, MODELS[model])
    prob.driver.opt_settings[
        'Summary file'] = output_directory + 'SNOPT_summary_multistart_%iturbs_%sWindRose_%idirs_%sModel.out' % (
        nTurbs, wind_rose_file, size, MODELS[model])

    prob.driver.add_constraint('sc', lower=np.zeros(int(((nTurbs - 1.) * nTurbs / 2.))), scaler=1E-2,
                               active_tol=(2. * rotor_diameter) ** 2)

    prob.driver.add_objective('obj', scaler=1E-3)

    # select design variables
    prob.driver.add_desvar('turbineX', scaler=1E1, lower=np.ones(nTurbines)*np.min(turbineX),
                           upper=np.ones(nTurbines)*np.max(turbineX))
    prob.driver.add_desvar('turbineY', scaler=1E1, lower=np.ones(nTurbines)*np.min(turbineY),
                           upper=np.ones(nTurbines)*np.max(turbineY))

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
    prob.root.ln_solver.options['mode'] = 'rev'

    tic = time.time()
    print("entering setup at time = ", tic)
    prob.setup(check=True)
    toc = time.time()
    mpi_print(prob, "setup complete at time = ", toc)

    # print the results
    mpi_print(prob, ('Problem setup took %.03f sec.' % (toc - tic)))

    ####################################################################################################################



    #####################################   initialize parameter values     ############################################

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
    prob['cut_in_speed'] = np.ones(nTurbines)*cut_in_speed
    prob['cut_out_speed'] = np.ones(nTurbines)*cut_out_speed
    prob['rated_wind_speed'] = np.ones(nTurbines)*rated_wind_speed
    prob['rated_power'] =np.ones(nTurbines)*rated_power
    ####################################################################################################################


    ##############################   run optimization problem and print results    #####################################

    # get initial AEP information
    prob.run_once()
    AEP_init = np.copy(prob['AEP'])

    mpi_print(prob, 'inital AEP (kWh):')
    mpi_print(prob, prob['AEP'])

    config.obj_func_calls_array[:] = 0.0
    config.sens_func_calls_array[:] = 0.0

    # run the problem
    mpi_print(prob, 'start %s run' % (MODELS[model]))
    tic = time.time()

    prob.run()

    toc = time.time()
    run_time = toc - tic

    # get final AEP and calulate improvement
    AEP_opt = prob['AEP']
    mpi_print(prob, "AEP improvement = ", AEP_opt / AEP_init)

    if prob.root.comm.rank == 0:

        if save_locations:
            np.savetxt(output_directory + 'locations_%iturbs_%sWindRose_%idirs_%s.txt' % (
                nTurbs, wind_rose_file, size, MODELS[model]),
                       np.c_[turbineX_init, turbineY_init, prob['turbineX'], prob['turbineY']],
                       header="initial turbineX, initial turbineY, final turbineX, final turbineY")
        if save_run_data:
            output_file = output_directory + 'rundata_%iturbs_%sWindRose_%idirs_%s.txt' \
                          % (nTurbs, wind_rose_file, size, MODELS[model])
            f = open(output_file, "a")

            header = "aep init (kW), aep opt (kW), run time (s), obj func calls, sens func calls"

            np.savetxt(f, np.c_[AEP_init, AEP_opt, run_time,
                                config.obj_func_calls_array[0], config.sens_func_calls_array[0]],
                       header=header)
            f.close()

    if prob.root.comm.rank == 0:

        # print the results
        mpi_print(prob, ('Optimization time: %.03f sec.' % (run_time)))

        mpi_print(prob, 'turbine X positions in wind frame (m): %s' % prob['turbineX'])
        mpi_print(prob, 'turbine Y positions in wind frame (m): %s' % prob['turbineY'])
        mpi_print(prob, 'wind farm power in each direction (kW): %s' % prob['dirPowers'])
        mpi_print(prob, 'Initial AEP (kWh): %s' % AEP_init)
        mpi_print(prob, 'Final AEP (kWh): %s' % AEP_opt)
        mpi_print(prob, 'AEP improvement: %s' % (AEP_opt / AEP_init))