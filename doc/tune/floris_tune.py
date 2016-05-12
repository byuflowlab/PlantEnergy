from openmdao.api import Problem, Group, IndepVarComp
from pyoptsparse import Optimization, OPT, SNOPT

from florisse.floris import DirectionGroup
from florisse import config

from scipy.io import loadmat
import time
import numpy as np
import cPickle as pickle
import pylab as plt


def objfunction(xdict):

    global prob
    global me
    global MU
    global initialWakeDisplacement

     # set tuning variables
    prob['gen_params:pP'] = params[0]
    prob['floris_params:kd'] = params[1]
    prob['floris_params:initialWakeAngle'] = params[2]
    # prob['floris_params:initialWakeDisplacement'] = initialWakeDisplacement
    prob['floris_params:bd'] = params[3]
    prob['floris_params:ke'] = params[4]
    prob['floris_params:me'] = np.append(params[5:7], me[2])
    # print prob['floris_params:me']
    prob['floris_params:MU'] = np.insert(params[7:9], 1, MU[1])
    # print prob['floris_params:MU']
    prob['floris_params:aU'] = params[9]
    prob['floris_params:bU'] = params[10]
    prob['floris_params:cos_spread'] = params[11]

    # Define turbine locations and orientation
    prob['turbineX'] = np.array([1118.1, 1881.9])
    prob['turbineY'] = np.array([1279.5, 1720.5])

    ICOWESdata = loadmat('YawPosResults.mat')
    yawrange = ICOWESdata['yaw'][0]

    FLORISpower = list()

    for yaw1 in yawrange:

        # assign values to yaw
        prob['yaw0'] = np.array([yaw1, 0.0])

        # run the problem at given conditions
        prob.run()

        # print np.sqrt((turbineY[0]-turbineY[1])**2+(turbineX[0]-turbineX[1])**2)/rotorDiameter[0] # print downwind distance

        FLORISpower.append(list(prob['wtPower']))

    FLORISpower = np.array(FLORISpower)

    SOWFApower = np.array([ICOWESdata['yawPowerT1'][0], ICOWESdata['yawPowerT2'][0]]).transpose()/1000.

    error_turbine2 = np.sum(np.abs(FLORISpower[:, 1] - SOWFApower[:, 1]))

    error_turbine1 = np.sum(np.abs(FLORISpower[:, 0] - SOWFApower[:, 0]))

    error_total = np.sum(np.abs(np.sum(FLORISpower) - np.sum(SOWFApower)))

    posrange = ICOWESdata['pos'][0]

    prob['yaw0'] = np.array([0.0, 0.0])

    FLORISpower = list()

    for pos2 in posrange:
        # Define turbine locations and orientation
        effUdXY = 0.523599

        Xinit = np.array([1118.1, 1881.9])
        Yinit = np.array([1279.5, 1720.5])
        XY = np.array([Xinit, Yinit]) + np.dot(np.array([[np.cos(effUdXY), -np.sin(effUdXY)],
                                                        [np.sin(effUdXY), np.cos(effUdXY)]]),
                                               np.array([[0., 0], [0, pos2]]))

        prob['turbineX'] = XY[0, :]
        prob['turbineY'] = XY[1, :]

        # run the problem at given conditions
        prob.run()

        # print np.sqrt((turbineY[0]-turbineY[1])**2+(turbineX[0]-turbineX[1])**2)/rotorDiameter[0] # print downwind distance

        FLORISpower.append(list(prob['wtPower']))

    FLORISpower = np.array(FLORISpower)

    SOWFApower = np.array([ICOWESdata['posPowerT1'][0], ICOWESdata['posPowerT2'][0]]).transpose()/1000.

    error_turbine2 += np.sum(np.abs(FLORISpower[:, 1] - SOWFApower[:, 1]))
    error_turbine1 += np.sum(np.abs(FLORISpower[:, 0] - SOWFApower[:, 0]))
    error_total += np.sum(np.abs(np.sum(FLORISpower) - np.sum(SOWFApower)))

    funcs = {'obj': error_turbine2}
    # funcs = {'obj': error_turbine1}
    # funcs = {'obj': error_total}

    # funcs = {}

    # funcs['obj'] = error_turbine2
    # funcs['con'] = min(FLORISvelocity[:, 1] 2.91046369992
    fail = False

    # print error_turbine1
    print error_turbine2

    return funcs, fail


def plotSOWFAvsFLORIS(prob, just_SOWFA=True, plot_prefix=""):

    # prob['floris_params:me'] = params[0:3]
    # prob['floris_params:MU'] = params[3:6]
    # prob['floris_params:cos_spread'] = params[6]

    # Define turbine locations and orientation
    prob['turbineX'] = np.array([1118.1, 1881.9])
    prob['turbineY'] = np.array([1279.5, 1720.5])

    ICOWESdata = loadmat('YawPosResults.mat')
    yawrange = ICOWESdata['yaw'][0]

    FLORISpower = list()

    for yaw1 in yawrange:

        # print 'yaw in = ', yaw1*np.pi/180.

        # assign values to yaw
        prob['yaw0'] = np.array([yaw1, 0.0])

        # run the problem at given conditions
        prob.run()

        # print np.sqrt((turbineY[0]-turbineY[1])**2+(turbineX[0]-turbineX[1])**2)/rotorDiameter[0] # print downwind distance

        FLORISpower.append(list(prob['wtPower0']))

        time.sleep(0.001)

    # time.sleep(10)
    # quit()
    FLORISpower = np.array(FLORISpower)

    SOWFApower = np.array([ICOWESdata['yawPowerT1'][0], ICOWESdata['yawPowerT2'][0]]).transpose()/1000.

    error_turbine2 = np.sum(np.abs(FLORISpower[:, 1] - SOWFApower[:, 1]))

    fig, axes = plt.subplots(ncols=2, nrows=1, sharey=False)

    axes[0].plot(yawrange.transpose(), SOWFApower[:, 0], 'o', markeredgecolor='r', markerfacecolor='None',
                    markersize=4, label='Turbine 1, SOWFA Results')
    axes[0].plot(yawrange.transpose(), SOWFApower[:, 1], 'o', markeredgecolor='b', markerfacecolor='None',
                    markersize=4, label='Turbine 2, SOWFA Results')
    axes[0].plot(yawrange.transpose(), SOWFApower[:, 0] + SOWFApower[:, 1], 'k+', label='Total, SOWFA results')
    axes[0].plot(yawrange.transpose(), FLORISpower[:, 0], 'k-')
    axes[0].plot(yawrange.transpose(), FLORISpower[:, 1], 'k-')
    axes[0].plot(yawrange.transpose(), FLORISpower[:, 0]+FLORISpower[:, 1], 'k-', label='FLORIS model')

    # axes[0, 0].plot(yawrange, FLORISpower[:, 1], '#7CFC00')
    axes[0].set_xlabel('yaw angle (deg.)')
    axes[0].set_ylabel('power (kW)')
    # axes[0, 0].set_ylim(500, 6000)
    # axes[0, 0].legend(loc=2)


    posrange = ICOWESdata['pos'][0]

    prob['yaw0'] = np.array([0.0, 0.0])

    FLORISpower = list()

    for pos2 in posrange:
        # Define turbine locations and orientation
        effUdXY = 0.523599

        Xinit = np.array([1118.1, 1881.9])
        Yinit = np.array([1279.5, 1720.5])
        XY = np.array([Xinit, Yinit]) + np.dot(np.array([[np.cos(effUdXY), -np.sin(effUdXY)],
                                                        [np.sin(effUdXY), np.cos(effUdXY)]]),
                                               np.array([[0., 0], [0, pos2]]))

        prob['turbineX'] = XY[0, :]
        prob['turbineY'] = XY[1, :]

        # run the problem at given conditions
        prob.run()

        # print np.sqrt((turbineY[0]-turbineY[1])**2+(turbineX[0]-turbineX[1])**2)/rotorDiameter[0] # print downwind distance

        FLORISpower.append(list(prob['wtPower0']))

    FLORISpower = np.array(FLORISpower)

    SOWFApower = np.array([ICOWESdata['posPowerT1'][0], ICOWESdata['posPowerT2'][0]]).transpose()/1000.

    error_turbine2 += np.sum(np.abs(FLORISpower[:, 1] - SOWFApower[:, 1]))

    # print error_turbine2
    axes[1].plot(posrange/rotorDiameter[0], FLORISpower[:, 0], 'k-')
    axes[1].plot(posrange/rotorDiameter[0], SOWFApower[:, 0], 'o', markeredgecolor='r', markerfacecolor='None',
                    markersize=4, label='Turbine 1, SOWFA Results')
    axes[1].plot(posrange/rotorDiameter[0], FLORISpower[:, 1], 'k-')
    axes[1].plot(posrange/rotorDiameter[0], SOWFApower[:, 1], 'o', markeredgecolor='b', markerfacecolor='None',
                    markersize=4, label='Turbine 2, SOWFA Results')
    axes[1].plot(posrange/rotorDiameter[0], FLORISpower[:, 0]+FLORISpower[:, 1], 'k-', label='FLORIS model')
    axes[1].plot(posrange/rotorDiameter[0], SOWFApower[:, 0]+SOWFApower[:, 1], 'k+', label='Total, SOWFA Results')
    axes[1].set_xlabel('y/D')
    axes[1].set_ylabel('power (kW)')
    axes[1].set_xlim(min(posrange/rotorDiameter[0]), max(posrange/rotorDiameter[0]))
    # lgd = axes[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    lgd = axes[0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.05),
          fancybox=False, shadow=False, ncol=2)
    plt.tight_layout()

    if not plot_prefix == "":
        plt.savefig(plot_prefix+"SOWFA.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

    # ############
    if not just_SOWFA:
        fig, axes = plt.subplots(ncols=2, nrows=1, sharey=False)

        posrange = np.linspace(-3.*rotorDiameter[0], 30.*rotorDiameter[0], num=1000)
        yaw = np.array([0.0, 0.0])
        wind_direction = 270.

        prob['yaw0'] = yaw
        prob['wind_direction'] = wind_direction

        FLORISpower = list()
        FLORISvelocity = list()

        for pos2 in posrange:

            # assign values to yaw
            prob['turbineX'] = np.array([0., pos2])
            prob['turbineY'] = np.array([0.0, 0.0])

            # run the problem at given conditions
            prob.run()

            # print np.sqrt((turbineY[0]-turbineY[1])**2+(turbineX[0]-turbineX[1])**2)/rotorDiameter[0] # print downwind distance

            FLORISpower.append(list(prob['wtPower0']))
            FLORISvelocity.append(list(prob['wtVelocity0']))

        FLORISpower = np.array(FLORISpower)
        FLORISvelocity = np.array(FLORISvelocity)

        # np.savetxt("downstream_velocity_original.txt", np.c_[posrange/rotorDiameter[0], FLORISvelocity[:, 1]], delimiter=',')

        axes[1].plot(posrange/rotorDiameter[0], FLORISpower[:, 1], '#7CFC00', label='FLORIS model')
        axes[1].plot(np.array([7, 7]), np.array([0, 1800]), '--k', label='tuning point')
        axes[1].set_xlabel('x/D')
        axes[1].set_ylabel('Power (kW)')
        axes[1].legend(loc=4)

        axes[0].plot(posrange/rotorDiameter[0], FLORISvelocity[:, 1], '#7CFC00', label='FLORIS model')
        axes[0].plot(np.array([7, 7]), np.array([2, 9]), '--k', label='tuning point')
        axes[0].set_xlabel('x/D')
        axes[0].set_ylabel('Valocity (m/s)')
        axes[0].legend(loc=4)
        # plt.show()

        plt.tight_layout()
        if not plot_prefix == "":
            plt.savefig(plot_prefix+"DownwindProfile.pdf")

        FLORIScenters = list()
        FLORISdiameters = list()
        FLORISoverlap = list()
        # prob['wakeCentersYT'][2] = 0.
        for pos2 in posrange:

            # assign values to yaw
            prob['turbineX'] = np.array([0., pos2])
            # prob['turbineY'] = np.array([0.0, prob['wakeCentersYT'][2]])
            prob['turbineY'] = np.array([0.0, 0.0])

            # run the problem at given conditions
            prob.run()

            # print np.sqrt((turbineY[0]-turbineY[1])**2+(turbineX[0]-turbineX[1])**2)/rotorDiameter[0] # print downwind distance
            # print prob['wakeCentersYT']
            FLORIScenters.append(list(prob['wakeCentersYT']))
            FLORISdiameters.append(list(prob['wakeDiametersT']))
            FLORISoverlap.append(list(prob['wakeOverlapTRel']))
            # print prob['wtVelocity0']

            # print prob['wakeOverlapTRel'][6:]

        FLORIScenters = np.array(FLORIScenters)
        FLORISdiameters = np.array(FLORISdiameters)
        FLORISoverlap = np.array(FLORISoverlap)


        fig, axes = plt.subplots(ncols=2, nrows=2, sharey=False, sharex=False)
        # plot wake center
        axes[0, 0].plot(posrange/rotorDiameter[0], FLORIScenters[:, 2]/rotorDiameter[0], 'k', label='Wake Center')
        # axes[0, 0].set_xlabel('x/D')
        axes[0, 0].set_ylabel('y/D')
        axes[0, 0].legend(loc=1)

        # plot wake diameters
        axes[0, 1].plot(posrange/rotorDiameter[0], FLORISdiameters[:, 3*nTurbines+0]/rotorDiameter[0],
                        'b', label='Near Wake')
        axes[0, 1].plot(posrange/rotorDiameter[0], FLORISdiameters[:, 3*nTurbines+2]/rotorDiameter[0],
                        'r', label='Far Wake')
        axes[0, 1].plot(posrange/rotorDiameter[0], FLORISdiameters[:, 3*nTurbines+4]/rotorDiameter[0],
                        'y', label='Mixing Zone')
        # axes[0, 1].set_xlabel('x/D')
        axes[0, 1].set_ylabel('Wake Diameter / Rotor Diameter')
        axes[0, 1].legend(loc=2)
        axes[0, 1].set_ylim([-1., 5.])

        # plot wake relative overlap
        axes[1, 0].plot(posrange/rotorDiameter[0], FLORISoverlap[:, 3*nTurbines+0],
                        'b', label='Near Wake')
        axes[1, 0].plot(posrange/rotorDiameter[0], FLORISoverlap[:, 3*nTurbines+2],
                        'r', label='Far Wake')
        axes[1, 0].plot(posrange/rotorDiameter[0], FLORISoverlap[:, 3*nTurbines+4],
                        'y', label='Mixing Zone')
        axes[1, 0].set_xlabel('x/D')
        axes[1, 0].set_ylabel('Relative Overlap')
        axes[1, 0].legend(loc=0)
        axes[1, 0].set_ylim([-0.1, 1.1])


        posrange = np.linspace(-3.*rotorDiameter[0], 7.*rotorDiameter[0], num=300)
        yaw = np.array([0.0, 0.0])
        wind_direction = 270.

        prob['yaw0'] = yaw
        prob['wind_direction'] = wind_direction

        FLORISpower = list()

        for pos2 in posrange:

            # assign values to yaw
            prob['turbineX'] = np.array([0., 5*rotorDiameter[0]])
            prob['turbineY'] = np.array([0.0, pos2])

            # run the problem at given conditions
            prob.run()

            # print np.sqrt((turbineY[0]-turbineY[1])**2+(turbineX[0]-turbineX[1])**2)/rotorDiameter[0] # print downwind distance

            FLORISpower.append(list(prob['wtPower0']))

        FLORISpower = np.array(FLORISpower)

        axes[1, 1].plot(posrange/rotorDiameter[0], FLORISpower[:, 1], '#7CFC00')#, label="FLORIS model")
        axes[1, 1].set_xlabel('x/D')
        axes[1, 1].set_ylabel('Power (kW)')
        axes[1, 1].legend(loc=4)

        if not plot_prefix == "":
            plt.savefig(plot_prefix+"WakeProfile.pdf")

    #####################


    # FLORISaxialInd = list()
    # windspeed = np.linspace(0, 30, 100)
    # # prob['wakeCentersYT'][2] = 0.
    # for pos2 in posrange:
    #
    #     # assign values to yaw
    #     prob['turbineX'] = np.array([0., pos2])
    #     # prob['turbineY'] = np.array([0.0, prob['wakeCentersYT'][2]])
    #     prob['turbineY'] = np.array([0.0, 0.0])
    #
    #     # run the problem at given conditions
    #     prob.run()
    #
    #     # print np.sqrt((turbineY[0]-turbineY[1])**2+(turbineX[0]-turbineX[1])**2)/rotorDiameter[0] # print downwind distance
    #     # print prob['wakeCentersYT']
    #     FLORIScenters.append(list(prob['wakeCentersYT']))
    #     FLORISdiameters.append(list(prob['wakeDiametersT']))
    #     FLORISoverlap.append(list(prob['wakeOverlapTRel']))
    #     # print prob['wtVelocity0']
    #
    #     # print prob['wakeOverlapTRel'][6:]
    #
    # FLORIScenters = np.array(FLORIScenters)
    # FLORISdiameters = np.array(FLORISdiameters)
    # FLORISoverlap = np.array(FLORISoverlap)


    # fig, axes = plt.subplots(ncols=2, nrows=2, sharey=False, sharex=True)
    # # plot wake center
    # axes[0, 0].plot(posrange/rotorDiameter[0], FLORIScenters[:, 2], 'k', label='Wake Center')
    # # axes[0, 0].set_xlabel('x/D')
    # axes[0, 0].set_ylabel('Position')
    # axes[0, 0].legend(loc=1)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    config.floris_single_component = True

    model = 'smooth'  # options: 'original', 'smooth'

    gradients = 'fd'    # options: 'fd', 'exact'
    flat = True        # if False, will use cosine smoothing factor
    rotor = False       # if True, will use rotor coupled data
    tune = False        # if True, will optimize parameters starting with provided values

    # plot option
    just_SOWFA = False
    # plot_file_prefix = "cosineRotorHandTuned"
    plot_file_prefix = ""

    # #############################  Initialize parameters ###############################

     # power
    pP = 1.88   # control Cp adjustment to yaw

    # deflection
    kd = 0.15                       # adjust yaw deflection
    global initialWakeDisplacement
    initialWakeDisplacement = -4.5  # initial rotational displacement
    bd = -0.01                      # continued deflection from rotation as separation increases
    # initialWakeAngle = 0.5*3.0
    initialWakeAngle = 1.5

    # expansion
    ke = 0.065                          # adjust overall rate of wake expansion
    global me
    me = np.array([-0.5, 0.22, 1.0])    # adjust individual wake expansion

    # velocity
    global MU
    MU = np.array([0.5, 1.0, 5.5])      # zone velocity deficit decay rate

    aU = 5.0                            # offset in decay adjustment
    bU = 1.66                           # parameter of yaw on decay adjustment
    cos_spread = 1e12                   # additional deficit based on crosswind relative location

    if rotor:
        # deflection
        kd = 0.17                   # adjust yaw deflection

        # expansion
        ke = 0.05                   # adjust overall rate of wake expansion

        # velocity
        aU = 12.                    # offset in decay adjustment
        bU = 1.3                    # parameter of yaw on decay adjustment

    if not flat:
        # velocity
        cos_spread = 2.0                    # additional deficit based on crosswind relative location
        MU = np.array([0.5, 1.0, 5.5])      # zone velocity deficit decay rate
        me = np.array([-0.5, 0.3, 1.0])    # adjust individual wake expansion
        initialWakeAngle = 1.5

    # ###################################################################################

    params = np.array([pP, kd, initialWakeAngle, bd, ke,   me[0], me[1], MU[0], MU[2], aU,  bU,  cos_spread])

    if rotor:
        NREL5MWCPCT = pickle.load(open('NREL5MWCPCT_smooth_dict.p'))
        # NREL5MWCPCT = pickle.load(open('NREL5MWCPCT_dict.p'))
        datasize = NREL5MWCPCT['CP'].size
    else:
        datasize = 0

    if model == 'smooth':
        differentiable = True
    else:
        differentiable = False

    # initialize input variable arrays
    nTurbines = 2
    rotorDiameter = np.zeros(nTurbines)
    axialInduction = np.zeros(nTurbines)
    Ct = np.zeros(nTurbines)
    Cp = np.zeros(nTurbines)
    generatorEfficiency = np.zeros(nTurbines)
    yaw = np.zeros(nTurbines)

    # define initial values
    for turbI in range(0, nTurbines):
        rotorDiameter[turbI] = 126.4            # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = (0.7737/0.944) * 4.0 * 1.0/3.0 * np.power((1. - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 0.944
        yaw[turbI] = 0.     # deg.
    print "initial Cp", Cp
    print "initial Ct", Ct

    # Define flow properties
    if rotor:
        wind_speed = 8.1        # m/s
    else:
        wind_speed = 8.0        # m/s
    air_density = 1.1716        # kg/m^3
    wind_direction = 270.-0.523599*180./np.pi    # deg (N = 0 deg., using direction FROM, as in met-mast data)
    print wind_direction, -0.523599*180./np.pi

    # initialize problem
    global prob
    prob = Problem(root=Group())

    prob.root.add('FLORIS', DirectionGroup(nTurbines, use_rotor_components=rotor,
                                           datasize=datasize, differentiable=differentiable), promotes=['*'])

    # set up problem
    prob.setup(check=False)
    print 'setup complete'

    if rotor:
        prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
        prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
        prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
    else:
        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp
        prob['gen_params:CTcorrected'] = False
        prob['gen_params:CPcorrected'] = False

    if model is not "original":
        prob['floris_params:useWakeAngle'] = True
        prob['floris_params:adjustInitialWakeDiamToYaw'] = False
        prob['floris_params:useaUbU'] = True

    if rotor:
        prob['floris_params:useWakeAngle'] = True
        prob['floris_params:adjustInitialWakeDiamToYaw'] = False
        prob['floris_params:axialIndProvided'] = False
        prob['floris_params:useaUbU'] = True


    # set tuning variables
    prob['gen_params:pP'] = params[0]
    prob['floris_params:kd'] = params[1]
    prob['floris_params:initialWakeAngle'] = params[2]
    prob['floris_params:initialWakeDisplacement'] = initialWakeDisplacement
    prob['floris_params:bd'] = params[3]
    prob['floris_params:ke'] = params[4]
    prob['floris_params:me'] = np.append(params[5:7], me[2])
    prob['floris_params:MU'] = np.insert(params[7:9], 1, MU[1])
    prob['floris_params:aU'] = params[9]
    prob['floris_params:bU'] = params[10]
    prob['floris_params:cos_spread'] = params[11]

    # assign values to constant inputs
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['wind_speed'] = wind_speed
    prob['air_density'] = air_density
    prob['wind_direction'] = wind_direction

    if tune:

        # define optimization object
        optProb = Optimization('FLORIS Tuning', objfunction)

        # define design variables
        lower = [0.0, 0.0,            -10.0, -1.0, 0.0,  -10.,     0.,   0.0,   1.5,   0.,  0.0,         0.0]
        upper = [5.0, 1.0,             10.0,  1.0, 1.0,    0.,    0.3,   1.0,  20.0,  20.,  5.0,         2.0]
        value = [pP,   kd, initialWakeAngle,   bd,  ke, me[0],  me[1], MU[0], MU[2],   aU,   bU,  cos_spread]


        if flat:
            # turn off cos_spread effect
            lower[-1] = 1E12
            upper[-1] = 1E14
            value[-1] = 1E13

        optProb.addVarGroup('xvars', 12, lower=lower, upper=upper, value=value)

        # Objective
        optProb.addObj('obj')

        # Optimizer
        snopt = SNOPT()
        # opt = OPT('NSGA2')
        # opt.setOption('PopSize', 200)

        # Solve
        sol = snopt(optProb, sens='CD')

        # print results
        print sol

        value = sol.variables['xvars']
        params = np.zeros(14)
        count = 0
        for val in value['xvars']:
            params[count] = val.value

            print params[count]
            count += 1

         # set tuning variables
        prob['gen_params:pP'] = params[0]
        prob['floris_params:kd'] = params[1]
        prob['floris_params:initialWakeAngle'] = params[2]
        prob['floris_params:initialWakeDisplacement'] = initialWakeDisplacement
        prob['floris_params:bd'] = params[3]
        prob['floris_params:ke'] = params[4]
        prob['floris_params:me'] = np.append(params[5:7], me[2])
        prob['floris_params:MU'] = np.insert(params[7:9], 1, MU[1])
        prob['floris_params:aU'] = params[9]
        prob['floris_params:bU'] = params[10]
        prob['floris_params:cos_spread'] = params[11]

    # params = value
    plotSOWFAvsFLORIS(prob, just_SOWFA=just_SOWFA, plot_prefix=plot_file_prefix)
