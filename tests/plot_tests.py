import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
from akima import akima_interp

def plot_power_curve_with_spline_on_cut_in_speed():

    nSpeeds = 100000
    vmax = 30.

    # import component
    from plantenergy.GeneralWindFarmComponents import WindDirectionPower

    # load cp data
    power_data = np.loadtxt('./input_files/niayifar_vestas_v80_power_curve_observed.txt', delimiter=',')
    for i in np.arange(20, 25, .1):
        power_data = np.append(power_data, np.array([[i, 2.0]]), axis=0)
    power_data = np.append(power_data, np.array([[25.01, 0.0]]), axis=0)
    rotor_diameter = 80.0
    air_density = 1.1716
    Ar = 0.25 * np.pi * rotor_diameter**2
    cp_curve_cp = 2.0*power_data[:, 1]*1E6/(air_density*Ar*(power_data[:, 0])**3)
    cp_curve_v = power_data[:, 0]
    # print(0.5*air_density*cp_curve_cp*Ar*cp_curve_v**3)
    # quit()

    # power_data = np.loadtxt("./input_files/v80powercurve.csv", delimiter=",")
    # print(power_data)
    # rotor_diameter = 80.
    # air_density = 1.225
    # Ar = 0.25 * np.pi * rotor_diameter ** 2
    # cp_curve_cp = power_data[:, 1] * (1E3) / (0.5 * air_density * Ar * power_data[:, 0] ** 3)
    # cp_curve_v = power_data[:, 0]

    print(cp_curve_cp)
    # power_data = np.loadtxt("./input_files/NREL5MWCPCT.txt")
    # cp_curve_cp = power_data[:, 1]
    # cp_curve_v = power_data[:, 0]
    # define openmdao problem
    prob = om.Problem()
    prob.model.add_subsystem('power', WindDirectionPower(nTurbines=nSpeeds, cp_points=cp_curve_cp.size), promotes=['*'])

    # initialize openmdao problem
    prob.setup(check=False)


    # set inflow velocity
    prob['wtVelocity0'] = np.linspace(0, vmax, nSpeeds)

    # set up power curve
    prob['cp_curve_cp'] = cp_curve_cp
    prob['cp_curve_wind_speed'] = cp_curve_v

    cutin = 4.0
    cutout = 25.
    ratedws = 16.
    ratedp = 2E3

    # cutin = 3.  # m/s
    # cutout = 25.  # m/s
    # ratedws = 11.4  # m/s
    # ratedp = 5000.  # kW

    # assign values to turbine specs
    prob['cut_in_speed'] = np.ones(nSpeeds) * cutin
    prob['cut_out_speed'] = np.ones(nSpeeds) * cutout
    prob['rated_power'] = np.ones(nSpeeds) * ratedp
    prob['rated_wind_speed'] = np.ones(nSpeeds) * ratedws
    prob['generatorEfficiency'] = np.ones(nSpeeds)
    prob['rotorDiameter'] = np.ones(nSpeeds)*rotor_diameter
    prob['use_power_curve_definition'] = False

    # run the problem
    prob.run_model()

    def power_curve(velocity, cutin, cutout, ratedws, ratedp):

        power = np.zeros_like(velocity)
        for i in np.arange(0, velocity.size):
            if velocity[i] > cutin and velocity[i] < ratedws:
                power[i] = ratedp * ((velocity[i] ) / (ratedws - cutin)) ** 3
            elif velocity[i] > ratedws and velocity[i] < cutout:
                power[i] = ratedp

            if power[i]>ratedp:
                power[i]=ratedp

        return power

    velocities = prob['wtVelocity0']
    prob_power = prob['wtPower0']
    func_power = power_curve(velocities, cutin, cutout, ratedws, ratedp)

    check_power = 0.5*cp_curve_cp*air_density*Ar*cp_curve_v**3

    # data = np.loadtxt('./input_files/v80powercurve.csv', delimiter=",")
    data = np.loadtxt('./input_files/niayifar_vestas_v80_power_curve_observed.txt', delimiter=",")

    scaler = 1E-3
    plt.plot(velocities, prob_power*scaler, 'k', label='Model')
    # plt.plot(cp_curve_v, cp_curve_cp, 'b:')
    # plt.plot(velocities, func_power*scaler, 'r--', label="function")
    plt.plot(data[:, 0], data[:, 1], '--r', label='Data')
    # plt.plot(cp_curve_v, check_power*1E-6, ':r', label='Check')
    # plt.plot(data[:, 0], data[:, 1], '--k', label='Data')
    plt.legend(frameon=False)
    plt.ylabel('Wind Turbine Power (MW)')
    plt.xlabel('Inflow Velocity (m/s)')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.tick_params(top='none', right='none')
    plt.tight_layout()
    plt.savefig('v80power.pdf', transparent=True)

    plt.show()

if __name__ == "__main__":


    font = {'size': 15}
    plt.rc('font', **font)

    plt.rcParams['xtick.major.pad'] = '8'
    plt.rcParams['ytick.major.pad'] = '8'
    plot_power_curve_with_spline_on_cut_in_speed()