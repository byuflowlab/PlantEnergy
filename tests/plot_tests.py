import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt

def plot_power_curve_with_spline_on_cut_in_speed():

    nSpeeds = 100000
    vmax = 30.

    # import component
    from plantenergy.GeneralWindFarmComponents import WindDirectionPower

    # load cp data
    power_data = np.loadtxt('./input_files/niayifar_vestas_v80_power_curve_observed.txt', delimiter=',')
    rotor_diameter = 80.
    air_density = 1.225
    Ar = 0.25 * np.pi * rotor_diameter ** 2
    cp_curve_cp = power_data[:, 1] * (1E6) / (0.5 * air_density * power_data[:, 0] ** 3 * Ar)
    cp_curve_v = power_data[:, 0]

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
    ratedws = 9.8
    ratedp = 3.35E6

    # assign values to turbine states
    prob['cut_in_speed'] = np.ones(nSpeeds) * cutin
    prob['cut_out_speed'] = np.ones(nSpeeds) * cutout
    prob['rated_power'] = np.ones(nSpeeds) * ratedp
    prob['rated_wind_speed'] = np.ones(nSpeeds) * ratedws
    prob['use_power_curve_definition'] = False

    # run the problem
    prob.run_model()

    def power_curve(velocity, cutin, cutout, ratedws, ratedp):

        power = np.zeros_like(velocity)

        for i in np.arange(0, velocity.size):
            if velocity[i] > cutin and velocity[i] < ratedws:
                power[i] = ratedp * ((velocity[i] - cutin) / (ratedws - cutin)) ** 3
            elif velocity[i] > ratedws and velocity[i] < cutout:
                power[i] = ratedp

        return power

    velocities = np.linspace(0., vmax, nSpeeds)
    prob_power = np.zeros_like(velocities)
    func_power = np.zeros_like(velocities)

    velocities = prob['wtVelocity0']
    prob_power = prob['wtPower0']
    func_power = power_curve(velocities, cutin, cutout, ratedws, ratedp)

    scaler = 1E-6
    plt.plot(velocities, prob_power*scaler, 'r')
    plt.plot(velocities, func_power*scaler, 'k--')
    plt.ylabel('MW')
    plt.xlabel('m/s')
    plt.show()

if __name__ == "__main__":
    plot_power_curve_with_spline_on_cut_in_speed()