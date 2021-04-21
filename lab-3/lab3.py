from Data import params
from ParameterEstimator import ParameterEstimator
from Plotter import Plotter
from RealObject import RealObject
from SignalGenerator import SignalGenerator
import matplotlib.pyplot as plt
from sympy import *


def object_free_movements():
    sol, t = obj.calcODE()

    Plotter.draw([t, sol])

    return sol, t


def do_experiment():
    obj.set_u_fcn(u_func)

    sol, t = obj.calcODE()
    u = u_func(0, t)

    Plotter.draw([t, [sol, u]])

    return sol, t


def model_and_analyzing():
    def ode_ideal():
        return obj.getODE()

    def ode_lin():
        def decor(x, t, k):
            y = x
            [K, T] = k

            u = u_func(x, t)

            dydt = (K*u-y)/T
            return dydt

        return decor

    def ode_non_lin():
        def decor(x, t, k):
            y = x
            [K, T] = k

            u = u_func(x, t)

            import numpy as np
            dydt = (K*np.cos(u)-y)/T
            return dydt

        return decor

    def analyze(guess, y0, func):
        to_estimate_init_values = {'guess': guess, 'y0': [y0, ]}

        estimator = ParameterEstimator(
            experiments, to_estimate_init_values, func)

        sol_ideal = estimator.get_ideal_solution(func)

        Plotter.draw([t, [sol, sol_ideal]])

    ode_func_map = {
        'ideal': ode_ideal(),
        'lin': ode_lin(),
        # 'non_lin': ode_non_lin(),
    }

    experiments = [[t, sol], ]

    for ode_type in ode_func_map.keys():
        guess = params['models'][ode_type]['guess']
        y0 = params['models'][ode_type]['initial_condition']

        analyze(guess, y0, ode_func_map[ode_type])


obj = RealObject()
sig = SignalGenerator()

sol, t = object_free_movements()

for sig_name in ['monoharm', 'impulse', 'square']:
    print("Signal name is {}".format(sig_name))

    u_func = sig.get_u(sig_name)

    sol, t = do_experiment()

    model_and_analyzing()


plt.show()
