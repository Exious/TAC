from re import U
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


def do_experiment():
    # obj.set_u_fcn(monoharm_u)
    # obj.set_u_fcn(impulse_u)
    obj.set_u_fcn(meandr_u)

    sol, t = obj.calcODE()

    #u = monoharm_u(0, t)
    #u = impulse_u(0, t)
    u = meandr_u(0, t)

    Plotter.draw([t, [sol, u]])

    return sol, t


def model_and_analyzing():
    def ode_ideal():
        return obj.getODE()

    def ode_lin():
        def decor(x, t, k):
            y = x
            [K, T] = k
            u = meandr_u(0, t)
            dydt = (K*u-y)/T
            return dydt

        return decor

    def analyze(guess, y0, func):
        experiments = [[t, sol], ]

        to_estimate_init_values = {'guess': guess, 'y0': [y0, ]}

        estimator = ParameterEstimator(
            experiments, to_estimate_init_values, func)

        sol_ideal = estimator.get_ideal_solution(func)

        Plotter.draw([t, [sol, sol_ideal]])

    ode_func_map = {
        'ode_ideal': ode_ideal(),
        'ode_lin': ode_lin()
    }

    guess = params['models']['ideal']['guess']
    y0 = params['models']['ideal']['initial_condition']
    func = params['models']['ideal']['func']

    analyze(guess, y0, ode_func_map[str(func)])

    guess = params['models']['linear']['guess']
    y0 = params['models']['linear']['initial_condition']
    func = params['models']['linear']['func']

    analyze(guess, y0, ode_func_map[str(func)])


obj = RealObject()
sig = SignalGenerator()

object_free_movements()
monoharm_u = sig.get_u('monoharm')
impulse_u = sig.get_u('impulse')
meandr_u = sig.get_u('meandr')
sol, t = do_experiment()
model_and_analyzing()


plt.show()
