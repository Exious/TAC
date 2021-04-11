from ParameterEstimator import ParameterEstimator
from Plotter import Plotter
from RealObject import RealObject
from SignalGenerator import SignalGenerator
import matplotlib.pyplot as plt

variant = 5


def object_free_movements():
    sol, t = obj.calcODE()

    Plotter.draw([t, sol])


def return_monoharm_u():
    return SignalGenerator.get_monoharm_u()


def do_experiment():
    obj.set_u_fcn(monoharm_u)

    sol, t = obj.calcODE()

    u = monoharm_u(0, t)

    Plotter.draw([t, [sol, u]])

    return sol, t


def model_and_analyzing():
    def ode_lin(x, t, k):
        y = x
        [K, T] = k
        u = monoharm_u(0, t)
        dydt = (K*u-y)/T
        return dydt

    def analyze(guess, y0, func):
        experiments = [[t, sol], ]

        to_estimate_init_values = {'guess': guess, 'y0': y0}

        estimator = ParameterEstimator(
            experiments, to_estimate_init_values, func)

        sol_ideal = estimator.get_ideal_solution(func)

        Plotter.draw([t, [sol, sol_ideal]])

    guess = [1.1, 0.25, 1.75, 2.2]
    y0 = [1, ]

    analyze(guess, y0, obj.getODE())

    guess = [0.2, 0.3]
    y0 = [0, ]

    analyze(guess, y0, ode_lin)


obj = RealObject(variant)

object_free_movements()
monoharm_u = return_monoharm_u()
sol, t = do_experiment()
model_and_analyzing()


plt.show()
