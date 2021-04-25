from Data import params
from Neural import Neural
from Plotter import Plotter
from RealObject import RealObject
from SignalGenerator import SignalGenerator
from ParameterEstimator import ParameterEstimator
import matplotlib.pyplot as plt


class Wrapper():
    def __init__(self):
        self.obj = RealObject()
        self.sig = SignalGenerator()
        self.neural = Neural()

        self.u_func = None
        self.sol = None
        self.t = None
        self.u = None

        self.to_predict = 'monoharm'

    def object_free_movements(self):
        sol, t = self.obj.calcODE()

        Plotter.draw([t, sol])

        return self

    def do_experiment(self):
        self.obj.set_u_fcn(self.u_func)

        self.sol, self.t = self.obj.calcODE()
        self.u = self.u_func(0, self.t)

        Plotter.draw([self.t, [self.sol, self.u]])

        return self

    def model_and_analyzing(self):
        def ode_ideal():
            return self.obj.getODE()

        def ode_lin():
            def decor(x, t, k):
                y = x
                [K, T] = k

                u = self.u_func(x, t)

                dydt = (K*u-y)/T
                return dydt

            return decor

        def ode_non_lin():
            def decor(x, t, k):
                y = x
                [K, T, non_lin_1, non_lin_2] = k

                u = self.u_func(x, t)

                dydt = (K*self.obj.nonlin_fcns[self.obj.nonlin_type]
                        (u, non_lin_1, non_lin_2) - y)/T
                return dydt

            return decor

        def analyze(guess, y0, func):
            to_estimate_init_values = {'guess': guess, 'y0': [y0, ]}

            estimator = ParameterEstimator(
                self.experiments, to_estimate_init_values, func)

            sol_ideal = estimator.get_ideal_solution(func)

            Plotter.draw([self.t, [self.sol, sol_ideal]])

        ode_func_map = {
            'ideal': ode_ideal(),
            'lin': ode_lin(),
            'non_lin': ode_non_lin(),
        }

        self.experiments = [[self.t, self.sol], ]

        for ode_type in ode_func_map.keys():
            guess = params['models'][ode_type]['guess']
            y0 = params['models'][ode_type]['initial_condition']

            analyze(guess, y0, ode_func_map[ode_type])

        return self

    def neural_model_and_analyzing(self, to_drop=False):
        self.neural.setSignal(self.u_func, self.sol, self.t)
        self.neural.series2dataset().expandDimensions()
        self.neural.setInput(to_drop).setOutput(to_drop)
        self.neural.separateSequenses()
        self.neural.modelConstruct()
        self.neural.invertedScaling()
        self.neural.predict()

    def start(self):
        self.object_free_movements()

        # for sig_name in params['signals'].keys():
        # for sig_name in ['step', 'monoharm']:
        for sig_name in ['step']:
            print("Signal name is {}".format(sig_name))

            self.u_func = self.sig.get_u(sig_name)

            self.do_experiment()

            # self.model_and_analyzing()

            to_drop = False

            if sig_name == self.to_predict:
                to_drop = True
            #    continue

            self.neural_model_and_analyzing(to_drop=to_drop)

        #print(neural.common['input'][996], neural.common['output'][996])
        print(self.neural.common['input'].shape,
              self.neural.scaler.inverse_transform(self.neural.common['output']))
        # plt.show()
