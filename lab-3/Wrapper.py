import copy
from Data import params
from Neural import Neural
from Plotter import Plotter
from Utility import Utility
from RealObject import RealObject
from SignalGenerator import SignalGenerator
from ParameterEstimator import ParameterEstimator


class Wrapper():
    def __init__(self):
        self.obj = RealObject()
        self.sig = SignalGenerator()
        self.neural = Neural()

        self.u_func = None
        self.sol = None
        self.t = None
        self.u = None

        self.options = {
            "title": None,
            "labels": {
                "x": "t",
                "y": "Amplitude",
            },
            "limits": {
                "x": None,
                "y": None,
            },
            "legend": {
                "blue": None,
                "orange": None,
            },
        }

        self.data = {}

        self.signalInstance = None

        self.to_predict = params['models']['neural']['want_to_predict'][1:]

    def object_free_movements(self):
        sol, t = self.obj.calcODE()
        options = copy.deepcopy(self.options)
        options['title'] = "Object free movements"
        Plotter.draw([t, sol], options=options)

        return self

    def do_experiment(self):
        self.obj.set_u_fcn(self.u_func)

        signalParams = params['signals'][self.signalInstance]

        self.sol, self.t = self.obj.calcODE()
        self.u = self.u_func(0, self.t)

        options = copy.deepcopy(self.options)
        options['title'] = "Experiment with " + \
            self.signalInstance.replace('_', ' ') + " signal"
        options['legend']['blue'] = "Response"
        options['legend']['orange'] = "Input"

        Plotter.draw([self.t, [self.sol, self.u]], options=options)

        Utility.checkProperty('experiment', self.data)
        Utility.checkProperty('sig_params', self.data)
        Utility.checkProperty(self.signalInstance, self.data['sig_params'])

        self.data['experiment'][self.signalInstance] = {
            "guess": [self.obj.lin_par_1, self.obj.lin_par_2, self.obj.nonlin_par_1, self.obj.nonlin_par_2],
            "y0": [self.obj.y0],
        }

        for prop in signalParams.keys():
            self.data['sig_params'][self.signalInstance][prop] = signalParams[prop] or None

        return self

    def model_and_analyzing(self):
        def ode_ideal():
            return self.obj.getODE()

        def ode_linear():
            def decor(x, t, k):
                y = x
                [K, T] = k

                u = self.u_func(x, t)

                dydt = (K*u-y)/T
                return dydt

            return decor

        def ode_non_linear():
            def decor(x, t, k):
                y = x
                [K, T, non_lin_1, non_lin_2] = k

                u = self.u_func(x, t)

                dydt = (K*self.obj.nonlin_fcns[self.obj.nonlin_type]
                        (u, non_lin_1, non_lin_2) - y)/T
                return dydt

            return decor

        def analyze(guess, y0, ode_type):
            func = ode_func_map[ode_type]
            to_estimate_init_values = {'guess': guess, 'y0': [y0, ]}

            estimator = ParameterEstimator(
                self.experiments, to_estimate_init_values, func)

            sol_ideal, sol_guess, sol_y0 = estimator.get_ideal_solution(func)

            options = copy.deepcopy(self.options)
            options['title'] = "System and " + \
                ode_type + " model response to " + self.signalInstance + " signal"
            options['legend']['blue'] = "System response"
            options['legend']['orange'] = ode_type + " model response"

            Plotter.draw([self.t, [self.sol, sol_ideal]], options=options)

            Utility.checkProperty('model', self.data)
            Utility.checkProperty(self.signalInstance, self.data['model'])
            '''if not 'model' in self.data:
                self.data['model'] = {}
            if not self.signalInstance in self.data['model']:
                self.data['model'][self.signalInstance] = {}'''
            self.data['model'][self.signalInstance][ode_type] = {
                "initial": {
                    "guess": guess,
                    "y0": [y0],
                },
                "estimated": {
                    "guess": sol_guess.tolist(),
                    "y0": sol_y0.tolist(),
                },
            }

        ode_func_map = {
            'ideal': ode_ideal(),
            'linear': ode_linear(),
            'non_linear': ode_non_linear(),
        }

        self.experiments = [[self.t, self.sol], ]

        for ode_type in ode_func_map.keys():
            guess = params['models'][ode_type]['guess']
            y0 = params['models'][ode_type]['initial_condition']

            analyze(guess, y0, ode_type)

        return self

    def neural_analyzing(self, to_drop=False):
        self.neural.setSignal(self.u_func, self.sol, self.t)
        self.neural.series2dataset().expandDimensions()
        self.neural.setInput(to_drop).setOutput(to_drop).scaleOutput(-1, 1)
        self.neural_construct_model_and_predict()

    def neural_common_invoke(self):
        self.neural.setCommonInvoke()
        self.neural.scaleOutput()
        self.neural_construct_model_and_predict()
        self.neural.setCommonInvoke()

    def neural_construct_model_and_predict(self):
        history_options = copy.deepcopy(self.options)
        history_options['title'] = "Train and test result of all signals without " + self.to_predict + " signal" if self.neural.isCommonInvokeTime else "Train and test result of " + \
            self.signalInstance + " signal"
        history_options['labels']['x'] = "Epoch"
        history_options['labels']['y'] = "Loss"
        history_options['legend']['blue'] = "Train"
        history_options['legend']['orange'] = "Test"

        predict_options = copy.deepcopy(self.options)
        predict_options['title'] = "Predicted with common neural model " + self.to_predict + " signal" if self.neural.isCommonInvokeTime else "Predicted with neural model " + \
            self.signalInstance + " signal"
        predict_options['labels']['x'] = "Count"
        if not self.neural.isCommonInvokeTime:
            predict_options['legend']['blue'] = "System response"
            predict_options['legend']['orange'] = "Neural model response"

        self.neural.separateSequenses()
        self.neural.modelConstruct(history_options)
        self.neural.invertedScaling()
        self.neural.predict(predict_options)

    def start(self):
        for param in ['variant', 'duration', 'discretization']:
            self.data[param] = params['numeric'][param]

        self.data['want_to_predict'] = params['models']['neural']['want_to_predict']
        self.data['method'] = params['models']['neural']['method']

        self.object_free_movements()

        for sig_name in params['signals'].keys():
            self.signalInstance = sig_name

            print("Signal name is {}".format(sig_name))

            self.u_func = self.sig.get_u(sig_name)

            self.do_experiment()

            self.model_and_analyzing()

            to_drop = False

            if sig_name == self.to_predict:
                to_drop = True

            self.neural_analyzing(to_drop=to_drop)

        self.neural_common_invoke()

        Utility.saveData(self.data)
