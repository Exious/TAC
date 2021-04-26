import numpy as np
from scipy import signal
from Data import params


class SignalGenerator:
    def __init__(self):
        numeric_params = params['numeric']
        self.duration = numeric_params['duration']
        self.discretization = numeric_params['discretization']
        self.signals = params['signals']

        self.u = {
            'monoharm': self.monoharm_u,
            'impulse': self.impulse_u,
            'sinc_impulse': self.sinc_impulse_u,
            'square': self.square_u,
            'step': self.step_u,
        }

    def monoharm_u(self, x, t):
        ins = self.signals['monoharm']

        return ins['amplitude']*np.sin(ins['to_approximate'] * t * 2 * np.pi)

    def impulse_u(self, x, t):
        def decor(x, t):
            if t == 0:
                return ins['amplitude']
            else:
                return 0

        ins = self.signals['impulse']

        return np.vectorize(decor)(x, t)

    def sinc_impulse_u(self, x, t):
        def suppression(sig):
            return sig if np.abs(sig) > ins['amplitude']/2 else 0

        ins = self.signals['sinc_impulse']
        shift = ins['shift'] * self.duration

        to_suppres = np.vectorize(suppression)

        return to_suppres(ins['amplitude'] * np.sinc(ins['to_approximate'] * (t - shift) * 2 * np.pi))

    def square_u(self, x, t):
        ins = self.signals['square']

        return ins['amplitude']*signal.square(ins['to_approximate'] * t * 2 * np.pi)

    def step_u(self, x, t):
        def decor(x, t):
            if t > shift:
                return ins['amplitude']
            else:
                return 0

        ins = self.signals['step']
        shift = ins['shift'] * self.duration

        return np.vectorize(decor)(x, t)

    def get_u(self, sig):
        return self.u[sig]
