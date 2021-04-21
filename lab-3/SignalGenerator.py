from Data import params
import numpy as np
from scipy import signal


class SignalGenerator:
    def __init__(self):
        self.duration = params['numeric']['duration']
        self.signals = params['signals']

        self.u = {
            'monoharm': self.monoharm_u,
            'impulse': self.impulse_u,
            'square': self.square_u,
        }

    def monoharm_u(self, x, t):
        ins = self.signals['monoharm']

        return ins['amplitude']*np.sin(ins['to_approximate'] * t * 2 * np.pi)

    def impulse_u(self, x, t):
        def suppression(sig):
            return sig if np.abs(sig) > ins['amplitude']/2 else 0

        ins = self.signals['impulse']
        shift = ins['shift'] * self.duration

        to_suppres = np.vectorize(suppression)

        return to_suppres(ins['amplitude'] * np.sinc(ins['to_approximate'] * (t - shift) * 2 * np.pi))

    def square_u(self, x, t):
        ins = self.signals['square']

        return ins['amplitude']*signal.square(ins['to_approximate'] * t * 2 * np.pi)

    def get_u(self, sig):
        return self.u[sig]
