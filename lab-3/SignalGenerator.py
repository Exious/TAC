from numpy.lib import isin
from Data import params
import numpy as np
from scipy import signal


class SignalGenerator:
    def __init__(self):
        self.amplitude = params['numeric']['amplitude']
        self.duration = params['numeric']['duration']
        self.shift = params['numeric']['shift_coeff'] * self.duration
        self.to_approximate = params['numeric']['to_approximate']
        self.u = {
            'monoharm': self.monoharm_u,
            'impulse': self.impulse_u,
        }

    def monoharm_u(self, x, t):
        return self.amplitude*np.sin(self.to_approximate * t * 2 * np.pi)

    def impulse_u(self, x, t):
        def suppression(sig):
            if not isinstance(sig, np.ndarray):
                return sig if np.abs(sig) > self.amplitude/2 else 0
            return [dot if np.abs(dot) > self.amplitude/2 else 0 for dot in sig]

        return suppression(self.amplitude * np.sinc(self.to_approximate * (t - self.shift) * 2 * np.pi))

    def get_u(self, sig):
        return self.u[sig]
