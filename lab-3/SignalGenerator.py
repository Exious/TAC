from numpy.lib import isin
from Data import params
import numpy as np
from scipy import signal


class SignalGenerator:
    def __init__(self):
        self.amplitude = params['numeric']['amplitude']
        self.duration = params['numeric']['duration']
        self.shift = 0.5 * self.duration
        self.to_approximate = 15

    def monoharm_u(self, x, t):
        return self.amplitude*np.sin(self.to_approximate * t * 2 * np.pi)

    def impulse_u(self, x, t):
        def suppression(sig):
            if not isinstance(sig, np.ndarray):
                return sig if np.abs(sig) > self.amplitude/2 else 0
            return [dot if np.abs(dot) > self.amplitude/2 else 0 for dot in sig]

        return suppression(self.amplitude * np.sinc(self.to_approximate * (t - self.shift) * 2 * np.pi))

    def get_monoharm_u(self):
        return self.monoharm_u

    def get_impulse_u(self):
        return self.impulse_u
