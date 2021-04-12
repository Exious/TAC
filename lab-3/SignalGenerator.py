from numpy.lib import isin
from Data import params
import numpy as np
from scipy import signal


class SignalGenerator:
    def __init__(self, pike_index=None):
        self.amplitude = params['numeric']['amplitude']
        self.discretization = params['numeric']['discretization']
        self.duration = params['numeric']['duration']
        self.shift = 0.5 * self.duration
        self.to_approximate = 15
        self.pike_index = int(
            np.random.sample()*self.discretization) if not bool(pike_index) else pike_index
        self.impulse_pike_counter = 0
        self.random_coeff = 0.3

    def monoharm_u(self, x, t):
        return self.amplitude*np.sin(self.to_approximate * t * 2 * np.pi)

    def impulse_u(self, x, t):
        '''if np.random.sample() < self.random_coeff:
            self.impulse_pike_counter += 1

        if not isinstance(t, np.ndarray):
            return self.amplitude if self.impulse_pike_counter == 1 else 0

        return [dot*self.amplitude for dot in signal.unit_impulse(
            self.discretization, self.pike_index)]'''
        def suppression(sig):
            if not isinstance(sig, np.ndarray):
                return sig if np.abs(sig) > self.amplitude/2 else 0
            return [dot if np.abs(dot) > self.amplitude/2 else 0 for dot in sig]

        return suppression(self.amplitude * np.sinc(self.to_approximate * (t - self.shift) * 2 * np.pi))
        # sig = self.amplitude * \
        #    np.sinc(self.to_approximate * (t - self.shift) * 2 * np.pi)
        # return [dot if abs(dot) > self.amplitude/2 else 0 for dot in sig]
        # return self.amplitude*(x if np.abs(x) < self.amplitude/2 else 0 for x in lam)

    def get_monoharm_u(self):
        return self.monoharm_u

    def get_impulse_u(self):
        return self.impulse_u
