from Data import params
import numpy as np
from scipy import signal


class SignalGenerator:
    def __init__(self, pike_index=None):
        self.amplitude = params['numeric']['amplitude']
        self.discretization = params['numeric']['discretization']
        self.pike_index = int(
            np.random.sample()*self.discretization) if not bool(pike_index) else pike_index
        self.impulse_pike_counter = 0
        self.random_coeff = 0.3

    def monoharm_u(self, x, t):
        return self.amplitude*np.sin(0.1 * t * 2 * np.pi)

    def impulse_u(self, x, t):
        if np.random.sample() < self.random_coeff:
            self.impulse_pike_counter += 1

        if not isinstance(t, np.ndarray):
            return self.amplitude if self.impulse_pike_counter == 1 else 0

        return [dot*self.amplitude for dot in signal.unit_impulse(
            self.discretization, self.pike_index)]

    def get_monoharm_u(self):
        return self.monoharm_u

    def get_impulse_u(self):
        return self.impulse_u
