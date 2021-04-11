from Data import params
import numpy as np


class SignalGenerator:
    def monoharm_u(x, t):
        amplitude = params['numeric']['amplitude']
        return amplitude*np.sin(0.1 * t * 2 * np.pi)

    def get_monoharm_u():
        return SignalGenerator.monoharm_u
