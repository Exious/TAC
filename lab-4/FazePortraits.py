import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import constants
from mpl_toolkits.mplot3d import Axes3D
from sympy import *
from Utility import Utility
import re


class FazePortrait2D:
    def __init__(self):
        self.v_stroke = 'v'
        self.theta_stroke = 'theta'
        self.deltaX = 1
        self.deltaDX = 1
        self.startX = 0
        self.stopX = 5
        self.startDX = 0
        self.stopDX = 0
        self.ts = 10
        self.nt = 101

    def ode(self, X, t, volts=None, sys=None):
        v, theta = X

        if volts and sys:
            u = volts * sys(theta)

        dydt = [eval(self.v_stroke),
                eval(self.theta_stroke)]
        return dydt

    def calcODE(self, y0, dy0, sys_op):
        y0 = [y0, dy0]
        t = np.linspace(0, self.ts, self.nt)
        sol = odeint(self.ode, y0, t, sys_op)
        return sol

    def draw(self, name, extra_dot=None, volts=None,sys=None):
        figure = plt.figure(name)
        ax = figure.add_subplot()

        for y0 in np.arange(self.startX, self.stopX, self.deltaX):
            for dy0 in np.arange(self.startDX, self.stopDX, self.deltaDX):
                sol = self.calcODE(y0, dy0, (volts,sys))
                ax.plot(sol[:, 0], sol[:, 1], 'b')

        if (extra_dot is not None):
            circle = plt.Circle(
                extra_dot['dot'], extra_dot['radial'], color='r')

            ax.add_patch(circle)

        plt.xlabel('v')
        plt.ylabel('dv/dt')
        plt.grid()

        Utility.saveFile(figure, name)

    def changeParams(self, params):
        for key in params:
            self.__dict__[key] = params[key]

    def linearize(self):
        for stroke in {'v_stroke', 'theta_stroke'}:
            self.__dict__[stroke] = re.sub(
                r'(\*(v|theta)){2,}', '*0', self.__dict__[stroke])


class FazePortrait3D(FazePortrait2D):
    def __init__(self):
        self.x_stroke = 'x'
        self.y_stroke = 'y'
        self.z_stroke = 'z'
        self.deltaX = 4
        self.deltaY = 4
        self.deltaZ = 4
        self.startX = -10
        self.stopX = 10
        self.startY = -10
        self.stopY = 10
        self.startZ = -10
        self.stopZ = 10
        self.ts = 10
        self.nt = 1001

    def ode(self, Y, t):
        x, y, z = Y

        dxdt = eval(self.x_stroke)
        dydt = eval(self.y_stroke)
        dzdt = eval(self.z_stroke)

        return [dxdt, dydt, dzdt]

    def calcODE(self, x, y, z):
        y0 = [x, y, z]
        t = np.linspace(0, self.ts, self.nt)
        sol = odeint(self.ode, y0, t)
        return sol

    def draw(self, name):
        figure = plt.figure(name)
        spacing = 0.5
        plt.subplots_adjust(hspace=spacing, wspace=spacing)

        ax = figure.add_subplot(2, 2, 1, projection='3d')
        ax.set_title("3D")
        plt.subplot(2, 2, 2)
        plt.title("X-Y")
        plt.grid()
        plt.subplot(2, 2, 3)
        plt.title("X-Z")
        plt.grid()
        plt.subplot(2, 2, 4)
        plt.title("Y-Z")
        plt.grid()

        for x in range(self.startX, self.stopX, self.deltaX):
            for y in range(self.startY, self.stopY, self.deltaY):
                for z in range(self.startZ, self.stopZ, self.deltaZ):
                    sol = self.calcODE(x, y, z)

                    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2])
                    plt.subplot(2, 2, 2)
                    plt.plot(sol[:, 0], sol[:, 1])
                    plt.subplot(2, 2, 3)
                    plt.plot(sol[:, 0], sol[:, 2])
                    plt.subplot(2, 2, 4)
                    plt.plot(sol[:, 1], sol[:, 2])

        Utility.saveFile(figure, name)

    def changeParams(self, params):
        for key in params:
            self.__dict__[key] = params[key]
