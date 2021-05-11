from numpy.core.numeric import NaN
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from Utility import Utility

class Real_Object:
    def __init__(self, ts=None, nt=None,T_1=None, T_2=None, temperature=None, volts=None, Q_MIN=None, Q_MAX=None, y0=None):
        self.v_stroke = 'v'
        self.theta_stroke = 'theta'
        self.ts = ts or 250
        self.nt = nt or 1001
        self.T_1 = T_1 or 20
        self.T_2 = T_2 or 40
        self.temperature = temperature or 40
        self.volts = volts or 220
        self.K = self.temperature / self.volts
        self.Q_MIN = Q_MIN or 21
        self.Q_MAX = Q_MAX or 23
        self.y0 = y0 or (0,0)
        self.state = 0
        self.control_type = 'default'

    def ode_sys_control(self, x):
        c_1 = x >= self.Q_MIN
        c_2 = x >= self.Q_MAX

        if self.control_type == 'relay':
            if not c_1 and not c_2:
                self.state = 1
            elif c_2:
                self.state = 0
            else:
                self.state = 1
        else:
            if not c_1 and not c_2:
                self.state = 1
            elif c_1 and c_2:
                self.state = 0
            elif c_1 and not c_2 and self.state:
                self.state = 1
            elif c_1 and not c_2 and not self.state:
                self.state = 0

        return self.state

    def ode(self, X, t):
        """
        Функция, рализующая систему из первого примера
        """
        v, theta = X

        u = self.volts * self.ode_sys_control(theta)
        #dv_dt = (u - v)/self.T_1
        #dtheta_dt = (self.K*v - theta) / self.T_2
        #return (dv_dt, dtheta_dt)

        dydt = [eval(self.v_stroke),
                eval(self.theta_stroke)]

        return dydt

    def calcODE(self):
        t = np.linspace(0,self.ts,self.nt)

        sol = odeint(self.ode, self.y0, t)

        return (t, sol)

    def draw(self, name, scaled=None):
        t, sol = self.calcODE()

        self.graphifize(name,t,sol)

        self.findFrequency(t,sol)

        if scaled:
            print(t.shape,sol.shape)
            x_scale,y_scale = scaled
            t_len=self.ts
            sol_max=max(sol[:,1])
            limits = {
                "x": (int(x_scale * t_len), t_len),
                "y": (int(y_scale * sol_max), sol_max)
                }
            self.graphifize('scaled '+name,t,sol,limits)

    def findFrequency(self, t,sol):
        T = None
        frequency = None

        local_max_val = None
        local_min_val = None
        local_max_t = None
        local_min_t = None
        
        amplitude = None

        step = int(len(t)*0.001)
        digits = 0

        for i in range(len(t)-2*step):
            prev = sol[len(t)-1-i-2*step][1:]
            current_index=len(t)-1-i-step
            current = sol[current_index][1:]
            next = sol[len(t)-1-i][1:]

            if prev < current and current > next: 
                if not local_max_val:
                    local_max_val = current
                    local_max_t = t[current_index]
                elif np.round(local_max_val,digits) == np.round(current, digits):
                    T = local_max_t - t[current_index]
                    break

            if prev > current and current < next: 
                local_min_val = current
                local_min_t = t[current_index]

        if T:
            frequency = 1/T

        if local_max_val and local_min_val:
            amplitude = (local_max_val - local_min_val)/2
        
        print(T,frequency,amplitude)
    
    def graphifize(self,name,t,sol,limits=None):
        figure = plt.figure(name)

        plt.plot(t, sol[:, 1])
        plt.ylabel('temperature, $^\circ$C')
        plt.xlabel('time, s')
        plt.grid()
        if limits:
            plt.xlim(limits['x'])
            plt.ylim(limits['y'])

        Utility.saveFile(figure, name)

    def change_control(self, control_type):
        self.control_type = control_type

    def replace_with_object_data(self, stroke):
        for key in self.__dict__:
            stroke = stroke.replace(f"{key}",str(self.__dict__[key]))
        return stroke