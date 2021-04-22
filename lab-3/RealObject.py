from Data import params
from scipy.integrate import odeint
import numpy as np


class RealObject:
    def __init__(self):
        self._var = params['numeric']['variant']
        self.start_time = params['numeric']['start_time']
        self.duration = params['numeric']['duration']
        self.discretization = params['numeric']['discretization']
        self.y0 = params['numeric']['initial_condition']
        self._ctrl_fcn = RealObject._default_control
        self.lin_par_1 = params['numeric']['lin_par_1']
        self.lin_par_2 = params['numeric']['lin_par_2']
        self.nonlin_par_1 = params['numeric']['nonlin_par_1']
        self.nonlin_par_2 = params['numeric']['nonlin_par_2']
        #self.nonlin_fcns = [self.deadZone, self.saturation, self.relay]
        self.nonlin_fcns = [self.saturation,  self.relay, self.deadZone]
        #self.nonlin_fcns = [self.deadZone,  self.relay, self.saturation]
        self.nonlin_names = ['deadZone', 'saturation', 'relay']
        self.nonlin_type = params['numeric']['nonlin_type']
        self._params = [self.lin_par_1, self.lin_par_2,
                        self.nonlin_par_1, self.nonlin_par_2]
        print(self.lin_par_1, self.lin_par_2,
              self.nonlin_par_1, self.nonlin_par_2)

    def deadZone(self, x, p1, p2):
        if np.abs(x) < p1:
            x = 0
        elif x > 0:
            x = x - p1
        elif x < 0:
            x = x + p2
        return x

    def saturation(self, x, p1, p2):
        if x > p1:
            x = p1
        elif x < -p2:
            x = -p2
        return x

    def relay(self, x, p1, p2):
        if x > 0:
            return p1
        else:
            return -p2

    def _ode(self, x, t, k):
        '''
        Функция принимает на вход вектор переменных состояния и реализует по сути систему в форме Коши
        x -- текущий вектор переменных состояния
        t -- текущее время
        k -- значения параметров
        '''
        y = x
        u = self._get_u(x, t)
        lin_par_1, lin_par_2, nonlin_par_1, nonlin_par_2 = k

        dydt = (lin_par_1 * self.nonlin_fcns[self.nonlin_type]
                (u, nonlin_par_1, nonlin_par_2) - y) / lin_par_2
        return dydt

    def _default_control(x, t):
        """
        Управление по умолчанию. Нулевой вход
        """
        return params['numeric']['default_initial_condition']

    def _get_u(self, x, t):
        """
        Получить значение управления при значениях переменных состояния x и времени t
        """
        return self._ctrl_fcn(x, t)

    def set_u_fcn(self, new_u):
        """
        Установить новую функцию управления
        формат функции: fcn(x, t)
        """
        self._ctrl_fcn = new_u

    def calcODE(self):
        """
        Вспомогательная функция для получения решения систему ДУ, "Проведение эксперимента" с заданным воздействием
        """
        t = np.linspace(self.start_time, self.duration, self.discretization)
        sol = odeint(self._ode, [self.y0, ], t, (self._params, ))
        return sol, t

    def getODE(self):
        """
        Получить "идеальную" модель объекта без параметров
        """
        return self._ode

    def get_nonlinear_element_type(self):
        return self.nonlin_names[self.nonlin_type]

    def change_initial_conditions(self, y0):
        self.y0 = y0
