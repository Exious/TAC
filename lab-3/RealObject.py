from scipy.integrate import odeint
import numpy as np

class RealObject:
    def __init__(self, variant):
        self._var = variant
        self.y0 = 1
        self._ctrl_fcn = RealObject._default_control
        self.lin_par_1 = variant % 10 * 0.2
        self.lin_par_2 = ((32-variant) % 9 + 0.1) * 2.5
        self.nonlin_par_1 = variant % 15 * 0.35
        self.nonlin_par_2 = variant % 12 * 0.45
        self.nonlin_fcns = [self.deadZone, self.saturation, self.relay]
        self.nonlin_names = ['deadZone', 'saturation', 'relay']
        self.nonlin_type = variant % 3
        self._params = [self.lin_par_1, self.lin_par_2, self.nonlin_par_1, self.nonlin_par_2]
        print(self.lin_par_1, self.lin_par_2, self.nonlin_par_1, self.nonlin_par_2)
        
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
        
        dydt = (lin_par_1 * self.nonlin_fcns[self.nonlin_type](u, nonlin_par_1, nonlin_par_2) - x) / lin_par_2
        return dydt
    
    def _default_control(x, t):
        """
        Управление по умолчанию. Нулевой вход
        """
        return 0
    
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
    
    def calcODE(self, ts=10, nt=1001):
        """
        Вспомогательная функция для получения решения систему ДУ, "Проведение эксперимента" с заданным воздействием
        """
        y0 = [self.y0,]
        t = np.linspace(0, ts, nt)
        args = (self._params, )
        sol = odeint(self._ode, y0, t, args)
        return sol, t
    
    def getODE(self):        
        """
        Получить "идеальную" модель объекта без параметров
        """
        return self._ode
    
    def get_nonlinear_element_type(self):
        return self.nonlin_names[self.nonlin_type]

    def change_initial_conditions(self,y0):
        self.y0 = y0