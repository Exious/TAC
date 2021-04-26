import numpy as np
from Data import params
from scipy import optimize
from scipy import integrate
from scipy.integrate import odeint


class ParameterEstimator():
    def __init__(self, experiments, to_estimate_init_values, f):
        """
        experiments -- список кортежей с данными экспериментов в формате [x_data, y_data] (вход, выход)
        f -- функция, реализующая дифференциальное уравнение модели
        """
        self._experiments = experiments

        self.guess = to_estimate_init_values['guess']
        self.y0 = to_estimate_init_values['y0']

        self._f = f
        # Предполагаем, что все переменные состояния наблюдаемые, однако в общем случае это не так
        x_data, y_data = experiments[0]
        self.n_observed = params['numeric']['n_observed']

    def my_ls_func(self, x, teta):
        """
        Определение функции, возвращающей значения решения ДУ в
        процессе оценки параметров
        x заданные (временные) точки, где известно решение
        (экспериментальные данные)
        teta -- массив с текущим значением оцениваемых параметров.
        Первые self._y0_len элементов -- начальные условия,
        остальные -- параметры ДУ
        """
        # Для передачи функуии используем ламбда-выражение с подставленными
        # параметрами
        # Вычислим значения дифференциального уравления в точках "x"
        r = integrate.odeint(lambda y, t: self._f(y, t, teta[self._y0_len:]),
                             teta[0:self._y0_len], x)
        # Возвращаем только наблюдаемые переменные
        return r[:, 0:self.n_observed]

    def estimate_ode(self):
        """
        Произвести оценку параметров дифференциального уравнения с заданными
        начальными значениями параметров:
            y0 -- начальные условия ДУ
            guess -- параметры ДУ
        """
        # Сохраняем число начальных условий
        self._y0_len = len(self.y0)
        # Создаем вектор оцениваемых параметров,
        # включающий в себя начальные условия
        est_values = np.concatenate((self.y0, self.guess))
        c = self.estimate_param(est_values)
        # В возвращаемом значении разделяем начальные условия и параметры
        return c[self._y0_len:], c[0:self._y0_len]

    def f_resid(self, p):
        """
        Функция для передачи в optimize.leastsq
        При дальнейших вычислениях значения, возвращаемые этой функцией,
        будут возведены в квадрат и просуммированы.

        """
        delta = []
        # Получаем ошибку для всех экспериментов при заданных параметрах модели
        for data in self._experiments:
            x_data, y_data = data
            d = y_data - self.my_ls_func(x_data, p)
            d = d.flatten()
            delta.append(d)
        delta = np.array(delta)

        return delta.flatten()  # Преобразуем в одномерный массив

    def estimate_param(self, guess):
        """
        Произвести оценку параметров ДУ
            guess -- параметры ДУ
        """
        self._est_values = guess
        # Решить оптимизационную задачу - решение в переменной c
        res = optimize.least_squares(self.f_resid, self._est_values)
        return res.x

    def get_ideal_solution(self, func):
        est_par = self.estimate_ode()

        self.args, self.y0 = est_par

        print("Estimated parameter: {}".format(self.args))
        print("Estimated initial condition: {}".format(self.y0))

        [[t, sol]] = self._experiments

        sol_ideal = odeint(func, self.y0, t, (self.args,))

        return sol_ideal
