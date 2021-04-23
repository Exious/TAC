from Data import params
from ParameterEstimator import ParameterEstimator
from Plotter import Plotter
from RealObject import RealObject
from SignalGenerator import SignalGenerator
import matplotlib.pyplot as plt
from sympy import *

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt


def object_free_movements():
    sol, t = obj.calcODE()

    Plotter.draw([t, sol])

    return sol, t


def do_experiment():
    obj.set_u_fcn(u_func)

    sol, t = obj.calcODE()
    u = u_func(0, t)

    Plotter.draw([t, [sol, u]])

    return sol, t


def model_and_analyzing():
    def ode_ideal():
        return obj.getODE()

    def ode_lin():
        def decor(x, t, k):
            y = x
            [K, T] = k

            u = u_func(x, t)

            dydt = (K*u-y)/T
            return dydt

        return decor

    def ode_non_lin():
        def decor(x, t, k):
            y = x
            [K, T, non_lin_1, non_lin_2] = k

            u = u_func(x, t)

            dydt = (K*obj.nonlin_fcns[obj.nonlin_type]
                    (u, non_lin_1, non_lin_2) - y)/T
            return dydt

        return decor

    def analyze(guess, y0, func):
        to_estimate_init_values = {'guess': guess, 'y0': [y0, ]}

        estimator = ParameterEstimator(
            experiments, to_estimate_init_values, func)

        sol_ideal = estimator.get_ideal_solution(func)

        Plotter.draw([t, [sol, sol_ideal]])

    ode_func_map = {
        'ideal': ode_ideal(),
        'lin': ode_lin(),
        'non_lin': ode_non_lin(),
    }

    experiments = [[t, sol], ]

    for ode_type in ode_func_map.keys():
        guess = params['models'][ode_type]['guess']
        y0 = params['models'][ode_type]['initial_condition']

        analyze(guess, y0, ode_func_map[ode_type])


obj = RealObject()
sig = SignalGenerator()

sol, t = object_free_movements()

for sig_name in params['signals'].keys():
    # for sig_name in ['step']:
    print("Signal name is {}".format(sig_name))

    u_func = sig.get_u(sig_name)

    sol, t = do_experiment()

    model_and_analyzing()

    def neural():
        import numpy as np
        scaler = MinMaxScaler(feature_range=(-1, 1))

        def series2dataset(data, seq_len):
            """
            Преобразование временной последовательнсти к формату датасета
            Шаг дискретизации должен быть постоянным
            """
            dataset = []
            for i in range(data.shape[0]-seq_len):
                r = np.copy(data[i:i+seq_len])
                dataset.append(r)
            return np.array(dataset)

        values = u_func(0, t)
        model_order = 5

        # Этой командой получается массив обучающих последовательностей с одного
        # эксперимента, для работы с несколькими экспериментами, можно получить
        # последовательно массивы отдельно по каждому эксперименту, а затем объединить
        # массивы
        x_values = series2dataset(values, model_order)
        x_values = np.expand_dims(x_values, 2)
        print(x_values.shape)

        # Разделим на тестовую и обучающие выборки
        # В случае использования нескольких экспериментов (>5), в качестве тестового
        # лучше взять один из экспериментов целиком
        n_train_samples = int(x_values.shape[0] * 0.7)

        train_X = x_values[:n_train_samples, :]
        test_X = x_values[n_train_samples:, :]

        y_values = scaler.fit_transform(sol)
        y_values = y_values[model_order:]
        train_y = y_values[:n_train_samples]
        test_y = y_values[n_train_samples:]

        print("Shape of train is {}, {}, shape of test is {}, {}".format(train_X.shape,
                                                                         train_y.shape,
                                                                         test_X.shape,
                                                                         test_y.shape))
        model = Sequential()

        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # fit network
        history = model.fit(train_X,
                            train_y,
                            epochs=100,
                            batch_size=72,
                            validation_data=(test_X, test_y),
                            verbose=0,  # выключить или включить
                            shuffle=False)
        # plot history
        plt.figure()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()

        # делаем предсказание
        yhat = model.predict(test_X)

        # обратное масштабирование для прогноза
        inv_yhat = yhat  # scaler.inverse_transform(yhat)
        #inv_yhat = inv_yhat[:,0]
        # обратное масштабирование для фактического
        inv_y = test_y  # scaler.inverse_transform(test_y)
        #inv_y = inv_y[:,0]
        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)

        # Соединяем обратно входные данные
        X = np.concatenate((train_X, test_X), axis=0)
        Y_real = np.concatenate((train_y, test_y), axis=0)
        # делаем предсказание
        Y = model.predict(X)

        print(X.shape, Y.shape)

        plt.figure()
        plt.plot(Y_real)
        plt.plot(Y)

    neural()


plt.show()
