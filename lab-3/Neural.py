import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from Plotter import Plotter
from Data import params


class Neural():
    def __init__(self):
        neural_params = params['models']['neural']
        self.scaler = MinMaxScaler(
            feature_range=neural_params['min_max_scaler'])
        self.estimated_model_order = int(
            neural_params['order_factor'] * neural_params['model_order'])
        self.separate_coeff = neural_params['separate_factor']
        self.dims2expand = neural_params['dimensions_to_expand']

        self.common = {
            'input': None,
            'output': None,
        }

        self.signal = None
        self.sol = None

        self.dataset = []
        self.input = None
        self.train = None
        self.test = None
        self.output = None

        self.model = None
        self.history = None

        self.LSTM_units = neural_params['LSTM_units']
        self.Dense_units = neural_params['Dense_units']
        self.epochs = neural_params['epochs_count']
        self.batch_size = neural_params['batch_size']
        self.enable_verbose = neural_params['enable_verbose']

    def concatenateSequences(self, first, second):
        if first is not None:
            return np.concatenate((first, second), axis=0)
        return second

    def setSignal(self, u_func, sol, t):
        self.signal = u_func(0, t)
        self.sol = sol
        return self

    def series2dataset(self):
        for i in range(self.signal.shape[0]-self.estimated_model_order):
            r = np.copy(self.signal[i:i+self.estimated_model_order])
            self.dataset.append(r)
        self.dataset = np.array(self.dataset)
        return self

    def expandDimensions(self):
        self.dataset = np.expand_dims(self.dataset, self.dims2expand)
        return self

    def setInput(self):
        self.input = self.dataset
        self.common['input'] = self.concatenateSequences(
            self.common['input'], self.input)
        self.dataset = []
        return self

    def setOutput(self):
        self.output = self.scaler.fit_transform(
            self.sol)[self.estimated_model_order:]
        self.common['output'] = self.concatenateSequences(
            self.common['output'], self.output)
        return self

    def separateSequenses(self):
        n_train_samples = int(self.input.shape[0] * self.separate_coeff)
        self.train = [self.input[:n_train_samples, :],
                      self.output[:n_train_samples]]
        self.test = [self.input[n_train_samples:, :],
                     self.output[n_train_samples:]]
        return self

    def modelConstruct(self):
        train_X, train_y = self.train
        test_X, test_y = self.test

        shapes_X = train_X.shape

        self.model = Sequential()
        self.model.add(LSTM(self.LSTM_units, input_shape=(
            shapes_X[1], shapes_X[2])))
        self.model.add(Dense(self.Dense_units))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        self.history = self.model.fit(train_X,
                                      train_y,
                                      epochs=self.epochs,
                                      batch_size=self.batch_size,
                                      validation_data=(test_X, test_y),
                                      verbose=self.enable_verbose,
                                      shuffle=False)

        Plotter.draw(
            [None, [self.history.history['loss'], self.history.history['val_loss']]])

    def invertedScaling(self):
        test_X, test_y = self.test
        yhat = self.model.predict(test_X)

        inverted_yhat = yhat

        i1 = self.scaler.inverse_transform(yhat)
        i2 = inverted_yhat[:, 0]

        inverted_y = test_y

        t1 = self.scaler.inverse_transform(test_y)
        t2 = inverted_y[:, 0]

        rmse = np.sqrt(mean_squared_error(inverted_y, inverted_yhat))
        print('Test RMSE: %.3f' % rmse)

        rmse1 = np.sqrt(mean_squared_error(i1, t1))
        rmse2 = np.sqrt(mean_squared_error(i2, t2))
        print(rmse)
        print(rmse1)
        print(rmse2)

    def predict(self):
        train_X, train_y = self.train
        test_X, test_y = self.test

        X = np.concatenate((train_X, test_X), axis=0)
        Y_real = np.concatenate((train_y, test_y), axis=0)

        Y_predicted = self.model.predict(X)

        Plotter.draw([None, [Y_real, Y_predicted]])
