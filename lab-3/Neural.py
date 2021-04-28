import numpy as np
from Utility import Utility
from Plotter import Plotter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from Data import params


class Neural():
    def __init__(self):
        neural_params = params['models']['neural']
        self.scaler = MinMaxScaler(
            feature_range=neural_params['min_max_scaler'])
        self.estimated_model_order = int(
            neural_params['order_factor'] * neural_params['model_order'])
        self.dims2expand = neural_params['dimensions_to_expand']

        self.method = neural_params['method']
        self.separate_coeff = neural_params['separate_factor']
        self.separate_over = neural_params['separate_over']

        self.common = {
            'input': None,
            'output': None,
            'to_predict': None,
        }

        self.dataset = []

        self.LSTM_units = neural_params['LSTM_units']
        self.Dense_units = neural_params['Dense_units']
        self.epochs = neural_params['epochs_count']
        self.batch_size = neural_params['batch_size']
        self.enable_verbose = neural_params['enable_verbose']

        self.isCommonInvokeTime = False

    def setCommonInvoke(self):
        self.isCommonInvokeTime = True if not self.isCommonInvokeTime else False
        return self

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

    def setInput(self, to_drop):
        self.input = self.dataset
        if not to_drop:
            self.common['input'] = Utility.concatenateSequences(
                self.common['input'], self.input)
        else:
            self.common['to_predict'] = self.input
        self.dataset = []
        return self

    def setOutput(self, to_drop):
        self.output = self.sol[self.estimated_model_order:]

        if not to_drop:
            self.common['output'] = Utility.concatenateSequences(
                self.common['output'], self.output)

        return self

    def scaleOutput(self, data_min=None, data_max=None):
        scaler = self.scaler.fit(self.output)
        self.scaler.feature_range = (
            data_min or scaler.data_min_, data_max or scaler.data_max_)
        self.output = self.scaler.fit_transform(self.output)

        return self

    def separateSequenses(self):
        if self.isCommonInvokeTime:
            self.input = self.common['input']
            self.output = self.common['output']

        if self.method == 'm:n_flat':
            n_train_samples = int(self.input.shape[0] * self.separate_coeff)
            self.train = [self.input[:n_train_samples, :],
                          self.output[:n_train_samples]]
            self.test = [self.input[n_train_samples:, :],
                         self.output[n_train_samples:]]
        if self.method == 'every_n':
            train_X, test_X = Utility.separateArray(
                self.input, self.separate_over)
            train_y, test_y = Utility.separateArray(
                self.output, self.separate_over)
            self.train = [train_X, train_y]
            self.test = [test_X, test_y]

        return self

    def modelConstruct(self, options=None):
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
            [None, [self.history.history['loss'], self.history.history['val_loss']]], options=options)

    def invertedScaling(self):
        test_X, test_y = self.test
        yhat = self.model.predict(test_X)

        inverted_yhat = yhat
        inverted_y = test_y

        rmse = np.sqrt(mean_squared_error(inverted_y, inverted_yhat))
        print('Test RMSE: %.3f' % rmse)

    def predict(self, options=None):
        train_X, train_y = self.train
        test_X, test_y = self.test

        to_plot = []

        if self.isCommonInvokeTime:
            X = self.common['to_predict']
        else:
            if self.method == 'm:n_flat':
                X = np.concatenate((train_X, test_X), axis=0)
                Y_real = np.concatenate((train_y, test_y), axis=0)
            if self.method == 'every_n':
                X = Utility.mergeArrays(train_X, test_X, self.separate_over)
                Y_real = Utility.mergeArrays(
                    train_y, test_y, self.separate_over)
            to_plot.append(Y_real)

        Y_predicted = self.model.predict(X)
        to_plot.append(Y_predicted)

        Plotter.draw([None, to_plot], options=options)
