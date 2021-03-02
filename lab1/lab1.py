import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import re

variant = 14  # Изменяйте ТОЛЬКО значение варианта


class Signal:
    def __init__(self, variant):
        self.variant = variant
        self.dt = 0.00001
        self.test_signal_start = 0
        self.test_signal_duration = 100
        self.test_sig_ampl = 1 + self.variant * 0.1
        self.test_sig_freq = 1 + self.variant * 3.5
        self.non_lin_param_1 = 0.5 + self.variant * 0.1
        self.lin_param_k = 0.5 + self.variant * 0.3
        self.lin_param_T = 0.1 + self.variant * 0.2
        self.sig_sin = {
            'data': None,
            'after_relay': None,
            'after_dead_zone': None,
            'after_saturation': None
        }
        self.sig_meandr = {
            'data': None,
            'after_relay': None,
            'after_dead_zone': None,
            'after_saturation': None
        }
        self.sig_sawtooth = {
            'data': None,
            'after_relay': None,
            'after_dead_zone': None,
            'after_saturation': None
        }

    def getParams(self):
        print("Вариант номер: {}".format(self.variant))
        print("Амплитуда тестового сигнала: {:.2}".format(self.test_sig_ampl))
        print("Частота тестового сигнала: {} Гц".format(self.test_sig_freq))
        print("Длительность тестового сигнала: {} с".format(
            self.test_signal_duration))
        print("Параметр нелинейностей 1: {:.2}".format(self.non_lin_param_1))
        print("Коэффициент усиления линейного звена: {:.2}".format(
            self.lin_param_k))
        print("Постоянная времени линейного звена: {:.2}".format(
            self.lin_param_T))
        print("Частота дискретизации: {}".format(self.dt))

    def getTestVector(self):
        self.t = np.arange(0, self.test_signal_duration, self.dt)

    def getSignals(self):
        self.getTestVector()

        self.phaze = self.test_sig_freq * self.t * 2 * np.pi

        self.sig_sin['data'] = self.test_sig_ampl * np.sin(self.phaze)
        self.sig_meandr['data'] = self.test_sig_ampl * \
            signal.square(self.phaze)
        self.sig_sawtooth['data'] = self.test_sig_ampl * \
            signal.sawtooth(self.phaze)

    def getSignalSpec(self, sig):
        sig_spec = np.abs(np.fft.fft(sig))
        freqs = np.fft.fftfreq(sig.shape[0], self.dt)

        return [sig_spec, freqs]

    def getSignalAfterNonLinElem(self, sig):
        def dead_zone_scalar(x, width=0.5):
            if np.abs(x) < width:
                return 0
            elif x > 0:
                return x-width
            else:
                return x+width

        dead_zone = np.vectorize(dead_zone_scalar, otypes=[
            np.float64], excluded=['width'])

        def saturation_scalar(x, height=0.5):
            if np.abs(x) < height:
                return x
            elif x > 0:
                return height
            else:
                return -height

        saturation = np.vectorize(saturation_scalar, otypes=[
                                  np.float64], excluded=['hight'])

        sig_props = self.__dict__['sig_'+sig]

        sig_props['after_relay'] = np.sign(sig_props['data'])
        sig_props['after_dead_zone'] = dead_zone(
            sig_props['data'], self.non_lin_param_1)
        sig_props['after_saturation'] = saturation(
            sig_props['data'], self.non_lin_param_1)


class Utility:
    def __init__(self, signals, transliteration):
        self.signals = signals
        self.non_lin_elems = ['relay', 'dead_zone', 'saturation']
        self.transliteration = transliteration

    def pretify(self, str):
        return re.sub(r'^([a-zёа-я])', lambda tmp: tmp.group().upper(), str.lower())

    def signalPlot(self, this, sig, options, title):
        self.getPlotDetails(options, title)

        plt.plot(this.t, sig)

    def specPlot(self, this, sig, options, title):
        sig_spec, freqs = this.getSignalSpec(sig)
        self.getPlotDetails(options, title)

        plt.plot(freqs, sig_spec)

    def getPlotDetails(self, options, title):
        limits = options['limits']
        if(limits['x'] is not None):
            plt.xlim(limits['x'])
        if(limits['y'] is not None):
            plt.ylim(limits['y'])

        labels = options['labels']
        if(labels['x'] is not None):
            plt.xlabel(self.pretify(labels['x']))
        if(labels['y'] is not None):
            plt.ylabel(self.pretify(labels['y']))

        plt.title(self.pretify(title))

    def getSignalPlots(self, this, non_lin_elem=None):
        to_plot = ['sig', 'spec']

        for graph in self.signals:
            options = self.signals[graph]

            sig = this.__dict__['sig_'+graph]['data']

            if(non_lin_elem is None):
                figure_name = '{} и её спектр'.format(
                    self.pretify(self.transliteration['signal'][graph]['name']))
                to_spec = sig
            else:
                sig_after_non_lim = this.__dict__[
                    'sig_'+graph]['after_'+non_lin_elem]
                figure_name = 'График {} и отклика на него {}. Спектр сигнала'.format(
                    self.transliteration['signal'][graph]['case'],
                    self.transliteration['non_lin_elem'][non_lin_elem]['case'])
                to_spec = sig_after_non_lim

            plt.figure(figure_name)

            for type in to_plot:
                plt.subplot(len(to_plot), 1,
                            to_plot.index(type) + 1)
                plt.grid()
                if(type == 'spec'):
                    if(non_lin_elem is None):
                        title = 'Спектр {}'.format(
                            self.transliteration['signal'][graph]['case'])
                    else:
                        title = 'Спектр {} после {}'.format(self.transliteration['signal'][graph]['case'],
                                                            self.transliteration['non_lin_elem'][non_lin_elem]['case'])
                    self.specPlot(this, to_spec, options['spec'], title)
                else:
                    if(non_lin_elem is None):
                        title = self.pretify(
                            self.transliteration['signal'][graph]['name'])
                    else:
                        title = self.pretify('{} до и после {}'.format(self.transliteration['signal'][graph]['name'],
                                                                       self.transliteration['non_lin_elem'][non_lin_elem]['case']))
                    self.signalPlot(this, sig, options['sig'], title)
                    if(non_lin_elem is not None):
                        plt.plot(this.t, sig_after_non_lim)

    def getNonLinearPlots(self, this):
        for non_lin_elem in self.non_lin_elems:
            self.getStaticPlots(
                this, 'Статическая характеристика {} от {} на входе', non_lin_elem)
            self.getSignalPlots(this, non_lin_elem)

    def getStaticPlots(self, this, title, non_lin_type):
        non_lin_type_translit = self.transliteration['non_lin_elem'][non_lin_type]['case']
        plt.figure('Статическая характеристика {}'.format(
            non_lin_type_translit))

        for sig in self.signals:
            this.getSignalAfterNonLinElem(sig)
            sig_props = this.__dict__['sig_'+sig]

            plt.subplot(len(self.signals), 1,
                        list(self.signals).index(sig) + 1)
            plt.grid()

            self.getPlotDetails(
                self.signals[sig]['static'],
                title.format(
                    self.transliteration['non_lin_elem'][non_lin_type]['case'],
                    self.transliteration['signal'][sig]['case']
                ))

            plt.plot(sig_props['data'],
                     sig_props['after_'+non_lin_type])


sin_options = {
    'signal_name': 'sin',
    'sig': {
        'labels': {'x': 'Время, с', 'y': 'Амплитуда', },
        'limits': {'x': [0, 0.15], 'y': None, },
    },
    'spec': {
        'labels': {'x': 'Частота, Гц', 'y': None, },
        'limits': {'x': [-1500, 1500], 'y': None, },
    },
    'static': {
        'labels': {'x': 'Вход', 'y': 'Выход', },
        'limits': {'x': None, 'y': None, },
    },
}

meandr_options = {
    'signal_name': 'meandr',
    'sig': {
        'labels': {'x': 'Время, с', 'y': 'Амплитуда', },
        'limits': {'x': [0, 0.15], 'y': None, },
    },
    'spec': {
        'labels': {'x': 'Частота, Гц', 'y': None, },
        'limits': {'x': [-1500, 1500], 'y': None, },
    },
    'static': {
        'labels': {'x': 'Вход', 'y': 'Выход', },
        'limits': {'x': None, 'y': None, },
    },
}

sawtooth_options = {
    'signal_name': 'sawtooth',
    'sig': {
        'labels': {'x': 'Время, с', 'y': 'Амплитуда', },
        'limits': {'x': [0, 0.15], 'y': None, },
    },
    'spec': {
        'labels': {'x': 'Частота, Гц', 'y': None, },
        'limits': {'x': [-1500, 1500], 'y': None, },
    },
    'static': {
        'labels': {'x': 'Вход', 'y': 'Выход', },
        'limits': {'x': None, 'y': None, },
    },
}

transliteration = {
    'signal': {
        'sin': {
            'name': 'синусоида',
            'case': 'синусоиды',
        },
        'meandr': {
            'name': 'меандр',
            'case': 'меандра',
        },
        'sawtooth': {
            'name': 'пила',
            'case': 'пилы',
        },
    },
    'non_lin_elem': {
        'relay': {
            'name': 'идеальное реле',
            'case': 'идеального реле',
        },
        'dead_zone': {
            'name': 'мёртвая зона',
            'case': 'мёртвой зоны',
        },
        'saturation': {
            'name': 'насыщение',
            'case': 'насыщения',
        },
    },
}

sig = Signal(variant)

utility = Utility(
    {
        'sin': sin_options,
        'meandr': meandr_options,
        'sawtooth': sawtooth_options,
    }, transliteration)

sig.getParams()
sig.getSignals()

utility.getSignalPlots(sig)
utility.getNonLinearPlots(sig)

plt.show()
