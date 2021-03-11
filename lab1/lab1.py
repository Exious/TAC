import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import re
import os

variant = 14  # Изменяйте ТОЛЬКО значение варианта


class Signal:
    def __init__(self, variant, sig_template):
        self.variant = variant
        self.dt = 0.00001
        self.test_signal_start = 0
        self.test_signal_duration = 100
        self.test_sig_ampl = 1 + self.variant * 0.1
        self.test_sig_freq = 1 + self.variant * 3.5
        self.non_lin_param_1 = 0.5 + self.variant * 0.1
        self.lin_param_k = 0.5 + self.variant * 0.3
        self.lin_param_T = 0.1 + self.variant * 0.2
        self.sig = {
            'sine': sig_template.copy(),
            'meandr': sig_template.copy(),
            'sawtooth': sig_template.copy(),
        }
        self.dead_zone_vec = None
        self.saturation_vec = None

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
        self.phaze = self.test_sig_freq * self.t * 2 * np.pi

        self.sig['sine']['data'] = self.test_sig_ampl * np.sin(self.phaze)
        self.sig['meandr']['data'] = self.test_sig_ampl * \
            signal.square(self.phaze)
        self.sig['sawtooth']['data'] = self.test_sig_ampl * \
            signal.sawtooth(self.phaze)

    def getSignalSpec(self, sig):
        sig_spec = np.abs(np.fft.fft(sig))
        freqs = np.fft.fftfreq(sig.shape[0], self.dt)

        return [sig_spec, freqs]

    def getSignalAfterNonLinElem(self, non_lin_type):
        def dead_zone_scalar(x, width=0.5):
            if np.abs(x) < width:
                return 0
            elif x > 0:
                return x-width
            else:
                return x+width

        if (self.dead_zone_vec is None):
            self.dead_zone_vec = np.vectorize(dead_zone_scalar, otypes=[
                                              np.float64], excluded=['width'])

        def saturation_scalar(x, height=0.5):
            if np.abs(x) < height:
                return x
            elif x > 0:
                return height
            else:
                return -height

        if (self.saturation_vec is None):
            self.saturation_vec = np.vectorize(saturation_scalar, otypes=[
                                               np.float64], excluded=['hight'])

        def relay(sig):
            return np.sign(sig)

        def dead_zone(sig):
            return self.dead_zone_vec(sig, self.non_lin_param_1)

        def saturation(sig):
            return self.saturation_vec(sig, self.non_lin_param_1)

        if(non_lin_type == 'relay'):
            return relay
        if(non_lin_type == 'dead zone'):
            return dead_zone
        if(non_lin_type == 'saturation'):
            return saturation

    def getSignalAfterFiltration(self):
        B = [self.lin_param_k/(1+self.lin_param_T/self.dt)]
        A = [1, -1/(1+self.dt/self.lin_param_T)]

        def filtred(sig):
            return signal.lfilter(B, A, sig)

        return filtred


class Utility:
    def __init__(self, options):
        self.options = options
        self.non_lin_elems = ['relay', 'dead zone', 'saturation']

    def pretify(self, str):
        return re.sub(r'^([a-zёа-я])', lambda tmp: tmp.group().upper(), str.lower())

    def renameImage(self, str):
        return '-'.join(str.split())

    def saveFile(self, figure, figure_name):
        path = './LaTeX/body/images/' + \
            self.renameImage(figure_name) + '.png'

        if(os.path.isfile(path)):
            os.remove(path)

        figure.savefig(path)

    def signalPlot(self, this, sigs, options, title):
        self.getPlotDetails(options, title)
        for sig in sigs:
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

    def getSignalPlots(self, this, extra, scaled_one=None):
        to_plot = ['sig', 'spec']

        if (scaled_one is not None):
            to_plot.insert(1, 'scaled_sig')

        if(len(to_plot) == 3):
            h_space = 1
        else:
            h_space = 0.45

        for graph in self.options:
            options = self.options[graph]

            sig = [this.sig[graph]['data']]

            if(extra['data'] is not None):
                for func in extra['data']:
                    sig.extend([func(sig[len(sig)-1])])

            to_spec = sig[len(sig) - 1]

            figure_name = extra['figure_name'].format(graph)
            spectre_title = extra['spectre_title'].format(graph)
            title = extra['title'].format(graph)

            figure = plt.figure(self.pretify(figure_name))

            for type in to_plot:
                plt.subplot(len(to_plot), 1,
                            to_plot.index(type) + 1)
                plt.subplots_adjust(hspace=h_space)
                plt.grid()

                if(type == 'spec'):
                    self.specPlot(
                        this, to_spec, options['spec'], spectre_title)
                if(type == 'sig'):
                    self.signalPlot(this, sig, options['sig'], title)
                if(type == 'scaled_sig'):
                    self.signalPlot(
                        this, sig, options['filtred'], 'scaled' + title)

            self.saveFile(figure, figure_name)

    def getNonLinearPlots(self, this):
        for non_lin_elem in self.non_lin_elems:
            after_non_lin_elem = this.getSignalAfterNonLinElem(non_lin_elem)

            self.getStaticPlots(
                this, non_lin_elem, after_non_lin_elem)

            params = {
                'data': [after_non_lin_elem],
                'figure_name': '{} after ' + non_lin_elem + ' and its spectre ',
                'title': '{} and its result after ' + non_lin_elem,
                'spectre_title': '{} after ' + non_lin_elem + ' spectre',
            }

            self.getSignalPlots(this, params)

            filtred = this.getSignalAfterFiltration()

            params = {
                'data': [after_non_lin_elem, filtred],
                'figure_name': '{} after filtred ' + non_lin_elem + ' and its spectre',
                'title': '{}, same after ' + non_lin_elem + 'and same filtred after ' + non_lin_elem,
                'spectre_title': '{} filtred after ' + non_lin_elem + ' spectre',
            }

            self.getSignalPlots(this, params, True)

            params = {
                'data': [filtred, after_non_lin_elem],
                'figure_name': 'filtred {} after ' + non_lin_elem + ' and its spectre',
                'title': '{}, filtred same and filtred same after ' + non_lin_elem,
                'spectre_title': 'filtred {} after ' + non_lin_elem + ' spectre',
            }

            self.getSignalPlots(this, params, True)

    def getStaticPlots(self, this, non_lin_elem, func):
        figure_name = '{} static characteristics'.format(
            non_lin_elem)

        figure = plt.figure(self.pretify(figure_name))

        for graph in self.options:
            sig_props = this.sig[graph]['data']

            sig_after_non_lin = func(
                sig_props)

            plt.subplot(len(self.options), 1,
                        list(self.options).index(graph) + 1)
            plt.subplots_adjust(hspace=1)
            plt.grid()

            title = '{} static characteristic with a {} at the entrance'.format(
                non_lin_elem, graph)

            self.getPlotDetails(
                self.options[graph]['static'],
                self.pretify(title))

            plt.plot(sig_props, sig_after_non_lin)

        self.saveFile(figure, figure_name)


sig_options_template = {
    'signal_name': None,
    'sig': {
        'labels': {'x': 'Time, s', 'y': 'Amplitude', },
        'limits': {'x': [0, 0.15], 'y': None, },
    },
    'spec': {
        'labels': {'x': 'Frequency, Hz', 'y': None, },
        'limits': {'x': [-1500, 1500], 'y': None, },
    },
    'static': {
        'labels': {'x': 'In', 'y': 'Out', },
        'limits': {'x': None, 'y': None, },
    },
    'filtred': {
        'labels': {'x': 'Time, s', 'y': 'Amplitude', },
        'limits': {'x': [0, 0.1], 'y': [-0.05, 0.05], },
    }
}

options = {
    'sine': sig_options_template.copy(),
    'meandr': sig_options_template.copy(),
    'sawtooth': sig_options_template.copy(),
}

sig_template = {
    'data': None,
    'relay': None,
    'dead_zone': None,
    'saturation': None,
    'filtration': {
        'data': None,
        'relay': {
            'after': None,
            'reversed': None
        },
        'dead_zone': {
            'after': None,
            'reversed': None
        },
        'saturation': {
            'after': None,
            'reversed': None
        },
    },
}

params = {
    'data': None,
    'figure_name': '{} and its spectre',
    'title': '{}',
    'spectre_title': '{} spectre',
}

sig = Signal(variant, sig_template)

utility = Utility(options)

sig.getParams()

sig.getTestVector()
sig.getSignals()


utility.getSignalPlots(sig, params)
utility.getNonLinearPlots(sig)

plt.show()
