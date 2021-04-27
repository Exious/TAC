import numpy as np
import matplotlib.pyplot as plt
from Utility import Utility


class Plotter:
    def draw(to_plot, options=None):
        x, y = to_plot

        if not isinstance(y, list):
            y = [y]

        options['title'] = Utility.pretify(
            Utility.replaceUnderscore(options['title']))

        figure_name = options['title']

        figure = plt.figure(figure_name)

        Plotter.getPlotDetails(options)

        legend = options['legend']
        legend_values = list(legend.values())
        if all(legend_values):
            pretify = np.vectorize(Utility.pretify)
            replaceUnderscore = np.vectorize(Utility.replaceUnderscore)
            legend = pretify(replaceUnderscore(legend_values))
        else:
            legend = None

        for every_y in range(len(y)):
            if x is None:
                if legend is not None:
                    plt.plot(y[every_y], label=legend[every_y])
                    plt.legend()
                else:
                    plt.plot(y[every_y])
            else:
                if legend is not None:
                    plt.plot(x, y[every_y], label=legend[every_y])
                    plt.legend()
                else:
                    plt.plot(x, y[every_y])

        plt.grid()

        Utility.saveImage(figure, figure_name)

    def getPlotDetails(options):
        if not options:
            return

        plt.title(options['title'])

        limits = options['limits']
        if(limits['x'] is not None):
            plt.xlim(limits['x'])
        if(limits['y'] is not None):
            plt.ylim(limits['y'])

        labels = options['labels']
        if(labels['x'] is not None):
            plt.xlabel(labels['x'])
        if(labels['y'] is not None):
            plt.ylabel(labels['y'])
