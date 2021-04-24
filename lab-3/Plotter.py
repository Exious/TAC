import matplotlib.pyplot as plt


class Plotter:
    def draw(to_plot, options=None):
        # print(to_plot)
        x, y = to_plot
        # print('x:',x)
        # print('y:',y)

        if not isinstance(y, list):
            y = [y]

        plt.figure()

        for every_y in y:
            if x is None:
                plt.plot(every_y)
            else:
                plt.plot(x, every_y)

        plt.grid()
