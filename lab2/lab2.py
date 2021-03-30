from fazePortraits import FazePortrait2D, FazePortrait3D, Utility
import matplotlib.pyplot as plt
import json
from sympy import *

variant = 14


fazePortrait2D = FazePortrait2D(variant, b=0.25, c=5)
fazePortrait3D = FazePortrait3D(sigma=10, r=28, b=8/3)

with open('data.json', 'r', encoding='utf-8') as readfile:
    file_data = readfile.read()
    data = json.loads(file_data)

for task in data:
    params = data[task]

    if task == '1':

        for param in params:
            non_linear_params = params[param]['non-linear']
            linear_params = params[param]['linear']
            non_linear_dot_params = params[param]['non-linear-dot']

            fazePortrait2D.changeParams(non_linear_params)
            fazePortrait2D.draw('System ' + param)

            dots = fazePortrait2D.getAllDots(non_linear_params)

            for dot in dots:
                print('Additional stability dot for system {} is {}'.format(param, dot))

                figure_name = 'System ' + param + \
                    ' additional stability dot №' + str(dots.index(dot)+1)
                dot_params = non_linear_dot_params.copy()

                dot_params = fazePortrait2D.shiftParams(dot, dot_params)

                fazePortrait2D.changeParams(dot_params)
                fazePortrait2D.draw(figure_name, {'dot': dot, 'radial': 0.01})

            fazePortrait2D.linearize()
            fazePortrait2D.changeParams(linear_params)
            fazePortrait2D.draw('Linearized system ' + param)

    if task == '2':

        fazePortrait2D.changeParams(params)
        fazePortrait2D.draw('Simulink system')

    if task == '3':

        for drag in ['', '-b*y']:
            new_params = params.copy()
            new_params['y_stroke'] = drag + new_params['y_stroke']

            figure_name = ''

            if drag:
                figure_name = ' with drag'

            fazePortrait2D.changeParams(new_params)
            fazePortrait2D.draw('Pendulum system' + figure_name)

    if task == '4':

        for mu in ['mu*', '0.5*mu*', '2*mu*']:
            new_params = params.copy()
            new_params['y_stroke'] = mu + new_params['y_stroke']

            fazePortrait2D.changeParams(new_params)
            fazePortrait2D.draw('Oscillator with ' +
                                Utility.star_replacer(mu))

    if task == '5':

        for param in params:
            fazePortrait3D.changeParams(params[param])
            fazePortrait3D.draw('Lorenz attractor type ' + param)

plt.show()
