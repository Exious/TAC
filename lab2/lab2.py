import re
from fazePortraits import FazePortrait2D, FazePortrait3D, Utility
import matplotlib.pyplot as plt
import json

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
            fazePortrait2D.changeParams(params[param]['non-linear'])
            fazePortrait2D.draw('System ' + param)

            fazePortrait2D.linearize()
            fazePortrait2D.changeParams(params[param]['linear'])
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
