from FazePortraits import FazePortrait2D, FazePortrait3D, Utility
from RealObject import Real_Object
import json
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

with open('data.json', 'r', encoding='utf-8') as readfile:
    file_data = readfile.read()
    data = json.loads(file_data)

real_obj = Real_Object()
fazePortrait2D = FazePortrait2D()
fazePortrait3D = FazePortrait3D()

for system in ['default', 'relay']:
    params = data[system+'-system']

    for stroke in ['v_stroke','theta_stroke']:
        params[stroke] = real_obj.__dict__[stroke] = real_obj.replace_with_object_data(params[stroke])

    real_obj.change_control(system)
    real_obj.draw(f'{system} system')

    fazePortrait2D.changeParams(params)
    fazePortrait2D.draw(f'{system} system phase portrait',volts=real_obj.volts,sys=real_obj.ode_sys_control)

for type in ['non-linear', 'linear']:
    params = data['oscillator-system'][type]
    
    if type == 'linear':
        fazePortrait2D.linearize()

    fazePortrait2D.changeParams(params)
    fazePortrait2D.draw(f'{type} oscillator system phase portrait')

params = data['third-order-system']

fazePortrait3D.changeParams(params)
fazePortrait3D.draw('third order system')

plt.show()