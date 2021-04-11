import os
import json


class Data:
    def __init__(self):
        self.paths = ['numeric', 'models']
        #self.paths = ['numeric']
        self.format = '.json'
        self.data = {}
        self.raw_data = {}

        #self.numeric_keys = None
        #self.models_keys = None

        self.variant = None
        self.lin_par_1 = None
        self.lin_par_2 = None
        self.nonlin_par_1 = None
        self.nonlin_par_2 = None
        self.nonlin_type = None
        self.duration = None
        self.discretization = None
        self.initial_condition = None
        self.default_initial_condition = None
        self.n_observed = None

    def read(self):
        for path in self.paths:

            #raw_data = self.raw_data
            #print(raw_data, path)
            self.raw_data.update({path: None})

            with open(path + self.format, 'r', encoding='utf-8') as readfile:
                file_data = readfile.read()
                self.raw_data[path] = json.loads(file_data)
                self.__dict__[path+'_keys'] = self.raw_data[path].keys()

    def saveData(self, to_save=None, raw_data=None):
        self.read()

        if self.variant is None:
            self.variant = int(self.raw_data['numeric']['variant'])

        if not bool(self.data):
            self.data.update(self.raw_data)

        if to_save is None:
            to_save = self.data
        if raw_data is None:
            raw_data = self.raw_data

        print(to_save, raw_data)
        for stroke in raw_data:
            print(stroke)
            for substroke in raw_data[stroke]:
                print(substroke, to_save[stroke][substroke])

                if not isinstance(raw_data[stroke][substroke], dict):
                    if 'self.variant' in raw_data[stroke][substroke]:
                        to_save[stroke][substroke] = eval(
                            raw_data[stroke][substroke])
                    else:
                        splitted = raw_data[stroke][substroke].split(' ')

                        if len(splitted) is not 1:
                            to_save[stroke][substroke] = splitted
                        else:
                            to_save[stroke][substroke] = eval(
                                raw_data[stroke][substroke])
                else:
                    self.saveData(to_save[stroke],
                                  raw_data[stroke])

        print(self.data)

    def return_data(self):
        #data = {}
        # for stroke in self.numeric_keys:
        #    data[stroke] = self.__dict__[stroke]

        return self.data


data = Data()

data.saveData()

params = data.return_data()

print('-------------------------------------')
print(params)
print('-------------------------------------')

# print(data)
