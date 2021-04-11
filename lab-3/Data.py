import os
import json


class Data:
    def __init__(self):
        self.format = '.json'

        paths = []

        for file in os.listdir("."):
            if file.endswith(self.format):
                print(file)
                #print(os.path.join("/mydir", file))

        files = list(filter(lambda file: file.endswith(self.format), [
            file for file in os.listdir(".")]))
        print(files, type(files))
        self.paths = [file.replace(self.format, '') for file in files]
        print('--', self.paths)

        for file in files:
            print(file)

        #self.paths = ['numeric', 'models']
        self.data = {}
        self.raw_data = {}
        self.variant = None

    def read(self):
        for path in self.paths:
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

        for stroke in raw_data:
            for substroke in raw_data[stroke]:
                if not isinstance(raw_data[stroke][substroke], dict):
                    if 'self.variant' in raw_data[stroke][substroke]:
                        to_save[stroke][substroke] = eval(
                            raw_data[stroke][substroke])
                    else:
                        splitted = raw_data[stroke][substroke].split(' ')

                        if len(splitted) != 1:
                            to_save[stroke][substroke] = splitted
                        else:
                            to_save[stroke][substroke] = eval(
                                raw_data[stroke][substroke])
                else:
                    self.saveData(to_save[stroke],
                                  raw_data[stroke])

        return self

    def return_data(self):
        return self.data


data = Data()

params = data.saveData().return_data()
