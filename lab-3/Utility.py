import re
import os
import json
import numpy as np


class Utility():
    def concatenateSequences(first, second):
        if first is not None:
            return np.concatenate((first, second), axis=0)
        return second

    def separateArray(arr, over):
        train = []
        test = []
        for i in range(arr.shape[0]):
            if (i+1) % over:
                train.append(arr[i])
            else:
                test.append(arr[i])

        return np.array(train), np.array(test)

    def mergeArrays(first, second, over):
        arr = []
        k = 0

        for i in range(second.shape[0]):
            for j in range(over-1):
                arr.append(first[i+j+k])
            arr.append(second[i])
            k += 1

        return np.array(arr)

    def saveImage(figure, figure_name):
        directory = '/LaTeX/body/images/'
        Utility.checkDirectory(directory)
        file = Utility.renameImage(figure_name) + '.png'
        path = '.'+directory+file

        if(os.path.isfile(path)):
            os.remove(path)

        figure.savefig(path)

    def saveData(data):
        directory = '/LaTeX/body/data/'
        Utility.checkDirectory(directory)
        file = 'data.json'
        path = '.'+directory+file
        with open(path, 'w') as file:
            file.write(json.dumps(data, indent=4))

    def pretify(str):
        return re.sub(r'^([a-zёа-я])', lambda tmp: tmp.group().upper(), str.lower())

    def renameImage(str):
        return '-'.join(str.split())

    def replaceUnderscore(str):
        return str.replace('_', ' ')

    def checkDirectory(dir):
        folders = Utility.separatePath(dir)
        Utility.checkFolders(folders)

    def separatePath(path):
        return [el for el in path.split('/') if el]

    def checkFolders(folders, way='.'):
        if not folders:
            return

        if not os.path.exists(way+'/'+'/'.join(folders[:1])):
            os.mkdir(way+'/'+''.join(folders[:1]))
        Utility.checkFolders(folders[1:], way+'/'+''.join(folders[:1]))

    def checkProperty(property, data):
        if not property in data:
            data[property] = {}
