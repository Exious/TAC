import os
import re

class Utility:
    def __init__(self):
        self = self

    def renameImage(name):
        return '-'.join(name.split())

    def saveFile(figure, figure_name):
        path = './LaTeX/body/images/' + \
            Utility.renameImage(figure_name) + '.png'

        if(os.path.isfile(path)):
            os.remove(path)

        figure.savefig(path)

    def star_replacer(mu):
        return re.sub(r'\*', '', mu)