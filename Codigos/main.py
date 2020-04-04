import Caracteristicas
import Weka
import numpy as np
import ArffManager as am
import os

features = Caracteristicas.VideoEnParte(False, np.array(['cejaizq', 'cejader', 'ojoizq', 'ojoder', 'boca']))
features('01', '1')
# am.ConcatenaArff('Resultado', np.array(['01']), np.array(['1']), True)
# clasificador = Weka.Clasificacion('RForest')
# clasificador('Caracteristicas' + os.sep + 'Resultado.arff')
