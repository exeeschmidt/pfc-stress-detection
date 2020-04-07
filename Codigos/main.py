import traceback

import Codigos.Caracteristicas as carac
import Codigos.Weka as wek
import weka.core.jvm as jvm
import numpy as np
import Codigos.ArffManager as am
import os

# Solamente la arranco, si la detengo a lo ultimo tira una excepcion MATLAB, se ve que la detiene por su cuenta
jvm.start()
features = carac.VideoEntero(False, np.array(['ojoizq', 'ojoder', 'boca']), np.array(['LBP', 'AU', 'HOG']))
features('03', '1')
# features = carac.VideoEnParte(False, np.array(['ojoizq', 'ojoder', 'boca']), np.array(['LBP', 'AU', 'HOG']))
# features('03', '1')
# features = carac.Audio(False)
# features('01', '1')

# am.ConcatenaArff('Resultado Video', np.array(['03']), np.array(['1']), True, False)
# am.ConcatenaArff('Resultado Audio', np.array(['01']), np.array(['1']), True, True)
#
path = 'Caracteristicas' + os.sep + 'Sujeto_03_1.arff'
carga = wek.CargaYFiltrado()
# selec = wek.SeleccionCaracteristicas('Firsts')
clasi = wek.Clasificacion('RForest')
data = carga(path)
# data = selec(data)
clasi(data)
