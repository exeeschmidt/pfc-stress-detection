import traceback

import Codigos.Caracteristicas as carac
import Codigos.Herramientas as hrm
import Codigos.Weka as wek
import weka.core.jvm as jvm
import numpy as np
import Codigos.ArffManager as am
import os

# Solamente la arranco, si la detengo a lo ultimo tira una excepcion MATLAB, se ve que la detiene por su cuenta
jvm.start()
# features = carac.VideoEntero(False, np.array(['ojoizq', 'ojoder', 'boca']), np.array(['LBP', 'AU']))
# features('03', '1')
# features = carac.VideoEnParte(False, np.array(['ojoizq', 'ojoder', 'boca']), np.array(['LBP', 'AU']))
# features('03', '1')
# features = carac.Audio(False)
# features('03', '1')

# am.ConcatenaArff('Resultado Video', np.array(['03']), np.array(['1']), True, False)
# am.ConcatenaArff('Resultado Audio', np.array(['03']), np.array(['1']), True, True)

path1 = 'Caracteristicas' + os.sep + 'Resultado Video.arff'
path2 = 'Caracteristicas' + os.sep + 'Resultado Audio.arff'

carga = wek.CargaYFiltrado()
clasi = wek.Clasificacion()

data = carga(path1)
predicciones = clasi(data, 'RForest')
vec_predic1 = hrm.convPrediccion(predicciones)

data = carga(path2)
predicciones = clasi(data, 'RForest')
vec_predic2 = hrm.convPrediccion(predicciones)

print(vec_predic1.shape[0], vec_predic2.shape[0])
[n_pre_1, n_pre_2] = hrm.segmentaPrediccion(vec_predic1, vec_predic2)
# print(n_pre_1)