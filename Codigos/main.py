# import Codigos.Caracteristicas as carac
# import Codigos.Herramientas as hrm
# import Codigos.Weka as wek
# import weka.core.jvm as jvm
# import numpy as np
# import Codigos.ArffManager as am
# import os

# Solamente la arranco, si la detengo a lo ultimo tira una excepcion MATLAB, se ve que la detiene por su cuenta
# jvm.start()
# features = carac.Audio(False)
# segmentos_audibles = features('03', '1', eliminar_silencios=True)
# am.ConcatenaArff('Resultado Audio', np.array(['03']), np.array(['1']), bool_audio=True, rangos_audibles=segmentos_audibles)
#
# features = carac.Video(False, np.array(['ojoizq', 'ojoder', 'boca', 'nariz']), np.array(['LBP', 'AU']))
# features('03', '1', completo=False, rangos_audibles=segmentos_audibles)
# am.ConcatenaArff('Resultado Video', np.array(['03']), np.array(['1']))

# path1 = 'Caracteristicas' + os.sep + 'Resultado Video.arff'
# path2 = 'Caracteristicas' + os.sep + 'Resultado Audio.arff'
# data = wek.CargaYFiltrado(path1)
# train, test = wek.ParticionaDatos(data)
# predicciones = wek.Clasificacion(train, test, 'RForest', sumario=True)
# vec_predic_video = hrm.prediccionCSVtoArray(predicciones)
# print(hrm.prediccionCSVtoArray(predicciones))
# predicciones = clasi(data, 'SVM', sumario=True)
# vec_predic_video = np.concatenate([vec_predic_video, np.array([hrm.prediccionCSVtoArray(predicciones)])])
#
# data = carga(path2)
# predicciones = clasi(data, 'RForest', sumario=True)
# vec_predic_audio = np.array([hrm.prediccionCSVtoArray(predicciones)])
# predicciones = clasi(data, 'SVM', sumario=True)
# vec_predic_audio = np.concatenate([vec_predic_audio, np.array([hrm.prediccionCSVtoArray(predicciones)])])
#
#
# print(vec_predic_video.shape, '/', vec_predic_audio.shape)
# [new_predic_video, new_predic_audio] = hrm.segmentaPrediccion(vec_predic_video, vec_predic_audio)
# print(new_predic_video.shape, '/', new_predic_audio.shape)

import Codigos.Experimentos as exp
import numpy as np
from tabulate import tabulate

personas = np.array(['01'])
etapas = np.array(['1'])
zonas = np.array(['ojoizq', 'ojoder', 'boca', 'nariz'])
met_caracteristicas = np.array(['LBP', 'AU'])
met_seleccion = np.array(['Firsts', 'PCA'])
# met_seleccion = np.array([])
met_clasificacion = np.array(['RForest', 'J48', 'SVM'])

resultados = exp.Unimodal(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion)
headers = resultados[0, :]
table = tabulate(resultados[1:, :], headers, tablefmt="fancy_grid")
print(table)

print('Fin de ejecucion')