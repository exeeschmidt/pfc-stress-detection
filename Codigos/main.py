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
# features = carac.VideoCompleto(False, np.array(['ojoizq', 'ojoder', 'boca', 'nariz']), np.array(['LBP', 'HOP', 'AU']))
# features('05', '1')

# features = carac.VideoPorRespuesta(False, np.array(['ojoizq', 'ojoder', 'boca']), np.array(['LBP', 'AU']))
# features('03', '1')
am.ConcatenaArff('Resultado Video', np.array(['03']), np.array(['1']), True, False)

features = carac.Audio(False)
features('03', '1', True)
# am.ConcatenaArff('Resultado Audio', np.array(['03']), np.array(['1']), True, True)

# path1 = 'Caracteristicas' + os.sep + 'Resultado Video.arff'
# path2 = 'Caracteristicas' + os.sep + 'Resultado Audio.arff'
#
# carga = wek.CargaYFiltrado()
# clasi = wek.Clasificacion()
#
# data = carga(path1)
# predicciones = clasi(data, 'RForest')
# vec_predic_video = np.array([hrm.convPrediccion(predicciones)])
# predicciones = clasi(data, 'SVM')
# vec_predic_video = np.concatenate([vec_predic_video, np.array([hrm.convPrediccion(predicciones)])])
#
# data = carga(path2)
# predicciones = clasi(data, 'RForest')
# vec_predic_audio = np.array([hrm.convPrediccion(predicciones)])
# predicciones = clasi(data, 'SVM')
# vec_predic_audio = np.concatenate([vec_predic_audio, np.array([hrm.convPrediccion(predicciones)])])
#
#
# print(vec_predic_video.shape, '/', vec_predic_audio.shape)
# [new_predic_video, new_predic_audio] = hrm.segmentaPrediccion(vec_predic_video, vec_predic_audio)
# print(new_predic_video.shape, '/', new_predic_audio.shape)
print('Fin de ejecucion')