import os
import numpy as np
import weka.core.jvm as jvm
import Codigos.Caracteristicas as carac
import Codigos.Herramientas as hrm
import Codigos.Weka as wek
import Codigos.ArffManager as am
import Codigos.Datos as datos


# zonas = np.array(['ojoizq', 'ojoder', 'boca', 'nariz'])
# met_caracteristicas = np.array(['LBP', 'AU'])
# met_seleccion = np.array(['Firsts', 'PCA'])
# met_clasificacion = np.array(['RForest', 'J48', 'SVM', 'MLP'])

def unimodal(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion, binarizo_etiquetas=False):
    jvm.start(packages=True)
    selecciono_caracteristicas = False
    if len(met_seleccion) > 0:
        selecciono_caracteristicas = True

    print('Extracción de caracteristicas en progreso')
    features = carac.Video(zonas, met_caracteristicas, binarizo_etiquetas)
    for i in personas:
        for j in etapas:
            features(i, j, completo=True)
            print('...')
    print('Completada extraccion de caracteristicas')

    am.concatenaArff('Resultado Video', personas, etapas, bool_partes=False)
    path = os.path.join(datos.PATH_CARACTERISTICAS, 'Resultado Video.arff')
    data = wek.cargaYFiltrado(path)

    cant_met_seleccion = 1
    if selecciono_caracteristicas:
        cant_met_seleccion = len(met_seleccion)

    vec_predicciones = np.array([])
    lista_metodos = np.empty(0)

    print('Clasificación en progreso')
    for i in range(0, cant_met_seleccion):
        if selecciono_caracteristicas:
            metodo_actual = met_seleccion[i] + ' + '
            data = wek.seleccionCaracteristicas(data, met_seleccion[i])
        else:
            metodo_actual = ''
        train, test = wek.particionaDatos(data)
        print('..')
        for j in range(0, len(met_clasificacion)):
            lista_metodos = np.append(lista_metodos, np.array([metodo_actual + met_clasificacion[j]]))
            predicciones = wek.clasificacion(train, test, met_clasificacion[j])
            if len(vec_predicciones) == 0:
                vec_predicciones = np.array([hrm.prediccionCSVtoArray(predicciones)])
            else:
                vec_predicciones = np.concatenate([vec_predicciones, np.array([hrm.prediccionCSVtoArray(predicciones)])])
            print('...')

    resultados = hrm.resumoPredicciones(vec_predicciones, lista_metodos)
    return resultados


def multimodalCompleto(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion,
                       binarizo_etiquetas=False, elimino_silencios=False):
    jvm.start(packages=True)
    if len(met_seleccion) > 0:
        selecciono_caracteristicas = True
    else:
        selecciono_caracteristicas = False

    print('Extracción de caracteristicas en progreso')
    features_v = carac.Video(zonas, met_caracteristicas, binarizo_etiquetas)
    features_a = carac.Audio(binarizo_etiquetas)
    for persona in personas:
        for etapa in etapas:
            rang_audibles = features_a(persona, etapa, eliminar_silencios=elimino_silencios)
            features_v(persona, etapa, completo=False, rangos_audibles=rang_audibles)
            print('...')
    print('Completada extraccion de caracteristicas')

    am.concatenaArff('Resultado Video', personas, etapas)
    am.concatenaArff('Resultado Audio', personas, etapas, bool_audio=True)
    path_v = os.path.join(datos.PATH_CARACTERISTICAS, 'Resultado Video.arff')
    path_a = os.path.join(datos.PATH_CARACTERISTICAS, 'Resultado Audio.arff')

    data_v = wek.cargaYFiltrado(path_v)
    data_a = wek.cargaYFiltrado(path_a)

    cant_met_seleccion = 1
    if selecciono_caracteristicas:
        cant_met_seleccion = len(met_seleccion)

    vec_predicciones_v = np.array([])
    vec_predicciones_a = np.array([])
    lista_metodos = np.empty(0)

    print('Clasificación en progreso')
    for i in range(0, cant_met_seleccion):
        if selecciono_caracteristicas:
            metodo_actual = met_seleccion[i] + ' + '
            data_v = wek.seleccionCaracteristicas(data_v, met_seleccion[i])
            data_a = wek.seleccionCaracteristicas(data_a, met_seleccion[i])
        else:
            metodo_actual = ''
        train_v, test_v = wek.particionaDatos(data_v)
        train_a, test_a = wek.particionaDatos(data_a)
        print('..')
        for j in range(0, len(met_clasificacion)):
            lista_metodos = np.append(lista_metodos, np.array([metodo_actual + met_clasificacion[j]]))
            predicciones_v = wek.clasificacion(train_v, test_v, met_clasificacion[j])
            predicciones_a = wek.clasificacion(train_a, test_a, met_clasificacion[j])
            if len(vec_predicciones_v) == 0 or len(vec_predicciones_a) == 0:
                vec_predicciones_v = np.array([hrm.prediccionCSVtoArray(predicciones_v)])
                vec_predicciones_a = np.array([hrm.prediccionCSVtoArray(predicciones_a)])
            else:
                vec_predicciones_v = np.concatenate(
                    [vec_predicciones_v, np.array([hrm.prediccionCSVtoArray(predicciones_v)])])
                vec_predicciones_a = np.concatenate(
                    [vec_predicciones_a, np.array([hrm.prediccionCSVtoArray(predicciones_a)])])
            print('...')

    resultados_v = hrm.resumoPredicciones(vec_predicciones_v, lista_metodos)
    resultados_a = hrm.resumoPredicciones(vec_predicciones_a, lista_metodos)
    resultados = hrm.segmentaResumen(resultados_v, resultados_a)
    return resultados
