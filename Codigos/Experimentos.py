import Codigos.Caracteristicas as carac
import Codigos.Herramientas as hrm
import Codigos.Weka as wek
import weka.core.jvm as jvm
import numpy as np
import Codigos.ArffManager as am
import os


# zonas = np.array(['ojoizq', 'ojoder', 'boca', 'nariz'])
# met_caracteristicas = np.array(['LBP', 'AU'])
# met_seleccion = np.array(['Firsts', 'PCA'])
# met_clasificacion = np.array(['RForest', 'J48', 'SVM', 'MLP'])

def Unimodal(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion, binarizo_etiquetas=False):
    jvm.start()
    selecciono_caracteristicas = False
    if len(met_seleccion) > 0:
        selecciono_caracteristicas = True

    print('Extracción de caracteristicas en progreso')
    features = carac.Video(binarizo_etiquetas, zonas, met_caracteristicas)
    for i in personas:
        for j in etapas:
            features(i, j, completo=True)
            print('...')
    print('Completada extraccion de caracteristicas')

    am.ConcatenaArff('Resultado Video', personas, etapas, bool_partes=False)
    path = 'Caracteristicas' + os.sep + 'Resultado Video.arff'
    data = wek.CargaYFiltrado(path)

    cant_met_seleccion = 1
    if selecciono_caracteristicas:
        cant_met_seleccion = len(met_seleccion)

    vec_predicciones = np.array([])
    lista_metodos = np.empty((0))

    print('Clasificación en progreso')
    for i in range(0, cant_met_seleccion):
        if selecciono_caracteristicas:
            metodo_actual = met_seleccion[i] + ' + '
            data = wek.SeleccionCaracteristicas(data, met_seleccion[i])
        else:
            metodo_actual = ''
        train, test = wek.ParticionaDatos(data)
        print('..')
        for j in range(0, len(met_clasificacion)):
            lista_metodos = np.append(lista_metodos, np.array([metodo_actual + met_clasificacion[j]]))
            predicciones = wek.Clasificacion(train, test, met_clasificacion[j])
            if len(vec_predicciones) == 0:
                vec_predicciones = np.array([hrm.prediccionCSVtoArray(predicciones)])
            else:
                vec_predicciones = np.concatenate([vec_predicciones, np.array([hrm.prediccionCSVtoArray(predicciones)])])
            print('...')

    resultados = hrm.resumoPredicciones(vec_predicciones, lista_metodos)
    return resultados

def MultimodalCompleto(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion, binarizo_etiquetas=False):
    jvm.start()
    selecciono_caracteristicas = False
    if len(met_seleccion) > 0:
        selecciono_caracteristicas = True

    print('Extracción de caracteristicas en progreso')
    features_v = carac.Video(binarizo_etiquetas, zonas, met_caracteristicas)
    features_a = carac.Audio(binarizo_etiquetas)
    for i in personas:
        for j in etapas:
            features_v(i, j, completo=False)
            features_a(i, j, eliminar_silencios=False)
            print('...')
    print('Completada extraccion de caracteristicas')

    am.ConcatenaArff('Resultado Video', personas, etapas)
    am.ConcatenaArff('Resultado Audio', personas, etapas, bool_audio=True)
    path_v = 'Caracteristicas' + os.sep + 'Resultado Video.arff'
    path_a = 'Caracteristicas' + os.sep + 'Resultado Audio.arff'

    data_v = wek.CargaYFiltrado(path_v)
    data_a = wek.CargaYFiltrado(path_a)

    cant_met_seleccion = 1
    if selecciono_caracteristicas:
        cant_met_seleccion = len(met_seleccion)

    vec_predicciones_v = np.array([])
    vec_predicciones_a = np.array([])
    lista_metodos = np.empty((0))

    print('Clasificación en progreso')
    for i in range(0, cant_met_seleccion):
        if selecciono_caracteristicas:
            metodo_actual = met_seleccion[i] + ' + '
            data_v = wek.SeleccionCaracteristicas(data_v, met_seleccion[i])
            data_a = wek.SeleccionCaracteristicas(data_a, met_seleccion[i])
        else:
            metodo_actual = ''
        train_v, test_v = wek.ParticionaDatos(data_v)
        train_a, test_a = wek.ParticionaDatos(data_a)
        print('..')
        for j in range(0, len(met_clasificacion)):
            lista_metodos = np.append(lista_metodos, np.array([metodo_actual + met_clasificacion[j]]))
            predicciones_v = wek.Clasificacion(train_v, test_v, met_clasificacion[j])
            predicciones_a = wek.Clasificacion(train_a, test_a, met_clasificacion[j])
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

# def MultimodalSinSilencios():