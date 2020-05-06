import os
import numpy as np
import weka.core.jvm as jvm
import Codigos.Caracteristicas as carac
import Codigos.Herramientas as hrm
import Codigos.Weka as wek
import Codigos.ArffManager as am
import Codigos.Datos as datos
import time

# zonas = np.array(['ojoizq', 'ojoder', 'boca', 'nariz'])
# met_caracteristicas = np.array(['LBP', 'AU'])
# met_seleccion = np.array(['Firsts', 'PCA'])
# met_clasificacion = np.array(['RForest', 'J48', 'SVM', 'MLP'])

def Unimodal(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion, binarizo_etiquetas=False):
    start_total = time.time()
    jvm.start(packages=True)
    selecciono_caracteristicas = False
    if len(met_seleccion) > 0:
        selecciono_caracteristicas = True

    print('Extracción de caracteristicas en progreso')
    features = carac.Video(binarizo_etiquetas, zonas, met_caracteristicas)
    for i in personas:
        for j in etapas:
            start2 = time.time()
            print(i + ' ' + j)
            features(i, j, completo=True)
            print(time.time() - start2)

    print('Completada extraccion de caracteristicas')
    print(time.time() - start_total)

    am.ConcatenaArff('Resultado Video', personas, etapas, partes=-1)
    path = os.path.join(datos.PATH_CARACTERISTICAS, 'Resultado Video.arff')
    data_ori = wek.CargaYFiltrado(path)

    cant_met_seleccion = 1
    if selecciono_caracteristicas:
        cant_met_seleccion = len(met_seleccion)

    vec_predicciones = np.array([])
    lista_metodos = list()
    lista_errores = list()

    print('Clasificación en progreso')
    for i in range(0, cant_met_seleccion):
        if selecciono_caracteristicas:
            metodo_actual = met_seleccion[i] + ' + '
            data = wek.SeleccionCaracteristicas(data_ori, met_seleccion[i])
        else:
            metodo_actual = ''
            data = data_ori
        train, test = wek.ParticionaDatos(data)
        print(time.time() - start_total)
        print('..')
        for j in range(0, len(met_clasificacion)):
            lista_metodos.append(metodo_actual + met_clasificacion[j])
            predicciones, error = wek.Clasificacion(train, test, met_clasificacion[j])
            lista_errores.append(error)
            if len(vec_predicciones) == 0:
                vec_predicciones = np.array([hrm.prediccionCSVtoArray(predicciones)])
            else:
                vec_predicciones = np.concatenate([vec_predicciones, np.array([hrm.prediccionCSVtoArray(predicciones)])])
            print(time.time() - start_total)
            print('...')

    resultados = hrm.resumePredicciones(vec_predicciones, lista_metodos, lista_errores)
    jvm.stop()
    print(time.time() - start_total)
    return resultados


def PrimerMultimodalCompleto(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion,
                       binarizo_etiquetas=False, elimino_silencios=False):
    start_total = time.time()
    jvm.start(packages=True)
    selecciono_caracteristicas = False
    if len(met_seleccion) > 0:
        selecciono_caracteristicas = True

    print('Extracción de caracteristicas en progreso')
    features_v = carac.Video(binarizo_etiquetas, zonas, met_caracteristicas)
    features_a = carac.Audio(binarizo_etiquetas)
    for i in personas:
        for j in etapas:
            start2 = time.time()
            print(i + ' ' + j)
            rang_audibles = features_a(i, j, eliminar_silencios=elimino_silencios)
            features_v(i, j, completo=False, rangos_audibles=rang_audibles)
            print(time.time() - start2)
    print('Completada extraccion de caracteristicas')

    am.ConcatenaArff('Resultado Video', personas, etapas)
    am.ConcatenaArff('Resultado Audio', personas, etapas, bool_wav=True)
    path_v = os.path.join(datos.PATH_CARACTERISTICAS, 'Resultado Video.arff')
    path_a = os.path.join(datos.PATH_CARACTERISTICAS, 'Resultado Audio.arff')

    data_v_ori = wek.CargaYFiltrado(path_v)
    data_a_ori = wek.CargaYFiltrado(path_a)

    cant_met_seleccion = 1
    if selecciono_caracteristicas:
        cant_met_seleccion = len(met_seleccion)

    vec_predicciones_v = np.array([])
    vec_predicciones_a = np.array([])
    lista_metodos = list()
    lista_errores_v = list()
    lista_errores_a = list()

    print('Clasificación en progreso')
    for i in range(0, cant_met_seleccion):
        if selecciono_caracteristicas:
            metodo_actual = met_seleccion[i] + ' + '
            data_v = wek.SeleccionCaracteristicas(data_v_ori, met_seleccion[i])
            data_a = wek.SeleccionCaracteristicas(data_a_ori, met_seleccion[i])
        else:
            metodo_actual = ''
            data_v = data_v_ori
            data_a = data_a_ori
        train_v, test_v = wek.ParticionaDatos(data_v)
        train_a, test_a = wek.ParticionaDatos(data_a)
        print('..')
        for j in range(0, len(met_clasificacion)):
            lista_metodos.append(metodo_actual + met_clasificacion[j])
            predicciones_v, error = wek.Clasificacion(train_v, test_v, met_clasificacion[j])
            lista_errores_v.append(error)
            predicciones_a, error = wek.Clasificacion(train_a, test_a, met_clasificacion[j])
            lista_errores_a.append(error)
            if len(vec_predicciones_v) == 0 or len(vec_predicciones_a) == 0:
                vec_predicciones_v = np.array([hrm.prediccionCSVtoArray(predicciones_v)])
                vec_predicciones_a = np.array([hrm.prediccionCSVtoArray(predicciones_a)])
            else:
                vec_predicciones_v = np.concatenate(
                    [vec_predicciones_v, np.array([hrm.prediccionCSVtoArray(predicciones_v)])])
                vec_predicciones_a = np.concatenate(
                    [vec_predicciones_a, np.array([hrm.prediccionCSVtoArray(predicciones_a)])])
            print('...')

    resultados_v = hrm.resumePredicciones(vec_predicciones_v, lista_metodos, lista_errores_v)
    resultados_a = hrm.resumePredicciones(vec_predicciones_a, lista_metodos, lista_errores_a)
    resultados = hrm.uneResumenes(resultados_v, resultados_a)
    jvm.stop()
    print(time.time() - start_total)
    return resultados

def SegundoMultimodalCompleto(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion,
                       binarizo_etiquetas=False):
    start_total = time.time()
    jvm.start(packages=True)
    selecciono_caracteristicas = False
    if len(met_seleccion) > 0:
        selecciono_caracteristicas = True

    print('Extracción de caracteristicas en progreso')
    features_v = carac.Video(binarizo_etiquetas, zonas, met_caracteristicas)
    features_a = carac.Audio(binarizo_etiquetas)
    for i in personas:
        for j in etapas:
            start2 = time.time()
            print(i + ' ' + j)
            rang_audibles = features_a(i, j, eliminar_silencios=False)
            features_v(i, j, completo=False, rangos_audibles=rang_audibles)
            print(time.time() - start2)

    print('Completada extraccion de caracteristicas')

    am.ConcatenaArff('Resultado Video', personas, etapas)
    am.ConcatenaArff('Resultado Audio', personas, etapas, bool_wav=True)
    am.ConcatenaArffv2('Resultado Audiovisual', 'Resultado Audio', 'Resultado Video')

    path = os.path.join(datos.PATH_CARACTERISTICAS, 'Resultado Audiovisual.arff')

    data_ori = wek.CargaYFiltrado(path)

    cant_met_seleccion = 1
    if selecciono_caracteristicas:
        cant_met_seleccion = len(met_seleccion)

    vec_predicciones = np.array([])
    lista_metodos = list()
    lista_errores = list()

    print('Clasificación en progreso')
    for i in range(0, cant_met_seleccion):
        if selecciono_caracteristicas:
            metodo_actual = met_seleccion[i] + ' + '
            data = wek.SeleccionCaracteristicas(data_ori, met_seleccion[i])
        else:
            metodo_actual = ''
            data = data_ori
        train, test = wek.ParticionaDatos(data)
        print('..')
        for j in range(0, len(met_clasificacion)):
            lista_metodos.append(metodo_actual + met_clasificacion[j])
            predicciones, error = wek.Clasificacion(train, test, met_clasificacion[j])
            lista_errores.append(error)
            if len(vec_predicciones) == 0:
                vec_predicciones = np.array([hrm.prediccionCSVtoArray(predicciones)])
            else:
                vec_predicciones = np.concatenate(
                    [vec_predicciones, np.array([hrm.prediccionCSVtoArray(predicciones)])])
            print('...')

    resultados = hrm.resumePredicciones(vec_predicciones, lista_metodos, lista_errores)
    jvm.stop()
    print(time.time() - start_total)
    return resultados

