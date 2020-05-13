import os
import numpy as np
import weka.core.jvm as jvm
import Codigos.Caracteristicas as carac
import Codigos.Herramientas as hrm
import Codigos.Weka as wek
import Codigos.ArffManager as am
import Codigos.Datos as datos
import time
from tabulate import tabulate

# zonas = np.array(['ojoizq', 'ojoder', 'boca', 'nariz'])
# met_caracteristicas = np.array(['LBP', 'AU'])
# met_seleccion = np.array(['Firsts', 'PCA'])
# met_clasificacion = np.array(['RForest', 'J48', 'SVM', 'MLP'])

def Unimodal(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion, binarizo_etiquetas=False, folds=-1):
    start_total = time.time()
    jvm.start(max_heap_size="8G", packages=True)

    # print('Extracción de caracteristicas en progreso')
    # features = carac.Video(binarizo_etiquetas, zonas, met_caracteristicas)
    # for i in personas:
    #     for j in etapas:
    #         start2 = time.time()
    #         print(i + ' ' + j)
    #         features(i, j, completo=True)
    #         print(time.time() - start2)
    #
    # print('Completada extraccion de caracteristicas')
    # print(time.time() - start_total)

    if folds == -1:
        vueltas = 1
    else:
        # Contando que cuando se usa folds siempre se trabaja con toda la bd
        vueltas = int(21 / folds)
        resumen_folds = np.empty(0)

    for k in range(0, vueltas):
        if folds == -1:
            nro_instancias = am.ConcatenaArff('Resultado Video', personas, etapas, partes=-1)
            orden_instancias = np.array(range(0, nro_instancias))
            np.random.shuffle(orden_instancias)
            am.MezclaInstanciasArff('Resultado Video', orden_instancias)
            path = os.path.join(datos.PATH_CARACTERISTICAS, 'Resultado Video.arff')
            data_ori = wek.CargaYFiltrado(path)
            train_ori, test_ori = wek.ParticionaDatos(data_ori)
        else:
            print('Vuelta: ' + str(k + 1) + '/' + str(vueltas + 1))
            # Defino el conjunto de test. El de entrenamiento se define a partir de lo que no son de test
            personas_train = np.empty(0, dtype=int)
            personas_test = k * folds + np.array(range(1, folds + 1), dtype=int)
            for i in range(1, 22):
                if np.where(personas_test == i)[0].size == 0:
                    personas_train = np.append(personas_train, i)
            # Casteo a string
            personas_train = personas_train.astype(np.str)
            personas_test = personas_test.astype(np.str)
            # Los que son numero de una cifra se les tiene que agregar un 0 a la izquierda
            for i in range(0, personas_train.size):
                if int(personas_train[i]) < 10:
                    personas_train[i] = '0' + personas_train[i]
            for i in range(0, personas_test.size):
                if int(personas_test[i]) < 10:
                    personas_test[i] = '0' + personas_test[i]
            am.ConcatenaArff('Data train', personas_train, etapas, partes=-1)
            am.ConcatenaArff('Data test', personas_test, etapas, partes=-1)
            path_train = os.path.join(datos.PATH_CARACTERISTICAS, 'Data train.arff')
            path_test = os.path.join(datos.PATH_CARACTERISTICAS, 'Data test.arff')
            train_ori = wek.CargaYFiltrado(path_train)
            test_ori = wek.CargaYFiltrado(path_test)

        vec_predicciones = np.array([])
        lista_metodos = list()
        lista_errores = list()

        print('Seleccion y clasificación en progreso')
        for i in range(0, len(met_seleccion)):
            print(met_seleccion[i])
            start2 = time.time()

            if met_seleccion[i] != '':
                metodo_actual = met_seleccion[i] + ' + '
                train, test = wek.SeleccionCaracteristicas(train_ori, test_ori, met_seleccion[i])
                print(time.time() - start2)
            else:
                metodo_actual = ''
                train = train_ori
                test = test_ori
            for j in range(0, len(met_clasificacion)):
                print(met_clasificacion[j])
                start2 = time.time()
                lista_metodos.append(metodo_actual + met_clasificacion[j])
                predicciones, error = wek.Clasificacion(train, test, met_clasificacion[j])
                lista_errores.append(error)
                if len(vec_predicciones) == 0:
                    vec_predicciones = np.array([hrm.prediccionCSVtoArray(predicciones)])
                else:
                    vec_predicciones = np.concatenate([vec_predicciones, np.array([hrm.prediccionCSVtoArray(predicciones)])])
                print(time.time() - start2)

        resultados = hrm.resumePredicciones(vec_predicciones, lista_metodos, lista_errores)
        resumen_final = hrm.Fusion(resultados, 'Voto', mejores=4)
        if folds != -1:
            resumen_folds = np.append(resumen_folds, resumen_final[1, 1])
        _mostrar_tabla(resultados, resumen_final)
    if folds != -1:
        print(resumen_folds)
    jvm.stop()
    print(time.time() - start_total)


def PrimerMultimodalCompleto(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion, binarizo_etiquetas=False, elimino_silencios=False):
    start_total = time.time()
    jvm.start(packages=True)

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
    print(time.time() - start_total)

    am.ConcatenaArff('Resultado Video', personas, etapas)
    am.ConcatenaArff('Resultado Audio', personas, etapas, bool_wav=True)

    nro_instancias = am.NormalizaArff('Resultado Video', 'Resultado Audio')
    orden_instancias = np.array(range(0, nro_instancias))
    np.random.shuffle(orden_instancias)
    am.MezclaInstanciasArff('Resultado Video', orden_instancias)
    am.MezclaInstanciasArff('Resultado Audio', orden_instancias)

    path_v = os.path.join(datos.PATH_CARACTERISTICAS, 'Resultado Video.arff')
    path_a = os.path.join(datos.PATH_CARACTERISTICAS, 'Resultado Audio.arff')

    data_v_ori = wek.CargaYFiltrado(path_v)
    data_a_ori = wek.CargaYFiltrado(path_a)
    train_v_ori, test_v_ori = wek.ParticionaDatos(data_v_ori)
    train_a_ori, test_a_ori = wek.ParticionaDatos(data_a_ori)

    vec_predicciones_v = np.array([])
    vec_predicciones_a = np.array([])
    lista_metodos = list()
    lista_errores_v = list()
    lista_errores_a = list()

    print('Seleccion y clasificación en progreso')
    for i in range(0, len(met_seleccion)):
        print(met_seleccion[i])
        start2 = time.time()
        if met_seleccion[i] != '':
            metodo_actual = met_seleccion[i] + ' + '
            train_v, test_v = wek.SeleccionCaracteristicas(train_v_ori, test_v_ori, met_seleccion[i])
            train_a, test_a = wek.SeleccionCaracteristicas(train_a_ori, test_a_ori, met_seleccion[i])
            print(time.time() - start2)
        else:
            metodo_actual = ''
            train_v = train_v_ori
            test_v = test_v_ori
            train_a = train_a_ori
            test_a = test_a_ori
        for j in range(0, len(met_clasificacion)):
            print(met_clasificacion[j])
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

    resultados_v = hrm.resumePredicciones(vec_predicciones_v, lista_metodos, lista_errores_v)
    resultados_a = hrm.resumePredicciones(vec_predicciones_a, lista_metodos, lista_errores_a)
    resultados = hrm.uneResumenes(resultados_v, resultados_a)
    resumen_final = hrm.Fusion(resultados, 'Voto', mejores=2, por_modalidad=True)
    jvm.stop()
    print(time.time() - start_total)
    return resultados, resumen_final


def SegundoMultimodalCompleto(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion, binarizo_etiquetas=False):
    start_total = time.time()
    jvm.start(packages=True)

    # print('Extracción de caracteristicas en progreso')
    # features_v = carac.Video(binarizo_etiquetas, zonas, met_caracteristicas)
    # features_a = carac.Audio(binarizo_etiquetas)
    # for i in personas:
    #     for j in etapas:
    #         start2 = time.time()
    #         print(i + ' ' + j)
    #         rang_audibles = features_a(i, j, eliminar_silencios=False)
    #         features_v(i, j, completo=False, rangos_audibles=rang_audibles)
    #         print(time.time() - start2)
    #
    # print('Completada extraccion de caracteristicas')
    # print(time.time() - start_total)
    #
    # am.ConcatenaArff('Resultado Video', personas, etapas)
    # am.ConcatenaArff('Resultado Audio', personas, etapas, bool_wav=True)
    #
    # nro_instancias = am.ConcatenaArffv2('Resultado Audiovisual', 'Resultado Audio', 'Resultado Video')
    # orden_instancias = np.array(range(0, nro_instancias))
    # np.random.shuffle(orden_instancias)
    # am.MezclaInstanciasArff('Resultado Audiovisual', orden_instancias)

    path = os.path.join(datos.PATH_CARACTERISTICAS, 'Resultado Audiovisual.arff')

    data_ori = wek.CargaYFiltrado(path)
    train_ori, test_ori = wek.ParticionaDatos(data_ori)

    vec_predicciones = np.array([])
    lista_metodos = list()
    lista_errores = list()

    print('Seleccion y clasificación en progreso')
    for i in range(0, len(met_seleccion)):
        print(met_seleccion[i])
        start2 = time.time()
        if met_seleccion[i] != '':
            metodo_actual = met_seleccion[i] + ' + '
            train, test = wek.SeleccionCaracteristicas(train_ori, test_ori, met_seleccion[i])
            print(time.time() - start2)
        else:
            metodo_actual = ''
            train = train_ori
            test = test_ori
        for j in range(0, len(met_clasificacion)):
            print(met_clasificacion[j])
            lista_metodos.append(metodo_actual + met_clasificacion[j])
            predicciones, error = wek.Clasificacion(train, test, met_clasificacion[j])
            lista_errores.append(error)
            if len(vec_predicciones) == 0:
                vec_predicciones = np.array([hrm.prediccionCSVtoArray(predicciones)])
            else:
                vec_predicciones = np.concatenate(
                    [vec_predicciones, np.array([hrm.prediccionCSVtoArray(predicciones)])])

    resultados = hrm.resumePredicciones(vec_predicciones, lista_metodos, lista_errores)
    resumen_final = hrm.Fusion(resultados, 'Voto', mejores=4)
    jvm.stop()
    print(time.time() - start_total)
    return resultados, resumen_final

def _mostrar_tabla(resultados, resumen_final):
    headers = resultados[0, :]
    table = tabulate(resultados[1:2, :], headers, tablefmt="fancy_grid")
    print(table)
    headers = resumen_final[0, :]
    table = tabulate(resumen_final[1:2, :], headers, tablefmt="fancy_grid")
    print(table)
