import numpy as np
import weka.core.jvm as jvm
import Codigos.ExtraccionCaracteristicas as carac
import Codigos.Herramientas as hrm
import Codigos.Weka as wek
import Codigos.ArffManager as am
import Codigos.Datos as datos
import Codigos.LogManager as log
import time


def Unimodal():
    start_total = time.time()

    personas = datos.PERSONAS
    etapas = datos.ETAPAS
    zonas = datos.ZONAS
    met_extraccion = datos.MET_EXTRACCION
    met_seleccion = datos.MET_SELECCION
    met_clasificacion = datos.MET_CLASIFICACION
    binarizo_etiquetas = datos.BINARIZO_ETIQUETA
    out = datos.OUT

    jvm.start(max_heap_size="9G", packages=True)

    # print('Adaptación de caracteristicas en progreso')
    # log.agrega('Adaptación de caracteristicas en progreso')
    # features = carac.Video(binarizo_etiquetas, zonas, met_extraccion)
    # for i in personas:
    #     for j in etapas:
    #         start2 = time.time()
    #         print('Persona ' + i + ' -> Etapa ' + j)
    #         log.agrega('Persona ' + i + ' -> Etapa ' + j)
    #         features(i, j, completo=True)
    #         print(time.time() - start2)
    #         log.agrega(time.time() - start2)
    #
    # print('Completada adaptación de caracteristicas')
    # print(time.time() - start_total)
    # log.agrega('Completada adaptación de caracteristicas en progreso')
    # log.agrega(time.time() - start_total)

    vec_resultados = np.empty((0, 3, met_seleccion.size * met_clasificacion.size + 1))
    vec_resultados_fusionado = np.empty((0, 3, 2))
    vec_resultados_fusionado_2 = np.empty((0, 3, 2))

    if out == -1:
        vueltas = 1
    else:
        # Contando que cuando se usa out siempre se trabaja con toda la bd
        vueltas = int(21 / out)

    orden_instancias = np.empty(0)
    for k in range(0, vueltas):
        datos.defineFoldActual(k + 1)
        if out == -1:
            data = am.Concatena(personas, etapas, 'VCom')
            orden_instancias = am.GeneraOrdenInstancias(data, datos.INSTANCIAS_POR_PERIODOS)
            data_ori = am.MezclaInstancias(data, orden_instancias)
            train_ori, test_ori = wek.ParticionaDatos(data_ori)
        else:
            print('Vuelta: ' + str(k + 1) + '/' + str(vueltas))
            log.agrega('Vuelta: ' + str(k + 1) + '/' + str(vueltas))
            # Defino el conjunto de test. El de entrenamiento se define a partir de lo que no son de test
            personas_train = np.empty(0, dtype=int)
            personas_test = k * out + np.array(range(1, out + 1), dtype=int)
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
            train_ori = am.Concatena(personas_train, etapas, 'VCom')
            test_ori = am.Concatena(personas_test, etapas, 'VCom')

        vec_predicciones = np.array([])
        lista_metodos = list()
        lista_acu = list()
        lista_uar = list()

        print('Seleccion y clasificación en progreso')
        for i in range(0, len(met_seleccion)):
            start2 = time.time()

            if met_seleccion[i] != '':
                print(met_seleccion[i])
                log.agrega(met_seleccion[i])
                metodo_actual = met_seleccion[i] + ' + '
                train, test = wek.SeleccionCaracteristicas(train_ori, test_ori, met_seleccion[i])
                print(time.time() - start2)
                log.agrega(time.time() - start2)
            else:
                print('Sin selección de caracteristicas')
                metodo_actual = ''
                train = train_ori
                test = test_ori
            for j in range(0, len(met_clasificacion)):
                # Si no se selecciona caracteristicas y esta MLP, que no lo haga porque va a demorar demasiado
                if metodo_actual != '' or (met_clasificacion[j] != 'MLP' and met_clasificacion[j] != 'SVM'):
                    print(met_clasificacion[j])
                    log.agrega(met_clasificacion[j])
                    start2 = time.time()
                    lista_metodos.append(metodo_actual + met_clasificacion[j])
                    predi_csv = wek.Clasificacion(train, test, met_clasificacion[j], met_seleccion[i])
                    prediccion = hrm.prediccionCSVtoArray(predi_csv)
                    lista_acu.append(hrm.Accuracy(prediccion[:, 1], prediccion[:, 2]))
                    lista_uar.append(hrm.UAR(prediccion[:, 1], prediccion[:, 2]))
                    if len(vec_predicciones) == 0:
                        vec_predicciones = np.array([prediccion])
                    else:
                        vec_predicciones = np.concatenate([vec_predicciones, np.array([prediccion])])
                    print(time.time() - start2)
                    log.agrega(time.time() - start2)

        resultados = hrm.resumePredicciones(vec_predicciones, lista_metodos, lista_acu, lista_uar)
        resultados_fusionado = hrm.Fusion(resultados, 'Voto', mejores=datos.VOTO_MEJORES_X)

        if out == -1:
            resultados_fusionado, desfase = hrm.OrdenaInstancias(resultados_fusionado, orden_instancias)
            resultados_fusionado_2 = hrm.VotoPorSegmento(resultados_fusionado, datos.INSTANCIAS_POR_PERIODOS, desfase)
        else:
            resultados_fusionado_2 = hrm.VotoPorSegmento(resultados_fusionado, datos.INSTANCIAS_POR_PERIODOS)

        vec_resultados = np.concatenate([vec_resultados,  np.array([resultados[0:3, :]])], axis=0)
        vec_resultados_fusionado = np.concatenate([vec_resultados_fusionado, np.array([resultados_fusionado[0:3, :]])], axis=0)
        vec_resultados_fusionado_2 = np.concatenate([vec_resultados_fusionado_2, np.array([resultados_fusionado_2[0:3, :]])], axis=0)

        hrm.muestraTabla(resultados)
        hrm.muestraTabla(resultados_fusionado)
        hrm.muestraTabla(resultados_fusionado_2)
    if out != -1:
        resumen_final = hrm.generaResumenFinal(vec_resultados, vec_resultados_fusionado, vec_resultados_fusionado_2)
        hrm.muestraTabla(resumen_final)
    jvm.stop()
    print(time.time() - start_total)
    log.agrega('Tiempo final')
    log.agrega(time.time() - start_total)


def PrimerMultimodalCompleto(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion, binarizo_etiquetas=False, elimino_silencios=False):
    start_total = time.time()
    jvm.start(packages=True)

    print('Adaptación de caracteristicas en progreso')
    features_v = carac.Video(binarizo_etiquetas, zonas, met_caracteristicas)
    features_a = carac.Audio(binarizo_etiquetas)
    for i in personas:
        for j in etapas:
            start2 = time.time()
            print(i + ' ' + j)
            rang_audibles = features_a(i, j, eliminar_silencios=elimino_silencios)
            features_v(i, j, completo=False, rangos_audibles=rang_audibles)
            print(time.time() - start2)
    print('Completada Adaptación de caracteristicas')
    print(time.time() - start_total)

    # data_v = am.Concatena(personas, etapas, 'VResp')
    # data_a = am.Concatena(personas, etapas, 'AResp')
    # orden_instancias = am.GeneraOrdenInstancias(data_v)
    # data_v_ori = am.MezclaInstancias(data_v, orden_instancias)
    # data_a_ori = am.MezclaInstancias(data_a, orden_instancias)
    #
    # train_v_ori, test_v_ori = wek.ParticionaDatos(data_v_ori)
    # train_a_ori, test_a_ori = wek.ParticionaDatos(data_a_ori)
    #
    # vec_predicciones_v = np.array([])
    # vec_predicciones_a = np.array([])
    # lista_metodos = list()
    # lista_errores_v = list()
    # lista_errores_a = list()
    #
    # print('Seleccion y clasificación en progreso')
    # for i in range(0, len(met_seleccion)):
    #     print(met_seleccion[i])
    #     start2 = time.time()
    #     if met_seleccion[i] != '':
    #         metodo_actual = met_seleccion[i] + ' + '
    #         train_v, test_v = wek.SeleccionCaracteristicas(train_v_ori, test_v_ori, met_seleccion[i])
    #         train_a, test_a = wek.SeleccionCaracteristicas(train_a_ori, test_a_ori, met_seleccion[i])
    #         print(time.time() - start2)
    #     else:
    #         metodo_actual = ''
    #         train_v = train_v_ori
    #         test_v = test_v_ori
    #         train_a = train_a_ori
    #         test_a = test_a_ori
    #     for j in range(0, len(met_clasificacion)):
    #         print(met_clasificacion[j])
    #         lista_metodos.append(metodo_actual + met_clasificacion[j])
    #         predicciones_v, error = wek.Clasificacion(train_v, test_v, met_clasificacion[j])
    #         lista_errores_v.append(error)
    #         predicciones_a, error = wek.Clasificacion(train_a, test_a, met_clasificacion[j])
    #         lista_errores_a.append(error)
    #         if len(vec_predicciones_v) == 0 or len(vec_predicciones_a) == 0:
    #             vec_predicciones_v = np.array([hrm.prediccionCSVtoArray(predicciones_v)])
    #             vec_predicciones_a = np.array([hrm.prediccionCSVtoArray(predicciones_a)])
    #         else:
    #             vec_predicciones_v = np.concatenate(
    #                 [vec_predicciones_v, np.array([hrm.prediccionCSVtoArray(predicciones_v)])])
    #             vec_predicciones_a = np.concatenate(
    #                 [vec_predicciones_a, np.array([hrm.prediccionCSVtoArray(predicciones_a)])])
    #
    # resultados_v = hrm.resumePredicciones(vec_predicciones_v, lista_metodos, lista_errores_v)
    # resultados_a = hrm.resumePredicciones(vec_predicciones_a, lista_metodos, lista_errores_a)
    # resultados = hrm.uneResumenes(resultados_v, resultados_a)
    # resumen_final = hrm.Fusion(resultados, 'Voto', mejores=2, por_modalidad=True)
    jvm.stop()
    print(time.time() - start_total)
    # return resultados, resumen_final


def SegundoMultimodalCompleto(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion, binarizo_etiquetas=False):
    start_total = time.time()
    jvm.start(packages=True)

    # print('Adaptación de caracteristicas en progreso')
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
    # print('Completada Adaptación de caracteristicas')
    # print(time.time() - start_total)
    #
    data = am.Concatena(personas, etapas, 'VResp', 'AResp')
    orden_instancias = am.GeneraOrdenInstancias(data, datos.INSTANCIAS_POR_PERIODOS)
    data_ori = am.MezclaInstancias(data, orden_instancias)
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


def ExtractorDeCaracteristicas(personas, etapas, zonas):
    start_total = time.time()
    jvm.start()
    print('Extracción de caracteristicas en progreso')
    features = carac.CaracteristicasVideo(zonas)
    for i in personas:
        for j in etapas:
            if i != '09' or j != '1':
                start2 = time.time()
                print('Persona ' + i + ' -> Etapa ' + j)
                features(i, j)
                print(time.time() - start2)
    print('Completada extraccion de caracteristicas')
    print(time.time() - start_total)
    # jvm.stop()
