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
    nro_val = datos.VAL
    nro_test = datos.TEST

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
    # log.agrega('Completada adaptación de caracteristicas')
    # log.agrega(time.time() - start_total)

    vec_resultados = np.empty((0, 3, met_seleccion.size * met_clasificacion.size + 1))
    vec_resultados_fusionado = np.empty((0, 3, 2))
    vec_resultados_fusionado_2 = np.empty((0, 3, 2))

    if nro_test == -1:
        vueltas = 1
    else:
        # Contando que cuando se usa test siempre se trabaja con toda la bd
        vueltas = int(21 / nro_test)

    orden_instancias = np.empty(0)
    for k in range(0, vueltas):
        datos.defineFoldActual(k + 1)
        if nro_test == -1:
            data = am.Concatena(personas, etapas, 'VCom')
            orden_instancias = am.GeneraOrdenInstancias(data, datos.INSTANCIAS_POR_PERIODOS)
            data_ori = am.MezclaInstancias(data, orden_instancias)
            train_ori, test_ori = wek.ParticionaDatos(data_ori)
            train_ori, val_ori = wek.ParticionaDatos(train_ori)
        else:
            print('Vuelta: ' + str(k + 1) + '/' + str(vueltas))
            log.agrega('Vuelta: ' + str(k + 1) + '/' + str(vueltas))
            personas_train, personas_val, personas_test = GeneraConjuntos(k, nro_val, nro_test)
            train_ori = am.Concatena(personas_train, etapas, 'VCom')
            val_ori = am.Concatena(personas_val, etapas, 'VCom')
            test_ori = am.Concatena(personas_test, etapas, 'VCom')

        vec_predicciones_val = np.array([])
        vec_predicciones_tst = np.array([])
        lista_metodos = list()
        lista_acu_tst = list()
        lista_uar_tst = list()
        lista_acu_val = list()
        lista_uar_val = list()

        print('Seleccion y clasificación en progreso')
        for i in range(0, len(met_seleccion)):
            start2 = time.time()

            if met_seleccion[i] != '':
                print(met_seleccion[i])
                log.agrega(met_seleccion[i])
                metodo_actual = met_seleccion[i] + ' + '
                train, val, test = wek.SeleccionCaracteristicas(train_ori, val_ori, test_ori, met_seleccion[i])
                print(time.time() - start2)
                log.agrega(time.time() - start2)
            else:
                print('Sin selección de caracteristicas')
                metodo_actual = ''
                train = train_ori
                val = val_ori
                test = test_ori
            for j in range(0, len(met_clasificacion)):
                # Si no se selecciona caracteristicas y esta MLP, que no lo haga porque va a demorar demasiado
                if metodo_actual != '' or met_clasificacion[j] != 'MLP':
                    print(met_clasificacion[j])
                    log.agrega(met_clasificacion[j])
                    start2 = time.time()
                    lista_metodos.append(metodo_actual + met_clasificacion[j])
                    predi_csv_val, predi_csv_tst = wek.Clasificacion(train, val, test, met_clasificacion[j], met_seleccion[i])
                    prediccion_val = hrm.prediccionCSVtoArray(predi_csv_val)
                    prediccion_tst = hrm.prediccionCSVtoArray(predi_csv_tst)
                    lista_acu_val.append(hrm.Accuracy(prediccion_val[:, 1], prediccion_val[:, 2]))
                    lista_acu_tst.append(hrm.Accuracy(prediccion_tst[:, 1], prediccion_tst[:, 2]))
                    lista_uar_val.append(hrm.UAR(prediccion_val[:, 1], prediccion_val[:, 2]))
                    lista_uar_tst.append(hrm.UAR(prediccion_tst[:, 1], prediccion_tst[:, 2]))
                    if len(vec_predicciones_val) == 0:
                        vec_predicciones_val = np.array([prediccion_val])
                        vec_predicciones_tst = np.array([prediccion_tst])
                    else:
                        vec_predicciones_val = np.concatenate([vec_predicciones_val, np.array([prediccion_val])])
                        vec_predicciones_tst = np.concatenate([vec_predicciones_tst, np.array([prediccion_tst])])
                    print(time.time() - start2)
                    log.agrega(time.time() - start2)

        resultados_val = hrm.resumePredicciones(vec_predicciones_val, lista_metodos, lista_acu_val, lista_uar_val)
        resultados_tst = hrm.resumePredicciones(vec_predicciones_tst, lista_metodos, lista_acu_tst, lista_uar_tst)

        indice_mejores = hrm.EleccionFusion(resultados_val, mejores=datos.VOTO_MEJORES_X)
        aux_mejores_metodos = 'Mejores combinaciones para la fusion: '
        for i in range(0, indice_mejores.size):
            aux_mejores_metodos = aux_mejores_metodos + '[' + str(resultados_tst[0, indice_mejores[i]]) + ']'
        print(aux_mejores_metodos)
        log.agrega(aux_mejores_metodos)
        resultados_fusionado = hrm.Fusion(resultados_tst, 'Voto', indice_mejores)

        if test == -1:
            resultados_fusionado, desfase = hrm.OrdenaInstancias(resultados_fusionado, orden_instancias)
            resultados_fusionado_2 = hrm.VotoPorSegmento(resultados_fusionado, datos.INSTANCIAS_POR_PERIODOS, desfase)
        else:
            resultados_fusionado_2 = hrm.VotoPorSegmento(resultados_fusionado, datos.INSTANCIAS_POR_PERIODOS)

        vec_resultados = np.concatenate([vec_resultados,  np.array([resultados_tst[0:3, :]])], axis=0)
        vec_resultados_fusionado = np.concatenate([vec_resultados_fusionado, np.array([resultados_fusionado[0:3, :]])], axis=0)
        vec_resultados_fusionado_2 = np.concatenate([vec_resultados_fusionado_2, np.array([resultados_fusionado_2[0:3, :]])], axis=0)

        hrm.muestraTabla(resultados_tst)
        hrm.muestraTabla(resultados_fusionado)
        hrm.muestraTabla(resultados_fusionado_2)
    if test != -1:
        resumen_final = hrm.generaResumenFinal(vec_resultados, vec_resultados_fusionado, vec_resultados_fusionado_2)
        hrm.muestraTabla(resumen_final)
    jvm.stop()
    print(time.time() - start_total)
    log.agrega('Tiempo final')
    log.agrega(time.time() - start_total)


def PrimerMultimodalCompleto(elimino_silencios=False):
    start_total = time.time()
    personas = datos.PERSONAS
    etapas = datos.ETAPAS
    zonas = datos.ZONAS
    met_extraccion = datos.MET_EXTRACCION
    met_seleccion = datos.MET_SELECCION
    met_clasificacion = datos.MET_CLASIFICACION
    binarizo_etiquetas = datos.BINARIZO_ETIQUETA
    nro_val = datos.VAL
    nro_test = datos.TEST

    jvm.start(max_heap_size="9G", packages=True)

    # print('Adaptación de caracteristicas en progreso')
    # log.agrega('Adaptación de caracteristicas en progreso')
    # features_v = carac.Video(binarizo_etiquetas, zonas, met_extraccion)
    # features_a = carac.Audio(binarizo_etiquetas)
    # for i in personas:
    #     for j in etapas:
    #         start2 = time.time()
    #         print('Persona ' + i + ' -> Etapa ' + j)
    #         log.agrega('Persona ' + i + ' -> Etapa ' + j)
    #         rang_audibles = features_a(i, j, eliminar_silencios=elimino_silencios)
    #         features_v(i, j, completo=False, rangos_audibles=rang_audibles)
    #         print(time.time() - start2)
    #         log.agrega(time.time() - start2)
    #
    # print('Completada adaptación de caracteristicas')
    # print(time.time() - start_total)
    # log.agrega('Completada adaptación de caracteristicas')
    # log.agrega(time.time() - start_total)

    vec_resultados = np.empty((0, 3, 2 * met_seleccion.size * met_clasificacion.size + 1))
    vec_resultados_fusionado = np.empty((0, 3, 2))
    vec_resultados_fusionado_2 = np.empty((0, 3, 2))

    if nro_test == -1:
        vueltas = 1
    else:
        # Contando que cuando se usa test siempre se trabaja con toda la bd
        vueltas = int(21 / nro_test)

    orden_instancias = np.empty(0)
    for k in range(0, vueltas):
        datos.defineFoldActual(k + 1)
        if nro_test == -1:
            data_v = am.Concatena(personas, etapas, 'VResp')
            data_a = am.Concatena(personas, etapas, 'AResp')
            orden_instancias = am.GeneraOrdenInstancias(data_v, datos.INSTANCIAS_POR_PERIODOS)
            data_v_ori = am.MezclaInstancias(data_v, orden_instancias)
            data_a_ori = am.MezclaInstancias(data_a, orden_instancias)
            train_v_ori, test_v_ori = wek.ParticionaDatos(data_v_ori)
            train_v_ori, val_v_ori = wek.ParticionaDatos(train_v_ori)
            train_a_ori, test_a_ori = wek.ParticionaDatos(data_a_ori)
            train_a_ori, val_a_ori = wek.ParticionaDatos(train_a_ori)
        else:
            print('Vuelta: ' + str(k + 1) + '/' + str(vueltas))
            log.agrega('Vuelta: ' + str(k + 1) + '/' + str(vueltas))
            personas_train, personas_val, personas_test = GeneraConjuntos(k, nro_val, nro_test)
            train_v_ori = am.Concatena(personas_train, etapas, 'VResp')
            val_v_ori = am.Concatena(personas_val, etapas, 'VResp')
            test_v_ori = am.Concatena(personas_test, etapas, 'VResp')
            train_a_ori = am.Concatena(personas_train, etapas, 'AResp')
            val_a_ori = am.Concatena(personas_val, etapas, 'AResp')
            test_a_ori = am.Concatena(personas_test, etapas, 'AResp')

        vec_predicciones_v_val = np.array([])
        vec_predicciones_v_tst = np.array([])
        vec_predicciones_a_val = np.array([])
        vec_predicciones_a_tst = np.array([])
        lista_metodos = list()
        lista_acu_v_val = list()
        lista_acu_v_tst = list()
        lista_acu_a_val = list()
        lista_acu_a_tst = list()
        lista_uar_v_val = list()
        lista_uar_v_tst = list()
        lista_uar_a_val = list()
        lista_uar_a_tst = list()

        print('Seleccion y clasificación en progreso')
        for i in range(0, len(met_seleccion)):
            start2 = time.time()

            if met_seleccion[i] != '':
                print(met_seleccion[i])
                log.agrega(met_seleccion[i])
                metodo_actual = met_seleccion[i] + ' + '
                train_v, val_v, test_v = wek.SeleccionCaracteristicas(train_v_ori, val_v_ori, test_v_ori, met_seleccion[i])
                train_a, val_a, test_a = wek.SeleccionCaracteristicas(train_a_ori, val_a_ori, test_a_ori, met_seleccion[i])
                print(time.time() - start2)
                log.agrega(time.time() - start2)
            else:
                print('Sin selección de caracteristicas')
                metodo_actual = ''
                train_v = train_v_ori
                val_v = val_v_ori
                test_v = test_v_ori
                train_a = train_a_ori
                val_a = val_a_ori
                test_a = test_a_ori
            for j in range(0, len(met_clasificacion)):
                # Si no se selecciona caracteristicas y esta MLP, que no lo haga porque va a demorar demasiado
                if metodo_actual != '' or met_clasificacion[j] != 'MLP':
                    print(met_clasificacion[j])
                    log.agrega(met_clasificacion[j])
                    start2 = time.time()
                    lista_metodos.append(metodo_actual + met_clasificacion[j])
                    predi_csv_v_val, predi_csv_v_tst = wek.Clasificacion(train_v, val_v, test_v, met_clasificacion[j], met_seleccion[i])
                    predi_csv_a_val, predi_csv_a_tst = wek.Clasificacion(train_a, val_a, test_a, met_clasificacion[j], met_seleccion[i])
                    prediccion_v_val = hrm.prediccionCSVtoArray(predi_csv_v_val)
                    prediccion_v_tst = hrm.prediccionCSVtoArray(predi_csv_v_tst)
                    prediccion_a_val = hrm.prediccionCSVtoArray(predi_csv_a_val)
                    prediccion_a_tst = hrm.prediccionCSVtoArray(predi_csv_a_tst)
                    lista_acu_v_val.append(hrm.Accuracy(prediccion_v_val[:, 1], prediccion_v_val[:, 2]))
                    lista_acu_a_val.append(hrm.Accuracy(prediccion_a_val[:, 1], prediccion_a_val[:, 2]))
                    lista_acu_v_tst.append(hrm.UAR(prediccion_v_tst[:, 1], prediccion_v_tst[:, 2]))
                    lista_acu_a_tst.append(hrm.UAR(prediccion_a_tst[:, 1], prediccion_a_tst[:, 2]))
                    lista_uar_v_val.append(hrm.Accuracy(prediccion_v_val[:, 1], prediccion_v_val[:, 2]))
                    lista_uar_a_val.append(hrm.Accuracy(prediccion_a_val[:, 1], prediccion_a_val[:, 2]))
                    lista_uar_v_tst.append(hrm.UAR(prediccion_v_tst[:, 1], prediccion_v_tst[:, 2]))
                    lista_uar_a_tst.append(hrm.UAR(prediccion_a_tst[:, 1], prediccion_a_tst[:, 2]))
                    if len(vec_predicciones_v_val) == 0:
                        vec_predicciones_v_val = np.array([prediccion_v_val])
                        vec_predicciones_a_val = np.array([prediccion_a_val])
                        vec_predicciones_v_tst = np.array([prediccion_v_tst])
                        vec_predicciones_a_tst = np.array([prediccion_a_tst])
                    else:
                        vec_predicciones_v_val = np.concatenate([vec_predicciones_v_val, np.array([prediccion_v_val])])
                        vec_predicciones_a_val = np.concatenate([vec_predicciones_a_val, np.array([prediccion_a_val])])
                        vec_predicciones_v_tst = np.concatenate([vec_predicciones_v_tst, np.array([prediccion_v_tst])])
                        vec_predicciones_a_tst = np.concatenate([vec_predicciones_a_tst, np.array([prediccion_a_tst])])
                    print(time.time() - start2)
                    log.agrega(time.time() - start2)

        resultados_v_val = hrm.resumePredicciones(vec_predicciones_v_val, lista_metodos, lista_acu_v_val, lista_uar_v_val)
        resultados_a_val = hrm.resumePredicciones(vec_predicciones_a_val, lista_metodos, lista_acu_a_val, lista_uar_a_val)
        resultados_v_tst = hrm.resumePredicciones(vec_predicciones_v_tst, lista_metodos, lista_acu_v_tst, lista_uar_v_tst)
        resultados_a_tst = hrm.resumePredicciones(vec_predicciones_a_tst, lista_metodos, lista_acu_a_tst, lista_uar_a_tst)

        resultados_val = hrm.uneResumenes(resultados_v_val, resultados_a_val)
        resultados_tst = hrm.uneResumenes(resultados_v_tst, resultados_a_tst)

        indice_mejores = hrm.EleccionFusion(resultados_val, mejores=datos.VOTO_MEJORES_X)
        aux_mejores_metodos = 'Mejores combinaciones para la fusion: '
        for i in range(0, indice_mejores.size):
            aux_mejores_metodos = aux_mejores_metodos + '[' + str(resultados_tst[0, indice_mejores[i]]) + ']'
        print(aux_mejores_metodos)
        log.agrega(aux_mejores_metodos)

        resultados_fusionado = hrm.Fusion(resultados_tst, 'Voto', indice_mejores)

        if nro_test == -1:
            resultados_fusionado, desfase = hrm.OrdenaInstancias(resultados_fusionado, orden_instancias)
            resultados_fusionado_2 = hrm.VotoPorSegmento(resultados_fusionado, datos.INSTANCIAS_POR_PERIODOS, desfase)
        else:
            resultados_fusionado_2 = hrm.VotoPorSegmento(resultados_fusionado, datos.INSTANCIAS_POR_PERIODOS)

        vec_resultados = np.concatenate([vec_resultados,  np.array([resultados_tst[0:3, :]])], axis=0)
        vec_resultados_fusionado = np.concatenate([vec_resultados_fusionado, np.array([resultados_fusionado[0:3, :]])], axis=0)
        vec_resultados_fusionado_2 = np.concatenate([vec_resultados_fusionado_2, np.array([resultados_fusionado_2[0:3, :]])], axis=0)

        hrm.muestraTabla(resultados_tst)
        hrm.muestraTabla(resultados_fusionado)
        hrm.muestraTabla(resultados_fusionado_2)
    if nro_test != -1:
        resumen_final = hrm.generaResumenFinal(vec_resultados, vec_resultados_fusionado, vec_resultados_fusionado_2)
        hrm.muestraTabla(resumen_final)
    jvm.stop()
    print(time.time() - start_total)
    log.agrega('Tiempo final')
    log.agrega(time.time() - start_total)


def SegundoMultimodalCompleto(elimino_silencios=False):
    start_total = time.time()
    personas = datos.PERSONAS
    etapas = datos.ETAPAS
    zonas = datos.ZONAS
    met_extraccion = datos.MET_EXTRACCION
    met_seleccion = datos.MET_SELECCION
    met_clasificacion = datos.MET_CLASIFICACION
    binarizo_etiquetas = datos.BINARIZO_ETIQUETA
    nro_val = datos.VAL
    nro_test = datos.TEST

    jvm.start(max_heap_size="9G", packages=True)

    # print('Adaptación de caracteristicas en progreso')
    # log.agrega('Adaptación de caracteristicas en progreso')
    # features_v = carac.Video(binarizo_etiquetas, zonas, met_extraccion)
    # features_a = carac.Audio(binarizo_etiquetas)
    # for i in personas:
    #     for j in etapas:
    #         start2 = time.time()
    #         print('Persona ' + i + ' -> Etapa ' + j)
    #         log.agrega('Persona ' + i + ' -> Etapa ' + j)
    #         rang_audibles = features_a(i, j, eliminar_silencios=elimino_silencios)
    #         features_v(i, j, completo=False, rangos_audibles=rang_audibles)
    #         print(time.time() - start2)
    #         log.agrega(time.time() - start2)
    #
    # print('Completada adaptación de caracteristicas')
    # print(time.time() - start_total)
    # log.agrega('Completada adaptación de caracteristicas')
    # log.agrega(time.time() - start_total)

    vec_resultados = np.empty((0, 3, met_seleccion.size * met_clasificacion.size + 1))
    vec_resultados_fusionado = np.empty((0, 3, 2))
    vec_resultados_fusionado_2 = np.empty((0, 3, 2))

    if nro_test == -1:
        vueltas = 1
    else:
        # Contando que cuando se usa test siempre se trabaja con toda la bd
        vueltas = int(21 / nro_test)

    orden_instancias = np.empty(0)
    for k in range(0, vueltas):
        datos.defineFoldActual(k + 1)
        if nro_test == -1:
            data = am.Concatena(personas, etapas, 'VResp', 'AResp')
            orden_instancias = am.GeneraOrdenInstancias(data, datos.INSTANCIAS_POR_PERIODOS)
            data_ori = am.MezclaInstancias(data, orden_instancias)
            train_ori, test_ori = wek.ParticionaDatos(data_ori)
            train_ori, val_ori = wek.ParticionaDatos(train_ori)
        else:
            print('Vuelta: ' + str(k + 1) + '/' + str(vueltas))
            log.agrega('Vuelta: ' + str(k + 1) + '/' + str(vueltas))
            personas_train, personas_val, personas_test = GeneraConjuntos(k, nro_val, nro_test)
            train_ori = am.Concatena(personas_train, etapas, 'VResp', 'AResp')
            val_ori = am.Concatena(personas_val, etapas, 'VResp', 'AResp')
            test_ori = am.Concatena(personas_test, etapas, 'VResp', 'AResp')

        vec_predicciones_val = np.array([])
        vec_predicciones_tst = np.array([])
        lista_metodos = list()
        lista_acu_tst = list()
        lista_uar_tst = list()
        lista_acu_val = list()
        lista_uar_val = list()

        print('Seleccion y clasificación en progreso')
        for i in range(0, len(met_seleccion)):
            start2 = time.time()

            if met_seleccion[i] != '':
                print(met_seleccion[i])
                log.agrega(met_seleccion[i])
                metodo_actual = met_seleccion[i] + ' + '
                train, val, test = wek.SeleccionCaracteristicas(train_ori, val_ori, test_ori, met_seleccion[i])
                print(time.time() - start2)
                log.agrega(time.time() - start2)
            else:
                print('Sin selección de caracteristicas')
                metodo_actual = ''
                train = train_ori
                val = val_ori
                test = test_ori
            for j in range(0, len(met_clasificacion)):
                # Si no se selecciona caracteristicas y esta MLP, que no lo haga porque va a demorar demasiado
                if metodo_actual != '' or met_clasificacion[j] != 'MLP':
                    print(met_clasificacion[j])
                    log.agrega(met_clasificacion[j])
                    start2 = time.time()
                    lista_metodos.append(metodo_actual + met_clasificacion[j])
                    predi_csv_val, predi_csv_tst = wek.Clasificacion(train, val, test, met_clasificacion[j], met_seleccion[i])
                    prediccion_val = hrm.prediccionCSVtoArray(predi_csv_val)
                    prediccion_tst = hrm.prediccionCSVtoArray(predi_csv_tst)
                    lista_acu_val.append(hrm.Accuracy(prediccion_val[:, 1], prediccion_val[:, 2]))
                    lista_acu_tst.append(hrm.Accuracy(prediccion_tst[:, 1], prediccion_tst[:, 2]))
                    lista_uar_val.append(hrm.UAR(prediccion_val[:, 1], prediccion_val[:, 2]))
                    lista_uar_tst.append(hrm.UAR(prediccion_tst[:, 1], prediccion_tst[:, 2]))
                    if len(vec_predicciones_val) == 0:
                        vec_predicciones_val = np.array([prediccion_val])
                        vec_predicciones_tst = np.array([prediccion_tst])
                    else:
                        vec_predicciones_val = np.concatenate([vec_predicciones_val, np.array([prediccion_val])])
                        vec_predicciones_tst = np.concatenate([vec_predicciones_tst, np.array([prediccion_tst])])
                    print(time.time() - start2)
                    log.agrega(time.time() - start2)

        resultados_val = hrm.resumePredicciones(vec_predicciones_val, lista_metodos, lista_acu_val, lista_uar_val)
        resultados_tst = hrm.resumePredicciones(vec_predicciones_tst, lista_metodos, lista_acu_tst, lista_uar_tst)

        indice_mejores = hrm.EleccionFusion(resultados_val, mejores=datos.VOTO_MEJORES_X)
        aux_mejores_metodos = 'Mejores combinaciones para la fusion: '
        for i in range(0, indice_mejores.size):
            aux_mejores_metodos = aux_mejores_metodos + '[' + str(resultados_tst[0, indice_mejores[i]]) + ']'
        print(aux_mejores_metodos)
        log.agrega(aux_mejores_metodos)
        resultados_fusionado = hrm.Fusion(resultados_tst, 'Voto', indice_mejores)

        if nro_test == -1:
            resultados_fusionado, desfase = hrm.OrdenaInstancias(resultados_fusionado, orden_instancias)
            resultados_fusionado_2 = hrm.VotoPorSegmento(resultados_fusionado, datos.INSTANCIAS_POR_PERIODOS, desfase)
        else:
            resultados_fusionado_2 = hrm.VotoPorSegmento(resultados_fusionado, datos.INSTANCIAS_POR_PERIODOS)

        vec_resultados = np.concatenate([vec_resultados,  np.array([resultados_tst[0:3, :]])], axis=0)
        vec_resultados_fusionado = np.concatenate([vec_resultados_fusionado, np.array([resultados_fusionado[0:3, :]])], axis=0)
        vec_resultados_fusionado_2 = np.concatenate([vec_resultados_fusionado_2, np.array([resultados_fusionado_2[0:3, :]])], axis=0)

        hrm.muestraTabla(resultados_tst)
        hrm.muestraTabla(resultados_fusionado)
        hrm.muestraTabla(resultados_fusionado_2)
    if nro_test != -1:
        resumen_final = hrm.generaResumenFinal(vec_resultados, vec_resultados_fusionado, vec_resultados_fusionado_2)
        hrm.muestraTabla(resumen_final)
    jvm.stop()
    print(time.time() - start_total)
    log.agrega('Tiempo final')
    log.agrega(time.time() - start_total)


def ExtractorDeCaracteristicas(personas, etapas, zonas):
    start_total = time.time()
    jvm.start()
    print('Extracción de caracteristicas en progreso')
    features = carac.CaracteristicasVideo(zonas)
    for i in personas:
        for j in etapas:
            # if i != '09' or j != '1':
            start2 = time.time()
            print('Persona ' + i + ' -> Etapa ' + j)
            features(i, j)
            print(time.time() - start2)
    print('Completada extraccion de caracteristicas')
    print(time.time() - start_total)
    # jvm.stop()


def GeneraConjuntos(vuelta_actual, val, test):
    personas_train = np.empty(0, dtype=int)
    personas_val = np.empty(0, dtype=int)
    personas_test = vuelta_actual * test + np.array(range(1, test + 1), dtype=int)
    for i in range(0, val):
        ind_val = personas_test[0] - 1 - i
        if ind_val < 1:
            ind_val = 21 - abs(ind_val)
        personas_val = np.append(personas_val, ind_val)
    for i in range(1, 22):
        if np.where(personas_test == i)[0].size == 0 and np.where(personas_val == i)[0].size == 0:
            personas_train = np.append(personas_train, i)
    # Casteo a string
    personas_train = personas_train.astype(np.str)
    personas_val = personas_val.astype(np.str)
    personas_test = personas_test.astype(np.str)
    # Los que son numero de una cifra se les tiene que agregar un 0 a la izquierda
    for i in range(0, personas_train.size):
        if int(personas_train[i]) < 10:
            personas_train[i] = '0' + personas_train[i]
    for i in range(0, personas_val.size):
        if int(personas_val[i]) < 10:
            personas_val[i] = '0' + personas_val[i]
    for i in range(0, personas_test.size):
        if int(personas_test[i]) < 10:
            personas_test[i] = '0' + personas_test[i]
    return personas_train, personas_val, personas_test
