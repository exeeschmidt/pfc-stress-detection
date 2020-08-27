import numpy as np
import weka.core.jvm as jvm
import ExtraccionCaracteristicas as Extrc
import Herramientas as Hrm
import Weka
import ArffManager as Am
import Datos
import LogManager as Log
import time


def Unimodal():
    Log.create()
    start_total = time.time()

    persons = Datos.PERSONAS
    stages = Datos.ETAPAS
    zones = Datos.ZONAS
    extraction_methods = Datos.MET_EXTRACCION
    selection_methods = Datos.MET_SELECCION
    classification_methods = Datos.MET_CLASIFICACION
    binarize_labels = Datos.BINARIZO_ETIQUETA
    num_to_validation = Datos.VAL
    num_to_test = Datos.TEST

    print('Adaptación de caracteristicas en progreso')
    Log.add('Adaptación de caracteristicas en progreso')
    features = Extrc.VideoFeaturesUnification(binarize_labels, zones, extraction_methods)
    answers_limits_list = list()
    for i in persons:
        for j in stages:
            start2 = time.time()
            print('Persona ' + i + ' -> Etapa ' + j)
            Log.add('Persona ' + i + ' -> Etapa ' + j)
            video_name = Hrm.buildFileName(i, j)
            video_path = Hrm.buildFilePath(i, j, video_name, extension=Datos.EXTENSION_VIDEO)
            labels_list, answers_limits = Hrm.mapLabelsOwnBD(i, j, binarize_labels, complete_mode=True)
            answers_limits_list.append(answers_limits)
            features(video_name, video_path, labels_list, complete_mode=True)
            print(time.time() - start2)
            Log.add(time.time() - start2)

    print('Completada adaptación de caracteristicas')
    Log.add('Completada adaptación de caracteristicas')
    print(time.time() - start_total)
    Log.add(time.time() - start_total)

    result_raw_vector = np.empty((0, 3, selection_methods.size * classification_methods.size + 1))
    result_first_fusion_vector = np.empty((0, 3, 2))
    result_second_fusion_vector = np.empty((0, 3, 2))

    if num_to_test == -1:
        laps = 1
    else:
        # Contando que cuando se usa test siempre se trabaja con toda la bd
        laps = int(21 / num_to_test)

    instances_order = np.empty(0)
    for k in range(0, laps):
        Datos.defineActualValidationFold(k + 1)
        if num_to_test == -1:
            data = Am.joinPersonStageData(persons, stages, 'VCom')
            instances_order = Am.generateInstancesOrder(data, Datos.INSTANCIAS_POR_PERIODOS)
            data_ori = Am.mixInstances(data, instances_order)
            train_ori, val_ori, test_ori = Weka.partitionData(data_ori)
        else:
            print('Vuelta: ' + str(k + 1) + '/' + str(laps))
            Log.add('Vuelta: ' + str(k + 1) + '/' + str(laps))
            persons_train, persons_validation, persons_test = generateSetsOwnBD(k, num_to_validation, num_to_test)
            train_ori = Am.joinPersonStageData(persons_train, stages, 'VCom')
            val_ori = Am.joinPersonStageData(persons_validation, stages, 'VCom')
            test_ori = Am.joinPersonStageData(persons_test, stages, 'VCom')
            Hrm.writeLimitsOwnBD(persons_test, answers_limits_list)

        Datos.calculateAttributesToCut(train_ori.num_attributes)
        validation_predic_vector = np.array([])
        test_predic_vector = np.array([])
        methods_list = list()
        test_accuracy = list()
        test_uar = list()
        validation_accuracy = list()
        validation_uar = list()

        print('Seleccion y clasificación en progreso')
        for i in range(0, len(selection_methods)):
            start2 = time.time()

            if selection_methods[i] != '':
                print(selection_methods[i])
                Log.add(selection_methods[i])
                actual_selection_method = selection_methods[i] + ' + '
                train, val, test = Weka.featuresSelection(train_ori, val_ori, test_ori, selection_methods[i])
                print(time.time() - start2)
                Log.add(time.time() - start2)
            else:
                print('Sin selección de caracteristicas')
                actual_selection_method = ''
                train = train_ori
                val = val_ori
                test = test_ori
            for j in range(0, len(classification_methods)):
                # Si no se selecciona caracteristicas y esta MLP, que no lo haga porque va a demorar demasiado
                if actual_selection_method != '' or classification_methods[j] != 'MLP':
                    print(classification_methods[j])
                    Log.add(classification_methods[j])
                    start2 = time.time()
                    methods_list.append(actual_selection_method + classification_methods[j])
                    validation_predic_in_csv, test_predic_in_csv = Weka.classification(train, val, test,
                                                                                       classification_methods[j],
                                                                                       selection_methods[i])
                    validation_prediction = Hrm.predictionCSVtoArray(validation_predic_in_csv)
                    test_prediction = Hrm.predictionCSVtoArray(test_predic_in_csv)
                    validation_accuracy.append(Hrm.Accuracy(validation_prediction[:, 1], validation_prediction[:, 2]))
                    test_accuracy.append(Hrm.Accuracy(test_prediction[:, 1], test_prediction[:, 2]))
                    validation_uar.append(Hrm.UAR(validation_prediction[:, 1], validation_prediction[:, 2]))
                    test_uar.append(Hrm.UAR(test_prediction[:, 1], test_prediction[:, 2]))
                    if len(validation_predic_vector) == 0:
                        validation_predic_vector = np.array([validation_prediction])
                        test_predic_vector = np.array([test_prediction])
                    else:
                        validation_predic_vector = np.concatenate([validation_predic_vector,
                                                                   np.array([validation_prediction])])
                        test_predic_vector = np.concatenate([test_predic_vector, np.array([test_prediction])])
                    print(time.time() - start2)
                    Log.add(time.time() - start2)

        validation_results = Hrm.summarizePredictions(validation_predic_vector, methods_list, validation_accuracy,
                                                      validation_uar)
        test_results = Hrm.summarizePredictions(test_predic_vector, methods_list, test_accuracy, test_uar)

        best_index = Hrm.indexsBestClassifiers(validation_results, best_of=Datos.VOTO_MEJORES_X)
        best_methods = 'Mejores combinaciones para la fusion según la validación: '
        for i in range(0, best_index.size):
            best_methods = best_methods + '[' + str(test_results[0, best_index[i]]) + ']'
        print(best_methods)
        Log.add(best_methods)
        results_first_fusion = Hrm.fusionClassifiers(test_results, 'Voto', best_index)

        if num_to_test == -1:
            results_first_fusion, desfase = Hrm.sortInstances(results_first_fusion, instances_order)
            results_second_fusion = Hrm.voteForPeriod(results_first_fusion, Datos.INSTANCIAS_POR_PERIODOS, desfase)
        else:
            results_second_fusion = Hrm.voteForPeriod(results_first_fusion, Datos.INSTANCIAS_POR_PERIODOS)

        result_raw_vector = np.concatenate([result_raw_vector, np.array([test_results[0:3, :]])], axis=0)
        result_first_fusion_vector = np.concatenate([result_first_fusion_vector,
                                                     np.array([results_first_fusion[0:3, :]])], axis=0)
        result_second_fusion_vector = np.concatenate([result_second_fusion_vector,
                                                      np.array([results_second_fusion[0:3, :]])], axis=0)

        Hrm.showTable(test_results)
        Hrm.showTable(results_first_fusion)
        Hrm.showTable(results_second_fusion)
    if num_to_test != -1:
        final_summary = Hrm.createFinalSummary(result_raw_vector, result_first_fusion_vector,
                                               result_second_fusion_vector)
        Hrm.showTable(final_summary)
    print(time.time() - start_total)
    Log.add('Tiempo final')
    Log.add(time.time() - start_total)
    print('Fin de experimento')
    Log.add('Fin de experimento')


def PrimerMultimodal():
    Log.create()
    start_total = time.time()
    persons = Datos.PERSONAS
    stages = Datos.ETAPAS
    zones = Datos.ZONAS
    extraction_methods = Datos.MET_EXTRACCION
    selection_methods = Datos.MET_SELECCION
    classification_methods = Datos.MET_CLASIFICACION
    binarize_labels = Datos.BINARIZO_ETIQUETA
    num_to_validation = Datos.VAL
    num_to_test = Datos.TEST

    # print('Adaptación de caracteristicas en progreso')
    # Log.add('Adaptación de caracteristicas en progreso')
    # video_features = Extrc.VideoFeaturesUnification(binarize_labels, zones, extraction_methods)
    # audio_features = Extrc.AudioFeaturesExtraction(binarize_labels)
    # answers_limits_list = list()
    # for i in persons:
    #     for j in stages:
    #         start2 = time.time()
    #         print('Persona ' + i + ' -> Etapa ' + j)
    #         Log.add('Persona ' + i + ' -> Etapa ' + j)
    #         video_name = Hrm.buildFileName(i, j)
    #         video_path = Hrm.buildFilePath(i, j, video_name, extension=Datos.EXTENSION_VIDEO)
    #
    #         labels_list, answers_limits = Hrm.mapLabelsOwnBD(i, j, binarize_labels, complete_mode=False)
    #         audio_features(video_name, video_path, labels_list, complete_mode=False, extract_from_video=True)
    #         final_answer_limits = video_features(video_name, video_path, labels_list, complete_mode=False)
    #         answers_limits_list.append(final_answer_limits)
    #
    #         print(time.time() - start2)
    #         Log.add(time.time() - start2)
    #
    # print('Completada adaptación de caracteristicas')
    # print(time.time() - start_total)
    # Log.add('Completada adaptación de caracteristicas')
    # Log.add(time.time() - start_total)

    result_raw_vector = np.empty((0, 3, 2 * selection_methods.size * classification_methods.size + 1))
    result_first_fusion_vector = np.empty((0, 3, 2))
    result_second_fusion_vector = np.empty((0, 3, 2))

    if num_to_test == -1:
        laps = 1
    else:
        # Contando que cuando se usa test siempre se trabaja con toda la bd
        laps = int(21 / num_to_test)

    instances_order = np.empty(0)
    for k in range(0, laps):
        Datos.defineActualValidationFold(k + 1)
        if num_to_test == -1:
            data_v, data_a = Am.joinPersonStageData(persons, stages, 'VResp', 'AResp')
            instances_order = Am.generateInstancesOrder(data_v, Datos.INSTANCIAS_POR_PERIODOS)
            data_v_ori = Am.mixInstances(data_v, instances_order)
            data_a_ori = Am.mixInstances(data_a, instances_order)
            train_v_ori, val_v_ori, test_v_ori = Weka.partitionData(data_v_ori)
            train_a_ori, val_a_ori, test_a_ori = Weka.partitionData(data_a_ori)
        else:
            print('Vuelta: ' + str(k + 1) + '/' + str(laps))
            Log.add('Vuelta: ' + str(k + 1) + '/' + str(laps))
            persons_train, persons_validation, persons_test = generateSetsOwnBD(k, num_to_validation, num_to_test)
            # train_v_ori, train_a_ori, new_answers_limits_list = \
            #     Am.joinPersonStageData(persons_train, stages,
            #                            'VResp', 'AResp', join=False,
            #                            answer_limits_list=answers_limits_list)
            # val_v_ori, val_a_ori, new_answers_limits_list = \
            #     Am.joinPersonStageData(persons_validation, stages,
            #                            'VResp', 'AResp', join=False,
            #                            answer_limits_list=new_answers_limits_list)
            # test_v_ori, test_a_ori, new_answers_limits_list = \
            #     Am.joinPersonStageData(persons_test, stages,
            #                            'VResp', 'AResp', join=False,
            #                            answer_limits_list=new_answers_limits_list)
            train_v_ori, train_a_ori, new_answers_limits_list = \
                Am.joinPersonStageData(persons_train, stages,
                                       'VResp', 'AResp', join=False)
            val_v_ori, val_a_ori, new_answers_limits_list = \
                Am.joinPersonStageData(persons_validation, stages,
                                       'VResp', 'AResp', join=False)
            test_v_ori, test_a_ori, new_answers_limits_list = \
                Am.joinPersonStageData(persons_test, stages,
                                       'VResp', 'AResp', join=False)
            # Hrm.writeLimitsOwnBD(persons_test, new_answers_limits_list)

        Datos.calculateAttributesToCut(train_v_ori.num_attributes)
        validation_predic_vector_v = np.array([])
        test_predic_vector_v = np.array([])
        validation_predic_vector_a = np.array([])
        test_predic_vector_a = np.array([])
        methods_list = list()
        validation_accuracy_v = list()
        test_accuracy_v = list()
        validation_accuracy_a = list()
        test_accuracy_a = list()
        validation_uar_v = list()
        test_uar_v = list()
        validation_uar_a = list()
        test_uar_a = list()

        print('Seleccion y clasificación en progreso')
        for i in range(0, len(selection_methods)):
            start2 = time.time()

            if selection_methods[i] != '':
                print(selection_methods[i])
                Log.add(selection_methods[i])
                actual_selection_method = selection_methods[i] + ' + '
                print('Video')
                Log.add('Video')
                train_v, val_v, test_v = Weka.featuresSelection(train_v_ori, val_v_ori, test_v_ori,
                                                                selection_methods[i])
                print('Audio')
                Log.add('Audio')
                train_a, val_a, test_a = Weka.featuresSelection(train_a_ori, val_a_ori, test_a_ori,
                                                                selection_methods[i])
                print(time.time() - start2)
                Log.add(time.time() - start2)
            else:
                print('Sin selección de caracteristicas')
                actual_selection_method = ''
                train_v = train_v_ori
                val_v = val_v_ori
                test_v = test_v_ori
                train_a = train_a_ori
                val_a = val_a_ori
                test_a = test_a_ori
            for j in range(0, len(classification_methods)):
                # Si no se selecciona caracteristicas y esta MLP, que no lo haga porque va a demorar demasiado
                if actual_selection_method != '' or classification_methods[j] != 'MLP':
                    print(classification_methods[j])
                    Log.add(classification_methods[j])
                    start2 = time.time()
                    methods_list.append(actual_selection_method + classification_methods[j])
                    validation_predic_in_csv_v, test_predic_in_csv_v = Weka.classification(train_v, val_v, test_v,
                                                                                           classification_methods[j],
                                                                                           selection_methods[i])
                    validation_predic_in_csv_a, test_predic_in_csv_a = Weka.classification(train_a, val_a, test_a,
                                                                                           classification_methods[j],
                                                                                           selection_methods[i])
                    validation_prediction_v = Hrm.predictionCSVtoArray(validation_predic_in_csv_v)
                    test_prediction_v = Hrm.predictionCSVtoArray(test_predic_in_csv_v)
                    validation_prediction_a = Hrm.predictionCSVtoArray(validation_predic_in_csv_a)
                    test_prediction_a = Hrm.predictionCSVtoArray(test_predic_in_csv_a)
                    validation_accuracy_v.append(Hrm.Accuracy(validation_prediction_v[:, 1],
                                                              validation_prediction_v[:, 2]))
                    validation_accuracy_a.append(Hrm.Accuracy(validation_prediction_a[:, 1],
                                                              validation_prediction_a[:, 2]))
                    test_accuracy_v.append(Hrm.Accuracy(test_prediction_v[:, 1], test_prediction_v[:, 2]))
                    test_accuracy_a.append(Hrm.Accuracy(test_prediction_a[:, 1], test_prediction_a[:, 2]))
                    validation_uar_v.append(Hrm.UAR(validation_prediction_v[:, 1], validation_prediction_v[:, 2]))
                    validation_uar_a.append(Hrm.UAR(validation_prediction_a[:, 1], validation_prediction_a[:, 2]))
                    test_uar_v.append(Hrm.UAR(test_prediction_v[:, 1], test_prediction_v[:, 2]))
                    test_uar_a.append(Hrm.UAR(test_prediction_a[:, 1], test_prediction_a[:, 2]))
                    if len(validation_predic_vector_v) == 0:
                        validation_predic_vector_v = np.array([validation_prediction_v])
                        validation_predic_vector_a = np.array([validation_prediction_a])
                        test_predic_vector_v = np.array([test_prediction_v])
                        test_predic_vector_a = np.array([test_prediction_a])
                    else:
                        validation_predic_vector_v = np.concatenate([validation_predic_vector_v,
                                                                     np.array([validation_prediction_v])])
                        validation_predic_vector_a = np.concatenate([validation_predic_vector_a,
                                                                     np.array([validation_prediction_a])])
                        test_predic_vector_v = np.concatenate([test_predic_vector_v, np.array([test_prediction_v])])
                        test_predic_vector_a = np.concatenate([test_predic_vector_a, np.array([test_prediction_a])])
                    print(time.time() - start2)
                    Log.add(time.time() - start2)

        validation_results_v = Hrm.summarizePredictions(validation_predic_vector_v, methods_list, validation_accuracy_v,
                                                        validation_uar_v)
        validation_results_a = Hrm.summarizePredictions(validation_predic_vector_a, methods_list, validation_accuracy_a,
                                                        validation_uar_a)
        test_results_v = Hrm.summarizePredictions(test_predic_vector_v, methods_list, test_accuracy_v, test_uar_v)
        test_results_a = Hrm.summarizePredictions(test_predic_vector_a, methods_list, test_accuracy_a, test_uar_a)

        validation_results_joined = Hrm.joinSummaries(validation_results_v, validation_results_a)
        test_results_joined = Hrm.joinSummaries(test_results_v, test_results_a)

        best_index = Hrm.indexsBestClassifiers(validation_results_joined, best_of=Datos.VOTO_MEJORES_X)
        best_methods = 'Mejores combinaciones para la fusion: '
        for i in range(0, best_index.size):
            best_methods = best_methods + '[' + str(test_results_joined[0, best_index[i]]) + ']'
        print(best_methods)
        Log.add(best_methods)

        results_first_fusion = Hrm.fusionClassifiers(test_results_joined, 'Voto', best_index)

        if num_to_test == -1:
            results_first_fusion, gap = Hrm.sortInstances(results_first_fusion, instances_order)
            results_second_fusion = Hrm.voteForPeriod(results_first_fusion, Datos.INSTANCIAS_POR_PERIODOS, gap)
        else:
            results_second_fusion = Hrm.voteForPeriod(results_first_fusion, Datos.INSTANCIAS_POR_PERIODOS)

        result_raw_vector = np.concatenate([result_raw_vector, np.array([test_results_joined[0:3, :]])], axis=0)
        result_first_fusion_vector = np.concatenate([result_first_fusion_vector,
                                                     np.array([results_first_fusion[0:3, :]])], axis=0)
        result_second_fusion_vector = np.concatenate([result_second_fusion_vector,
                                                      np.array([results_second_fusion[0:3, :]])], axis=0)

        Hrm.showTable(test_results_joined)
        Hrm.showTable(results_first_fusion)
        Hrm.showTable(results_second_fusion)
    if num_to_test != -1:
        final_summary = Hrm.createFinalSummary(result_raw_vector, result_first_fusion_vector,
                                               result_second_fusion_vector)
        Hrm.showTable(final_summary)
    print(time.time() - start_total)
    Log.add('Tiempo final')
    Log.add(time.time() - start_total)


def SegundoMultimodal():
    Log.create()
    start_total = time.time()
    persons = Datos.PERSONAS
    stages = Datos.ETAPAS
    zones = Datos.ZONAS
    extraction_methods = Datos.MET_EXTRACCION
    selection_methods = Datos.MET_SELECCION
    classification_methods = Datos.MET_CLASIFICACION
    binarize_labels = Datos.BINARIZO_ETIQUETA
    num_to_validation = Datos.VAL
    num_to_test = Datos.TEST

    # print('Adaptación de caracteristicas en progreso')
    # Log.add('Adaptación de caracteristicas en progreso')
    # video_features = Extrc.VideoFeaturesUnification(binarize_labels, zones, extraction_methods)
    # audio_features = Extrc.AudioFeaturesExtraction(binarize_labels)
    # answers_limits_list = list()
    # for i in persons:
    #     for j in stages:
    #         start2 = time.time()
    #         print('Persona ' + i + ' -> Etapa ' + j)
    #         Log.add('Persona ' + i + ' -> Etapa ' + j)
    #         video_name = Hrm.buildFileName(i, j)
    #         video_path = Hrm.buildFilePath(i, j, video_name, extension=Datos.EXTENSION_VIDEO)
    #
    #         labels_list, answers_limits = Hrm.mapLabelsOwnBD(i, j, binarize_labels, complete_mode=False)
    #         audio_features(video_name, video_path, labels_list, complete_mode=False, extract_from_video=True)
    #         final_answer_limits = video_features(video_name, video_path, labels_list, complete_mode=False)
    #         answers_limits_list.append(final_answer_limits)
    #
    #         print(time.time() - start2)
    #         Log.add(time.time() - start2)
    #
    # print('Completada adaptación de caracteristicas')
    # print(time.time() - start_total)
    # Log.add('Completada adaptación de caracteristicas')
    # Log.add(time.time() - start_total)

    result_raw_vector = np.empty((0, 3, selection_methods.size * classification_methods.size + 1))
    result_first_fusion_vector = np.empty((0, 3, 2))
    result_second_fusion_vector = np.empty((0, 3, 2))

    if num_to_test == -1:
        laps = 1
    else:
        # Contando que cuando se usa test siempre se trabaja con toda la bd
        laps = int(21 / num_to_test)

    instances_order = np.empty(0)
    for k in range(0, laps):
        Datos.defineActualValidationFold(k + 1)
        if num_to_test == -1:
            data = Am.joinPersonStageData(persons, stages, 'VResp', 'AResp', join=True)
            instances_order = Am.generateInstancesOrder(data, Datos.INSTANCIAS_POR_PERIODOS)
            data_ori = Am.mixInstances(data, instances_order)
            train_ori, val_ori, test_ori = Weka.partitionData(data_ori)
        else:
            print('Vuelta: ' + str(k + 1) + '/' + str(laps))
            Log.add('Vuelta: ' + str(k + 1) + '/' + str(laps))
            persons_train, persons_validation, persons_test = generateSetsOwnBD(k, num_to_validation, num_to_test)
            # train_ori, new_answers_limits_list = Am.joinPersonStageData(persons_train, stages,
            #                                                             'VResp', 'AResp', join=True,
            #                                                             answer_limits_list=answers_limits_list)
            # val_ori, new_answers_limits_list = Am.joinPersonStageData(persons_validation, stages,
            #                                                           'VResp', 'AResp', join=True,
            #                                                           answer_limits_list=new_answers_limits_list)
            # test_ori, new_answers_limits_list = Am.joinPersonStageData(persons_test, stages,
            #                                                            'VResp', 'AResp', join=True,
            #                                                            answer_limits_list=new_answers_limits_list)
            train_ori, new_answers_limits_list = Am.joinPersonStageData(persons_train, stages,
                                               'VResp', 'AResp', join=True)
            val_ori, new_answers_limits_list = Am.joinPersonStageData(persons_validation, stages,
                                             'VResp', 'AResp', join=True)
            test_ori, new_answers_limits_list  = Am.joinPersonStageData(persons_test, stages,
                                              'VResp', 'AResp', join=True)
            # Hrm.writeLimitsOwnBD(persons_test, new_answers_limits_list)

        Datos.calculateAttributesToCut(train_ori.num_attributes)
        validation_predic_vector = np.array([])
        test_predic_vector = np.array([])
        methods_list = list()
        test_accuracy = list()
        test_uar = list()
        validation_accuracy = list()
        validation_uar = list()

        print('Seleccion y clasificación en progreso')
        for i in range(0, len(selection_methods)):
            start2 = time.time()

            if selection_methods[i] != '':
                print(selection_methods[i])
                Log.add(selection_methods[i])
                actual_selection_method = selection_methods[i] + ' + '
                train, val, test = Weka.featuresSelection(train_ori, val_ori, test_ori, selection_methods[i])
                print(time.time() - start2)
                Log.add(time.time() - start2)
            else:
                print('Sin selección de caracteristicas')
                actual_selection_method = ''
                train = train_ori
                val = val_ori
                test = test_ori
            for j in range(0, len(classification_methods)):
                # Si no se selecciona caracteristicas y esta MLP, que no lo haga porque va a demorar demasiado
                if actual_selection_method != '' or classification_methods[j] != 'MLP':
                    print(classification_methods[j])
                    Log.add(classification_methods[j])
                    start2 = time.time()
                    methods_list.append(actual_selection_method + classification_methods[j])
                    validation_predic_in_csv, test_predic_in_csv = Weka.classification(train, val, test,
                                                                                       classification_methods[j],
                                                                                       selection_methods[i])
                    validation_prediction = Hrm.predictionCSVtoArray(validation_predic_in_csv)
                    test_prediction = Hrm.predictionCSVtoArray(test_predic_in_csv)
                    validation_accuracy.append(
                        Hrm.Accuracy(validation_prediction[:, 1], validation_prediction[:, 2]))
                    test_accuracy.append(Hrm.Accuracy(test_prediction[:, 1], test_prediction[:, 2]))
                    validation_uar.append(Hrm.UAR(validation_prediction[:, 1], validation_prediction[:, 2]))
                    test_uar.append(Hrm.UAR(test_prediction[:, 1], test_prediction[:, 2]))
                    if len(validation_predic_vector) == 0:
                        validation_predic_vector = np.array([validation_prediction])
                        test_predic_vector = np.array([test_prediction])
                    else:
                        validation_predic_vector = np.concatenate([validation_predic_vector,
                                                                   np.array([validation_prediction])])
                        test_predic_vector = np.concatenate([test_predic_vector, np.array([test_prediction])])
                    print(time.time() - start2)
                    Log.add(time.time() - start2)

        validation_results = Hrm.summarizePredictions(validation_predic_vector, methods_list, validation_accuracy,
                                                      validation_uar)
        test_results = Hrm.summarizePredictions(test_predic_vector, methods_list, test_accuracy, test_uar)

        best_index = Hrm.indexsBestClassifiers(validation_results, best_of=Datos.VOTO_MEJORES_X)
        best_methods = 'Mejores combinaciones para la fusion según la validación: '
        for i in range(0, best_index.size):
            best_methods = best_methods + '[' + str(test_results[0, best_index[i]]) + ']'
        print(best_methods)
        Log.add(best_methods)
        results_first_fusion = Hrm.fusionClassifiers(test_results, 'Voto', best_index)

        if num_to_test == -1:
            results_first_fusion, desfase = Hrm.sortInstances(results_first_fusion, instances_order)
            results_second_fusion = Hrm.voteForPeriod(results_first_fusion, Datos.INSTANCIAS_POR_PERIODOS, desfase)
        else:
            results_second_fusion = Hrm.voteForPeriod(results_first_fusion, Datos.INSTANCIAS_POR_PERIODOS)

        result_raw_vector = np.concatenate([result_raw_vector, np.array([test_results[0:3, :]])], axis=0)
        result_first_fusion_vector = np.concatenate([result_first_fusion_vector,
                                                     np.array([results_first_fusion[0:3, :]])], axis=0)
        result_second_fusion_vector = np.concatenate([result_second_fusion_vector,
                                                      np.array([results_second_fusion[0:3, :]])], axis=0)

        Hrm.showTable(test_results)
        Hrm.showTable(results_first_fusion)
        Hrm.showTable(results_second_fusion)
    if num_to_test != -1:
        final_summary = Hrm.createFinalSummary(result_raw_vector, result_first_fusion_vector,
                                               result_second_fusion_vector)
        Hrm.showTable(final_summary)
    print(time.time() - start_total)
    Log.add('Tiempo final')
    Log.add(time.time() - start_total)
    print('Fin de experimento')
    Log.add('Fin de experimento')


def videoFeaturesExtractionWrapperOwnBD(persons, stages):
    start_total = time.time()
    jvm.start()
    zones = Datos.ZONAS
    print('Extracción de caracteristicas en progreso')
    features_extraction = Extrc.VideoFeaturesExtraction(zones)
    for i in persons:
        for j in stages:
            # if i != '09' or j != '1':
            start2 = time.time()
            print('Persona ' + i + ' -> Etapa ' + j)
            video_name = Hrm.buildFileName(i, j)
            video_path = Hrm.buildFilePath(i, j, video_name, extension=Datos.EXTENSION_VIDEO)
            features_extraction(video_name, video_path)
            print(time.time() - start2)
    print('Completada extraccion de caracteristicas')
    print(time.time() - start_total)
    # jvm.stop()


def generateSetsOwnBD(actual_lap, val, test):
    persons_train = np.empty(0, dtype=int)
    persons_validation = np.empty(0, dtype=int)
    persons_test = actual_lap * test + np.array(range(1, test + 1), dtype=int)
    for i in range(0, val):
        index_validation = persons_test[0] - 1 - i
        if index_validation < 1:
            index_validation = 21 - abs(index_validation)
        persons_validation = np.append(persons_validation, index_validation)
    for i in range(1, 22):
        if np.where(persons_test == i)[0].size == 0 and np.where(persons_validation == i)[0].size == 0:
            persons_train = np.append(persons_train, i)
    # Casteo a string
    persons_train = persons_train.astype(np.str)
    persons_validation = persons_validation.astype(np.str)
    persons_test = persons_test.astype(np.str)
    # Los que son numero de una cifra se les tiene que agregar un 0 a la izquierda
    for i in range(0, persons_train.size):
        if int(persons_train[i]) < 10:
            persons_train[i] = '0' + persons_train[i]
    for i in range(0, persons_validation.size):
        if int(persons_validation[i]) < 10:
            persons_validation[i] = '0' + persons_validation[i]
    for i in range(0, persons_test.size):
        if int(persons_test[i]) < 10:
            persons_test[i] = '0' + persons_test[i]
    return persons_train, persons_validation, persons_test


def testMSPImprov():
    Log.create()
    start_total = time.time()

    zones = Datos.ZONAS
    extraction_methods = Datos.MET_EXTRACCION
    best_configuration = Datos.MEJORES_CONFIGURACIONES
    binarize_labels = Datos.BINARIZO_ETIQUETA

    file_list = Hrm.processEvalutionFile()
    videoExtractionWrapperMSPImprov(file_list)

    print('Adaptación de caracteristicas en progreso')
    Log.add('Adaptación de caracteristicas en progreso')
    video_features = Extrc.VideoFeaturesUnification(binarize_labels, zones, extraction_methods)
    audio_features = Extrc.AudioFeaturesExtraction(binarize_labels)
    count = 1
    for row_file in file_list:
        start2 = time.time()
        print('Video nro ' + str(count))
        count += 1
        labels_list = Hrm.mapLabelsMSPImprov(row_file, binarize_labels)
        file_name = row_file[0]
        video_path = row_file[1]
        audio_path = row_file[2]
        video_features(file_name, video_path, labels_list, complete_mode=True, for_frames=False)
        audio_features(file_name, audio_path, labels_list, complete_mode=True, extract_from_video=False)
        print(time.time() - start2)

    print('Completada adaptación de caracteristicas')
    print(time.time() - start_total)
    Log.add('Completada adaptación de caracteristicas')
    Log.add(time.time() - start_total)

    data_tst, answer_limits = Am.joinListData(file_list)
    # for i in range(0, best_configuration.shape[0]):


def videoExtractionWrapperMSPImprov(file_list):
    start_total = time.time()
    jvm.start()
    zones = Datos.ZONAS
    print('Extracción de caracteristicas en progreso')
    features_extraction = Extrc.VideoFeaturesExtraction(zones)
    count = 1
    for row_file in file_list:
        # if i != '09' or j != '1':
        start2 = time.time()
        print('Video nro ' + str(count))
        count += 1
        video_name = row_file[0]
        video_path = row_file[1]
        features_extraction(video_name, video_path)
        print(time.time() - start2)
    print('Completada extraccion de caracteristicas')
    print(time.time() - start_total)
