import os
import numpy as np
import cv2 as cv
from weka.core import jvm
from termcolor import colored
import Herramientas as Hrm
import ArffManager as Am
import ExtraccionCaracteristicas as Extrc
import Weka
import Datos


def fileProcessing(video_path, video_name, audio_path, audio_name, binarize_labels):
    jvm.logger.disabled = True
    jvm.start(packages=True)

    # Definición de parámetros
    zones = Datos.ZONAS
    extraction_methods = Datos.MET_EXTRACCION
    best_configuration = Datos.MEJORES_CONFIGURACIONES
    instances_for_period = Datos.INSTANCIAS_POR_PERIODOS

    # Extracción de características
    print(colored('Extracción de características en progreso', 'green'))
    print(colored('     Extrayendo características del video...', 'yellow'))
    setExtension(video_path, is_video=True)
    features_video_extraction = Extrc.VideoFeaturesExtraction(zones)
    features_video_extraction(video_name, video_path)
    print(colored('     Adaptando las características del video...', 'yellow'))
    video_features = Extrc.VideoFeaturesUnification(binarize_labels, zones, extraction_methods)
    video_labels = createLabelList(video_path)
    video_features(video_name, video_path, video_labels, complete_mode=True, for_frames=False)
    print(colored('     Extrayendo características del audio...', 'yellow'))
    features_audio_extraction = Extrc.AudioFeaturesExtraction(binarize_labels)
    if audio_path is None:
        audio_labels = createLabelList(video_path)
        features_audio_extraction(video_name, video_path, audio_labels, complete_mode=True, extract_from_video=True)
    else:
        setExtension(audio_path, is_video=False)
        audio_labels = createLabelList(audio_path)
        features_audio_extraction(audio_name, audio_path, audio_labels, complete_mode=True, extract_from_video=False)
    print(colored('Extracción de características completada', 'green'))

    # Clasificación
    print(colored('Clasificación en progreso', 'green'))
    if audio_path is None:
        data_tst = joinData(video_name, video_name)
    else:
        data_tst = joinData(video_name, audio_name)

    prediction_vector = np.array([])
    methods_list = list()
    aux_list = list()
    for i in range(0, best_configuration.shape[0]):
        selection_method = best_configuration[i][0]
        classification_method = best_configuration[i][1]
        actual_configuration_name = selection_method + ' + ' + classification_method
        print(colored('     Aplicando clasificación ' + str(i + 1) + '...', 'yellow'))

        methods_list.append(actual_configuration_name)
        path_load = Hrm.buildSaveModelPath(actual_configuration_name)

        prediction_in_csv = Weka.classificationOnlyTest(data_tst, path_load, filter_attributes=True)
        prediction = Hrm.predictionCSVtoArray(prediction_in_csv)
        aux_list.append(1)
        if len(prediction_vector) == 0:
            prediction_vector = np.array([prediction])
        else:
            prediction_vector = np.concatenate([prediction_vector, np.array([prediction])])

    results = Hrm.summarizePredictions(prediction_vector, methods_list, aux_list, aux_list)
    best_index = np.array(range(1, best_configuration.shape[0] + 1))
    results_first_fusion = Hrm.fusionClassifiers(results, 'Voto', best_index)
    results_second_fusion = Hrm.voteForPeriod(results_first_fusion, instances_for_period)

    print(colored('Detección de estrés finalizada', 'green'))

    return


def setExtension(filename, is_video):
    idx = filename.rfind('.')
    extension = filename[idx:]
    if is_video:
        Datos.EXTENSION_VIDEO = extension
    else:
        Datos.EXTENSION_AUDIO = extension


def createLabelList(video_path):
    labels_list = list()
    video = cv.VideoCapture(video_path)
    instances_number = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    for i in range(0, instances_number):
        labels_list.append('N')
    return labels_list


def joinData(video_name, audio_name):
    path = Hrm.buildSubFilePath(video_name, 'VCompFus')
    data1 = Am.loadAndFiltered(path)
    path = Hrm.buildSubFilePath(audio_name, 'AComp')
    data2 = Am.loadAndFiltered(path)
    data_vec_norm = Am.normalizeDatasets(np.array([data1, data2]))
    data_vec_norm[0].no_class()
    data_vec_norm[0].delete_last_attribute()
    data_final = Am.joinDatasetByInstances(data_vec_norm)
    return data_final
