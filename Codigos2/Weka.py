from weka.attribute_selection import ASSearch, ASEvaluation
from weka.classifiers import Classifier, Evaluation, PredictionOutput
from weka.filters import Filter
from weka.core.dataset import Instances
from weka.core import serialization
import Datos
import LogManager as Log
import Herramientas as Hrm
import os
import numpy as np


def featuresSelection(selection_method, data_train, data_val=None, data_test=None):
    # opciones: 'PSO' , 'PCA', 'BF', 'PC'
    data_trn = Instances.copy_instances(data_train)
    if data_val is not None and data_test is not None:
        data_vld = Instances.copy_instances(data_val)
        data_tst = Instances.copy_instances(data_test)
    else:
        data_vld = None
        data_tst = None

    evaluation_options = ''
    search_options = ''

    # Segun el metodo elegido realiza una preseleccion de características usando Coeficientes de correlacion de Pearson
    # (PC). Para cada metodo en particular se preseleccionan un numero distinto de atributos.
    if selection_method == 'PCA' or selection_method == 'PC' or selection_method == 'PC-pca' or \
            selection_method == 'PC-pso' or selection_method == 'PC-bf':
        search_method = 'weka.attributeSelection.Ranker'
        if selection_method == 'PCA':
            evaluation_method = 'weka.attributeSelection.PrincipalComponents'
            evaluation_options = Datos.PARAMETROS_SELECCION_EVALUACION.get(selection_method)
            search_options = ['-N', str(Datos.ATRIBS_FINALES)]
            data_trn, data_vld, data_tst = featuresSelection('PC-pca', data_trn, data_vld, data_tst)
        else:
            evaluation_method = 'weka.attributeSelection.CorrelationAttributeEval'
        if selection_method == 'PC':
            search_options = ['-N', str(Datos.ATRIBS_FINALES)]
        elif selection_method == 'PC-pca':
            search_options = ['-N', str(Datos.ATRIBS_PCA)]
        elif selection_method == 'PC-pso':
            search_options = ['-N', str(Datos.ATRIBS_PSO)]
        elif selection_method == 'PC-bf':
            search_options = ['-N', str(Datos.ATRIBS_BF)]
    else:
        evaluation_method = 'weka.attributeSelection.CfsSubsetEval'
        evaluation_options = Datos.PARAMETROS_SELECCION_EVALUACION.get('CFS')
        if Datos.PRUEBA_PARAMETROS_SELECCION:
            search_options = Datos.PARAMETROS_SELECCION_BUSQUEDA.get(selection_method)
            selection_method = selection_method[0: len(selection_method) - 2]
        else:
            search_options = Datos.PARAMETROS_SELECCION_BUSQUEDA.get(selection_method + ' 1')
        if selection_method == 'PSO':
            search_method = 'weka.attributeSelection.PSOSearch'
        elif selection_method == 'BF':
            search_method = 'weka.attributeSelection.BestFirst'
        else:
            search_method = ''
        data_trn, data_vld, data_tst = featuresSelection('PC-' + selection_method.lower(), data_trn, data_vld, data_tst)

    flter = Filter(classname="weka.filters.supervised.attribute.AttributeSelection")
    aseval = ASEvaluation(evaluation_method, options=evaluation_options)
    assearch = ASSearch(search_method, options=search_options)
    flter.set_property("evaluator", aseval.jobject)
    flter.set_property("search", assearch.jobject)
    flter.inputformat(data_trn)
    data_trn_filtered = flter.filter(data_trn)
    print('Atributos ', selection_method, ' :', data_trn_filtered.num_attributes, '/', data_trn.num_attributes)
    Log.add('Atributos ' + selection_method + ' :' + str(data_trn_filtered.num_attributes) + '/' +
            str(data_trn.num_attributes))
    if data_vld is not None and data_tst is not None:
        data_vld_filtered = flter.filter(data_vld)
        data_tst_filtered = flter.filter(data_tst)
        return data_trn_filtered, data_vld_filtered, data_tst_filtered
    else:
        return data_trn_filtered, None, None


def classification(data_train, data_val, data_test, classification_method, selection_method, summary=False):
    # Opciones, metodo = 'J48', 'RF', 'RT', 'SVM', 'LR', 'MLP'
    translate_classifier_name = {
        'J48': 'weka.classifiers.trees.J48',
        'RF': 'weka.classifiers.trees.RandomForest',
        'RT': 'weka.classifiers.trees.RandomTree',
        'SVM': 'weka.classifiers.functions.LibSVM',
        'LR': 'weka.classifiers.functions.LinearRegression',
        'MLP': 'weka.classifiers.functions.MultilayerPerceptron'
    }

    # En el caso de probar parametros se agregan como opciones, sino toma todos los parametros por defecto
    if not Datos.PRUEBA_PARAMETROS_CLASIFICACION:
        method = translate_classifier_name.get(classification_method)
        options = Datos.PARAMETROS_CLASIFICADOR.get(classification_method + ' 1')
    else:
        method = translate_classifier_name.get(classification_method[0: len(classification_method) - 2])
        Datos.refreshParamsMLP(data_train.num_attributes)
        options = Datos.PARAMETROS_CLASIFICADOR.get(classification_method)

    classifier = Classifier(classname=method, options=options)
    classifier.build_classifier(data_train)

    pout_val = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")
    evl = Evaluation(data_val)
    evl.test_model(classifier, data_val, output=pout_val)

    pout_tst = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")
    evl = Evaluation(data_test)
    evl.test_model(classifier, data_test, output=pout_tst)

    if Datos.GUARDO_INFO_CLASIFICACION:
        file_name = os.path.join(Datos.PATH_LOGS, str(Datos.FOLD_ACTUAL) + '_' + selection_method + '-' +
                                 classification_method)

        attributes_list = list()
        for i in range(0, data_train.num_attributes - 1):
            attributes_list.append(data_train.attribute(i).name)

        wf = open(file_name + '.txt', 'w')
        attrib_list_m = map(lambda x: x + '\n', attributes_list)
        wf.writelines(attrib_list_m)
        wf.close()

        Hrm.writeCSVFile(file_name + '(VAL).csv', pout_val.buffer_content())
        Hrm.writeCSVFile(file_name + '(TEST).csv', pout_tst.buffer_content())

    if summary:
        print(evl.summary())
    # Las columnas de predicciones (5) indican: número de segmento, etiqueta real, etiqueta predicha, error (indica con
    # un '+' donde se presentan), y el porcentaje de confianza o algo asi
    return pout_val.buffer_content(), pout_tst.buffer_content()


def classificationOnlyTrain(data_train, path_save, classification_method):
    translate_classifier_name = {
        'J48': 'weka.classifiers.trees.J48',
        'RF': 'weka.classifiers.trees.RandomForest',
        'RT': 'weka.classifiers.trees.RandomTree',
        'SVM': 'weka.classifiers.functions.LibSVM',
        'LR': 'weka.classifiers.functions.LinearRegression',
        'MLP': 'weka.classifiers.functions.MultilayerPerceptron'
    }
    method = translate_classifier_name.get(classification_method)
    options = Datos.PARAMETROS_CLASIFICADOR.get(classification_method + ' 1')
    classifier = Classifier(classname=method, options=options)
    classifier.build_classifier(data_train)
    serialization.write_all(path_save + '.model', [classifier])

    attributes_list = list()
    for i in range(0, data_train.num_attributes - 1):
        attributes_list.append(data_train.attribute(i).name)

    wf = open(path_save + '.txt', 'w')
    attrib_list_m = map(lambda x: x + '\n', attributes_list)
    wf.writelines(attrib_list_m)
    wf.close()


def classificationOnlyTest(data_test, path_load, filter_attributes=True):
    classifier = readModel(path_load)
    if filter_attributes is True:
        data_test = filterModelAttributes(path_load, data_test)
    pout_tst = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")
    evl = Evaluation(data_test)
    evl.test_model(classifier, data_test, output=pout_tst)
    return pout_tst.buffer_content()


def partitionData(data):
    # El rnd None permite que no los mezcle ni desordene al dividirlo
    val_train, test = data.train_test_split(Datos.PORCENTAJE_VAL + Datos.PORCENTAJE_TRAIN, rnd=None)
    train, val = val_train.train_test_split((Datos.PORCENTAJE_TRAIN /
                                             (Datos.PORCENTAJE_VAL + Datos.PORCENTAJE_TRAIN) * 100), rnd=None)
    return train, val, test


def readModel(file_path):
    # Permite leer tanto el clasificador entrenado como los atributos del conjunto con el cual se entreno, luego con
    # esto filtra los datos de validacion y test para aplicar la misma seleccion de caracteristicas
    objects = serialization.read_all(file_path + '.model')
    classifier = Classifier(jobject=objects[0])
    return classifier


def filterModelAttributes(file_path, data_tst, data_val=None):
    rf = open(file_path + '.txt', 'r')
    read_attributes = rf.read().splitlines()
    rf.close()

    # Invierto la lista porque voy sacando del ultimo, asi quedan en el mismo orden que el original
    read_attributes.reverse()
    # Voy guardando los indices donde se encuentran los atributos que quiero quedarme
    indexs_keep = ""
    while np.size(read_attributes) != 0:
        attrib = data_tst.attribute_by_name(read_attributes.pop())
        indexs_keep = indexs_keep + str(attrib.index + 1) + ','
    indexs_keep = indexs_keep + str(data_tst.class_attribute.index + 1)

    # Uso el filtro para remover de forma invertida con tal de quedarme solo con los indices de los que les paso
    flter = Filter(classname="weka.filters.unsupervised.attribute.Remove")
    flter.set_property("attributeIndices", indexs_keep)
    flter.set_property("invertSelection", True)
    flter.inputformat(data_tst)
    data_tst_f = flter.filter(data_tst)
    data_tst_f.class_is_last()
    if data_val is not None:
        data_val_f = flter.filter(data_val)
        data_val_f.class_is_last()
        return data_tst_f, data_val_f
    else:
        return data_tst_f
