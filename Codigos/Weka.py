from weka.attribute_selection import ASSearch, ASEvaluation
from weka.classifiers import Classifier, Evaluation, PredictionOutput
from weka.filters import Filter
from weka.core.dataset import Instances
from weka.core import serialization
import Codigos.Datos as datos
import Codigos.LogManager as log
import Codigos.Herramientas as hrm
import os
import numpy as np


def SeleccionCaracteristicas(data_train, data_val, data_test, metodo_seleccion):
    # opciones: 'PSO' , 'PCA', 'Firsts', 'PC'
    data_trn = Instances.copy_instances(data_train)
    data_vld = Instances.copy_instances(data_val)
    data_tst = Instances.copy_instances(data_test)

    options_eval = ''
    options_search = ''

    # Segun el metodo elegido realiza una preseleccion de características usando Gain Ratio (GR). Para cada metodo en
    # particular se preseleccionan un numero distinto de atributos. Luego al aplicarse cada metodo, si este tiene mas
    # del numero de atributos finales, se realiza nuevamente una seleccion utilizando GR
    if metodo_seleccion == 'PCA' or metodo_seleccion == 'PC' or metodo_seleccion == 'PC-pca' or metodo_seleccion == 'PC-pso' or metodo_seleccion == 'PC-bf':
        met_search = 'weka.attributeSelection.Ranker'
        if metodo_seleccion == 'PCA':
            met_eval = 'weka.attributeSelection.PrincipalComponents'
            options_eval = ['-R', '0.95', '-A', '10']
            options_search = ['-N', str(datos.ATRIBS_FINALES)]
            data_trn, data_vld, data_tst = SeleccionCaracteristicas(data_trn, data_vld, data_tst, 'PC-pca')
        else:
            met_eval = 'weka.attributeSelection.CorrelationAttributeEval'
        if metodo_seleccion == 'PC':
            options_search = ['-N', str(datos.ATRIBS_FINALES)]
        elif metodo_seleccion == 'PC-pca':
            options_search = ['-N', str(datos.ATRIBS_PCA)]
        elif metodo_seleccion == 'PC-pso':
            options_search = ['-N', str(datos.ATRIBS_PSO)]
        elif metodo_seleccion == 'PC-bf':
            options_search = ['-N', str(datos.ATRIBS_BF)]
    else:
        met_eval = 'weka.attributeSelection.CfsSubsetEval'
        # Agregar -Z para activar el precalculo de la matriz de correlacion
        options_eval = ['-Z', '-P', '4', '-E', '8']
        if metodo_seleccion == 'PSO':
            met_search = 'weka.attributeSelection.PSOSearch'
            options_search = ['-N', '250', '-I', '1000', '-T', '0', '-M', '0.01', '-A', '0.15', '-B', '0.25', '-C', '0.6', '-S', '1']
            data_trn, data_vld, data_tst = SeleccionCaracteristicas(data_trn, data_vld, data_tst, 'PC-pso')
        else:
            met_search = 'weka.attributeSelection.BestFirst'
            options_search = ['-D', '1', '-N', '5']
            data_trn, data_vld, data_tst = SeleccionCaracteristicas(data_trn, data_vld, data_tst, 'PC-bf')

    flter = Filter(classname="weka.filters.supervised.attribute.AttributeSelection")
    aseval = ASEvaluation(met_eval, options=options_eval)
    assearch = ASSearch(met_search, options=options_search)
    flter.set_property("evaluator", aseval.jobject)
    flter.set_property("search", assearch.jobject)
    flter.inputformat(data_trn)
    data_trn_filtrada = flter.filter(data_trn)
    data_vld_filtrada = flter.filter(data_vld)
    data_tst_filtrada = flter.filter(data_tst)
    print('Atributos ', metodo_seleccion, ' :', data_trn_filtrada.num_attributes, '/', data_trn.num_attributes)
    log.agrega('Atributos ' + metodo_seleccion + ' :' + str(data_trn_filtrada.num_attributes) + '/' + str(data_trn.num_attributes))
    # saver = Saver()
    # saver.save_file(data_f, "filtrado.arff")
    # saver.save_file(data_v_f, "filtrado_v.arff")

    return data_trn_filtrada, data_vld_filtrada, data_tst_filtrada


def Clasificacion(data_train, data_val, data_test, metodo_clasificacion, metodo_seleccion, sumario=False):
    # Opciones, metodo = 'J48', 'RForest', 'RTree', 'SVM', 'LR', 'MLP'
    switcher = {
        'J48': 'weka.classifiers.trees.J48',
        'RForest': 'weka.classifiers.trees.RandomForest',
        'RTree': 'weka.classifiers.trees.RandomTree',
        'SVM': 'weka.classifiers.functions.LibSVM',
        'LR': 'weka.classifiers.functions.LinearRegression',
        'MLP': 'weka.classifiers.functions.MultilayerPerceptron'
    }

    # En el caso de probar parametros se agregan como opciones, sino toma todos los parametros por defecto
    if not datos.PRUEBA_PARAMETROS:
        met_clasificacion = switcher.get(metodo_clasificacion)
        classifier = Classifier(classname=met_clasificacion)
    else:
        met_clasificacion = switcher.get(metodo_clasificacion[0: len(metodo_clasificacion) - 2])
        opciones = datos.PARAMETROS_CLASIFICADOR(met_clasificacion)
        classifier = Classifier(classname=met_clasificacion, options=opciones)

    classifier.build_classifier(data_train)

    if datos.GUARDO_MODEL:
        nombre_archivo = os.path.join(datos.PATH_LOGS, str(datos.FOLD_ACTUAL) + '_' + metodo_seleccion + '-' + metodo_clasificacion)
        serialization.write_all(nombre_archivo + '.model', [classifier])

        attrib_list = list()
        it = data_train.attributes()
        attrib = it.next()
        while it.col != data_train.num_attributes:
            attrib_list.append(attrib.name)
            attrib = it.next()

        wf = open(nombre_archivo + '.txt', 'w')
        attrib_list_m = map(lambda x: x + '\n', attrib_list)
        wf.writelines(attrib_list_m)
        wf.close()

    pout_val = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")
    evl = Evaluation(data_val)
    evl.test_model(classifier, data_val, output=pout_val)
    hrm.escribeCSV(nombre_archivo + '(VAL).csv', pout_val.buffer_content())

    pout_tst = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")
    evl = Evaluation(data_test)
    evl.test_model(classifier, data_test, output=pout_tst)
    hrm.escribeCSV(nombre_archivo + '(TEST).csv', pout_tst.buffer_content())

    if sumario:
        print(evl.summary())
    # Las columnas de predicciones (5) indican: número de segmento, etiqueta real, etiqueta predicha, error (indica con
    # un '+' donde se presentan), y el porcentaje de confianza o algo asi
    return pout_val.buffer_content(), pout_tst.buffer_content()


def ParticionaDatos(data, porcentaje=66.0):
    # El rnd None permite que no los mezcle ni desordene al dividirlo
    train, test = data.train_test_split(porcentaje, rnd=None)
    return train, test


def LeeModelo(nombre_archivo, data_val, data_tst):
    # Permite leer tanto el clasificador entrenado como los atributos del conjunto con el cual se entreno, luego con
    # esto filtra los datos de validacion y test para aplicar la misma seleccion de caracteristicas
    objects = serialization.read(nombre_archivo + '.model')
    classifier = Classifier(objects[0])

    rf = open(nombre_archivo + '.txt', 'r')
    attrib_list_readed = rf.read().splitlines()
    rf.close()

    # Voy guardando los indices donde se encuentran los atributos que quiero quedarme
    ind_keep = ""
    while np.size(attrib_list_readed) != 0:
        attrib = data_val.attribute_by_name(attrib_list_readed.pop())
        ind_keep = ind_keep + str(attrib.index + 1) + ','
    ind_keep = ind_keep + str(data_val.class_attribute.index + 1)

    # Uso el filtro para remover de forma invertida con tal de quedarme solo con los indices de los que les paso
    flter = Filter(classname="weka.filters.unsupervised.attribute.Remove")
    flter.set_property("attributeIndices", ind_keep)
    flter.set_property("invertSelection", True)
    flter.inputformat(data_val)
    data_val_f = flter.filter(data_val)
    data_tst_f = flter.filter(data_tst)
    data_val_f.class_is_last()
    data_tst_f.class_is_last()

    return data_val_f, data_tst_f, classifier
