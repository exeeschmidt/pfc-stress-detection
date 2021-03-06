from weka.attribute_selection import ASSearch, ASEvaluation
from weka.classifiers import Classifier, Evaluation, PredictionOutput
from weka.filters import Filter
from weka.core.dataset import Instances
from weka.core import serialization
import Codigos.Datos as datos
import Codigos.LogManager as log
import os


def SeleccionCaracteristicas(data_train, data_test, metodo_seleccion):
    # opciones: 'PSO' , 'PCA', 'Firsts', 'PC'
    data = Instances.copy_instances(data_train)
    data_tt = Instances.copy_instances(data_test)

    options_eval = ''
    options_search = ''

    final = True
    # Segun el metodo elegido realiza una preseleccion de características usando Gain Ratio (GR). Para cada metodo en
    # particular se preseleccionan un numero distinto de atributos. Luego al aplicarse cada metodo, si este tiene mas
    # del numero de atributos finales, se realiza nuevamente una seleccion utilizando GR
    if metodo_seleccion == 'PCA' or metodo_seleccion == 'PC' or metodo_seleccion == 'PC-pca' or metodo_seleccion == 'PC-pso' or metodo_seleccion == 'PC-bf':
        met_search = 'weka.attributeSelection.Ranker'
        if metodo_seleccion == 'PCA':
            met_eval = 'weka.attributeSelection.PrincipalComponents'
            options_eval = ['-R', '0.95', '-A', '10']
            options_search = ['-N', str(datos.ATRIBS_FINALES)]
            data, data_tt = SeleccionCaracteristicas(data, data_tt, 'PC-pca')
        else:
            met_eval = 'weka.attributeSelection.CorrelationAttributeEval'
        if metodo_seleccion == 'PC':
            options_search = ['-N', str(datos.ATRIBS_FINALES)]
        elif metodo_seleccion == 'PC-pca':
            options_search = ['-N', str(datos.ATRIBS_PCA)]
            final = False
        elif metodo_seleccion == 'PC-pso':
            options_search = ['-N', str(datos.ATRIBS_PSO)]
            final = False
        elif metodo_seleccion == 'PC-bf':
            options_search = ['-N', str(datos.ATRIBS_BF)]
            final = False
    else:
        met_eval = 'weka.attributeSelection.CfsSubsetEval'
        # Agregar -Z para activar el precalculo de la matriz de correlacion
        options_eval = ['-Z', '-P', '4', '-E', '8']
        if metodo_seleccion == 'PSO':
            met_search = 'weka.attributeSelection.PSOSearch'
            options_search = ['-N', '250', '-I', '1000', '-T', '0', '-M', '0.01', '-A', '0.15', '-B', '0.25', '-C', '0.6', '-S', '1']
            data, data_tt = SeleccionCaracteristicas(data, data_tt, 'PC-pso')
        else:
            met_search = 'weka.attributeSelection.BestFirst'
            options_search = ['-D', '1', '-N', '5']
            data, data_tt = SeleccionCaracteristicas(data, data_tt, 'PC-bf')

    flter = Filter(classname="weka.filters.supervised.attribute.AttributeSelection")
    aseval = ASEvaluation(met_eval, options=options_eval)
    assearch = ASSearch(met_search, options=options_search)
    flter.set_property("evaluator", aseval.jobject)
    flter.set_property("search", assearch.jobject)
    flter.inputformat(data)
    data_filtrada = flter.filter(data)
    data_tt_filtrada = flter.filter(data_tt)
    print('Atributos ', metodo_seleccion, ' :', data_filtrada.num_attributes, '/', data.num_attributes)
    log.agrega('Atributos ' + metodo_seleccion + ' :' + str(data_filtrada.num_attributes) + '/' + str(data.num_attributes))
    # saver = Saver()
    # saver.save_file(data_f, "filtrado.arff")
    # saver.save_file(data_v_f, "filtrado_v.arff")
    # if data_filtrada.num_attributes > datos.ATRIBS_FINALES + 1 and final:
    #     data_filtrada, data_tt_filtrada = SeleccionCaracteristicas(data_filtrada, data_tt_filtrada, 'PC')

    return data_filtrada, data_tt_filtrada


def Clasificacion(data_train, data_test, metodo_clasificacion, metodo_seleccion, sumario=False):
    # Opciones, metodo = 'J48', 'RForest', 'RTree', 'SVM', 'LR', 'MLP'
    switcher = {
        'J48': 'weka.classifiers.trees.J48',
        'RForest': 'weka.classifiers.trees.RandomForest',
        'RTree': 'weka.classifiers.trees.RandomTree',
        'SVM': 'weka.classifiers.functions.LibSVM',
        'LR': 'weka.classifiers.functions.LinearRegression',
        'MLP': 'weka.classifiers.functions.MultilayerPerceptron'
    }

    met_clasificacion = switcher.get(metodo_clasificacion)
    classifier = Classifier(classname=met_clasificacion)
    classifier.build_classifier(data_train)

    serialization.write_all(os.path.join(datos.PATH_LOGS, str(datos.FOLD_ACTUAL) + '_' + metodo_seleccion + '-' + metodo_clasificacion + '.model'), [classifier, data_train])

    pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")
    evl = Evaluation(data_test)
    evl.test_model(classifier, data_test, output=pout)

    if sumario:
        print(evl.summary())
    # Las columnas de predicciones (5) indican: número de segmento, etiqueta real, etiqueta predicha, error (indica con
    # un '+' donde se presentan), y el porcentaje de confianza o algo asi
    return pout.buffer_content()


def ParticionaDatos(data, porcentaje=66.0):
    # El rnd None permite que no los mezcle ni desordene al dividirlo
    train, test = data.train_test_split(porcentaje, rnd=None)
    return train, test


def LeeModelo(path, data_val):
    objects = serialization.read_all(path)

    classifier = Classifier(jobject=objects[0])
    data_load = Instances(jobject=objects[1])
    print('Data cargada', data_load.num_attributes)

    ind_keep = ""
    for i in range(0, data_load.num_attributes):
        attrib = data_val.attribute_by_name(data_load.attribute(i).name)
    ind_keep = ind_keep + str(attrib.index + 1) + ','
    ind_keep = ind_keep[0:len(ind_keep) - 1]

    flter = Filter(classname="weka.filters.unsupervised.attribute.Remove")
    flter.set_property("attributeIndices", ind_keep)
    flter.set_property("invertSelection", True)
    flter.inputformat(data_val)
    data_val_f = flter.filter(data_val)
    data_val_f.class_is_last()
    return data_val_f, classifier
