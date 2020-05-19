import numpy as np
from weka.attribute_selection import ASSearch, AttributeSelection, ASEvaluation
from weka.classifiers import Classifier, Evaluation, PredictionOutput
from weka.core.converters import Loader, Saver
from weka.filters import Filter
from weka.core.dataset import Instances, Attribute, Instance
import Codigos.Datos as datos
import os


def CargaYFiltrado(path):
    # Cargo los datos
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(path)

    # Creo un filtro para eliminar atributos de tipo string en caso que se presenten
    remove = Filter(classname="weka.filters.unsupervised.attribute.RemoveType", options=["-T", "string"])
    remove.inputformat(data)
    data = remove.filter(data)

    # Indico que la clase es el último atributo
    data.class_is_last()
    return data


def SeleccionCaracteristicas(data_train, data_test, metodo_seleccion, sumario=False):
    # opciones: 'PSO' , 'PCA', 'Firsts'
    data = Instances.copy_instances(data_train)
    data_tt = Instances.copy_instances(data_test)

    if metodo_seleccion == 'PCA':
        met_eval = 'weka.attributeSelection.PrincipalComponents'
        met_search = 'weka.attributeSelection.Ranker'
    else:
        met_eval = 'weka.attributeSelection.CfsSubsetEval'
        if metodo_seleccion == 'PSO':
            met_search = 'weka.attributeSelection.PSOSearch'
        else:
            met_search = 'weka.attributeSelection.BestFirst'

    flter = Filter(classname="weka.filters.supervised.attribute.AttributeSelection")
    aseval = ASEvaluation(met_eval)
    assearch = ASSearch(met_search)
    flter.set_property("evaluator", aseval.jobject)
    flter.set_property("search", assearch.jobject)
    flter.inputformat(data)
    data_filtrada = flter.filter(data)
    data_tt_filtrada = flter.filter(data_tt)
    # saver = Saver()
    # saver.save_file(data_f, "filtrado.arff")
    # saver.save_file(data_v_f, "filtrado_v.arff")

    return data_filtrada, data_tt_filtrada


def Clasificacion(data_train, data_test, metodo_clasificacion, sumario=False):
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

    # Validación cruzada
    # evl.crossvalidate_model(classifier, data, 2, Random(1), pout)

    pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")
    evl = Evaluation(data_train)
    evl.test_model(classifier, data_test, output=pout)

    if sumario:
        print(evl.summary())
    # Las columnas de predicciones (5) indican: número de segmento, etiqueta real, etiqueta predicha, error (indica con
    # un '+' donde se presentan), y el porcentaje de confianza o algo asi
    return pout.buffer_content(), evl.error_rate


def ParticionaDatos(data, porcentaje=66.0):
    #El rnd None permite que no los mezcle ni desordene al dividirlo
    train, test = data.train_test_split(porcentaje, rnd=None)
    return train, test

def Cabecera(nombre_atrib, range_atrib, zonas):
    atrib = list()
    # Recorro dentro de los intervalos pasados por el rango seleccionando la zona correspondiente
    for j in range(0, len(range_atrib) - 1):
        cont = 1
        for i in range(range_atrib[j], range_atrib[j + 1]):
            atrib.append(Attribute.create_numeric(nombre_atrib + '_' + zonas[j] + '[' + str(cont) + ']'))
            cont = cont + 1

    data = Instances.create_instances(nombre_atrib + "features", atrib, 0)
    return data

def AgregaInstancia(data, features):
    inst = Instance.create_instance(features)
    data.add_instance(inst)
    return data

def Guarda(nombre_archivo, atrib, data):
    path = os.path.join(datos.PATH_CARACTERISTICAS, atrib, nombre_archivo + '_' + atrib + '.arff')
    saver = Saver()
    saver.save_file(data, path)
