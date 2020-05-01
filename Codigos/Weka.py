import numpy as np
from weka.attribute_selection import ASSearch, AttributeSelection, ASEvaluation
from weka.classifiers import Classifier, Evaluation, PredictionOutput
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.filters import Filter
from weka.core.dataset import Instances

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


def SeleccionCaracteristicas(data2, metodo_seleccion, sumario=False):
    # opciones: 'PSO' , 'PCA', 'Firsts'
    data = Instances.copy_instances(data2)

    if metodo_seleccion == 'PCA':
        met_eval = 'weka.attributeSelection.PrincipalComponents'
        met_search = 'weka.attributeSelection.Ranker'
    else:
        met_eval = 'weka.attributeSelection.CfsSubsetEval'
        if metodo_seleccion == 'PSO':
            met_search = 'weka.attributeSelection.PSOSearch'
        else:
            met_search = 'weka.attributeSelection.BestFirst'

    search = ASSearch(classname=met_search)
    evaluation = ASEvaluation(classname=met_eval)
    attsel = AttributeSelection()
    attsel.search(search)
    attsel.evaluator(evaluation)
    attsel.select_attributes(data)

    if sumario:
        print("# attributes: " + str(attsel.number_attributes_selected))
        print("result string:\n" + attsel.results_string)

    # Selecciono los atributos a partir de los resultados del método
    num_att = data.num_attributes
    # Empiezo por el final porque sino corro los índices del vector y no me sirven los resultados del método de selección
    ind = num_att - 1
    for i in range(num_att, 0, -1):
        # El where devuelve una tupla con el vector con los índices que contienen la data y el tipo
        # Tengo en cuenta también la clase porque el método de selección no lo hace
        if np.where(attsel.selected_attributes == ind)[0].size == 0 and data.class_index != ind:
            data.delete_attribute(ind)
        ind = ind - 1
    return data


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
    return pout.buffer_content(), evl.mean_absolute_error


def ParticionaDatos(data, porcentaje=66.0):
    train, test = data.train_test_split(porcentaje, Random(1))
    return train, test
