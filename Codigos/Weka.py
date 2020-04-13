import traceback

import numpy as np
import weka.core.jvm as jvm
from weka.attribute_selection import ASSearch, AttributeSelection, ASEvaluation
from weka.classifiers import Classifier, Evaluation, PredictionOutput
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.filters import Filter


class CargaYFiltrado:
    def __call__(self, path):
        # Cargo los datos
        loader = Loader("weka.core.converters.ArffLoader")
        data = loader.load_file(path)

        # Creo un filtro para eliminar atributos de tipo string en caso que se presenten
        remove = Filter(classname="weka.filters.unsupervised.attribute.RemoveType", options=["-T", "string"])
        remove.inputformat(data)
        data = remove.filter(data)

        # Indico que la clase es el ultimo atributo
        data.class_is_last()
        return data


class SeleccionCaracteristicas:
    def __call__(self, metodo_seleccion):
        # opciones: 'PSO' , 'PCA', 'Firsts'
        if metodo_seleccion == 'PCA':
            self.met_eval = 'weka.attributeSelection.PrincipalComponents'
            self.met_search = 'weka.attributeSelection.Ranker'
        else:
            self.met_eval = 'weka.attributeSelection.CfsSubsetEval'
            if metodo_seleccion == 'PSO':
                self.met_search = 'weka.attributeSelection.PSOSearch'
            else:
                self.met_search = 'weka.attributeSelection.BestFirst'

        search = ASSearch(classname=self.met_search, options=["-D", "1", "-N", "5"])
        evaluation = ASEvaluation(classname=self.met_eval, options=["-P", "1", "-E", "1"])
        attsel = AttributeSelection()
        attsel.search(search)
        attsel.evaluator(evaluation)
        attsel.select_attributes(data)

        # Selecciono los atributos a partir de los resultados del metodo
        num_att = data.num_attributes
        # Empiezo por el final porque sino corro los indices del vector y no me sirven los resultados del metodo de seleccion
        ind = num_att - 1
        for i in range(num_att, 0, -1):
            # El where devuelve una tupla con el vector con los indices que contienen la data y el tipo
            # Tengo en cuenta tambien la clase porque el metodo de seleccion no lo hace
            if np.where(attsel.selected_attributes == ind)[0].size == 0 and data.class_index != ind:
                data.delete_attribute(ind)
            ind = ind - 1
        return data

class Clasificacion:
    def __call__(self, data, metodo_clasificacion, sumario=False):
        # Opciones, metodo = 'J48', 'RForest', 'RTree', 'SVM', 'LR', 'MLP'
        switcher = {
            'J48': 'weka.classifiers.trees.J48',
            'RForest': 'weka.classifiers.trees.RandomForest',
            'RTree': 'weka.classifiers.trees.RandomTree',
            'SVM': 'weka.classifiers.functions.SMO',
            'LR': 'weka.classifiers.functions.LinearRegression',
            'MLP': 'weka.classifiers.functions.MultilayerPerceptron'
        }

        self.met_clasificacion = switcher.get(metodo_clasificacion)
        classifier = Classifier(classname=self.met_clasificacion)
        classifier.build_classifier(data)

        evaluation = Evaluation(data)
        # Solo usando entrenamiento
        # evaluation.test_model(classifier, data)
        # Dividiendo en un % para entrenamiento y para test
        # evaluation.evaluate_train_test_split(classifier, data, 90.0, Random(1))
        # print(evaluation.summary())
        pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")
        evl = Evaluation(data)
        evl.crossvalidate_model(classifier, data, 2, Random(1), pout)
        if sumario:
            print(evl.summary())
        return pout.buffer_content()



