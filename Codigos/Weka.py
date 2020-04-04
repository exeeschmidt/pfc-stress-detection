import weka.core.jvm as jvm
from weka.clusterers import Clusterer, ClusterEvaluation
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.attribute_selection import ASSearch, AttributeSelection, ASEvaluation
import traceback
import numpy as np


class SeleccionCaracteristicas:
    def __init__(self, metodo):
        # opciones: 'PSO' , 'PCA', 'Firsts'
        if metodo == 'PCA':
            self.met_eval = 'weka.attributeSelection.PrincipalComponents'
            self.met_search = 'weka.attributeSelection.Ranker'
        else:
            self.met_eval = 'weka.attributeSelection.CfsSubsetEval'
            if metodo == 'PSO':
                self.met_search = 'weka.attributeSelection.PSOSearch'
            else:
                self.met_search = 'weka.attributeSelection.BestFirst'

    def __call__(self, path):
        try:
            jvm.start()
            self.__ejecuta__(path)
        except Exception as e:
            print(traceback.format_exc(e))
        finally:
            jvm.stop()

    def __ejecuta__(self, path):
        # Cargo los datos
        loader = Loader("weka.core.converters.ArffLoader")
        data = loader.load_file(path)
        # Indico que la clase es el ultimo atributo
        data.class_is_last()

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
    def __init__(self, metodo):
        # Opciones, metodo = 'J48', 'RForest', 'RTree', 'SVM', 'RegreLineal', 'RedNeuronal'
        switcher = {
            'J48': 'weka.classifiers.trees.J48',
            'RForest': 'weka.classifiers.trees.RandomForest',
            'RTree': 'weka.classifiers.trees.RandomTree',
            'SVM': 'weka.classifiers.functions.SMO',
            'LR': 'weka.classifiers.functions.LinearRegression',
            'MLP': 'weka.classifiers.functions.MultilayerPerceptron'
        }
        self.metodo = switcher.get(metodo)
        # Clasificadores
        #   weka.classifiers.trees.J48
        #   weka.classifiers.trees.RandomForest
        #   weka.classifiers.trees.RandomTree
        # Este es una manera optimizada del vector de soporte lineal, no deja usar el de LibSVM desde aca parece
        #   weka.classifiers.functions.SMO (Sequential Minimal Optimization algorithm for training a support vector classifier)
        # Regresion lineal solo es si se usan más de 2 clases
        #   weka.classifiers.functions.LinearRegression
        #   weka.classifiers.functions.MultilayerPerceptron

    def __call__(self, path):
        try:
            jvm.start()
            self.__ejecuta__(path)
        except Exception as e:
            print(traceback.format_exc(e))
        finally:
            jvm.stop()

    def __ejecuta__(self, path):
        # Cargo los datos
        loader = Loader("weka.core.converters.ArffLoader")
        data = loader.load_file(path)
        # Indico que la clase es el ultimo atributo
        data.class_is_last()

        classifier = Classifier(classname=self.metodo)
        classifier.build_classifier(data)

        evaluation = Evaluation(data)
        # Solo usando entrenamiento
        # evl = evaluation.test_model(classifier, data)
        # print(evl.summary())
        # Dividiendo en un % para entrenamiento y para test
        evaluation.evaluate_train_test_split(classifier, data, 66.0, Random(1))
        print(evaluation.summary())
