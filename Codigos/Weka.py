from weka.attribute_selection import ASSearch, ASEvaluation
from weka.classifiers import Classifier, Evaluation, PredictionOutput
from weka.filters import Filter
from weka.core.dataset import Instances



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
