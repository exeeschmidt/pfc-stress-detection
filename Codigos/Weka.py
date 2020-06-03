from weka.attribute_selection import ASSearch, ASEvaluation
from weka.classifiers import Classifier, Evaluation, PredictionOutput
from weka.filters import Filter
from weka.core.dataset import Instances
import Codigos.Datos as datos


def SeleccionCaracteristicas(data_train, data_test, metodo_seleccion):
    # opciones: 'PSO' , 'PCA', 'Firsts', 'IG'
    data = Instances.copy_instances(data_train)
    data_tt = Instances.copy_instances(data_test)

    options_eval = ''
    options_search = ''

    final = True
    # Segun el metodo elegido realiza una preseleccion de características usando Gain Ratio (GR). Para cada metodo en
    # particular se preseleccionan un numero distinto de atributos. Luego al aplicarse cada metodo, si este tiene mas
    # del numero de atributos finales, se realiza nuevamente una seleccion utilizando GR
    if metodo_seleccion == 'PCA' or metodo_seleccion == 'GR' or metodo_seleccion == 'GR-pca' or metodo_seleccion == 'GR-pso' or metodo_seleccion == 'GR-bf':
        met_search = 'weka.attributeSelection.Ranker'
        if metodo_seleccion == 'PCA':
            met_eval = 'weka.attributeSelection.PrincipalComponents'
            options_eval = ['-R', '0.95', '-A', '10']
            options_search = ['-N', str(datos.ATRIBS_FINALES)]
            data, data_tt = SeleccionCaracteristicas(data, data_tt, 'GR-pca')
        else:
            met_eval = 'weka.attributeSelection.GainRatioAttributeEval'
        if metodo_seleccion == 'GR':
            options_search = ['-N', str(datos.ATRIBS_FINALES)]
        elif metodo_seleccion == 'GR-pca':
            options_search = ['-N', str(datos.ATRIBS_PCA)]
            final = False
        elif metodo_seleccion == 'GR-pso':
            options_search = ['-N', str(datos.ATRIBS_PSO)]
            final = False
        elif metodo_seleccion == 'GR-bf':
            options_search = ['-N', str(datos.ATRIBS_BF)]
            final = False
    else:
        met_eval = 'weka.attributeSelection.CfsSubsetEval'
        options_eval = ['-P', '4', '-E', '8']
        if metodo_seleccion == 'PSO':
            met_search = 'weka.attributeSelection.PSOSearch'
            options_search = ['-N', '250', '-I', '1000', '-T', '0', '-M', '0.01', '-A', '0.15', '-B', '0.25', '-C', '0.6', '-S', '1']
            data, data_tt = SeleccionCaracteristicas(data, data_tt, 'GR-pso')
        else:
            met_search = 'weka.attributeSelection.BestFirst'
            options_search = ['-D', '1', '-N', '5']
            data, data_tt = SeleccionCaracteristicas(data, data_tt, 'GR-bf')

    flter = Filter(classname="weka.filters.supervised.attribute.AttributeSelection")
    aseval = ASEvaluation(met_eval, options=options_eval)
    assearch = ASSearch(met_search, options=options_search)
    flter.set_property("evaluator", aseval.jobject)
    flter.set_property("search", assearch.jobject)
    flter.inputformat(data)
    data_filtrada = flter.filter(data)
    data_tt_filtrada = flter.filter(data_tt)
    print('Atributos ', metodo_seleccion, ' :', data_filtrada.num_attributes, '/', data.num_attributes)
    # saver = Saver()
    # saver.save_file(data_f, "filtrado.arff")
    # saver.save_file(data_v_f, "filtrado_v.arff")
    if data_filtrada.num_attributes > datos.ATRIBS_FINALES + 1 and final:
        data_filtrada, data_tt_filtrada = SeleccionCaracteristicas(data_filtrada, data_tt_filtrada, 'GR')

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
    # El rnd None permite que no los mezcle ni desordene al dividirlo
    train, test = data.train_test_split(porcentaje, rnd=None)
    return train, test
