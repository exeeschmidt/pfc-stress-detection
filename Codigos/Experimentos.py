import Codigos.Caracteristicas as carac
import Codigos.Herramientas as hrm
import Codigos.Weka as wek
import weka.core.jvm as jvm
import numpy as np
import Codigos.ArffManager as am
import os


# zonas = np.array(['ojoizq', 'ojoder', 'boca', 'nariz'])
# met_caracteristicas = np.array(['LBP', 'AU'])
# met_seleccion = np.array(['Firsts', 'PCA'])
# met_clasificacion = np.array(['RForest', 'J48', 'SVM', 'MLP'])

def Unimodal(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion, binarizo_etiquetas=False):
    jvm.start()
    selecciono_caracteristicas = False
    if len(met_seleccion) > 0:
        selecciono_caracteristicas = True

    print('Extracción de caracteristicas en progreso')
    for i in personas:
        for j in etapas:
            features = carac.Video(binarizo_etiquetas, zonas, met_caracteristicas)
            features(i, j, completo=True)
            print('...')
    print('Completada extraccion de caracteristicas')

    am.ConcatenaArff('Resultado Video', personas, etapas, bool_partes=False)
    path = 'Caracteristicas' + os.sep + 'Resultado Video.arff'
    data = wek.CargaYFiltrado(path)

    cant_met_seleccion = 1
    if selecciono_caracteristicas:
        cant_met_seleccion = len(met_seleccion)

    vec_predicciones = np.array([])
    lista_metodos = np.empty((0))

    print('Clasificación en progreso')
    for i in range(0, cant_met_seleccion):
        if selecciono_caracteristicas:
            metodo_actual = met_seleccion[i] + ' + '
            data = wek.SeleccionCaracteristicas(data, met_seleccion[i])
        else:
            metodo_actual = ''
        train, test = wek.ParticionaDatos(data)
        print('..')
        for j in range(0, len(met_clasificacion)):
            lista_metodos = np.append(lista_metodos, np.array([metodo_actual + met_clasificacion[j]]))
            predicciones = wek.Clasificacion(train, test, met_clasificacion[j])
            if len(vec_predicciones) == 0:
                vec_predicciones = np.array([hrm.prediccionCSVtoArray(predicciones)])
            else:
                vec_predicciones = np.concatenate([vec_predicciones, np.array([hrm.prediccionCSVtoArray(predicciones)])])
            print('...')
    #Aca llamar a alguna funcion que convierta toda esa matriz de 3 dimensiones a una matriz linda
    resultados = hrm.resumoPredicciones(vec_predicciones, lista_metodos)
    return resultados

# def MultimodalCompleto():
#
# def MultimodalSinSilencios():