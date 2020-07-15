import os
import numpy as np
from pathlib import Path

# Se definen las variables y constantes necesarias para la inicialización y ejecución de todas las funciones

ROOT_PATH = Path(__file__).parent.parent
PATH_BD = os.path.join(ROOT_PATH, 'Base de datos')
PATH_CARACTERISTICAS = os.path.join(ROOT_PATH, 'Caracteristicas')
PATH_CODIGOS = os.path.join(ROOT_PATH, 'Codigos')
PATH_LIBRERIAS = os.path.join(ROOT_PATH, 'Librerias')
PATH_PROCESADO = os.path.join(ROOT_PATH, 'Procesado')
PATH_LOGS = os.path.join(ROOT_PATH, 'Logs')

PATH_OPENFACE = os.path.join(PATH_LIBRERIAS, 'openface')
PATH_OPENSMILE = os.path.join(PATH_LIBRERIAS, 'opensmile')
PATH_FFMPEG = os.path.join(PATH_LIBRERIAS, 'ffmpeg', 'bin')
PATH_ETIQUETAS = os.path.join(PATH_BD, 'EtiquetadoConTiempo.csv')

# PATH_CONFIG_FILE = os.path.join('config', 'IS09_emotion.conf')
PATH_CONFIG_FILE = os.path.join('config', 'gemaps', 'eGeMAPSv01a.conf')

EXPERIMENTO = 'Unimodal'
TEST = 3
VAL = 4
BINARIZO_ETIQUETA = False
ELIMINA_SILENCIOS = False
GUARDO_MODEL = False

INSTANCIAS_POR_PERIODOS = 20
VOTO_MEJORES_X = 4
# ATRIBS_PCA = 3000
# ATRIBS_PSO = 500
# ATRIBS_BF = 1000
ATRIBS_PCA = 100
ATRIBS_PSO = 100
ATRIBS_BF = 100
ATRIBS_FINALES = 100
TIEMPO_MICROEXPRESION = 0.25

PERSONAS = np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
                     '17', '18', '19', '20', '21'])
ETAPAS = np.array(['1', '2'])
ZONAS = np.array(['ojoizq', 'ojoder', 'cejaizq', 'cejader', 'boca', 'nariz'])
MET_EXTRACCION = np.array(['LBP', 'HOG', 'HOP', 'AUS'])
MET_SELECCION = np.array(['BF', 'PSO'])
# MET_CLASIFICACION = np.array(['RForest', 'SVM', 'J48', 'MLP'])
# MET_CLASIFICACION = np.array(['RForest', 'J48', 'MLP'])

FOLD_ACTUAL = -1

# Esto permite probar distinto parametros de los clasificadores
PRUEBA_PARAMETROS = True
MET_CLASIFICACION = np.array(['SVM 1', 'SVM 2', 'SVM 3', 'SVM 4'])
PARAMETROS_CLASIFICADOR = {
    'SVM 1': ['-S', '0', '-K', '0', '-D', '3', '-G', '0.0', '-R', '0.0', '-N', '0.5', '-M', '40.0', '-C', '1.0', '-E', '0.001', '-P', '0.1'],
    'SVM 2': ['-S', '0', '-K', '1', '-D', '3', '-G', '0.0', '-R', '0.0', '-N', '0.5', '-M', '40.0', '-C', '1.0', '-E', '0.001', '-P', '0.1'],
    'SVM 3': ['-S', '0', '-K', '2', '-D', '3', '-G', '0.0', '-R', '0.0', '-N', '0.5', '-M', '40.0', '-C', '1.0', '-E', '0.001', '-P', '0.1'],
    'SVM 4': ['-S', '0', '-K', '3', '-D', '3', '-G', '0.0', '-R', '0.0', '-N', '0.5', '-M', '40.0', '-C', '1.0', '-E', '0.001', '-P', '0.1'],
    'MLP 1': ['-L', '0.3', '-M', '0.2', '-N', '500', '-V', '0', '-S', '0', '-E', '20', '-H', 'a'],
    'RF 1': ['-P', '100', '-I', '100', '-num-slots', '1', '-K', '0', '-M', '1.0', '-V', '0.001', '-S', '1'],
    'J48 1': ['-C', '0.25' '-M', '2']
}

# -M (Cache Size in MB) -G (Gamma) -E (Epsilon) -P(Lost)
# -K (Kernel Type: 0-linear 1-polynomial 2-radial 3-sigmoid
# weka.classifiers.functions.LibSVM -S 0 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1

# -L (Learning rate) -M (Momentum) -N (trainingTime) -V (Validation set size) -H (Hidden layers)
# H- 'a' = (attribs + classes) / 2  sino agregar 2, 3, 4 (crearía 3 capas ocultas con la cantidad
# especificada de nodos esa 2, 3, 4)
# 'i' = attribs, 'o' = classes, 't' = attribs .+ classes) for wildcard values to create hidden layers
# weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a

# -I (Number of trees) -K (numFeatures random) -num-slots(Threads)
# weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1

# -C (confidence factor) -M (minimun objects per leaf)
# weka.classifiers.trees.J48 -C 0.25 -M 2

def defineFoldActual(fold):
    global FOLD_ACTUAL
    FOLD_ACTUAL = fold


def defineCarpetaLog(nombre):
    global PATH_LOGS
    PATH_LOGS = os.path.join(PATH_LOGS, nombre)