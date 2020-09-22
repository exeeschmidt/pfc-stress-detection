import os
import numpy as np
from pathlib import Path

# Se definen las variables y constantes necesarias para la inicialización y ejecución de todas las funciones

ROOT_PATH = Path(__file__).parent.parent
PATH_BD_PROPIA = os.path.join(ROOT_PATH, 'Base de datos')
PATH_BD_MSP = os.path.join(ROOT_PATH, 'MSP-IMPROV')
PATH_CARACTERISTICAS = os.path.join(ROOT_PATH, 'Caracteristicas')
PATH_CODIGOS = os.path.join(ROOT_PATH, 'Codigos')
PATH_LIBRERIAS = os.path.join(ROOT_PATH, 'Librerias')
PATH_PROCESADO = os.path.join(ROOT_PATH, 'Procesado')
PATH_LOGS = ''

PATH_OPENFACE = os.path.join(PATH_LIBRERIAS, 'openface')
PATH_OPENSMILE = os.path.join(PATH_LIBRERIAS, 'opensmile')
PATH_FFMPEG = os.path.join(PATH_LIBRERIAS, 'ffmpeg', 'bin')
PATH_ETIQUETAS = os.path.join(PATH_BD_PROPIA, 'EtiquetadoConTiempo.csv')

CONFIG_FILE = 'IS09_emotion.conf'
# CONFIG_FILE = 'eGeMAPSv01a.conf'
PATH_CONFIG_FILE = os.path.join('config', CONFIG_FILE)

EXTENSION_VIDEO = '.mp4'
EXTENSION_AUDIO = '.wav'

EXPERIMENTO = ''
# TEST con valor -1 indica que se usara la lista de personas y por tanto el ordena instancia.
TEST = 1
VAL = 2
BINARIZO_ETIQUETA = False
ETIQUETAS_BINARIAS = np.array(['N', 'S'])
ETIQUETAS_MULTICLASES = np.array(['N', 'B', 'M', 'A'])
GUARDO_INFO_CLASIFICACION = True

INSTANCIAS_POR_PERIODOS = 20
VOTO_MEJORES_X = 4
# PORC_ATRIBS_PCA = 2
# PORC_ATRIBS_PSO = 2
# PORC_ATRIBS_BF = 2
PORC_ATRIBS_PCA = 50
PORC_ATRIBS_PSO = 10
PORC_ATRIBS_BF = 10
PORC_ATRIBS_FINALES = 2
ATRIBS_PCA = 0
ATRIBS_PSO = 0
ATRIBS_BF = 0
ATRIBS_FINALES = 0
NUM_ATRIBS = 0
TIEMPO_MICROEXPRESION = 0.25
LIMITE_FPS = 10

if TEST == -1:
    PERSONAS = np.array(['05', '13', '19'])
else:
    PERSONAS = np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
                         '17', '18', '19', '20', '21'])

ETAPAS = np.array(['1', '2'])
ZONAS = np.array(['ojoizq', 'ojoder', 'cejaizq', 'cejader', 'boca', 'nariz'])
MET_EXTRACCION = np.array(['LBP', 'HOG', 'HOP', 'AUS'])
MET_SELECCION = np.array(['PCA', 'BF', 'PSO'])
MET_CLASIFICACION = np.array(['RF', 'SVM', 'J48', 'MLP'])

MEJORES_CONFIGURACIONES = np.array([
    np.array(['PCA', 'SVM']),
    np.array(['PCA', 'MLP']),
    np.array(['BF', 'RF']),
    np.array(['PSO', 'RF'])
])

FOLD_ACTUAL = -1

# Definen la division de datos al usar la mezcla de instancias
PORCENTAJE_TRAIN = 50
PORCENTAJE_VAL = 30

# Esto permite probar distinto parametros de los clasificadores
PRUEBA_PARAMETROS_CLASIFICACION = False
PARAMETROS_CLASIFICADOR = {
    'SVM 1': ['-S', '0', '-K', '0', '-D', '3', '-G', '0.0', '-R', '0.0', '-N', '0.5', '-M', '1000.0', '-C', '1.0',
              '-E', '0.001', '-P', '0.1', '-Z'],
    'MLP 1': ['-L', '0.3', '-M', '0.2', '-N', '100', '-V', '0', '-S', '0', '-E', '20', '-H', 'a'],
    'RF 1': ['-P', '100', '-I', '500', '-num-slots', '16', '-K', '0', '-M', '1.0', '-V', '0.001', '-S', '1'],
    'J48 1': ['-C', '0.25', '-M', '8'],
}

# -M (Cache Size in MB) -G (Gamma) -E (Tolerance criteria) -P(Epsilon in Lost) -Z (Normalize data) -C (Cost)
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

PRUEBA_PARAMETROS_SELECCION = False
PARAMETROS_SELECCION_BUSQUEDA = {
    'BF 1': ['-D', '2', '-N', '5'],
    'BF 2': ['-D', '0', '-N', '5'],
    'PSO 1': ['-N', '1000', '-I', '1000', '-T', '0', '-M', '0.03', '-A', '0.33', '-B', '0.33', '-C', '0.34', '-S',
              '1'],
}

PARAMETROS_SELECCION_EVALUACION = {
    'PCA': ['-R', '0.95', '-A', '10', '-C', '-O'],
    'CFS': ['-Z', '-P', '4', '-E', '8']
}


# -D (Direction (0-Backward, 1-Forward, 2-Bidirectional)) -N (SearchTermination)how many expansion without changes
# for finish) weka.attributeSelection.BestFirst -D 1 -N 5

# -N (PopulationSize) -I (Nro Iterations) -B (Social Weigth) -A (Inertial Weigth) -C (Individual Weigth)
# -M (Mutation prob) -T (Mutation type)
# weka.attributeSelection.PSOSearch -N 20 -I 40 -T 0 -M 0.01 -A 0.33 -B 0.33 -C 0.34 -S 1 -L

# -T (Threshold) -N (Num to select)
# weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1

# EVALUADORES
# -P (NumThreads) -E (Size of pool) (nro of cores for example) -Z (Precalculate correlation matrix)
# weka.attributeSelection.CfsSubsetEval -P 1 -E 1 -Z

# -R (Variance covered) -C (Center data) (like normalize)
# weka.attributeSelection.PrincipalComponents -R 0.95 -A 5 -C


def defineActualValidationFold(fold):
    global FOLD_ACTUAL
    FOLD_ACTUAL = fold


def defineLogFolder(nombre):
    global PATH_LOGS
    PATH_LOGS = os.path.join(ROOT_PATH, 'Logs', nombre)


def calculateAttributesToCut(atributos):
    global ATRIBS_BF, ATRIBS_PCA, ATRIBS_PSO, ATRIBS_FINALES, NUM_ATRIBS
    NUM_ATRIBS = atributos
    ATRIBS_BF = int(NUM_ATRIBS * (PORC_ATRIBS_BF / 100))
    ATRIBS_PCA = int(NUM_ATRIBS * (PORC_ATRIBS_PCA / 100))
    ATRIBS_PSO = int(NUM_ATRIBS * (PORC_ATRIBS_PSO / 100))
    ATRIBS_FINALES = int(NUM_ATRIBS * (PORC_ATRIBS_FINALES / 100))


def classificationParams():
    global PRUEBA_PARAMETROS_SELECCION
    PRUEBA_PARAMETROS_SELECCION = False
    global PRUEBA_PARAMETROS_CLASIFICACION
    PRUEBA_PARAMETROS_CLASIFICACION = True
    global MET_CLASIFICACION
    MET_CLASIFICACION = np.array(['MLP 1', 'MLP 2', 'MLP 3', 'MLP 4', 'MLP 5', 'MLP 6'])
    global MET_SELECCION
    MET_SELECCION = np.array(['BF'])


def selectionParams():
    global PRUEBA_PARAMETROS_CLASIFICACION
    PRUEBA_PARAMETROS_CLASIFICACION = False
    global PRUEBA_PARAMETROS_SELECCION
    PRUEBA_PARAMETROS_SELECCION = True
    global MET_SELECCION
    MET_SELECCION = np.array(['BF 1', 'BF 2', 'BF 3', 'BF 4', 'PSO 1', 'PSO 2', 'PSO 3', 'PSO 4', 'PSO 5'])
    global MET_CLASIFICACION
    MET_CLASIFICACION = np.array(['SVM', 'RF'])


def refreshParamsMLP(atributos):
    # Los 4 vienen de la cantidad de clases
    r1 = int(np.sqrt(atributos * 4))
    r2 = float((atributos / 4) ** (1 / 3))
    actualizacion = {
        'MLP 4': ['-L', '0.3', '-M', '0.2', '-N', '500', '-V', '0', '-S', '0', '-E', '20', '-H',
                  str(int(atributos / 20)) + ',' + str(int(atributos / 40))],
        'MLP 5': ['-L', '0.3', '-M', '0.2', '-N', '500', '-V', '0', '-S', '0', '-E', '20', '-H',
                  str(r1)],
        'MLP 6': ['-L', '0.3', '-M', '0.2', '-N', '500', '-V', '0', '-S', '0', '-E', '20', '-H',
                  str(int(np.multiply(4, r2) ** 2)) + ',' + str(int(np.multiply(4, r2)))]
    }
    PARAMETROS_CLASIFICADOR.update(actualizacion)
    print(PARAMETROS_CLASIFICADOR.get('MLP 3'))
