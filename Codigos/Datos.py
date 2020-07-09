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

EXPERIMENTO = 'Primer multimodal'
TEST = 1
VAL = 3
BINARIZO_ETIQUETA = False
ELIMINA_SILENCIOS = False

INSTANCIAS_POR_PERIODOS = 20
VOTO_MEJORES_X = 4
ATRIBS_PCA = 3000
ATRIBS_PSO = 500
ATRIBS_BF = 1000
ATRIBS_FINALES = 100
TIEMPO_MICROEXPRESION = 0.25

PERSONAS = np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
                     '17', '18', '19', '20', '21'])
ETAPAS = np.array(['1', '2'])
ZONAS = np.array(['ojoizq', 'ojoder', 'cejaizq', 'cejader', 'boca', 'nariz'])
MET_EXTRACCION = np.array(['LBP', 'HOG', 'HOP', 'AUS'])
MET_SELECCION = np.array(['PCA', 'BF', 'PSO'])
MET_CLASIFICACION = np.array(['RForest', 'SVM', 'J48', 'MLP'])

FOLD_ACTUAL = -1


def defineFoldActual(fold):
    global FOLD_ACTUAL
    FOLD_ACTUAL = fold


def defineCarpetaLog(nombre):
    global PATH_LOGS
    PATH_LOGS = os.path.join(PATH_LOGS, nombre)