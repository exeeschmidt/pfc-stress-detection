import os
from pathlib import Path

# Se definen las variables y constantes necesarias para la inicialización y ejecución de todas las funciones

PATH_ROOT = Path(__file__).parent.parent
PATH_BD = os.path.join(PATH_ROOT, 'Base de datos')
PATH_CARACTERISTICAS = os.path.join(PATH_ROOT, 'Caracteristicas')
PATH_CODIGOS = os.path.join(PATH_ROOT, 'Codigos')
PATH_LIBRERIAS = os.path.join(PATH_ROOT, 'Librerias')
PATH_PROCESADO = os.path.join(PATH_ROOT, 'Procesado')
PATH_ETIQUETAS = os.path.join(PATH_BD, 'EtiquetadoConTiempo.csv')
PATH_OPENFACE = os.path.join(PATH_LIBRERIAS, 'openface')
PATH_OPENSMILE = os.path.join(PATH_LIBRERIAS, 'opensmile')
PATH_FFMPEG = os.path.join(PATH_LIBRERIAS, 'ffmpeg', 'bin')
