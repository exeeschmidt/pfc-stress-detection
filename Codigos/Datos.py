import os
from pathlib import Path

# Se definen las variables y constantes necesarias para la inicialización y ejecución de todas las funciones

ROOT_PATH = Path(__file__).parent.parent
PATH_BD = os.path.join(ROOT_PATH, 'Base de datos')
PATH_CARACTERISTICAS = os.path.join(ROOT_PATH, 'Caracteristicas')
PATH_CODIGOS = os.path.join(ROOT_PATH, 'Codigos')
PATH_LIBRERIAS = os.path.join(ROOT_PATH, 'Librerias')
PATH_PROCESADO = os.path.join(ROOT_PATH, 'Procesado')

PATH_OPENFACE = os.path.join(PATH_LIBRERIAS, 'openface')
PATH_OPENSMILE = os.path.join(PATH_LIBRERIAS, 'opensmile')
PATH_FFMPEG = os.path.join(PATH_LIBRERIAS, 'ffmpeg', 'bin')

# PATH_CONFIG_FILE = os.path.join('config', 'IS09_emotion.conf')
PATH_CONFIG_FILE = os.path.join('config', 'gemaps', 'eGeMAPSv01a.conf')

def buildVideoName(persona, etapa, parte=-1, extension=False):
    video_name = 'Sujeto_' + persona + '_' + etapa
    if parte != -1:
        video_name += '_r' + parte
    if extension:
        video_name += '.mp4'
    return video_name

def buildPathVideo(persona, etapa, nombre_video, extension=True):
    path_video = os.path.join(PATH_BD, 'Sujeto ' + persona, 'Etapa ' + etapa, nombre_video)
    if extension:
        path_video += '.mp4'
    return path_video

def buildPathSub(persona, etapa, sub):
    path = os.path.join(PATH_CARACTERISTICAS, sub, buildVideoName(persona, etapa) + '_' + sub + '.arff')
    return path
