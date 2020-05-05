import os
import numpy as np
import cv2 as cv
import weka.core.jvm as jvm
import Codigos.Datos as datos
import Codigos.Herramientas as herramientas
import Codigos.Metodos as metodos
import Codigos.Caracteristicas as caracteristicas
import Codigos.ArffManager as arffmanager
import Codigos.Weka as weka
import Codigos.Experimentos as experimentos

# ======================================================================================================================
# En este archivo se prueban primeramente todas las clases y funciones por separado, para luego ir concatenándolas
# ======================================================================================================================

# ELECCIÓN DEL SUJETO
persona = '01'
etapa = '1'
parte = '1'

""" DATOS """
nombre_video = herramientas.buildVideoName(persona, etapa, parte, extension=False)
path_video = herramientas.buildPathVideo(persona, etapa, nombre_video, extension=True)

# METODOS
"""OpenFace"""
open_face = metodos.OpenFace(cara=True, hog=True, landmarks=True, aus=True)
open_face(path_video)

# CARACTERISTICAS
# Video
# zonas = np.array(['cejaizq', 'cejader', 'ojoizq', 'ojoder', 'boca'])
# metodos_video = np.array(['LBP', 'HOP', 'HOG', 'AU'])
# binarizar_etiquetas = False
# completo = False
# rangos_audibles = None
#
# jvm.start(packages=True)
# feat_video = caracteristicas.Video(zonas, metodos_video, binarizar_etiquetas)  # init
# feat_video(persona, etapa, completo, rangos_audibles)                          # call
