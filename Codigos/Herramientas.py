import csv
import numpy as np
import cv2 as cv
import read_hog_file


# Calcula el histograma de una imagen o una matriz en escala de grises (valores de 0 a 255 por celda)
def Histograma(imagen):
    img = np.copy(imagen)
    f = img.shape[0]
    c = img.shape[1]
    histo = np.zeros(256)

    for i in range(0, f):
        for j in range(0, c):
            histo[img[i, j]] = histo[img[i, j]] + 1
    return histo


def ROI(img, landmarks_x, landmarks_y, region, expandir, resize):
    # Devuelve el minimo rectangulo segun la region de la cara que se elija

    # Landmarks deberia traer toda la lista de puntos faciales de un frame
    # Por ejemplo desde open face archivo[nro_frame][....]

    # LISTA DE NUMEROS DE PUNTOS FACIALES SEGUN LA REGION
    # Borde de la cara 0 al 16
    # Cejas 17 al 26 (izquierda 17 a 21 y derecha 22 a 26)
    # Nariz 27 al 35
    # Ojos 36 al 47 (izquierdo 36 a 41 y derecha 42 a 47)
    # Boca 48 al 59

    switcher = {
        'cara': list(range(0, 27)),
        'cejas': list(range(17, 27)),
        'cejaizq': list(range(17, 22)),
        'cejader': list(range(22, 27)),
        'nariz': list(range(27, 36)),
        'ojos': list(range(36, 48)),
        'ojoizq': list(range(36, 42)),
        'ojoder': list(range(42, 48)),
        'boca': list(range(48, 60))
    }

    rango = switcher.get(region)
    frame = np.copy(img)
    landmarks_propios = np.zeros(0)

    for i in rango:
        punto = np.array([[int(float(landmarks_x[i])), int(float(landmarks_y[i]))]])
        # Este if esta por problemas al ir concatenando cuando esta vacio
        if len(landmarks_propios) == 0:
            landmarks_propios = punto
        else:
            landmarks_propios = np.concatenate((landmarks_propios, punto), axis=0)

    x1, y1, w1, h1 = cv.boundingRect(landmarks_propios)
    x2 = x1 + w1
    y2 = y1 + h1
    if expandir:
        # Si tomamos un 5% de expansion para cada lado
        pix_y = int(w1 / 20)
        pix_x = int(h1 / 20)
        # Verificacion que no sobrepase los limites de la imagen
        if y1 - pix_y < 0:
            y1 = 0
        else:
            y1 = y1 - pix_y

        if y2 + pix_y > frame.shape[0]:
            y2 = frame.shape[0]
        else:
            y2 = y2 + pix_y

        if x1 - pix_x < 0:
            x1 = 0
        else:
            x1 = x1 - pix_x

        if x2 + pix_x > frame.shape[1]:
            x2 = frame.shape[1]
        else:
            x2 = x2 + pix_x

    roi = frame[y1:y2, x1:x2]

    if resize:
        roi = ResizeZona(roi, region)

    return roi


def ResizeZona(imagen, region):
    # Segun la region lo lleva a un tamaño fijo, estos numeros se sacaron manualmente a partir de la observacion de un frame
    switcher = {
        'cara': (200, 200),
        'cejas': (180, 30),
        'cejaizq': (80, 30),
        'cejader': (80, 30),
        'nariz': (60, 80),
        'ojos': (140, 30),
        'ojoizq': (50, 30),
        'ojoder': (50, 30),
        'boca': (80, 40)
    }
    tam = switcher.get(region)
    img = cv.resize(imagen, tam, interpolation=cv.INTER_AREA)
    return img


def leeHOG(ruta_archivo):
    # ruta_archivo = 'Procesado/Sujeto 01a.hog'
    # Devuelve la matriz con los hog por cuadro
    # El segundo parametro devuelve si en ese cuadro se extrajo correctamente
    rhf = read_hog_file.initialize()
    [hog, inds] = rhf.Read_HOG_file(ruta_archivo, nargout=2)
    rhf.terminate()
    return hog, inds


def leeCSV(ruta_archivo):
    # Devuelve del csv una lista con los datos
    archivo = open(ruta_archivo)
    leido = csv.reader(archivo, delimiter=',', skipinitialspace=True)
    leido = list(leido)
    return leido


def leeEtiqueta(archivo, persona, etapa, parte):
    # Cada persona tiene 13 videos, 7 partes en la etapa 1 y 6 partes en la etapa 2
    # El primer 1+ en persona va para saltear la fila donde estan las caratulas
    ind_persona = 1 + (int(persona) - 1) * 13
    ind_etapa = (int(etapa) - 1) * 7
    ind_parte = int(parte) - 1
    etiqueta = archivo[ind_persona + ind_etapa + ind_parte][5]
    return etiqueta


def leeTiemposRespuesta(archivo, persona, etapa, parte):
    # Cada persona tiene 13 videos, 7 partes en la etapa 1 y 6 partes en la etapa 2
    # El primer 1+ en persona va para saltear la fila donde estan las caratulas
    ind_persona = 1 + (int(persona) - 1) * 13
    ind_etapa = (int(etapa) - 1) * 7
    ind_parte = int(parte) - 1
    segundos = int(archivo[ind_persona + ind_etapa + ind_parte][3]) * 60 + int(
        archivo[ind_persona + ind_etapa + ind_parte][4])
    return segundos