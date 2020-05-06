import csv
import numpy as np
import cv2 as cv
import read_hog_file


def Histograma(imagen):
    """
    Calcula el histograma de una imagen o una matriz en escala de grises (valores de 0 a 255 por celda).
    """
    img = np.copy(imagen)
    f = img.shape[0]
    c = img.shape[1]
    histo = np.zeros(256)

    for i in range(0, f):
        for j in range(0, c):
            histo[img[i, j]] = histo[img[i, j]] + 1
    return histo


def ROI(img, landmarks_x, landmarks_y, region, expandir=True, resize=True):
    """
    Devuelve el mínimo rectángulo según la región de la cara que se elija. Landmarks debería traer toda la lista de
    puntos faciales de un frame. Por ejemplo desde open face: archivo[nro_frame][....]

    LISTA DE NUMEROS DE PUNTOS FACIALES SEGUN LA REGION
        Borde de la cara 0 al 16
        Cejas 17 al 26 (izquierda 17 a 21 y derecha 22 a 26)
        Nariz 27 al 35
        Ojos 36 al 47 (izquierdo 36 a 41 y derecha 42 a 47)
        Boca 48 al 59
    """

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
    landmarks_propios = np.empty((0, 2), dtype=int)

    for i in rango:
        punto = np.array([[int(float(landmarks_x[i])), int(float(landmarks_y[i]))]])
        # Este if esta por problemas al ir concatenando cuando está vacío
        landmarks_propios = np.append(landmarks_propios, punto, axis=0)

    x1, y1, w1, h1 = cv.boundingRect(landmarks_propios)
    x2 = x1 + w1
    y2 = y1 + h1
    if expandir:
        # Si tomamos un 5% de expansión para cada lado
        pix_y = int(w1 / 20)
        pix_x = int(h1 / 20)
        # Verificación que no sobrepase los límites de la imagen
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

    # Si las coordenadas dan ambas nulas es invalido, suplantando la roi con
    # (en algoritmo de open face al tener landmarks invalidos da puntos fuera del tamaño de la imagen, pero al hacer
    # la comprobacion anterior la llevamos siempre al limite de la imagen)
    if (x1 - x2 == 0) or (y1 - y2 == 0):
        roi = np.zeros(frame.shape, frame.dtype)
    else:
        roi = frame[y1:y2, x1:x2]

    if resize:
        roi = ResizeZona(roi, region)

    return roi


def ResizeZona(imagen, region):
    """
    Según la región lo lleva a un tamaño fijo, estos números se sacaron manualmente a partir de la observación de un
    frame.
    """
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
    """
    Devuelve dos valores. El primero corresponde a la matriz con los hog por cuadro y el segundo devuelve si en ese
    cuadro se extrajo correctamente.
    Ejemplo: ruta_archivo = 'Procesado/Sujeto 01a.hog'
    """
    rhf = read_hog_file.initialize()
    [hog, inds] = rhf.Read_HOG_file(ruta_archivo, nargout=2)
    rhf.terminate()
    return hog, inds


def leeCSV(ruta_archivo):
    """
    Devuelve una lista con los datos a partir de un csv.
    """
    archivo = open(ruta_archivo)
    leido = csv.reader(archivo, delimiter=',', skipinitialspace=True)
    leido = list(leido)
    return leido


def leeEtiqueta(archivo, persona, etapa, parte):
    """
    Cada persona tiene 13 videos, 7 partes en la etapa 1 y 6 partes en la etapa 2. El primer 1+ en persona va para
    saltear la fila donde están las carátulas.
    """
    ind_persona = 1 + (int(persona) - 1) * 13
    ind_etapa = (int(etapa) - 1) * 7
    ind_parte = int(parte) - 1
    etiqueta = archivo[ind_persona + ind_etapa + ind_parte][5]
    return etiqueta


def leeTiemposRespuesta(archivo, persona, etapa, parte):
    """
    Cada persona tiene 13 videos, 7 partes en la etapa 1 y 6 partes en la etapa 2. El primer 1+ en persona va para
    saltear la fila donde están las carátulas.
    """
    ind_persona = 1 + (int(persona) - 1) * 13
    ind_etapa = (int(etapa) - 1) * 7
    ind_parte = int(parte) - 1
    segundos = int(archivo[ind_persona + ind_etapa + ind_parte][3]) * 60 + int(archivo[ind_persona + ind_etapa + ind_parte][4])
    return segundos


def prediccionCSVtoArray(predi):
    """
    Sirve para convertir los csv de las predicciones en vectores de numpy formato: [ [...,...,...], [....,...,...]...]
    """
    vec = np.array([])
    fila = np.array([])
    dato = ''
    # Recorro cada char
    for i in predi:
        # Si es una coma o salto de línea agrego el dato a la fila y lo reinicio
        if i == ',' or i == '\n':
            # En caso de dos comas seguidas o datos incompletos que el dato sea un espacio
            if dato == '':
                dato = ' '
            fila = np.append(fila, dato)
            dato = ''
            # En caso de ser salto de línea agrego la fila entera al vector y la reinicio
            if i == '\n':
                # Como no conozco la cantidad de columnas para inicializar un vector vacío, tengo que hacer esto
                if len(vec) == 0:
                    vec = np.array([fila])
                    fila = np.array([])
                else:
                    vec = np.concatenate([vec, np.array([fila])], axis=0)
                    fila = np.array([])
        else:
            # Concateno el dato a partir de los char
            dato = dato + i
    return vec

def uneResumenes(resu1, resu2):
    """
    A partir de los dos resumenes de predicciones los une en uno solo cortando al tamaño del menor
    """
    filas1 = resu1.shape[0]
    filas2 = resu2.shape[0]

    if filas1 < filas2:
        corte = filas1
    else:
        corte = filas2

    new_resu = np.concatenate([resu1[0:corte, :], resu2[0:corte, 1:]], axis=1)
    return new_resu

def resumePredicciones(predi, metodos, errores):
    """
    El primer parámetro representa el vector de matrices con las predicciones, el segundo un vector con el nombre de los
    métodos usados. Por ejemplo, para la predicción en la posicion 0 se utilizó 'PCA + SVM'. Con esto creo la cabecera.
    """

    # Número de métodos
    num_metodos = predi.shape[0]

    # Cantidad de segmentos de cada modalidad
    tam_pre = predi.shape[1]

    new_predi = np.empty((tam_pre + 2, 0))
    # Del primer método además de obtener la predicción saco la columna con las etiquetas (iguales en todos los métodos)
    new_predi = np.append(new_predi, np.array([np.append(np.array(['Etiqueta', 'Error %']), predi[0, :, 1])]).T, axis=1)
    for i in range(0, num_metodos):
        new_predi = np.append(new_predi, np.array([np.append(np.array([metodos[i], errores[i]*100]), predi[i, :, 2])]).T, axis=1)
    return new_predi
