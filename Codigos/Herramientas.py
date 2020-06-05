import csv
import numpy as np
import cv2 as cv
import read_hog_file
# import Codigos.Datos as datos


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
    new_predi = np.append(new_predi, np.array([np.append(np.array(['Etiqueta', 'Error medio %']), predi[0, :, 1])]).T, axis=1)
    for i in range(0, num_metodos):
        new_predi = np.append(new_predi, np.array([np.append(np.array([metodos[i], errores[i]*100]), predi[i, :, 2])]).T, axis=1)
    return new_predi


def uneResumenes(resu1, resu2):
    """
    A partir de los dos resumenes de predicciones los une en uno solo cortando al tamaño del menor.
    Se espera que el primer resumen sea el del video por el hecho de personalizar el nombre de las columnas
    """
    filas1 = resu1.shape[0]
    filas2 = resu2.shape[0]

    if filas1 < filas2:
        corte = filas1
    else:
        corte = filas2

    new_resu = np.concatenate([resu1[0:corte, :], resu2[0:corte, 1:]], axis=1)

    # Al nombre de los metodos les agrego la modalidad a la que refiere, se espera que empiece con los de video
    for i in range(1, new_resu.shape[1]):
        if i < resu1.shape[1]:
            new_resu[0, i] = new_resu[0, i] + '(V)'
        else:
            new_resu[0, i] = new_resu[0, i] + '(A)'
    return new_resu


def Fusion(resumen, metodo, mejores=-1, por_modalidad=False):
    """
    Recibiendo el resumen de todas las predicciones, errores y etiquetas. Utiliza el metodo mencionado para fusionar y
    los mejores x clasificadores para esto. En caso de que mejores sea -1 utiliza todos
    """
    indice_mejores = np.empty(0, dtype=np.int)
    valores_mejores = np.empty(0)

    # Creo el resumen final
    new_resu = np.array([np.array(['Etiqueta', metodo])])
    # Agrego la fila con el error y el valor 0, despues este se tiene que reemplazar al calcular el error al final
    new_resu = np.append(new_resu, np.array([np.array(['Error medio %', '0'])]), axis=0)

    # Si los mejores son por modalidad, guardo los indices de donde se encuentran cada uno
    modalidad_audio = list()
    modalidad_video = list()
    if por_modalidad:
        for i in range(1, resumen.shape[1]):
            # Si encuentro el (V) es video y sino supongo que es audio
            if resumen[0, i].find('(V)') != -1:
                modalidad_video.append(i)
            else:
                modalidad_audio.append(i)

    if mejores > 0:
        # Agrego al mejor de que tantos era
        new_resu[0, 1] = new_resu[0, 1] + ' M' + str(mejores)

        if por_modalidad:
            indice_mejores_video = np.empty(0, dtype=np.int)
            valores_mejores_video = np.empty(0)
            indice_mejores_audio = np.empty(0, dtype=np.int)
            valores_mejores_audio = np.empty(0)
            # Los primeros los uso como inicializacion
            for i in range(0, mejores):
                indice_mejores_video = np.append(indice_mejores_video, modalidad_video[i])
                valores_mejores_video = np.append(valores_mejores_video, float(resumen[1, modalidad_video[i]]))
                indice_mejores_audio = np.append(indice_mejores_audio, modalidad_audio[i])
                valores_mejores_audio = np.append(valores_mejores_audio, float(resumen[1, modalidad_audio[i]]))
            # Recien ahora recorro el resto
            for i in range(mejores, int((resumen.shape[1] - 1) / 2)):
                # Si tiene menor error que reemplace el menor tanto en indice como en valores
                if float(resumen[1, modalidad_video[i]]) < max(valores_mejores_video):
                    indice_mejores_video[valores_mejores_video.argmax()] = modalidad_video[i]
                    valores_mejores_video[valores_mejores_video.argmax()] = float(resumen[1, modalidad_video[i]])
                if float(resumen[1, modalidad_audio[i]]) < max(valores_mejores_audio):
                    indice_mejores_audio[valores_mejores_audio.argmax()] = modalidad_audio[i]
                    valores_mejores_audio[valores_mejores_audio.argmax()] = float(resumen[1, modalidad_audio[i]])
            indice_mejores = np.concatenate([indice_mejores_audio, indice_mejores_video])
        else:
            # Los primeros los uso como inicializacion
            for i in range(1, mejores + 1):
                valores_mejores = np.append(valores_mejores, float(resumen[1, i]))
                indice_mejores = np.append(indice_mejores, i)
            # Recien ahora recorro el resto
            for i in range(mejores + 1, resumen.shape[1]):
                # Si tiene menor error que reemplace el menor tanto en indice como en valores
                if float(resumen[1, i]) < max(valores_mejores):
                    indice_mejores[valores_mejores.argmax()] = i
                    valores_mejores[valores_mejores.argmax()] = float(resumen[1, i])
    else:
        indice_mejores = np.array(range(1, resumen.shape[1]))

    cont_errores = 0
    if metodo == 'Voto':
        for j in range(2, resumen.shape[0]):
            votos = list()
            # Recorro solo las columnas con los mejores clasificadores
            for i in indice_mejores:
                # Busco la posicion de la clase que corresponde la etiqueta que predice
                votos.append(resumen[j, i])
            # Agrego a la fila del resumen final la etiqueta y la prediccion final despues del voto
            mas_votado = max(set(votos), key=votos.count)
            new_resu = np.append(new_resu, np.array([np.array([resumen[j, 0], mas_votado])]), axis=0)
            # Si la prediccion no es igual a la etiqueta que sume uno al contador de errores
            if mas_votado != resumen[j, 0]:
                cont_errores = cont_errores + 1
        # Luego de terminar de realizar la fusion calculo el porcentaje de error medio
        error = (cont_errores / (resumen.shape[0] - 2)) * 100
        new_resu[1, 1] = str(error)

    return new_resu


def VotoPorSegmento(resumen, instancias_intervalos, desfase=0):
    """
    Aplica voto a los intervalos de tiempo contiguos, de manera que no se produzan cambios bruscos en las etiquetas por
    cada intervalo.
    """
    new_resu = np.copy(resumen)
    new_resu[0, 1] = new_resu[0, 1] + '-' + str(instancias_intervalos)
    cont_errores = 0
    rango = list([2, 2 + desfase])
    rango.extend(range(desfase + 2 + instancias_intervalos, resumen.shape[0], instancias_intervalos))
    for i in rango:
        votos = list()
        if desfase == 0:
            if i + instancias_intervalos < resumen.shape[0]:
                hasta = i + instancias_intervalos
            else:
                hasta = resumen.shape[0] - 1
        else:
            hasta = i + desfase
            desfase = 0
        for j in range(i, hasta):
            votos.append(resumen[j, 1])
        if votos:
            mas_votado = max(set(votos), key=votos.count)
        for j in range(i, hasta):
            new_resu[j, 1] = mas_votado
            if mas_votado != resumen[j, 0]:
                cont_errores = cont_errores + 1
    error = (cont_errores / (resumen.shape[0] - 2)) * 100
    new_resu[1, 1] = str(error)

    return new_resu


def OrdenaInstancias(resumen, orden_instancias):
    aux = np.empty(0, dtype=np.int)
    for i in range(0, orden_instancias.size):
        ind = np.where(orden_instancias == i)[0]
        if ind >= orden_instancias.size - (resumen.shape[0] - 2):
            aux = np.append(aux, ind)
    # Sobre lista de aux buscar el menor y ponerlo en primer indice de una nueva lista y asi
    maximo = max(aux)
    ordenado = np.empty(0, dtype=np.int)
    desfase = 2
    comienzo = True
    for i in range(0, aux.size):
        ordenado = np.append(ordenado, aux.argmin())
        aux[aux.argmin()] = maximo + 1
        if comienzo and i > 1:
            if ordenado[i] - 1 == ordenado[i - 1]:
                desfase = desfase + 1
            else:
                comienzo = False
    aux_resumen = resumen[2:]
    resumen[2:] = aux_resumen[ordenado]
    return resumen, desfase
