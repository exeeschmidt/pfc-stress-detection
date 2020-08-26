import csv
import os

import cv2 as cv
import numpy as np
from sklearn.metrics import recall_score, accuracy_score

import Datos
import LogManager as Log


def buildFileName(person, stages, part=-1):
    file_name = 'Sujeto_' + person + '_' + stages
    if part != -1:
        file_name += '_r' + str(part)
    return file_name


def buildFilePath(person, stage, file_name, extension=''):
    path_video = os.path.join(Datos.PATH_BD, 'Sujeto ' + person, 'Etapa ' + stage, file_name)
    path_video += extension
    return path_video


def buildSubFilePath(file_name, sub_name):
    path = os.path.join(Datos.PATH_CARACTERISTICAS, sub_name, file_name + '_' + sub_name + '.arff')
    return path


def buildFilePartName(file_name, parte, extension=''):
    if file_name.find('.mp4') == -1:
        file_name += '_r' + str(parte)
    else:
        file_name = file_name[0:file_name.find('.mp4')] + '_r' + str(parte)

    file_name += extension
    return file_name


def buildOpenSmileFilePath(file_name):
    return os.path.join(Datos.PATH_CARACTERISTICAS, file_name + '.arff')


def buildOutputPathFFMPEG(file_name):
    return os.path.join(Datos.PATH_PROCESADO, file_name + Datos.EXTENSION_AUDIO)


def extractStageFromFileName(file_name):
    if file_name.find(Datos.EXTENSION_VIDEO) == -1 and file_name.find(Datos.EXTENSION_AUDIO) == -1:
        stage = file_name[len(file_name) - 1]
    elif file_name.find(Datos.EXTENSION_VIDEO) == -1:
        stage = file_name[file_name.find(Datos.EXTENSION_AUDIO) - 1]
    else:
        stage = file_name[file_name.find(Datos.EXTENSION_VIDEO) - 1]
    return stage


def extractPersonFromVideoName(file_name):
    if file_name.find(Datos.EXTENSION_VIDEO) == -1 and file_name.find(Datos.EXTENSION_AUDIO) == -1:
        person = file_name[len(file_name) - 4] + file_name[len(file_name) - 3]
    elif file_name.find(Datos.EXTENSION_VIDEO) == -1:
        person = file_name[file_name.find(Datos.EXTENSION_AUDIO) - 4] + \
                 file_name[file_name.find(Datos.EXTENSION_AUDIO) - 3]
    else:
        person = file_name[file_name.find(Datos.EXTENSION_VIDEO) - 4] + \
                 file_name[file_name.find(Datos.EXTENSION_VIDEO) - 3]
    return person


def generateHistogram(image):
    """
    Calcula el histograma de una imagen o una matriz en escala de grises (valores de 0 a 255 por celda).
    """
    img = np.copy(image)
    f = img.shape[0]
    c = img.shape[1]
    hist = np.zeros(256)

    for i in range(0, f):
        for j in range(0, c):
            hist[img[i, j]] = hist[img[i, j]] + 1
    return hist


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

    region_range = switcher.get(region)
    frame = np.copy(img)
    landmarks_points = np.empty((0, 2), dtype=int)

    for i in region_range:
        punto = np.array([[int(float(landmarks_x[i])), int(float(landmarks_y[i]))]])
        # Este if esta por problemas al ir concatenando cuando está vacío
        landmarks_points = np.append(landmarks_points, punto, axis=0)

    x1, y1, w1, h1 = cv.boundingRect(landmarks_points)
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

    # Si ambas coordenadas dan nulas es invalido, suplantando la roi con ceros
    # (en algoritmo de open face al tener landmarks invalidos da puntos fuera del tamaño de la imagen, pero al hacer
    # la comprobacion anterior la llevamos siempre al limite de la imagen)
    if (x1 - x2 == 0) or (y1 - y2 == 0):
        roi = np.zeros(frame.shape, frame.dtype)
    else:
        roi = frame[y1:y2, x1:x2]

    if resize:
        roi = resizeByZone(roi, region)

    return roi


def resizeByZone(image, region):
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
    img = cv.resize(image, tam, interpolation=cv.INTER_AREA)
    return img


def readCSVFile(path, delimiter=','):
    """
    Devuelve una lista con los datos de un archivo csv.
    """
    file = open(path)
    readed = csv.reader(file, delimiter=delimiter, skipinitialspace=True)
    return list(readed)


def writeCSVFile(path, data):
    """
    Guarda un csv a partir de un vector con valores separados por coma
    """
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    return


def readLabel(file, person, stage, part):
    """
    Cada persona tiene 13 videos, 7 partes en la etapa 1 y 6 partes en la etapa 2. El primer 1+ en persona va para
    saltear la fila donde están las carátulas.
    """
    ind_persona = 1 + (int(person) - 1) * 13
    ind_etapa = (int(stage) - 1) * 7
    ind_parte = int(part) - 1
    etiqueta = file[ind_persona + ind_etapa + ind_parte][5]
    return etiqueta


def readAnswersTime(file, person, stage, part):
    """
    Cada persona tiene 13 videos, 7 partes en la etapa 1 y 6 partes en la etapa 2. El primer 1+ en persona va para
    saltear la fila donde están las carátulas.
    """
    person_index = 1 + (int(person) - 1) * 13
    stage_index = (int(stage) - 1) * 7
    part_index = int(part) - 1
    seconds = int(file[person_index + stage_index + part_index][3]) * 60 + int(file[person_index + stage_index +
                                                                                    part_index][4])
    return seconds


def mapLabels(person, stage, label_binarization, complete_mode=False):
    """
    Se encarga de generar un vector con la etiqueta que deberia ir por cada instancia, ademas se incorpo la devolucion
    de los limites de respuesta, mapeando los tiempos de este con las instancias
    """
    # Defino los nombres de la clase según si se binariza o no
    if label_binarization:
        labels = np.array(['N', 'E'])
    else:
        labels = np.array(['N', 'B', 'M', 'A'])

    # Cargo el archivo con las etiquetas
    data_labels = readCSVFile(Datos.PATH_ETIQUETAS)

    if int(stage) == 1:
        parts = 7
    else:
        parts = 6

    labels_list = list()
    answers_limits = list()
    if complete_mode:
        video_name = buildFileName(person, stage)
        video_path = buildFilePath(person, stage, video_name, extension=Datos.EXTENSION_VIDEO)
        # Cargo los tiempos donde termina cada respuesta, para saber en que intervalos va cada etiqueta,
        # esto está en segundos
        seconds_by_answer = np.zeros(parts)
        for i in range(0, parts):
            seconds_by_answer[i] = readAnswersTime(data_labels, person, stage, str(i + 1))
        # Permite saber en que respuesta voy para saber cuando cambiar la etiqueta
        interval_number = 1

        # Leo la etiqueta correspondiente a la primera parte para empezar en caso de ser completo, o la de la
        # respuesta segpun el caso
        actual_label = readLabel(data_labels, person, stage, str(1))
        if label_binarization:
            if actual_label != 'N':
                actual_label = labels[1]

        # Leo el video solo para saber el total de frames, este es igual a la cantidad de instancias para el etiquetado
        video = cv.VideoCapture(video_path)
        instances_number = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv.CAP_PROP_FPS))

        for i in range(0, instances_number):
            # Para definir intervalo de etiqueta
            # Si paso el tiempo donde termina la respuesta, leo la siguiente etiqueta
            # Me fijo también si el nro de intervalo no es el último, en ese caso debe etiquetarse hasta el
            # final. Por esa razón no debe cambiar más de etiqueta. Esta verificación está por si hay error
            # numérico al calcular los fps y se detecte un cambio de etiqueta unos cuadros antes de la
            # última etiqueta, lo que provocaría que quiera leer la etiqueta de un número de intervalo que
            # no existe
            if (i >= seconds_by_answer[interval_number - 1] * fps) and (interval_number != -1):
                answers_limits.append(answers_limits[len(answers_limits) - 1] + i)
                interval_number = interval_number + 1
                actual_label = readLabel(data_labels, person, stage, interval_number)
                # Paso a usar nro_intervalo como bandera por si es la última etiqueta de la última parte
                if interval_number == parts:
                    interval_number = -1
                if label_binarization:
                    if actual_label != 'N':
                        actual_label = labels[1]
            labels_list.append(actual_label)
    else:
        for j in range(0, parts):
            # Diferencias en los nombres de archivo y llamada a open face
            video_name = buildFileName(person, stage, part=(j + 1))
            video_path = buildFilePath(person, stage, video_name, extension=Datos.EXTENSION_VIDEO)

            # Leo el video solo para saber el total de frames, este es igual a la cantidad de instancias para el
            # etiquetado
            video = cv.VideoCapture(video_path)
            instances_number = int(video.get(cv.CAP_PROP_FRAME_COUNT))
            actual_label = readLabel(data_labels, person, stage, str(j + 1))
            if len(answers_limits) == 0:
                answers_limits.append(instances_number)
            else:
                answers_limits.append(answers_limits[len(answers_limits) - 1] + instances_number)
            if label_binarization:
                if actual_label != 'N':
                    actual_label = labels[1]
            for i in range(0, instances_number):
                labels_list.append(actual_label)
    return labels_list, answers_limits


def predictionCSVtoArray(prediction, from_csv_file=False):
    """
    Sirve para convertir los csv de las predicciones en vectores de numpy formato: [ [...,...,...], [....,...,...]...]
    Si el csv viene de uno que genero escribeCSV, este lo deja como listas, asi que hay que ingresar a cada char
    como si fuera una lista
    """
    prediction_vector = np.array([])
    file = np.array([])
    data = ''
    # Recorro cada char
    for i in prediction:
        if from_csv_file:
            i = i.pop()
        # Si es una coma o salto de línea agrego el dato a la fila y lo reinicio
        if i == ',' or i == '\n':
            # En caso de dos comas seguidas o datos incompletos que el dato sea un espacio
            if data == '':
                data = ' '
            file = np.append(file, data)
            data = ''
            # En caso de ser salto de línea agrego la fila entera al vector y la reinicio
            if i == '\n':
                # Como no conozco la cantidad de columnas para inicializar un vector vacío, tengo que hacer esto
                if len(prediction_vector) == 0:
                    prediction_vector = np.array([file])
                    file = np.array([])
                else:
                    prediction_vector = np.concatenate([prediction_vector, np.array([file])], axis=0)
                    file = np.array([])
        else:
            # Concateno el dato a partir de los char
            data = data + i
    return prediction_vector


def summarizePredictions(predictions, methods, accuracy, uar):
    """
    El primer parámetro representa el vector de matrices con las predicciones, el segundo un vector con el nombre de los
    métodos usados. Por ejemplo, para la predicción en la posicion 0 se utilizó 'PCA + SVM'. Con esto creo la cabecera.
    """

    # Número de métodos
    methods_number = predictions.shape[0]

    # Cantidad de segmentos de cada modalidad
    instances = predictions.shape[1]

    prediction_summary = np.empty((instances + 3, 0))
    # Del primer método además de obtener la predicción saco la columna con las etiquetas (iguales en todos los métodos)
    prediction_summary = np.append(prediction_summary, np.array([np.append(np.array(['Etiqueta', 'Accuracy', 'UAR']),
                                                                           predictions[0, :, 1])]).T, axis=1)
    for i in range(0, methods_number):
        prediction_summary = np.append(prediction_summary, np.array([np.append(np.array(
            [methods[i], accuracy[i], uar[i]]), predictions[i, :, 2])]).T, axis=1)
    return prediction_summary


def joinSummaries(first_summary, second_summary):
    """
    A partir de los dos resumenes de predicciones los une en uno solo cortando al tamaño del menor.
    Se espera que el primer resumen sea el del video por el hecho de personalizar el nombre de las columnas
    """
    row_first = first_summary.shape[0]
    row_second = second_summary.shape[0]

    if row_first < row_second:
        row_cut = row_first
    else:
        row_cut = row_second

    new_summary = np.concatenate([first_summary[0:row_cut, :], second_summary[0:row_cut, 1:]], axis=1)

    # Al nombre de los metodos les agrego la modalidad a la que refiere, se espera que empiece con los de video
    for i in range(1, new_summary.shape[1]):
        if i < first_summary.shape[1]:
            new_summary[0, i] = new_summary[0, i] + '(V)'
        else:
            new_summary[0, i] = new_summary[0, i] + '(A)'
    return new_summary


def indexsBestClassifiers(summary, best_of=-1, by_modality=False):
    """
     A partir del resumen de validacion devuelve los indices de las mejores combinaciones de metodos
    """
    best_index = np.empty(0, dtype=np.int)
    best_values = np.empty(0)

    # Si los mejores son por modalidad, guardo los indices de donde se encuentran cada uno
    audio_modality = list()
    video_modality = list()
    if by_modality:
        for i in range(1, summary.shape[1]):
            # Si encuentro el (V) es video y sino supongo que es audio
            if summary[0, i].find('(V)') != -1:
                video_modality.append(i)
            else:
                audio_modality.append(i)

    if best_of > 0:
        if by_modality:
            indice_mejores_video = np.empty(0, dtype=np.int)
            valores_mejores_video = np.empty(0)
            indice_mejores_audio = np.empty(0, dtype=np.int)
            valores_mejores_audio = np.empty(0)
            # Los primeros los uso como inicializacion
            for i in range(0, best_of):
                indice_mejores_video = np.append(indice_mejores_video, video_modality[i])
                valores_mejores_video = np.append(valores_mejores_video, float(summary[2, video_modality[i]]))
                indice_mejores_audio = np.append(indice_mejores_audio, audio_modality[i])
                valores_mejores_audio = np.append(valores_mejores_audio, float(summary[2, audio_modality[i]]))
            # Recien ahora recorro el resto
            for i in range(best_of, int((summary.shape[1] - 1) / 2)):
                # Si tiene mas uar que reemplace el menor tanto en indice como en valores
                if float(summary[2, video_modality[i]]) > min(valores_mejores_video):
                    indice_mejores_video[valores_mejores_video.argmin()] = video_modality[i]
                    valores_mejores_video[valores_mejores_video.argmin()] = float(summary[2, video_modality[i]])
                if float(summary[2, audio_modality[i]]) > min(valores_mejores_audio):
                    indice_mejores_audio[valores_mejores_audio.argmin()] = audio_modality[i]
                    valores_mejores_audio[valores_mejores_audio.argmin()] = float(summary[2, audio_modality[i]])
            best_index = np.concatenate([indice_mejores_audio, indice_mejores_video])
        else:
            # Los primeros los uso como inicializacion
            for i in range(1, best_of + 1):
                best_values = np.append(best_values, float(summary[2, i]))
                best_index = np.append(best_index, i)
            # Recien ahora recorro el resto
            for i in range(best_of + 1, summary.shape[1]):
                # Si tiene mayor uar que reemplace el menor tanto en indice como en valores
                if float(summary[2, i]) > min(best_values):
                    best_index[best_values.argmin()] = i
                    best_values[best_values.argmin()] = float(summary[2, i])
    else:
        best_index = np.array(range(1, summary.shape[1]))

    return best_index


def fusionClassifiers(summary, method, best_index):
    """
    Recibiendo el resumen de todas las predicciones, metricas y etiquetas fusiona las mejores con el metodo indicado
    """
    # Creo el resumen final
    new_summary = np.array([np.array(['Etiqueta', method])])
    # Agrego la fila con el error y el valor 0, despues este se tiene que reemplazar al calcular el error al final
    new_summary = np.append(new_summary, np.array([np.array(['Accuracy', '0'])]), axis=0)
    new_summary = np.append(new_summary, np.array([np.array(['UAR', '0'])]), axis=0)

    new_summary[0, 1] = new_summary[0, 1] + ' M' + str(best_index.size)

    if method == 'Voto':
        for j in range(3, summary.shape[0]):
            votes = list()
            # Recorro solo las columnas con los mejores clasificadores
            for i in best_index:
                # Busco la posicion de la clase que corresponde la etiqueta que predice
                votes.append(summary[j, i])
            # Agrego a la fila del resumen final la etiqueta y la prediccion final despues del voto
            most_voted = max(set(votes), key=votes.count)
            new_summary = np.append(new_summary, np.array([np.array([summary[j, 0], most_voted])]), axis=0)
        # Luego de terminar de realizar la fusion calculo la metricas
        new_summary[1, 1] = str(Accuracy(new_summary[3:, 0], new_summary[3:, 1]))
        new_summary[2, 1] = str(UAR(new_summary[3:, 0], new_summary[3:, 1]))

    return new_summary


def voteForPeriod(summary, instances_for_interval, gap=0):
    """
    Aplica voto a los intervalos de tiempo contiguos, de manera que no se produzan cambios bruscos en las etiquetas por
    cada intervalo.
    """
    new_summary = np.copy(summary)
    new_summary[0, 1] = new_summary[0, 1] + '-' + str(instances_for_interval)
    periods_range = list([3, 2 + gap])
    periods_range.extend(range(gap + 2 + instances_for_interval, summary.shape[0], instances_for_interval))
    for i in periods_range:
        votes = list()
        if gap == 0:
            if i + instances_for_interval < summary.shape[0]:
                to = i + instances_for_interval
            else:
                to = summary.shape[0] - 1
        else:
            to = i + gap
            gap = 0
        for j in range(i, to):
            votes.append(summary[j, 1])
        if votes:
            most_voted = max(set(votes), key=votes.count)
            for j in range(i, to):
                new_summary[j, 1] = most_voted
    new_summary[1, 1] = str(Accuracy(new_summary[3:, 0], new_summary[3:, 1]))
    new_summary[2, 1] = str(UAR(new_summary[3:, 0], new_summary[3:, 1]))

    return new_summary


def sortInstances(summary, instances_order):
    aux = np.empty(0, dtype=np.int)
    for i in range(0, instances_order.size):
        ind = np.where(instances_order == i)[0]
        if ind >= instances_order.size - (summary.shape[0] - 3):
            aux = np.append(aux, ind)
    # Sobre lista de aux buscar el menor y ponerlo en primer indice de una nueva lista y asi
    maximum = max(aux)
    sorted_data = np.empty(0, dtype=np.int)
    gap = 2
    begin = True
    for i in range(0, aux.size):
        sorted_data = np.append(sorted_data, aux.argmin())
        aux[aux.argmin()] = maximum + 1
        if begin and i > 1:
            if sorted_data[i] - 1 == sorted_data[i - 1]:
                gap = gap + 1
            else:
                begin = False
    aux_summary = summary[3:]
    summary[3:] = aux_summary[sorted_data]
    return summary, gap


def Accuracy(ground_truth, prediction):
    return accuracy_score(ground_truth, prediction)


def UAR(ground_truth, prediction):
    # Unweighted average recall
    return recall_score(ground_truth, prediction, average='macro')


def createFinalSummary(vec_res, vec_res_fus, vec_res_fus_2):
    folds = vec_res.shape[0]
    methods_number = vec_res.shape[2] - 1
    final_summary = np.empty((3, methods_number + 3), dtype='U20')
    # Creo la cabecera
    final_summary[0, 0] = ''
    final_summary[1, 0] = 'Accuracy promedio'
    final_summary[2, 0] = 'UAR promedio'
    final_summary[0, 1] = vec_res_fus[0, 0, 1]
    final_summary[0, 2] = vec_res_fus_2[0, 0, 1]
    final_summary[0, 3:] = vec_res[0, 0, 1:]

    aux_summary = np.zeros((2, methods_number + 2), dtype=np.float)
    for i in range(0, folds):
        for k in range(0, 2):
            aux_summary[k, 0] = aux_summary[k, 0] + float(vec_res_fus[i, k + 1, 1])
            aux_summary[k, 1] = aux_summary[k, 1] + float(vec_res_fus_2[i, k + 1, 1])
        for j in range(0, methods_number):
            for k in range(0, 2):
                aux_summary[k, j + 2] = aux_summary[k, j + 2] + float(vec_res[i, k + 1, j + 1])
    aux_summary = aux_summary / folds
    for i in range(0, aux_summary.shape[0]):
        for j in range(0, aux_summary.shape[1]):
            final_summary[i + 1, j + 1] = str(aux_summary[i, j])
    return final_summary


def showTable(summary):
    for j in range(0, summary.shape[0]):
        file = ''
        for i in range(0, summary.shape[1] - 1):
            file = file + summary[j, i] + ' | '
        file = file + summary[j, summary.shape[1] - 1]
        print(file)
        if j < 3:
            Log.add(file)
        Log.addToTable(file)


def writeLimits(persons_test, answers_limits_list):
    aux_answer_limits_list = list()
    offset_answers = 0
    for i in persons_test:
        aux_answer_limits = list()
        for j in range(0, len(answers_limits_list[int(i)])):
            aux_answer_limits.append(answers_limits_list[int(i)][j] + offset_answers)
        aux_answer_limits_list.extend(aux_answer_limits)
        offset_answers += answers_limits_list[int(i)][len(answers_limits_list[int(i)]) - 1]

    file = open(os.path.join(Datos.PATH_LOGS, str(Datos.FOLD_ACTUAL) + '_limites' + '.txt'), 'a+', encoding="utf-8")
    file.writelines(str(aux_answer_limits_list))
