import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
from colorama import Fore
import ArffManager as Am
import Metodos as Met
import Herramientas as Hrm
import Datos


# ======================================================= VIDEO ========================================================

class VideoFeaturesUnification:
    def __init__(self, label_binarization, zones, extract_methods):
        self.label_binarization = label_binarization
        self.microexpression_duration = Datos.TIEMPO_MICROEXPRESION

        # Defino las zonas donde quiero calcular lbp y hop, las opciones son las de abajo
        # cejas, cejaizq, cejader, ojos, ojoizq, ojoder, cara, nariz, boca
        # zonas = np.array(['cejaizq', 'cejader', 'ojoizq', 'ojoder', 'boca'])
        self.zones = zones

        # Defino cuáles características quiero utilizar
        # metodos = np.array(['LBP', 'HOP', 'HOG', 'AUS'])
        # Creo un vector booleano donde cada posición significa si un método va a ser o no tenido en cuenta
        self.bool_methods = np.zeros(4, dtype=bool)
        # El switch define que método se relaciona con que posicion del vector booleano
        switcher = {
            'LBP': 0,
            'HOP': 1,
            'HOG': 2,
            'AUS': 3
        }
        self.extract_methods = extract_methods
        # Si encuentro el método, uso el switch para definir la posición del vector que pasa a ser verdadera
        for i in extract_methods:
            self.bool_methods[switcher.get(i)] = True

        # Defino los nombres de la clase según si se binariza o no
        if self.label_binarization:
            self.classes = np.array(['N', 'E'])
        else:
            self.classes = np.array(['N', 'B', 'M', 'A'])

        self.zones_number = len(self.zones)

    def __call__(self, video_name, video_path, labels_list, complete_mode=False, for_frames=True):
        # Segun que metodo utilice concateno la data de cada arff que sea necesario
        data_vector = np.empty(0)
        for i in range(0, self.bool_methods.size):
            if self.bool_methods[i]:
                method = self.extract_methods[i]
                data_aux = Am.loadAndFiltered(Hrm.buildSubFilePath(video_name, method))
                data_vector = np.append(data_vector, data_aux)

        data = Am.joinDatasetByInstances(data_vector)
        data = Am.filterZones(data, self.zones)
        au_begin, au_end = Am.ausRange(data)

        if complete_mode:
            if for_frames:
                self.completeForFrame(video_name, video_path, data, labels_list)
            else:
                self.completeFusionFrames(video_name, video_path, data, labels_list,
                                                                  au_begin, au_end)
        else:
            processed_labels_list = self.forAnswer(video_name, data, labels_list, au_begin, au_end)
            return processed_labels_list

    def completeForFrame(self, video_name, video_path, data, labels_list):
        # Numero de instancia desde la que recorro
        # instancia_desde = 0

        if not os.path.exists(video_path):
            raise Exception("Ruta de archivo incorrecta o no válida")

        video = cv.VideoCapture(video_path)
        frames_totales = int(video.get(cv.CAP_PROP_FRAME_COUNT))

        # Numero de instancia hasta la que recorro
        instances_number = Am.instancesNumber(data)

        if (frames_totales - 1) != instances_number:
            print('Frames totales distintos a numero de instancias: ' + str(frames_totales - 1) + ' - '
                  + str(instances_number))

        data = Am.addClassAttribute(data, self.classes)

        for i in range(0, instances_number):
            if i < len(labels_list):
                data = Am.addLabel(data, i, np.where(self.classes == labels_list[i])[0][0])
            else:
                data = Am.addLabel(data, i, np.where(self.classes == labels_list[len(labels_list) - 1])[0][0])

        Am.changeRelationName(data, 'VideoFeatures Completos')
        Am.saveInSubfolder(video_name, 'VCom', data)
        return data

    def completeFusionFrames(self, video_name, video_path, data, labels_list, au_begin, au_end):
        if not os.path.exists(video_path):
            raise Exception("Ruta de archivo incorrecta o no válida")

        video = cv.VideoCapture(video_path)
        total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv.CAP_PROP_FPS))
        frame_duration = 1 / fps

        # Si es por respuesta necesito segmentar por el tiempo de las micro expresiones, en el completo el
        # análisis se hace por cuadro

        # Acumulador para indicar cuanto tiempo transcurre desde que empece a contar los frames para un segmento
        accumulated_time = 0
        # Vector para acumular las caracteristicas y luego promediarlas
        vector_features_to_promediate = np.empty(0)
        # Para ir guardando el ultimo vector de promedio valido en caso de tener periodos de tiempo invalidos
        vector_features_to_promediate_previous = np.empty(0)
        # Va guardando las maximas intensidades de cada AU
        vector_maximum_au = np.zeros((1, au_end - au_begin))
        # Para guardar la ultima etiqueta valida
        last_valid_label = ''
        # Vector para guardar las etiquetas y aplicar voto
        vector_label_to_vote = list()
        # Cuadros por segmento
        frames_per_segment = 0
        # Numero de periodos consecutivos invalidos en caso que se den
        invalids_count = 0

        new_data = Am.newDataset(data)
        new_data = Am.addClassAttribute(new_data, self.classes)

        # Numero de instancia hasta la que recorro
        instances_number = Am.instancesNumber(data)
        for i in range(0, instances_number):
            # Si no está eliminando silencios o encuentra que el frame se encuentra de los frames audibles
            # se extraen las caracteristicas.  Verfico también que la confidencialidad encuentra que se detecto
            # una cara en el cuadro segun un umbral interno
            if i < len(labels_list):
                label = labels_list[i]
            else:
                label = labels_list[len(labels_list) - 1]

            features_vector = Am.getValues(data, i)
            # Como puede que entre por no eliminar silencios pero en esa instancia no habia caras por
            # confidencialidad, verifico si no esta vacio
            if features_vector.size != 0:
                # Agrego las caracteristicas al vector que luego se promedia, la etiqueta
                # a la lista para luego hacer voto, y el contador de cuadros por segmento
                for k in range(0, au_end - au_begin):
                    if vector_maximum_au[0, k] < features_vector[k + au_begin]:
                        vector_maximum_au[0, k] = features_vector[k + au_begin]
                if vector_features_to_promediate.size == 0:
                    vector_features_to_promediate = features_vector
                else:
                    vector_features_to_promediate = vector_features_to_promediate + features_vector
                vector_label_to_vote.append(label)
                frames_per_segment = frames_per_segment + 1
            else:
                # Si no obtuve caracteristicas el vector queda vacio, para luego comprobar su tamaño
                features_vector = np.empty(0)

            accumulated_time = accumulated_time + frame_duration
            # Aunque no se extraigan caracteristicas si se hace análisis por períodos se debe verificar si termina
            # algun segmento de tiempo y se deba guardar
            if accumulated_time >= self.microexpression_duration:
                # Verifico si termino el segmento, si es asi debo promediar y agregar al arff
                if vector_features_to_promediate.size != 0:
                    vector_features_to_promediate = vector_features_to_promediate / frames_per_segment
                    voted_label = self.voting(vector_label_to_vote, self.classes)

                    # Si tengo el contador de invalidos hay que completar esas fila promediando el actual valido
                    # con el ultimo valido que hubo
                    if invalids_count > 0:
                        if vector_features_to_promediate_previous.size == 0:
                            # En caso de empezar con periodos invalidos simplemente lo igual al primer valido que
                            # se encuentre
                            approximate_features_vector = vector_features_to_promediate
                            approximate_label = voted_label
                        else:
                            # Creo el vector aproximando promediando
                            approximate_features_vector = (vector_features_to_promediate +
                                                           vector_features_to_promediate_previous) / 2
                            approximate_label = self.voting(list([last_valid_label, voted_label]), self.classes)
                        for k in range(0, invalids_count):
                            new_data = Am.addInstanceWithLabel(new_data,
                                                               approximate_features_vector,
                                                               np.where(self.classes == approximate_label)
                                                               [0][0])
                    # Reemplazo las aus promediadas con el maximo de las aus
                    vector_features_to_promediate[au_begin:au_end] = vector_maximum_au
                    vector_maximum_au = np.zeros((1, au_end - au_begin))
                    vector_features_to_promediate_previous = vector_features_to_promediate
                    last_valid_label = voted_label
                    invalids_count = 0
                    # Recien ahora agrego la fila del periodo actual despues de agregar las anteriores aproximadas
                    new_data = Am.addInstanceWithLabel(new_data, vector_features_to_promediate,
                                                       np.where(self.classes == voted_label)[0][0])
                else:
                    invalids_count = invalids_count + 1
                if accumulated_time > self.microexpression_duration:
                    # A su vez si es mayor es porque un cuadro se "corto" por lo que sus caracteristicas van a
                    # formar parte tambien del promediado del proximo segmento
                    accumulated_time = accumulated_time - self.microexpression_duration
                    # Verifico que se hayan extraido caracteristicas en el ultimo cuadro, y si se eliminan silencios
                    # que el proximo cuadro se encuentre dentro de lo audible, sino significaria que cambia de rango
                    # por lo que no serian cuadros consecutivos y no deberia tenerse en cuenta para el proximo
                    # segmento
                    if features_vector.size != 0:
                        # Si se extrajeron caracteristicas las tiene en cuenta
                        vector_features_to_promediate = features_vector
                        vector_label_to_vote = list(label)
                        frames_per_segment = 1
                    else:
                        # Si el cuadro era invalido que reinicie como si comenzara de cero
                        vector_features_to_promediate = np.empty(0)
                        vector_label_to_vote = list()
                        frames_per_segment = 0
                else:
                    # Si el segmento corta justo con el cuadro reinicio
                    accumulated_time = 0
                    vector_features_to_promediate = np.empty(0)
                    vector_label_to_vote = list()
                    frames_per_segment = 0

        # Si termino el video y tengo periodos de tiempo invalidos tengo que igualarlos con el ultimo periodo valido
        # que hubo
        if invalids_count > 0:
            for i in range(0, invalids_count):
                new_data = Am.addInstanceWithLabel(new_data, vector_features_to_promediate_previous,
                                                   np.where(self.classes == last_valid_label)[0][0])
        Am.saveInSubfolder(video_name, 'VCompProm', new_data)

    def forAnswer(self, video_name, data, labels_list, au_begin, au_end):
        data_parts_vector = np.empty(0)

        person = Hrm.extractPersonFromVideoName(video_name)
        stage = Hrm.extractStageFromFileName(video_name)
        if int(stage) == 1:
            parts = 7
        else:
            parts = 6

        from_instance = 0
        # Debo generar una nueva lista de limites por la fusion de cuadros
        final_answer_limits = list()
        for j in range(0, parts):
            # Diferencias en los nombres de archivo y llamada a open face
            video_name_part = Hrm.buildFileName(person, stage, part=(j + 1))
            video_path_part = Hrm.buildFilePath(person, stage, video_name_part, extension=Datos.EXTENSION_VIDEO)

            if not os.path.exists(video_path_part):
                raise Exception("Ruta de archivo incorrecta o no válida")

            video = cv.VideoCapture(video_path_part)
            total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
            fps = int(video.get(cv.CAP_PROP_FPS))
            frame_duration = 1 / fps

            # Si es por respuesta necesito segmentar por el tiempo de las micro expresiones, en el completo el
            # análisis se hace por cuadro

            # Numero de instancia hasta la que recorro
            to_instance = from_instance + total_frames + 1
            # Acumulador para indicar cuanto tiempo transcurre desde que empece a contar los frames para un segmento
            accumulated_time = 0
            # Vector para acumular las caracteristicas y luego promediarlas
            vector_features_to_promediate = np.empty(0)
            # Para ir guardando el ultimo vector de promedio valido en caso de tener periodos de tiempo invalidos
            vector_features_to_promediate_previous = np.empty(0)
            # Va guardando las maximas intensidades de cada AU
            vector_maximum_au = np.zeros((1, au_end - au_begin))
            # Para guardar la ultima etiqueta valida
            last_valid_label = ''
            # Vector para guardar las etiquetas y aplicar voto
            vector_label_to_vote = list()
            # Cuadros por segmento
            frames_per_segment = 0
            # Numero de periodos consecutivos invalidos en caso que se den
            invalids_count = 0

            data_current_part = Am.newDataset(data)
            data_current_part = Am.addClassAttribute(data_current_part, self.classes)
            for i in range(from_instance, to_instance):
                # Si no está eliminando silencios o encuentra que el frame se encuentra de los frames audibles
                # se extraen las caracteristicas.  Verfico también que la confidencialidad encuentra que se detecto
                # una cara en el cuadro segun un umbral interno
                if i < len(labels_list):
                    label = labels_list[i]
                else:
                    label = labels_list[len(labels_list) - 1]

                features_vector = Am.getValues(data, i)
                # Como puede que entre por no eliminar silencios pero en esa instancia no habia caras por
                # confidencialidad, verifico si no esta vacio
                if features_vector.size != 0:
                    # Agrego las caracteristicas al vector que luego se promedia, la etiqueta
                    # a la lista para luego hacer voto, y el contador de cuadros por segmento
                    for k in range(0, au_end - au_begin):
                        if vector_maximum_au[0, k] < features_vector[k + au_begin]:
                            vector_maximum_au[0, k] = features_vector[k + au_begin]
                    if vector_features_to_promediate.size == 0:
                        vector_features_to_promediate = features_vector
                    else:
                        vector_features_to_promediate = vector_features_to_promediate + features_vector
                    vector_label_to_vote.append(label)
                    frames_per_segment = frames_per_segment + 1
                else:
                    # Si no obtuve caracteristicas el vector queda vacio, para luego comprobar su tamaño
                    features_vector = np.empty(0)

                accumulated_time = accumulated_time + frame_duration
                # Aunque no se extraigan caracteristicas si se hace análisis por períodos se debe verificar si termina
                # algun segmento de tiempo y se deba guardar
                if accumulated_time >= self.microexpression_duration:
                    # Verifico si termino el segmento, si es asi debo promediar y agregar al arff
                    if vector_features_to_promediate.size != 0:
                        vector_features_to_promediate = vector_features_to_promediate / frames_per_segment
                        voted_label = self.voting(vector_label_to_vote, self.classes)

                        # Si tengo el contador de invalidos hay que completar esas fila promediando el actual valido
                        # con el ultimo valido que hubo
                        if invalids_count > 0:
                            if vector_features_to_promediate_previous.size == 0:
                                # En caso de empezar con periodos invalidos simplemente lo igual al primer valido que
                                # se encuentre
                                approximate_features_vector = vector_features_to_promediate
                                approximate_label = voted_label
                            else:
                                # Creo el vector aproximando promediando
                                approximate_features_vector = (vector_features_to_promediate +
                                                               vector_features_to_promediate_previous) / 2
                                approximate_label = self.voting(list([last_valid_label, voted_label]), self.classes)
                            for k in range(0, invalids_count):
                                data_current_part = Am.addInstanceWithLabel(data_current_part,
                                                                            approximate_features_vector,
                                                                            np.where(self.classes == approximate_label)
                                                                            [0][0])
                        # Reemplazo las aus promediadas con el maximo de las aus
                        vector_features_to_promediate[au_begin:au_end] = vector_maximum_au
                        vector_maximum_au = np.zeros((1, au_end - au_begin))
                        vector_features_to_promediate_previous = vector_features_to_promediate
                        last_valid_label = voted_label
                        invalids_count = 0
                        # Recien ahora agrego la fila del periodo actual despues de agregar las anteriores aproximadas
                        data_current_part = Am.addInstanceWithLabel(data_current_part, vector_features_to_promediate,
                                                                    np.where(self.classes == voted_label)[0][0])
                    else:
                        invalids_count = invalids_count + 1
                    if accumulated_time > self.microexpression_duration:
                        # A su vez si es mayor es porque un cuadro se "corto" por lo que sus caracteristicas van a
                        # formar parte tambien del promediado del proximo segmento
                        accumulated_time = accumulated_time - self.microexpression_duration
                        # Verifico que se hayan extraido caracteristicas en el ultimo cuadro, y si se eliminan silencios
                        # que el proximo cuadro se encuentre dentro de lo audible, sino significaria que cambia de rango
                        # por lo que no serian cuadros consecutivos y no deberia tenerse en cuenta para el proximo
                        # segmento
                        if features_vector.size != 0:
                            # Si se extrajeron caracteristicas las tiene en cuenta
                            vector_features_to_promediate = features_vector
                            vector_label_to_vote = list(label)
                            frames_per_segment = 1
                        else:
                            # Si el cuadro era invalido que reinicie como si comenzara de cero
                            vector_features_to_promediate = np.empty(0)
                            vector_label_to_vote = list()
                            frames_per_segment = 0
                    else:
                        # Si el segmento corta justo con el cuadro reinicio
                        accumulated_time = 0
                        vector_features_to_promediate = np.empty(0)
                        vector_label_to_vote = list()
                        frames_per_segment = 0

            # Si termino el video y tengo periodos de tiempo invalidos tengo que igualarlos con el ultimo periodo valido
            # que hubo
            if invalids_count > 0:
                for i in range(0, invalids_count):
                    data_current_part = Am.addInstanceWithLabel(data_current_part,
                                                                vector_features_to_promediate_previous,
                                                                np.where(self.classes == last_valid_label)[0][0])
            # Actualizo para tomar las instancias equivalentes a la proxima respuesta
            from_instance = to_instance
            data_parts_vector = np.append(data_parts_vector, data_current_part)
            # Agrego el numero de instancias de los datos de la parte actual a los limites
            if len(final_answer_limits) == 0:
                final_answer_limits.append(Am.instancesNumber(data_current_part))
            else:
                final_answer_limits.append(
                    final_answer_limits[len(final_answer_limits) - 1] + Am.instancesNumber(data_current_part))

        data_final = Am.joinDatasetByAttributes(data_parts_vector)
        Am.saveInSubfolder(video_name, 'VResp', data_final)
        return final_answer_limits

    @staticmethod
    def voting(labels, classes):
        # Simple algoritmo de votacion que devuelve la clase con mas etiquetas presentes
        votes = np.zeros(classes.shape)
        for i in range(0, classes.size):
            votes[i] = labels.count(classes[i])
        return classes[votes.argmax()]


# ======================================================= AUDIO ========================================================

class AudioFeaturesExtraction:
    def __init__(self, binarize_labels):
        self.binarize_labels = binarize_labels
        self.microexpression_duration = Datos.TIEMPO_MICROEXPRESION

    def __call__(self, file_name, file_path, labels_list, complete_mode=False, extract_from_video=True):
        # Defino los nombres de la clase según si binarizo o no
        if self.binarize_labels:
            self.classes = np.array(['N', 'E'])
        else:
            self.classes = np.array(['N', 'B', 'M', 'A'])

        if complete_mode:
            self.complete(file_name, file_path, labels_list, extract_from_video)
        else:
            self.forAnswer(file_name, labels_list)

    def complete(self, file_name, file_path, labels_list, extract_from_video):
        if not os.path.exists(file_path):
            raise Exception("Ruta de archivo incorrecta o no válida")

        # En el caso de extraer la pista de audio del video lo hago a traves de FFMPEG, sino utilizo el path pasado
        if extract_from_video:
            # Ejecuto el método para extraer el wav del video
            ffmpeg = Met.FFMPEG()
            ffmpeg(file_name, file_path + Datos.EXTENSION_VIDEO)
            audio_path = Hrm.buildOutputPathFFMPEG(file_name)
        else:
            audio_path = file_path

        # Ejecuto el método para extraer las caracteristicas del video
        open_smile = Met.OpenSmile(window=True)
        open_smile(file_name, audio_path, window_size=str(self.microexpression_duration))
        if extract_from_video:
            os.remove(Hrm.buildOutputPathFFMPEG(file_name))

        data = Am.loadAndFiltered(Hrm.buildOpenSmileFilePath(file_name))
        os.remove(Hrm.buildOpenSmileFilePath(file_name))
        # Modifico el arff devuelto por opensmile para agregarle la etiqueta a toda la respuesta
        data = Am.addClassAttribute(data, self.classes)
        for i in range(0, Am.instancesNumber(data)):
            if i < len(labels_list):
                data = Am.addLabel(data, i, np.where(self.classes ==
                                                     labels_list[i])[0][0])
            else:
                data = Am.addLabel(data, i, np.where(self.classes ==
                                                     labels_list[len(labels_list) - 1])[0][0])
        Am.saveInSubfolder(file_name, 'AComp', data)

    def forAnswer(self, file_name_general, labels_list):
        # Inicializaciones de los métodos
        ffmpeg = Met.FFMPEG()
        open_smile = Met.OpenSmile(window=True)

        person = Hrm.extractPersonFromVideoName(file_name_general)
        stage = Hrm.extractStageFromFileName(file_name_general)

        # Según la etapa, distinta cantidad de partes
        if int(stage) == 1:
            parts = 7
        else:
            parts = 6

        # Llevo el numero de instancias actual contando todas las respuestas acumuladas
        accumulate_instances = 0
        data_parts_vector = np.empty(0)
        for j in range(0, parts):
            # Me fijo si existe el archivo
            file_name = Hrm.buildFileName(person, stage, part=(j + 1))
            file_path = Hrm.buildFilePath(person, stage, file_name, extension=Datos.EXTENSION_VIDEO)
            if not os.path.exists(file_path):
                raise Exception("Ruta de archivo incorrecta o no válida")

            # Ejecuto los métodos para extraer el wav del video y luego el extractor de características
            ffmpeg(file_name, file_path)
            audio_path = Hrm.buildOutputPathFFMPEG(file_name)
            open_smile(file_name, audio_path, window_size=str(self.microexpression_duration))
            os.remove(Hrm.buildOutputPathFFMPEG(file_name))

            data = Am.loadAndFiltered(Hrm.buildOpenSmileFilePath(file_name))
            os.remove(Hrm.buildOpenSmileFilePath(file_name))
            # Modifico el arff devuelto por opensmile para agregarle la etiqueta a toda la respuesta
            data = Am.addClassAttribute(data, self.classes)
            for i in range(0, Am.instancesNumber(data)):
                accumulate_instances += 1
                if accumulate_instances < len(labels_list):
                    data = Am.addLabel(data, i, np.where(self.classes ==
                                                         labels_list[accumulate_instances])[0][0])
                else:
                    data = Am.addLabel(data, i, np.where(self.classes ==
                                                         labels_list[len(labels_list) - 1])[0][0])
            data_parts_vector = np.append(data_parts_vector, data)

        data_final = Am.joinDatasetByAttributes(data_parts_vector)
        Am.saveInSubfolder(file_name_general, 'AResp', data_final)


# =========================================== CARACTERISTICAS VIDEO ====================================================

class VideoFeaturesExtraction:
    def __init__(self, zones, tiempo_micro=0.25):
        self.tiempo_micro = tiempo_micro
        self.conf_threshold = 0.7

        # Defino las zonas donde quiero calcular lbp y hop, las opciones son
        # cejas, cejaizq, cejader, ojos, ojoizq, ojoder, cara, nariz, boca
        # zonas = np.array(['cejaizq', 'cejader', 'ojoizq', 'ojoder', 'boca'])
        self.zones = zones

    def __call__(self, video_name, video_path):
        zones_number = len(self.zones)

        if not os.path.exists(video_path):
            raise Exception("Ruta de archivo incorrecta o no válida")
        video = cv.VideoCapture(video_path)

        # Inicializo y ejecuto openface
        op_fa = Met.OpenFace(face=False, hog=False, landmarks=True, aus=True)
        op_fa(video_path)
        openface_data = Hrm.readCSVFile(os.path.join(Datos.PATH_PROCESADO, video_name + '.csv'))
        os.remove(Hrm.buildOutputPathFFMPEG(os.path.join(Datos.PATH_PROCESADO, video_name + '.csv')))
        os.remove(Hrm.buildOutputPathFFMPEG(os.path.join(Datos.PATH_PROCESADO, video_name + '_of_details.txt')))

        # Del 0 al 67 son los landmarks, guardo los índices de inicio y fin de cada coordenada de estos
        lim_landmarks_x1 = openface_data[0].index('x_0')
        lim_landmarks_x2 = openface_data[0].index('x_67')
        lim_landmarks_y1 = openface_data[0].index('y_0')
        dif_landmarks = lim_landmarks_x2 - lim_landmarks_x1

        au = np.array([])
        # Lo mismo con las intensidades de los AUs
        lim_aus_intensity_1 = openface_data[0].index('AU01_r')
        lim_aus_intensity_2 = openface_data[0].index('AU45_r')

        # Inicializo las clases de los métodos de extracción
        lbp = Met.OriginalLBP()
        hop = Met.HistogramOfPhase(plot=False, resize=False)
        win_size = (64, 64)
        block_size = (8, 8)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

        # Inicializo los rangos donde indican el inicio y fin de las características en cada zona según el método
        # Esto sirve para darles el nombre de zonas al guardar las características en los arff
        lbp_length = list([0])
        hop_length = list([0])
        hog_length = list([0])

        # Número de cuadro que va recorriendo
        nro_frame = 1
        # Booleano para saber si es el primer frame que extraigo características
        first_frame = True
        # Comienzo a recorrer el video por cada cuadro
        invalids_instances = 0

        # Inicialización de la barra de progreso
        total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        bar_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.GREEN)
        with tqdm(total=total_frames, unit='frame', desc="Frames", bar_format=bar_format) as frames_progress:
            frames_progress.update(1)

            while video.isOpened():
                ret, frame = video.read()
                if ret == 0:
                    break

                # Extraigo solo si la confidencialidad encuentra que se detecto una cara en el cuadro
                if self.confidenciality(frame):
                    # Obtengo los landmarks de los datos devueltos por openface
                    lm_x = openface_data[nro_frame][lim_landmarks_x1:lim_landmarks_x1 + dif_landmarks]
                    lm_y = openface_data[nro_frame][lim_landmarks_y1:lim_landmarks_y1 + dif_landmarks]

                    # Inicializo los vectores donde se van a ir concatenando las características de todas las zonas
                    lbp_hist = np.array([])
                    hop_hist = np.array([])
                    hog_hist = np.array([])

                    # Por cada zona repito
                    for i in range(0, zones_number):
                        # Recorto las roi, las expando y aplico un resize para que tengan tamaño constante en todos
                        # los frames
                        roi = Hrm.ROI(frame, lm_x, lm_y, self.zones[i])

                        # Calculo las distintas características
                        aux_lbp = np.array(Hrm.generateHistogram(lbp(roi)))
                        # Ravel lo uso para redimensionarlo en una sola fila
                        aux_hop = np.ravel(hop(roi))
                        aux_hog = np.ravel(hog.compute(cv.resize(roi, (64, 64))))

                        if first_frame:
                            # Agrego el indice hasta donde llegan las características de esta zona
                            lbp_length.append(lbp_length[len(lbp_length) - 1] + len(aux_lbp))
                            hop_length.append(hop_length[len(hop_length) - 1] + len(aux_hop))
                            hog_length.append(hog_length[len(hog_length) - 1] + len(aux_hog))

                        lbp_hist = np.concatenate([lbp_hist, aux_lbp])
                        hop_hist = np.concatenate([hop_hist, aux_hop])
                        hog_hist = np.concatenate([hog_hist, aux_hog])

                    # Obtengo las intensidades de las AUs de los datos de openface
                    au = openface_data[nro_frame][lim_aus_intensity_1:lim_aus_intensity_2]

                    # Agrego la cabecera del archivo arff en el caso de ser el primer frame
                    if first_frame:
                        data_lbp = Am.createHeader('LBP', lbp_length, self.zones)
                        data_hop = Am.createHeader('HOP', hop_length, self.zones)
                        data_hog = Am.createHeader('HOG', hog_length, self.zones)
                        data_aus = Am.createHeader('AUs', len(au), self.zones)
                        first_frame = False
                # En caso de no pasar la prueba de confidencialidad y no ser el primer frame, se definen las
                # características como valores faltantes
                elif not first_frame:
                    lbp_hist = np.zeros(lbp_length[len(lbp_length) - 1]) * Am.missingValue()
                    hop_hist = np.zeros(hop_length[len(hop_length) - 1]) * Am.missingValue()
                    hog_hist = np.zeros(hog_length[len(hog_length) - 1]) * Am.missingValue()
                    au = np.zeros(lim_aus_intensity_2 - lim_aus_intensity_1) * Am.missingValue()
                else:
                    invalids_instances = invalids_instances + 1

                if not first_frame:
                    # Al no tener antes el numero de atributos al tener el primer frame ya valido agrego todas
                    # las instancias que habian sido invalidas en los primeros frames
                    if invalids_instances > 0:
                        lbp_hist_aux = np.zeros(lbp_length[len(lbp_length) - 1]) * Am.missingValue()
                        hop_hist_aux = np.zeros(hop_length[len(hop_length) - 1]) * Am.missingValue()
                        hog_hist_aux = np.zeros(hog_length[len(hog_length) - 1]) * Am.missingValue()
                        au_aux = np.zeros(lim_aus_intensity_2 - lim_aus_intensity_1) * Am.missingValue()
                        for i in range(0, invalids_instances):
                            # Agrego la instancia con todas las caracteristicas de los frames invalidos
                            data_lbp = Am.addInstance(data_lbp, lbp_hist_aux)
                            data_hop = Am.addInstance(data_hop, hop_hist_aux)
                            data_hog = Am.addInstance(data_hog, hog_hist_aux)
                            data_aus = Am.addInstance(data_aus, au_aux)
                        invalids_instances = 0
                    # Agrego la instancia con todas las caracteristicas del frame actual
                    data_lbp = Am.addInstance(data_lbp, lbp_hist)
                    data_hop = Am.addInstance(data_hop, hop_hist)
                    data_hog = Am.addInstance(data_hog, hog_hist)
                    data_aus = Am.addInstance(data_aus, au)

                nro_frame = nro_frame + 1
                frames_progress.update(1)

        Am.saveInSubfolder(video_name, 'LBP', data_lbp)
        Am.saveInSubfolder(video_name, 'HOP', data_hop)
        Am.saveInSubfolder(video_name, 'HOG', data_hog)
        Am.saveInSubfolder(video_name, 'AUS', data_aus)

    def confidenciality(self, img):
        frame = np.copy(img)
        model_file = os.path.join(Datos.ROOT_PATH, 'opencv_face_detector_uint8.pb')
        config_file = os.path.join(Datos.ROOT_PATH, 'opencv_face_detector.pbtxt')
        net = cv.dnn.readNetFromTensorflow(model_file, config_file)

        # blobFromImage realiza un pequeño preprocesamiento, el tercer argumento es el nuevo tamaño de imagen,
        # el cuarto son las medias de cada canal de color para hacer la media - el valor de cada canal en cada pixel
        # el quinto argumento es si intercambia los canales rojos y azul, el sexto argumento es si recorta
        blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

        net.setInput(blob)
        detections = net.forward()
        # Devuelve muchas detecciones de posibles caras, como solo buscamos una, con que una confidencialidad sea alta
        # ya consideramos que la cara esta presente
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                return True
        return False
