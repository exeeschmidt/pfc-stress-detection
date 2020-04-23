import os
import cv2 as cv
import numpy as np
import time
import Codigos.ArffManager as am
import Codigos.Metodos as met
import Codigos.Herramientas as hrm
import Codigos.Datos as datos


# ======================================================= VIDEO ========================================================

class Video:
    def __init__(self, binarizar_etiquetas, zonas, metodos):
        self.binarizar_etiquetas = binarizar_etiquetas

        # Defino las zonas donde quiero calcular lbp y hop, las opciones son las de abajo
        # cejas, cejaizq, cejader, ojos, ojoizq, ojoder, cara, nariz, boca
        # zonas = np.array(['cejaizq', 'cejader', 'ojoizq', 'ojoder', 'boca'])
        self.zonas = zonas

        # Defino cuáles características quiero utilizar
        # metodos = np.array(['LBP', 'HOP', 'HOG', 'AU'])
        # Creo un vector booleano donde cada posición significa si un método va a ser o no tenido en cuenta
        self.bool_metodos = np.zeros(4, dtype=bool)
        # El switch define que método se relaciona con que posicion del vector booleano
        switcher = {
            'LBP': 0,
            'HOP': 1,
            'HOG': 2,
            'AU': 3
        }
        # Si encuentro el método, uso el switch para definir la posición del vector que pasa a ser verdadera
        for i in metodos:
            self.bool_metodos[switcher.get(i)] = True

    def __call__(self, persona, etapa, completo=False, rangos_audibles=None):
        start = time.time()

        if rangos_audibles is None:
            elimina_silencio = False
        else:
            elimina_silencio = True

        # Defino los nombres de la clase según si se binariza o no
        if self.binarizar_etiquetas:
            clases = np.array(['Estresado', 'No-Estresado'])
        else:
            clases = np.array(['N', 'B', 'M', 'A'])

        nro_zonas = len(self.zonas)

        # Inicializo OpenFace pasando las banderas
        op_fa = met.OpenFace(cara=False, hog=True, landmarks=True, aus=True)

        # Cargo el archivo con las etiquetas
        arch_etiquetas = hrm.leeCSV('EtiquetadoConTiempo.csv')
        if etapa == 1:
            partes = 7
        else:
            partes = 6

        # Si es por respuesta hago que recorra cada parte
        if completo:
            rango = 1
        else:
            rango = partes

        # Inicializo las clases de los métodos de extracción
        lbp = met.OriginalLBP()
        hop = met.HistogramOfPhase(plotear=False, resize=False)

        for j in range(0, rango):
            # Diferencias en los nombres de archivo y llamada a open face
            if completo:
                nombre = datos.buildVideoName(persona, etapa)
                op_fa(persona, etapa)
            else:
                nombre = datos.buildVideoName(persona, etapa, j+1)
                op_fa(persona, etapa, j+1)

            path = datos.buildPathVideo(persona, etapa, nombre, extension=True)

            if not os.path.exists(path):
                print("Ruta de archivo incorrecta o no válida")
                return

            video = cv.VideoCapture(path)
            frames_totales = int(video.get(cv.CAP_PROP_FRAME_COUNT))

            # Se toman los porcentajes de los segmentos audibles ya que al multiplicarlos por el número de frames
            # totales, se obtiene la traducción al rango de cuadros
            if elimina_silencio:
                rangos_cuadros = rangos_audibles[j] * frames_totales
                rangos_cuadros = rangos_cuadros.astype(int)
                cuadros_audibles = list()
                for i in rangos_cuadros:
                    cuadros_audibles.extend(range(i[0], i[1] + 1))

            # En el completo necesito esto es para definir los intervalos de etiqueta
            if completo:
                # Cargo los tiempos donde termina cada respuesta, para saber en que intervalos va cada etiqueta,
                # esto está en segundos
                tiempos = np.zeros(partes)
                for i in range(0, partes):
                    tiempos[i] = hrm.leeTiemposRespuesta(arch_etiquetas, persona, etapa, i+1)
                # Obtengo los fps para que al multiplicarlos por los tiempos sepa en que cuadro voy del video
                fps = frames_totales / tiempos[partes - 1]
                # Permite saber en que respuesta voy para saber cuando cambiar la etiqueta
                nro_intervalo = 1

            archivo = hrm.leeCSV(os.path.join(datos.PATH_PROCESADO, nombre + '.csv'))

            # Del 0 al 67 son los landmarks, guardo los índices de inicio y fin de cada coordenada de estos
            LimLandmarksX1 = archivo[0].index('x_0')
            LimLandmarksX2 = archivo[0].index('x_67')
            LimLandmarksY1 = archivo[0].index('y_0')
            dif_landmarks = LimLandmarksX2 - LimLandmarksX1

            if self.bool_metodos[2]:
                # Cargo el archivo con las características hog
                hog, inds_hog = hrm.leeHOG(os.path.join(datos.PATH_PROCESADO, nombre + '.hog'))
            else:
                hog = np.array([])

            AUs = np.array([])
            if self.bool_metodos[3]:
                # Lo mismo con las intensidades de los AUs
                LimIntAUs1 = archivo[0].index('AU01_r')
                LimIntAUs2 = archivo[0].index('AU45_r')

            # Inicializo los rangos donde indican el inicio y fin de las características en cada zona según el método
            # Esto sirve para darles el nombre de zonas al guardar las características en los arff
            lbp_range = list([0])
            hop_range = list([0])

            # Leo la etiqueta correspondiente a la primera parte para empezar en caso de ser completo, o la de la
            # respuesta en el caso
            etiqueta = hrm.leeEtiqueta(arch_etiquetas, persona, etapa, j+1)
            if self.binarizar_etiquetas:
                if etiqueta == 'N':
                    etiqueta = clases[1]
                else:
                    etiqueta = clases[0]

            # Número de cuadro que va recorriendo
            nro_frame = 1
            # Booleano para saber si es el primer frame que extraigo características
            primer_frame = True
            # Comienzo a recorrer el video por cada cuadro
            while video.isOpened():
                ret, frame = video.read()
                if ret == 0:
                    break

                # Si no está eliminando silencios o encuentra que el frame se encuentra de los frames audibles, se
                # extraen las características
                if (not elimina_silencio) or cuadros_audibles.count(nro_frame) > 0:
                    # Obtengo los landmarks del archivo
                    lm_x = archivo[nro_frame][LimLandmarksX1:LimLandmarksX1 + dif_landmarks]
                    lm_y = archivo[nro_frame][LimLandmarksY1:LimLandmarksY1 + dif_landmarks]

                    # Inicializo los vectores donde se van a ir concatenando las características de todas las zonas
                    lbp_hist = np.array([])
                    hop_hist = np.array([])

                    # Por cada zona repito
                    for i in range(0, nro_zonas):
                        # Recorto las roi, las expando y aplico un resize para que tengan tamaño constante en todos
                        # los frames
                        roi = hrm.ROI(frame, lm_x, lm_y, self.zonas[i], expandir=True, resize=True)

                        if self.bool_metodos[0]:
                            # Obtengo los patrones locales binarios y sus histogramas
                            aux_lbp = np.array(hrm.Histograma(lbp(roi)))
                            if primer_frame:
                                # A partir del anterior, le voy sumando el tamaño de este
                                lbp_range.append(lbp_range[len(lbp_range) - 1] + len(aux_lbp))
                            lbp_hist = np.concatenate([lbp_hist, aux_lbp])

                        if self.bool_metodos[1]:
                            # Obtengo los histogramas de fase, ravel lo uso para que quede en una sola fila
                            # DATASO: Al agregar más regiones para analizar con HOP, aunque estas impliquen menor tamaño
                            # que tomar una región más grande, demora más
                            # start2 = time.time()
                            # Obtengo los patrones locales binarios y sus histogramas
                            aux_hop = np.array(hrm.Histograma(hop(roi)))
                            if primer_frame:
                                # A partir del anterior, le voy sumando el tamaño de este
                                hop_range.append(hop_range[len(hop_range) - 1] + len(aux_hop))
                            hop_hist = np.concatenate([hop_hist, aux_hop])
                            # print("Tiempo HOP " + zonas[i] + ' ', time.time() - start2)

                    if self.bool_metodos[3]:
                        # Obtengo las intensidades de las AUs de OpenFace
                        AUs = archivo[nro_frame][LimIntAUs1:LimIntAUs2]

                    # Para definir intervalo de etiqueta
                    if completo:
                        # Si paso el tiempo donde termina la respuesta, leo la siguiente etiqueta
                        # Me fijo también si el nro de intervalo no es el último, en ese caso debe etiquetarse hasta el
                        # final. Por esa razón no debe cambiar más de etiqueta. Esta verificación está por si hay error
                        # numérico al calcular los fps y se detecte un cambio de etiqueta unos cuadros antes de la
                        # última etiqueta, lo que provocaría que quiera leer la etiqueta de un número de intervalo que
                        # no existe
                        if (nro_frame >= tiempos[nro_intervalo - 1] * fps) and (nro_intervalo != -1):
                            nro_intervalo = nro_intervalo + 1
                            etiqueta = hrm.leeEtiqueta(arch_etiquetas, persona, etapa, nro_intervalo)
                            print(etiqueta)
                            # print(nro_frame)
                            print("Tiempo: ", time.time() - start)
                            # Paso a usar nro_intervalo como bandera por si es la última etiqueta de la última parte
                            if nro_intervalo == partes:
                                nro_intervalo = -1
                            if self.binarizar_etiquetas:
                                if etiqueta == 'N':
                                    etiqueta = clases[1]
                                else:
                                    etiqueta = clases[0]

                    # Como hog se trata desde una lista extraída del archivo, tengo que cambiar si tengo la lista, sino
                    # le paso un vector vacío
                    if self.bool_metodos[2]:
                        # Si es el primer frame que genere la cabecera (lo hago aca porque tengo que saber el largo de
                        # los vectores de características)
                        if primer_frame:
                            am.CabeceraArff(nombre, lbp_range, hop_range, len(hog[nro_frame - 1]), len(AUs), clases,
                                            self.zonas)
                        # Agrego la fila con los vectores concatenados por método
                        am.FilaArff(nombre, lbp_hist, hop_hist, hog[nro_frame - 1], AUs, etiqueta)
                    else:
                        if primer_frame:
                            am.CabeceraArff(nombre, lbp_range, hop_range, len(hog), len(AUs), clases, self.zonas)
                        am.FilaArff(nombre, lbp_hist, hop_hist, hog, AUs, etiqueta)
                    if primer_frame:
                        primer_frame = False
                # print(nro_frame)
                nro_frame = nro_frame + 1


# ======================================================= AUDIO ========================================================

class Audio:
    def __init__(self, binarizar_etiquetas):
        self.binarizar_etiquetas = binarizar_etiquetas

    def __call__(self, persona, etapa, eliminar_silencios=False):
        # Defino los nombres de la clase según si binarizo o no
        if self.binarizar_etiquetas:
            clases = np.array(['Estresado', 'No-Estresado'])
        else:
            clases = np.array(['N', 'B', 'M', 'A'])

        # Inicializaciones de los métodos
        ffmpeg = met.FFMPEG()
        open_smile = met.OpenSmile(salida_csv=False, ventaneo=True, config_file='IS09_emotion.conf')
        eli_silencios = met.EliminaSilencios(plotear=False)
        arch_etiquetas = hrm.leeCSV('EtiquetadoConTiempo.csv')

        # Según la etapa, distinta cantidad de partes
        if etapa == 1:
            partes = 7
        else:
            partes = 6

        # Parámetro a retornar, en caso que no se eliminen los silencios quedará la lista vacía como retorno
        rangos_silencios = list()

        for j in range(0, partes):
            # Me fijo si existe el archivo
            nombre = datos.buildVideoName(persona, etapa, j+1)
            path = datos.buildPathVideo(persona, etapa, nombre, extension=True)
            if not os.path.exists(path):
                print("Ruta de archivo incorrecta o no válida")
                return

            # Leo la etiqueta correspondiente
            etiqueta = hrm.leeEtiqueta(arch_etiquetas, persona, etapa, j+1)
            if self.binarizar_etiquetas:
                if etiqueta == 'N':
                    etiqueta = clases[1]
                else:
                    etiqueta = clases[0]

            # Ejecuto los métodos para extraer el wav del video y luego el extractor de características
            ffmpeg(persona, etapa, j+1)

            if eliminar_silencios:
                # Obtengo los rangos donde hay segmentos audibles
                rango = eli_silencios(os.path.join(datos.PATH_PROCESADO, nombre + '.wav'))
                # Utilizo la cantidad de segmentos para saber cuantos archivos se generaron
                for i in range(0, rango.shape[0]):
                    nombre_archivo = nombre + '_' + str(i + 1)
                    open_smile(nombre_archivo, paso_ventaneo='0.3')
                    # Modifico el arff devuelto por opensmile para agregarle la etiqueta a toda la respuesta
                    am.AgregaEtiqueta(nombre_archivo, clases, etiqueta)
                # Lo agrego a la lista con los rangos de segmentos de cada respuesta
                rangos_silencios.append(rango)
            else:
                open_smile(nombre, paso_ventaneo='0.3')
                # Modifico el arff devuelto por opensmile para agregarle la etiqueta a toda la respuesta
                am.AgregaEtiqueta(nombre, clases, etiqueta)

        return rangos_silencios
