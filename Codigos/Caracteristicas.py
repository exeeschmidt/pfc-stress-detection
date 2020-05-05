import os
import cv2 as cv
import numpy as np
import Codigos.ArffManager as am
import Codigos.Metodos as met
import Codigos.Herramientas as hrm
import Codigos.Datos as datos


# ======================================================= VIDEO ========================================================

class Video:
    def __init__(self, zonas, metodos, binarizar_etiquetas=True, tiempo_micro=0.25):
        self.binarizar_etiquetas = binarizar_etiquetas
        self.tiempo_micro = tiempo_micro

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
        if rangos_audibles is None:
            rangos_audibles = list()

        if len(rangos_audibles) == 0:
            elimina_silencio = False
        else:
            elimina_silencio = True

        # Defino los nombres de la clase según si se binariza o no
        if self.binarizar_etiquetas:
            clases = np.array(['N', 'E'])
        else:
            clases = np.array(['N', 'B', 'M', 'A'])

        nro_zonas = len(self.zonas)

        # Inicializo OpenFace pasando las banderas
        op_fa = met.OpenFace(cara=False, hog=True, landmarks=True, aus=True)

        # Cargo el archivo con las etiquetas
        arch_etiquetas = hrm.readCSV(datos.PATH_ETIQUETAS)
        if int(etapa) == 1:
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
        winSize = (64, 64)
        blockSize = (8, 8)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

        for j in range(0, rango):
            # Diferencias en los nombres de archivo
            if completo:
                nombre = hrm.buildVideoName(persona, etapa)
            else:
                nombre = hrm.buildVideoName(persona, etapa, str(j+1))

            # Armo el path del archivo y verifico si existe o es válido
            path = hrm.buildPathVideo(persona, etapa, nombre, extension=True)
            if not os.path.exists(path):
                print("Ruta de archivo incorrecta o no válida")
                return

            # Llamada a open face
            op_fa(path)

            video = cv.VideoCapture(path)
            frames_totales = int(video.get(cv.CAP_PROP_FRAME_COUNT))
            fps = int(video.get(cv.CAP_PROP_FPS))
            duracion_cuadro = 1/fps

            # Se toman los porcentajes de los segmentos audibles ya que al multiplicarlos por el número de frames
            # totales, se obtiene la equivalencia al rango de cuadros
            if elimina_silencio:
                rangos_cuadros = rangos_audibles[j] * frames_totales
                rangos_cuadros = rangos_cuadros.astype(int)
                cuadros_audibles = list()
                # Convierto los rangos en una lista de cuadros
                for i in rangos_cuadros:
                    cuadros_audibles.extend(range(i[0], i[1] + 1))

            # En el completo necesito esto es para definir los intervalos de etiqueta
            if completo:
                # Cargo los tiempos donde termina cada respuesta, para saber en que intervalos va cada etiqueta,
                # esto está en segundos
                tiempos = np.zeros(partes)
                for i in range(0, partes):
                    tiempos[i] = hrm.readTiemposRespuesta(arch_etiquetas, persona, etapa, str(i+1))
                # Permite saber en que respuesta voy para saber cuando cambiar la etiqueta
                nro_intervalo = 1
            else:
                # Si es por respuesta necesito segmentar por el tiempo de las micro expresiones, en el completo el
                # análisis se hace por cuadro

                # Acumulador para indicar cuanto tiempo transcurre desde que empece a contar los frames para un segmento
                acu_tiempos = 0
                # Vector para acumular las caracteristicas y luego promediarlas
                vec_prom = np.empty(0)
                # Vector para guardar las etiquetas y aplicar voto
                vec_etiquetas = list()
                # Cuadros por segmento
                cps = 0

            archivo = hrm.readCSV(os.path.join(datos.PATH_PROCESADO, nombre + '.csv'))

            # Del 0 al 67 son los landmarks, guardo los índices de inicio y fin de cada coordenada de estos
            LimLandmarksX1 = archivo[0].index('x_0')
            LimLandmarksX2 = archivo[0].index('x_67')
            LimLandmarksY1 = archivo[0].index('y_0')
            dif_landmarks = LimLandmarksX2 - LimLandmarksX1

            AUs = np.array([])
            if self.bool_metodos[3]:
                # Lo mismo con las intensidades de los AUs
                LimIntAUs1 = archivo[0].index('AU01_r')
                LimIntAUs2 = archivo[0].index('AU45_r')

            # Inicializo los rangos donde indican el inicio y fin de las características en cada zona según el método
            # Esto sirve para darles el nombre de zonas al guardar las características en los arff
            lbp_range = list([0])
            hop_range = list([0])
            hog_range = list([0])

            # Leo la etiqueta correspondiente a la primera parte para empezar en caso de ser completo, o la de la
            # respuesta en el caso
            etiqueta = hrm.readEtiqueta(arch_etiquetas, persona, etapa, str(j+1))
            if self.binarizar_etiquetas:
                if etiqueta != 'N':
                    etiqueta = clases[1]

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
                if (not elimina_silencio) or (cuadros_audibles.count(nro_frame) > 0):
                    # Obtengo los landmarks del archivo
                    lm_x = archivo[nro_frame][LimLandmarksX1:LimLandmarksX1 + dif_landmarks]
                    lm_y = archivo[nro_frame][LimLandmarksY1:LimLandmarksY1 + dif_landmarks]

                    # Inicializo los vectores donde se van a ir concatenando las características de todas las zonas
                    lbp_hist = np.array([])
                    hop_hist = np.array([])
                    hog_hist = np.array([])

                    # Por cada zona repito
                    for i in range(0, nro_zonas):
                        # Recorto las roi, las expando y aplico un resize para que tengan tamaño constante en todos
                        # los frames
                        roi = hrm.getROI(frame, lm_x, lm_y, self.zonas[i], expandir=True, resize=True)

                        if self.bool_metodos[0]:
                            # Obtengo los patrones locales binarios y sus histogramas
                            aux_lbp = np.array(hrm.getHistograma(lbp(roi)))
                            if primer_frame:
                                # A partir del anterior, le voy sumando el tamaño de este
                                lbp_range.append(lbp_range[len(lbp_range) - 1] + len(aux_lbp))
                            lbp_hist = np.concatenate([lbp_hist, aux_lbp])

                        if self.bool_metodos[1]:
                            # Obtengo los histogramas de fase (ravel se usa para que quede en una sola fila)
                            # DATASO: Al agregar más regiones para analizar con HOP, aunque estas impliquen menor tamaño
                            # que tomar una región más grande, demora más
                            # start2 = time.time()
                            # Obtengo los patrones locales binarios y sus histogramas
                            aux_hop = np.ravel(hop(roi))
                            if primer_frame:
                                # A partir del anterior, le voy sumando el tamaño de este
                                hop_range.append(hop_range[len(hop_range) - 1] + len(aux_hop))
                            hop_hist = np.concatenate([hop_hist, aux_hop])
                            # print("Tiempo HOP " + zonas[i] + ' ', time.time() - start2)

                        if self.bool_metodos[2]:
                            # Obtengo los histogramas de gradiente
                            aux_hog = np.ravel(hog.compute(cv.resize(roi, (64, 64))))
                            if primer_frame:
                                hog_range.append(hog_range[len(hog_range) - 1] + len(aux_hog))
                            hog_hist = np.concatenate([hog_hist, aux_hog])

                    if self.bool_metodos[3]:
                        # Obtengo las intensidades de las AUs de OpenFace
                        AUs = archivo[nro_frame][LimIntAUs1:LimIntAUs2]

                    # Concateno todas las caracteristicas en un solo vector
                    vec_caracteristicas = np.concatenate([lbp_hist, hop_hist, hog_hist, AUs], axis=0)
                    vec_caracteristicas = vec_caracteristicas.astype('float64')

                    # Agrego la cabecera del archivo arff en el caso de ser el primer frame
                    if primer_frame:
                        am.cabeceraArff(nombre, lbp_range, hop_range, hog_range, len(AUs), clases, self.zonas)
                        primer_frame = False

                    # Si es completo analisis por cuadro, sino por periodos de tiempo
                    if completo:
                        # Para definir intervalo de etiqueta

                        # Si paso el tiempo donde termina la respuesta, leo la siguiente etiqueta
                        # Me fijo también si el nro de intervalo no es el último, en ese caso debe etiquetarse hasta el
                        # final. Por esa razón no debe cambiar más de etiqueta. Esta verificación está por si hay error
                        # numérico al calcular los fps y se detecte un cambio de etiqueta unos cuadros antes de la
                        # última etiqueta, lo que provocaría que quiera leer la etiqueta de un número de intervalo que
                        # no existe
                        if (nro_frame >= tiempos[nro_intervalo - 1] * fps) and (nro_intervalo != -1):
                            nro_intervalo = nro_intervalo + 1
                            etiqueta = hrm.readEtiqueta(arch_etiquetas, persona, etapa, nro_intervalo)
                            # print(etiqueta)
                            # print(nro_frame)
                            # print("Tiempo: ", time.time() - start)
                            # Paso a usar nro_intervalo como bandera por si es la última etiqueta de la última parte
                            if nro_intervalo == partes:
                                nro_intervalo = -1
                            if self.binarizar_etiquetas:
                                if etiqueta != 'N':
                                    etiqueta = clases[1]

                        # Agrego las caracteristicas y la etiqueta al arff
                        am.filaArffv2(nombre, vec_caracteristicas, etiqueta)
                    else:
                        # Al acumulador le agrego el tiempo del cuadro
                        acu_tiempos = acu_tiempos + duracion_cuadro

                        if acu_tiempos <= self.tiempo_micro:
                            # Si es menor o igual debo agregar las caracteristicas al vector que luego se promedia, la
                            # etiqueta a la lista para luego hacer voto, y el contador de cuadros por segmento
                            if vec_prom.size == 0:
                                vec_prom = vec_caracteristicas
                            else:
                                vec_prom = vec_prom + vec_caracteristicas
                            vec_etiquetas.append(etiqueta)
                            cps = cps + 1

                        if acu_tiempos >= self.tiempo_micro:
                            # NOTA: estamos promediando tambien la intensidad de las AUs, esto podemos volver a
                            # analizarlo
                            # Si es mayor o igual ya el segmento termino por lo que debo promediar y agregar al arff
                            vec_prom = vec_prom / cps
                            am.filaArffv2(nombre, vec_prom, self._voto(vec_etiquetas, clases))
                            if acu_tiempos > self.tiempo_micro:
                                # A su vez si es mayor es porque un cuadro se "corto" por lo que sus caracteristicas van
                                # a formar parte tambien del promediado del proximo segmento
                                acu_tiempos = acu_tiempos - self.tiempo_micro
                                vec_prom = vec_caracteristicas
                                vec_etiquetas = list(etiqueta)
                                cps = 1
                            else:
                                # Si el segmento corta justo con el cuadro reinicio all
                                acu_tiempos = 0
                                vec_prom = np.empty(0)
                                vec_etiquetas = list()
                                cps = 0
                # print(nro_frame)
                nro_frame = nro_frame + 1

    @staticmethod
    def _voto(etiquetas, clases):
        # Simple algoritmo de votacion que devuelve la clase con mas etiquetas presentes
        votos = np.zeros(clases.shape)
        for i in range(0, clases.size):
            votos[i] = etiquetas.count(clases[i])
        return clases[votos.argmax()]


# ======================================================= AUDIO ========================================================

class Audio:
    def __init__(self, binarizar_etiquetas, tiempo_micro=0.25):
        self.binarizar_etiquetas = binarizar_etiquetas
        self.tiempo_micro = tiempo_micro

    def __call__(self, persona, etapa, eliminar_silencios=False):
        # Defino los nombres de la clase según si binarizo o no
        if self.binarizar_etiquetas:
            clases = np.array(['N', 'E'])
        else:
            clases = np.array(['N', 'B', 'M', 'A'])

        # Inicializaciones de los métodos
        ffmpeg = met.FFMPEG()
        open_smile = met.OpenSmile(salida_csv=False, ventaneo=True, config_file='IS09_emotion.conf')
        eli_silencios = met.EliminaSilencios(plotear=False)
        arch_etiquetas = hrm.readCSV(datos.PATH_ETIQUETAS)

        # Según la etapa, distinta cantidad de partes
        if int(etapa) == 1:
            partes = 7
        else:
            partes = 6

        # Parámetro a retornar, en caso que no se eliminen los silencios quedará la lista vacía como retorno
        rangos_silencios = list()

        for j in range(0, partes):
            # Me fijo si existe el archivo
            nombre = hrm.buildVideoName(persona, etapa, str(j+1))
            path = hrm.buildPathVideo(persona, etapa, nombre, extension=True)
            if not os.path.exists(path):
                print("Ruta de archivo incorrecta o no válida")
                return

            # Leo la etiqueta correspondiente
            etiqueta = hrm.readEtiqueta(arch_etiquetas, persona, etapa, str(j+1))
            if self.binarizar_etiquetas:
                if etiqueta != 'N':
                    etiqueta = clases[1]

            # Ejecuto los métodos para extraer el wav del video y luego el extractor de características
            ffmpeg(persona, etapa, str(j+1))

            if eliminar_silencios:
                # Obtengo los rangos donde hay segmentos audibles
                rango = eli_silencios(os.path.join(datos.PATH_PROCESADO, nombre + '.wav'),
                                      tam_ventana=self.tiempo_micro)
                # Utilizo la cantidad de segmentos para saber cuantos archivos se generaron
                for i in range(0, rango.shape[0]):
                    nombre_archivo = nombre + '_' + str(i + 1) + '.wav'
                    open_smile(nombre_archivo, paso_ventaneo=str(self.tiempo_micro))
                    # Modifico el arff devuelto por opensmile para agregarle la etiqueta a toda la respuesta
                    am.agregaEtiqueta(nombre_archivo, clases, etiqueta)
                # Lo agrego a la lista con los rangos de segmentos de cada respuesta
                rangos_silencios.append(rango)
            else:
                open_smile(nombre + '.wav', paso_ventaneo=str(self.tiempo_micro))
                # Modifico el arff devuelto por opensmile para agregarle la etiqueta a toda la respuesta
                am.agregaEtiqueta(nombre + '.wav', clases, etiqueta)

        return rangos_silencios
