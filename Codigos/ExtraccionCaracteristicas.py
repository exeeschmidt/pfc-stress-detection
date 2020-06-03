import os
import cv2 as cv
import numpy as np
import time
from tqdm import tqdm
from colorama import Fore
import Codigos.ArffManager as am
import Codigos.Metodos as met
import Codigos.Herramientas as hrm
import Codigos.Datos as datos


# ======================================================= VIDEO ========================================================

class Video:
    def __init__(self, binarizar_etiquetas, zonas, metodos, ):
        self.binarizar_etiquetas = binarizar_etiquetas
        self.tiempo_micro = datos.TIEMPO_MICROEXPRESION

        # Defino las zonas donde quiero calcular lbp y hop, las opciones son las de abajo
        # cejas, cejaizq, cejader, ojos, ojoizq, ojoder, cara, nariz, boca
        # zonas = np.array(['cejaizq', 'cejader', 'ojoizq', 'ojoder', 'boca'])
        self.zonas = zonas

        # Defino cuáles características quiero utilizar
        # metodos = np.array(['LBP', 'HOP', 'HOG', 'AUS'])
        # Creo un vector booleano donde cada posición significa si un método va a ser o no tenido en cuenta
        self.bool_metodos = np.zeros(4, dtype=bool)
        # El switch define que método se relaciona con que posicion del vector booleano
        switcher = {
            'LBP': 0,
            'HOP': 1,
            'HOG': 2,
            'AUS': 3
        }
        self.nombres_metodos = metodos
        # Si encuentro el método, uso el switch para definir la posición del vector que pasa a ser verdadera
        for i in metodos:
            self.bool_metodos[switcher.get(i)] = True

        # Defino los nombres de la clase según si se binariza o no
        if self.binarizar_etiquetas:
            self.clases = np.array(['N', 'E'])
        else:
            self.clases = np.array(['N', 'B', 'M', 'A'])

        self.nro_zonas = len(self.zonas)

    def __call__(self, persona, etapa, completo=False, rangos_audibles=None):
        # start = time.time()
        if rangos_audibles is None:
            rangos_audibles = list()

        # Segun que metodo utilice concateno la data de cada arff que sea necesario
        nombre_aux = datos.buildVideoName(persona, etapa)
        data_vec = np.empty(0)
        for i in range(0, self.bool_metodos.size):
            if self.bool_metodos[i]:
                atrib = self.nombres_metodos[i]
                data_aux = am.CargaYFiltrado(os.path.join(datos.PATH_CARACTERISTICAS,
                                                           atrib, nombre_aux + '_' + atrib + '.arff'))
                data_vec = np.append(data_vec, data_aux)

        data = am.Une(data_vec)
        data = am.FiltraZonas(data, self.zonas)

        # Cargo el archivo con las etiquetas
        arch_etiquetas = hrm.leeCSV(datos.PATH_ETIQUETAS)

        if int(etapa) == 1:
            partes = 7
        else:
            partes = 6

        if completo:
            self.VideoCompleto(persona, etapa, partes, data, arch_etiquetas)
        else:
            self.VideoRespuestas(persona, etapa, partes, rangos_audibles, data, arch_etiquetas)

    def VideoCompleto(self, persona, etapa, partes, data, arch_etiquetas):
        # Numero de instancia desde la que recorro
        # instancia_desde = 0
        # Diferencias en los nombres de archivo y llamada a open face
        nombre = datos.buildVideoName(persona, etapa)
        path = datos.buildPathVideo(persona, etapa, nombre, extension=True)

        if not os.path.exists(path):
            print("Ruta de archivo incorrecta o no válida")
            return

        video = cv.VideoCapture(path)
        frames_totales = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv.CAP_PROP_FPS))

        # Cargo los tiempos donde termina cada respuesta, para saber en que intervalos va cada etiqueta,
        # esto está en segundos
        tiempos = np.zeros(partes)
        for i in range(0, partes):
            tiempos[i] = hrm.leeTiemposRespuesta(arch_etiquetas, persona, etapa, str(i+1))
        # Permite saber en que respuesta voy para saber cuando cambiar la etiqueta
        nro_intervalo = 1

        # Leo la etiqueta correspondiente a la primera parte para empezar en caso de ser completo, o la de la
        # respuesta en el caso
        etiqueta = hrm.leeEtiqueta(arch_etiquetas, persona, etapa, str(1))
        if self.binarizar_etiquetas:
            if etiqueta != 'N':
                etiqueta = self.clases[1]

        # Numero de instancia hasta la que recorro
        instancia_hasta = am.NroInstancias(data)

        if instancia_hasta != (frames_totales - 1):
            print('Instancias diferentes a numero de frames:', instancia_hasta, frames_totales)

        data = am.AgregaAtributoClase(data, self.clases)

        for i in range(0, instancia_hasta):
            # Para definir intervalo de etiqueta

            # Si paso el tiempo donde termina la respuesta, leo la siguiente etiqueta
            # Me fijo también si el nro de intervalo no es el último, en ese caso debe etiquetarse hasta el
            # final. Por esa razón no debe cambiar más de etiqueta. Esta verificación está por si hay error
            # numérico al calcular los fps y se detecte un cambio de etiqueta unos cuadros antes de la
            # última etiqueta, lo que provocaría que quiera leer la etiqueta de un número de intervalo que
            # no existe
            if (i >= tiempos[nro_intervalo - 1] * fps) and (nro_intervalo != -1):
                nro_intervalo = nro_intervalo + 1
                etiqueta = hrm.leeEtiqueta(arch_etiquetas, persona, etapa, nro_intervalo)
                # Paso a usar nro_intervalo como bandera por si es la última etiqueta de la última parte
                if nro_intervalo == partes:
                    nro_intervalo = -1
                if self.binarizar_etiquetas:
                    if etiqueta != 'N':
                        etiqueta = self.clases[1]
            data = am.AgregaEtiqueta(data, i, np.where(self.clases == etiqueta)[0][0])
        am.CambiarRelationName(data, 'VideoFeatures Completos')
        am.Guarda(persona, etapa, 'VCom', data)
        return data

    def VideoRespuestas(self, persona, etapa, partes, rangos_audibles, data, arch_etiquetas):
        # Si no hay rangos audibles especificados o se analiza el video completo no existe la eliminacion de silencios
        if len(rangos_audibles) == 0:
            elimina_silencio = False
        else:
            elimina_silencio = True

        data_vec_general = np.empty(0)

        instancia_desde = 0
        # Número de instancia que va recorriendo
        nro_instancia = 1
        for j in range(0, partes):
            # Diferencias en los nombres de archivo y llamada a open face
            nombre = datos.buildVideoName(persona, etapa, str(j + 1))

            path = datos.buildPathVideo(persona, etapa, nombre, extension=True)

            if not os.path.exists(path):
                print("Ruta de archivo incorrecta o no válida")
                return

            video = cv.VideoCapture(path)
            frames_totales = int(video.get(cv.CAP_PROP_FRAME_COUNT))
            fps = int(video.get(cv.CAP_PROP_FPS))
            duracion_cuadro = 1 / fps

            # Se toman los porcentajes de los segmentos audibles ya que al multiplicarlos por el número de frames
            # totales, se obtiene la equivalencia al rango de cuadros
            if elimina_silencio:
                # rangos audibles es una lista que en cada posicion tiene un vector de vectores por respuesta
                rangos_cuadros = rangos_audibles[j] * frames_totales
                rangos_cuadros = rangos_cuadros.astype(int)
                cuadros_audibles = list()
                # Convierto los rangos en una lista de cuadros
                for i in rangos_cuadros:
                    # Al ser rangos cuadros un vector de vectores, i en cada ciclo es un vector de dos componentes
                    cuadros_audibles.extend(range(i[0], i[1] + 1))

            # Si es por respuesta necesito segmentar por el tiempo de las micro expresiones, en el completo el
            # análisis se hace por cuadro

            # Numero de instancia hasta la que recorro
            instancia_hasta = instancia_desde + frames_totales + 1
            # Acumulador para indicar cuanto tiempo transcurre desde que empece a contar los frames para un segmento
            acu_tiempos = 0
            # Vector para acumular las caracteristicas y luego promediarlas
            vec_prom = np.empty(0)
            # Para ir guardando el ultimo vector de promedio valido en caso de tener periodos de tiempo invalidos
            vec_prom_ant = np.empty(0)
            # Para guardar la ultima etiqueta valida
            etiqueta_ant = ''
            # Vector para guardar las etiquetas y aplicar voto
            vec_etiquetas = list()
            # Cuadros por segmento
            cps = 0
            # Numero de periodos consecutivos invalidos en caso que se den
            invalidos = 0

            # Leo la etiqueta correspondiente a la primera parte para empezar en caso de ser completo, o la de la
            # respuesta en el caso
            etiqueta = hrm.leeEtiqueta(arch_etiquetas, persona, etapa, str(j + 1))
            if self.binarizar_etiquetas:
                if etiqueta != 'N':
                    etiqueta = self.clases[1]

            data_actual = am.NuevaData(data)
            data_actual = am.AgregaAtributoClase(data_actual, self.clases)
            for i in range(instancia_desde, instancia_hasta):
                # Si no está eliminando silencios o encuentra que el frame se encuentra de los frames audibles
                # se extraen las caracteristicas.  Verfico también que la confidencialidad encuentra que se detecto
                # una cara en el cuadro segun un umbral interno
                if (not elimina_silencio) or (cuadros_audibles.count(nro_instancia) > 0):
                    # Concateno todas las caracteristicas en un solo vector
                    vec_caracteristicas = am.getValores(data, i)

                    # Como puede que entre por no eliminar silencios pero en esa instancia no habia caras por
                    # confidencialidad, verifico si no esta vacio
                    if vec_caracteristicas.size != 0:
                        # Agrego las caracteristicas al vector que luego se promedia, la etiqueta
                        # a la lista para luego hacer voto, y el contador de cuadros por segmento
                        if vec_prom.size == 0:
                            vec_prom = vec_caracteristicas
                        else:
                            vec_prom = vec_prom + vec_caracteristicas
                        vec_etiquetas.append(etiqueta)
                        cps = cps + 1
                else:
                    # Si no obtuve caracteristicas doy valores vacios al vector este por el promediado
                    vec_caracteristicas = np.empty(0)

                # Aunque no se extraigan caracteristicas si se hace análisis por períodos se debe verificar si termina
                # algun segmento de tiempo y se deba guardar

                # Solo avanzo en el acumulador de tiempo si no estoy eliminando silencios o el cuadro es audible
                # En el caso de eliminar silencios los cuadros silenciosos deberian equivaler a no existir
                if not elimina_silencio or cuadros_audibles.count(nro_instancia) > 0:
                    acu_tiempos = acu_tiempos + duracion_cuadro

                # Verifico si termino el periodo, si es asi debo promediar y agregar al arff
                if acu_tiempos >= self.tiempo_micro:
                    if vec_prom.size != 0:
                        # NOTA: estamos promediando tambien la intensidad de las AUs, esto podemos volver a analizarlo
                        vec_prom = vec_prom / cps
                        etiqueta_prom = self._voto(vec_etiquetas, self.clases)

                        # Si tengo el contador de invalidos hay que completar esas fila promediando el actual valido
                        # con el ultimo valido que hubo
                        if invalidos > 0:
                            if vec_prom_ant.size == 0:
                                # En caso de empezar con periodos invalidos simplemente lo igual al primer valido que
                                # se encuentre
                                vec_aprox = vec_prom
                                etiqueta_aprox = etiqueta_prom
                            else:
                                # Creo el vector aproximando promediando
                                vec_aprox = (vec_prom + vec_prom_ant) / 2
                                etiqueta_aprox = self._voto(list([etiqueta_ant, etiqueta_prom]), self.clases)
                            for k in range(0, invalidos):
                                data_actual = am.AgregaInstanciaClase(data_actual, vec_aprox,
                                                                      np.where(self.clases == etiqueta_aprox)[0][0])
                        vec_prom_ant = vec_prom
                        etiqueta_ant = etiqueta_prom
                        invalidos = 0
                        # Recien ahora agrego la fila del periodo actual despues de agregar las anteriores aproximadas
                        data_actual = am.AgregaInstanciaClase(data_actual, vec_prom,
                                                               np.where(self.clases == etiqueta_prom)[0][0])
                    else:
                        invalidos = invalidos + 1
                    if acu_tiempos > self.tiempo_micro:
                        # A su vez si es mayor es porque un cuadro se "corto" por lo que sus caracteristicas van a
                        # formar parte tambien del promediado del proximo segmento
                        acu_tiempos = acu_tiempos - self.tiempo_micro
                        # Verifico que se hayan extraido caracteristicas en el ultimo cuadro, y si se eliminan silencios
                        # que el proximo cuadro se encuentre dentro de lo audible, sino significaria que cambia de rango
                        # por lo que no serian cuadros consecutivos y no deberia tenerse en cuenta para el proximo
                        # segmento
                        if (vec_caracteristicas.size != 0) or (
                                elimina_silencio and cuadros_audibles.count(nro_instancia + 1) == 0):
                            # Si se extrajeron las tiene en cuenta
                            vec_prom = vec_caracteristicas
                            vec_etiquetas = list(etiqueta)
                            cps = 1
                        else:
                            # Si el cuadro era invalido que reinicie todo a cero tambien
                            vec_prom = np.empty(0)
                            vec_etiquetas = list()
                            cps = 0
                    else:
                        # Si el segmento corta justo con el cuadro reinicio todo
                        acu_tiempos = 0
                        vec_prom = np.empty(0)
                        vec_etiquetas = list()
                        cps = 0

                # print(nro_instancia)
                nro_instancia = nro_instancia + 1
            # Si termino el video y tengo periodos de tiempo invalidos tengo que igualarlos con el ultimo periodo valido que hubo
            if invalidos > 0:
                for i in range(0, invalidos):
                    data_actual = am.AgregaInstanciaClase(data_actual, vec_prom_ant,
                                                           np.where(self.clases == etiqueta_ant)[0][0])
            # Actualizo para tomar las instancias equivalentes a la proxima respuesta
            instancia_desde = instancia_hasta
            data_vec_general = np.append(data_vec_general, data_actual)

        data_final = am.Unev2(data_vec_general)
        if elimina_silencio:
            am.Guarda(persona, etapa, 'VSil', data_final)
        else:
            am.Guarda(persona, etapa, 'VResp', data_final)

    @staticmethod
    def _voto(etiquetas, clases):
        # Simple algoritmo de votacion que devuelve la clase con mas etiquetas presentes
        votos = np.zeros(clases.shape)
        for i in range(0, clases.size):
            votos[i] = etiquetas.count(clases[i])
        return clases[votos.argmax()]


# ======================================================= AUDIO ========================================================

class Audio:
    def __init__(self, binarizar_etiquetas):
        self.binarizar_etiquetas = binarizar_etiquetas
        self.tiempo_micro = datos.TIEMPO_MICROEXPRESION

    def __call__(self, persona, etapa, eliminar_silencios=False):
        # Defino los nombres de la clase según si binarizo o no
        if self.binarizar_etiquetas:
            clases = np.array(['N', 'E'])
        else:
            clases = np.array(['N', 'B', 'M', 'A'])

        # Inicializaciones de los métodos
        ffmpeg = met.FFMPEG()
        open_smile = met.OpenSmile(salida_csv=False, ventaneo=True)
        eli_silencios = met.EliminaSilencios(plotear=False)
        arch_etiquetas = hrm.leeCSV(datos.PATH_ETIQUETAS)

        # Según la etapa, distinta cantidad de partes
        if int(etapa) == 1:
            partes = 7
        else:
            partes = 6

        # Parámetro a retornar, en caso que no se eliminen los silencios quedará la lista vacía como retorno
        rangos_silencios = list()

        data_vec_general = np.empty(0)
        for j in range(0, partes):
            # Me fijo si existe el archivo
            nombre = datos.buildVideoName(persona, etapa, str(j+1))
            path = datos.buildPathVideo(persona, etapa, nombre, extension=True)
            if not os.path.exists(path):
                print("Ruta de archivo incorrecta o no válida")
                return

            # Leo la etiqueta correspondiente
            etiqueta = hrm.leeEtiqueta(arch_etiquetas, persona, etapa, str(j+1))
            if self.binarizar_etiquetas:
                if etiqueta != 'N':
                    etiqueta = clases[1]

            # Ejecuto los métodos para extraer el wav del video y luego el extractor de características
            ffmpeg(persona, etapa, str(j+1))

            if eliminar_silencios:
                # Obtengo los rangos donde hay segmentos audibles
                rango = eli_silencios(os.path.join(datos.PATH_PROCESADO, nombre + '.wav'), tam_ventana=self.tiempo_micro)
                # rango es un vector de vectores, cada fila tiene un vector de dos componente con el principio y fin

                # Utilizo la cantidad de segmentos para saber cuantos archivos se generaron
                data_vec = np.empty(0)
                for i in range(0, rango.shape[0]):
                    nombre_archivo = nombre + '_' + str(i + 1) + '.wav'
                    open_smile(nombre_archivo, paso_ventaneo=str(self.tiempo_micro))
                    data_aux = am.CargaYFiltrado(os.path.join(datos.PATH_CARACTERISTICAS, nombre_archivo + '.arff'))
                    os.remove(os.path.join(datos.PATH_CARACTERISTICAS, nombre_archivo + '.arff'))
                    # Modifico el arff devuelto por opensmile para agregarle la etiqueta a toda la respuesta
                    data_aux = am.AgregaAtributoClase(data_aux, clases)
                    for k in range(0, am.NroInstancias(data_aux)):
                        data_aux = am.AgregaEtiqueta(data_aux, k, np.where(clases == etiqueta)[0][0])
                    data_vec = np.append(data_vec, data_aux)
                # Lo agrego a la lista con los rangos de segmentos de cada respuesta
                rangos_silencios.append(rango)
                # Concateno todas las subpartes en un arff por respuesta
                data = am.Une(data_vec)
            else:
                open_smile(nombre + '.wav', paso_ventaneo=str(self.tiempo_micro))
                data = am.CargaYFiltrado(os.path.join(datos.PATH_CARACTERISTICAS, nombre + '.wav.arff'))
                os.remove(os.path.join(datos.PATH_CARACTERISTICAS, nombre + '.wav.arff'))
                # Modifico el arff devuelto por opensmile para agregarle la etiqueta a toda la respuesta
                data = am.AgregaAtributoClase(data, clases)
                for i in range(0, am.NroInstancias(data)):
                    data = am.AgregaEtiqueta(data, i, np.where(clases == etiqueta)[0][0])
            data_vec_general = np.append(data_vec_general, data)

        data_final = am.Unev2(data_vec_general)
        if eliminar_silencios:
            am.Guarda(persona, etapa, 'ASil', data_final)
        else:
            am.Guarda(persona, etapa, 'AResp', data_final)
        return rangos_silencios


# =========================================== CARACTERISTICSA VIDEO ====================================================

class CaracteristicasVideo:
    def __init__(self, zonas, tiempo_micro=0.25):
        self.tiempo_micro = tiempo_micro

        # Defino las zonas donde quiero calcular lbp y hop, las opciones son
        # cejas, cejaizq, cejader, ojos, ojoizq, ojoder, cara, nariz, boca
        # zonas = np.array(['cejaizq', 'cejader', 'ojoizq', 'ojoder', 'boca'])
        self.zonas = zonas

    def __call__(self, persona, etapa):
        nro_zonas = len(self.zonas)

        # Inicializo y ejecuto openface
        op_fa = met.OpenFace(cara=False, hog=False, landmarks=True, aus=True)
        op_fa(persona, etapa)

        nombre = datos.buildVideoName(persona, etapa)
        path = datos.buildPathVideo(persona, etapa, nombre, extension=True)
        if not os.path.exists(path):
            print("Ruta de archivo incorrecta o no válida")
            return
        video = cv.VideoCapture(path)
        total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT)-1)

        arch_openface = hrm.leeCSV(os.path.join(datos.PATH_PROCESADO, nombre + '.csv'))

        # Del 0 al 67 son los landmarks, guardo los índices de inicio y fin de cada coordenada de estos
        LimLandmarksX1 = arch_openface[0].index('x_0')
        LimLandmarksX2 = arch_openface[0].index('x_67')
        LimLandmarksY1 = arch_openface[0].index('y_0')
        dif_landmarks = LimLandmarksX2 - LimLandmarksX1

        AUs = np.array([])
        # Lo mismo con las intensidades de los AUs
        LimIntAUs1 = arch_openface[0].index('AU01_r')
        LimIntAUs2 = arch_openface[0].index('AU45_r')

        # Inicializo las clases de los métodos de extracción
        lbp = met.OriginalLBP()
        hop = met.HistogramOfPhase(plotear=False, resize=False)
        winSize = (64, 64)
        blockSize = (8, 8)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

        # Inicializo los rangos donde indican el inicio y fin de las características en cada zona según el método
        # Esto sirve para darles el nombre de zonas al guardar las características en los arff
        lbp_range = list([0])
        hop_range = list([0])
        hog_range = list([0])

        # Número de cuadro que va recorriendo
        nro_frame = 1
        # Booleano para saber si es el primer frame que extraigo características
        primer_frame = True
        # Comienzo a recorrer el video por cada cuadro
        instancias_invalidas = 0
        bar_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.GREEN)
        with tqdm(total=total_frames, unit='frame', desc="Frames", bar_format=bar_format) as progreso_frames:
            while video.isOpened():
                ret, frame = video.read()
                if ret == 0:
                    break

                # Extraigo solo si la confidencialidad encuentra que se detecto una cara en el cuadro
                if self._confidencialidad(frame):
                    # Obtengo los landmarks del archivo
                    lm_x = arch_openface[nro_frame][LimLandmarksX1:LimLandmarksX1 + dif_landmarks]
                    lm_y = arch_openface[nro_frame][LimLandmarksY1:LimLandmarksY1 + dif_landmarks]

                    # Inicializo los vectores donde se van a ir concatenando las características de todas las zonas
                    lbp_hist = np.array([])
                    hop_hist = np.array([])
                    hog_hist = np.array([])

                    # Por cada zona repito
                    for i in range(0, nro_zonas):
                        # Recorto las roi, las expando y aplico un resize para que tengan tamaño constante en todos
                        # los frames
                        roi = hrm.ROI(frame, lm_x, lm_y, self.zonas[i])

                        # Obtengo los patrones locales binarios y sus histogramas
                        aux_lbp = np.array(hrm.Histograma(lbp(roi)))
                        if primer_frame:
                            # A partir del anterior, le voy sumando el tamaño de este
                            lbp_range.append(lbp_range[len(lbp_range) - 1] + len(aux_lbp))
                        lbp_hist = np.concatenate([lbp_hist, aux_lbp])

                        # Obtengo los histogramas de fase, ravel lo uso para que quede en una sola fila
                        aux_hop = np.ravel(hop(roi))
                        if primer_frame:
                            # A partir del anterior, le voy sumando el tamaño de este
                            hop_range.append(hop_range[len(hop_range) - 1] + len(aux_hop))
                        hop_hist = np.concatenate([hop_hist, aux_hop])
                        # print("Tiempo HOP " + zonas[i] + ' ', time.time() - start2)

                        # Obtengo los histogramas de gradiente
                        aux_hog = np.ravel(hog.compute(cv.resize(roi, (64, 64))))
                        if primer_frame:
                            # A partir del anterior, le voy sumando el tamaño de este
                            hog_range.append(hog_range[len(hog_range) - 1] + len(aux_hog))
                        hog_hist = np.concatenate([hog_hist, aux_hog])

                    # Obtengo las intensidades de las AUs de OpenFace
                    AUs = arch_openface[nro_frame][LimIntAUs1:LimIntAUs2]

                # Agrego la cabecera del archivo arff en el caso de ser el primer frame
                if primer_frame:
                    data_lbp = am.Cabecera('LBP', lbp_range, self.zonas)
                    data_hop = am.Cabecera('HOP', hop_range, self.zonas)
                    data_hog = am.Cabecera('HOG', hog_range, self.zonas)
                    data_aus = am.Cabecera('AUs', len(AUs), self.zonas)
                    primer_frame = False
            elif not primer_frame:
                lbp_hist = np.zeros(lbp_range[len(lbp_range) - 1]) * am.valorFaltante()
                hop_hist = np.zeros(hop_range[len(hop_range) - 1]) * am.valorFaltante()
                hog_hist = np.zeros(hog_range[len(hog_range) - 1]) * am.valorFaltante()
                AUs = np.zeros(LimIntAUs2 - LimIntAUs1) * am.valorFaltante()
            else:
                instancias_invalidas = instancias_invalidas + 1

                if not primer_frame:
                    # Al no tener antes el numero de atributos al tener el primer frame ya valido agrego todas las instancias
                    # que habian sido invalidas al principio
                    if instancias_invalidas > 0:
                        lbp_hist_aux = np.zeros(lbp_range[len(lbp_range) - 1]) * am.valorFaltante()
                        hop_hist_aux = np.zeros(hop_range[len(hop_range) - 1]) * am.valorFaltante()
                        hog_hist_aux = np.zeros(hog_range[len(hog_range) - 1]) * am.valorFaltante()
                        AUs_aux = np.zeros(LimIntAUs2 - LimIntAUs1) * am.valorFaltante()
                        for i in range(0, instancias_invalidas):
                            # Agrego la instancia con todas las caracteristicas de los frames invalidos
                            data_lbp = am.AgregaInstancia(data_lbp, lbp_hist_aux)
                            data_hop = am.AgregaInstancia(data_hop, hop_hist_aux)
                            data_hog = am.AgregaInstancia(data_hog, hog_hist_aux)
                            data_aus = am.AgregaInstancia(data_aus, AUs_aux)
                        instancias_invalidas = 0
                    # Agrego la instancia con todas las caracteristicas del frame actual
                    data_lbp = am.AgregaInstancia(data_lbp, lbp_hist)
                    data_hop = am.AgregaInstancia(data_hop, hop_hist)
                    data_hog = am.AgregaInstancia(data_hog, hog_hist)
                    data_aus = am.AgregaInstancia(data_aus, AUs)

                # print(nro_frame)
                nro_frame = nro_frame + 1
                progreso_frames.update(1)

        am.Guarda(persona, etapa, 'LBP', data_lbp)
        am.Guarda(persona, etapa, 'HOP', data_hop)
        am.Guarda(persona, etapa, 'HOG', data_hog)
        am.Guarda(persona, etapa, 'AUS', data_aus)

    @staticmethod
    def _confidencialidad(img):
        conf_threshold = 0.7
        frame = np.copy(img)
        modelFile = os.path.join(datos.ROOT_PATH, 'opencv_face_detector_uint8.pb')
        configFile = os.path.join(datos.ROOT_PATH, 'opencv_face_detector.pbtxt')
        net = cv.dnn.readNetFromTensorflow(modelFile, configFile)

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
            if confidence > conf_threshold:
                return True
        return False
