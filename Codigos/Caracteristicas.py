import os
import cv2 as cv
import numpy as np
import time
import Codigos.ArffManager as am
import Codigos.Metodos as met
import Codigos.Herramientas as hrm
import Codigos.Datos as datos


# ==================================================== VIDEO ====================================================

class VideoEntero:
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

    def __call__(self, persona, etapa):
        # persona = '01'
        # etapa = '1'
        start = time.time()

        # Defino los nombres de la clase según si se binariza o no
        if self.binarizar_etiquetas:
            clases = np.array(['Estresado', 'No-Estresado'])
        else:
            clases = np.array(['N', 'B', 'M', 'A'])

        nro_zonas = len(self.zonas)

        # Construyo los nombres y rutas necesarias
        nombre_video = datos.buildVideoName(persona, etapa)
        path_video = datos.buildPathVideo(persona, etapa, nombre_video)

        # Verifico si la ruta del archivo de video es válida
        if not os.path.exists(path_video):
            print("Ruta de archivo incorrecta o no válida")
            return

        # Cargo el video a analizar y obtengo su numero de frames
        video = cv.VideoCapture(path_video)
        frames_totales = int(video.get(cv.CAP_PROP_FRAME_COUNT))

        # Uso OpenFace para obtener los landmarks y las intensidades de las AUs
        op_fa = met.OpenFace(False, True, True, True)
        op_fa(persona, etapa)

        archivo = hrm.leeCSV(os.path.join(datos.PATH_PROCESADO, nombre_video + '.csv'))

        # Cargo el archivo con las etiquetas
        arch_etiquetas = hrm.leeCSV('EtiquetadoConTiempo.csv')
        if etapa == 1:
            partes = 7
        else:
            partes = 6

        # Cargo los tiempos donde termina cada respuesta para saber en que intervalos va cada etiqueta (en segundos)
        tiempos = np.zeros(partes)
        for i in range(0, partes):
            tiempos[i] = hrm.leeTiemposRespuesta(arch_etiquetas, persona, etapa, i + 1)
        # Obtengo los fps para que al multiplicarlos por los tiempos sepa en que cuadro voy del video
        fps = frames_totales / tiempos[partes - 1]

        # Del 0 al 67 son los landmarks, guardo los índices de inicio y fin de cada coordenada de estos
        LimLandmarksX1 = archivo[0].index('x_0')
        LimLandmarksX2 = archivo[0].index('x_67')
        LimLandmarksY1 = archivo[0].index('y_0')
        dif_landmarks = LimLandmarksX2 - LimLandmarksX1

        if self.bool_metodos[2]:
            # Cargo el archivo con las características hog
            hog, inds_hog = hrm.leeHOG(os.path.join(datos.PATH_PROCESADO, nombre_video + '.hog'))
        else:
            hog = np.array([])

        AUs = np.array([])
        if self.bool_metodos[3]:
            # Lo mismo con las intensidades de los AUs
            LimIntAUs1 = archivo[0].index('AU01_r')
            LimIntAUs2 = archivo[0].index('AU45_r')

        # Inicializo las clases de los métodos de extracción
        lbp = met.OriginalLBP()
        hop = met.HistogramOfPhase(False, False)

        # Inicializo los rangos donde indican el inicio y fin de las características en cada zona según el método
        # Esto sirve para darles el nombre de zonas al guardar las características en los arff
        lbp_range = list([0])
        hop_range = list([0])

        # Leo la etiqueta correspondiente a la primer parte para empezar
        etiqueta = hrm.leeEtiqueta(arch_etiquetas, persona, etapa, 1)
        if self.binarizar_etiquetas:
            # Binarizacion
            if etiqueta == 'N':
                etiqueta = clases[1]
            else:
                etiqueta = clases[0]

        # Permite saber en que respuesta voy para saber cuando cambiar la etiqueta
        nro_intervalo = 1
        # Número de cuadro que va recorriendo
        nro_frame = 1

        # Comienzo a recorrer el video por cada cuadro
        while video.isOpened():
            ret, frame = video.read()
            if ret == 0:
                break

            # Obtengo los landmarks del archivo
            lm_x = archivo[nro_frame][LimLandmarksX1:LimLandmarksX1 + dif_landmarks]
            lm_y = archivo[nro_frame][LimLandmarksY1:LimLandmarksY1 + dif_landmarks]

            # Inicializo los vectores donde se van a ir concatenando las características de todas las zonas
            lbp_hist = np.array([])
            hop_hist = np.array([])

            # Por cada zona repito
            for i in range(0, nro_zonas):
                # Recorto las roi, las expando y aplico un resize para que tengan tamaño constante en todos los frames
                roi = hrm.ROI(frame, lm_x, lm_y, self.zonas[i], True, True)

                if self.bool_metodos[0]:
                    # Obtengo los patrones locales binarios y sus histogramas
                    aux_lbp = np.array(hrm.Histograma(lbp(roi)))
                    if nro_frame == 1:
                        # A partir del anterior, le voy sumando el tamaño de este
                        lbp_range.append(lbp_range[len(lbp_range) - 1] + len(aux_lbp))
                    lbp_hist = np.concatenate([lbp_hist, aux_lbp])

                if self.bool_metodos[1]:
                    # Obtengo los histogramas de fase, ravel lo uso para que quede en una sola fila
                    # DATASO: Al agregar más regiones para analizar con HOP, aunque estas impliquen menor tamaño que
                    # tomar una region más grande, demora más
                    # start2 = time.time()
                    aux_hop = np.ravel(hop(roi))
                    if nro_frame == 1:
                        # A partir del anterior, le voy sumando el tamaño de este
                        hop_range.append(hop_range[len(hop_range) - 1] + len(aux_hop))
                    hop_hist = np.concatenate([hop_hist, aux_hop])
                    # print("Tiempo HOP " + zonas[i] + ' ', time.time() - start2)

            if self.bool_metodos[3]:
                # Obtengo las intensidades de las AUs de openface
                AUs = archivo[nro_frame][LimIntAUs1:LimIntAUs2]

            # Si pasó el tiempo donde termina la respuesta, leo la siguiente etiqueta
            # Me fijo también si el nro de intervalo no es el último, en ese caso debe etiquetarse hasta el final
            # por esa razón no debe cambiar más de etiqueta, esta verificación está por si el error numérico al calcular
            # los fps y detecte un cambio de etiqueta unos cuadros antes de la última etiqueta, lo que provocaria que
            # quiera leer la etiqueta de un nro de intervalo que no existe
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
                    # Binarización
                    if etiqueta == 'N':
                        etiqueta = clases[1]
                    else:
                        etiqueta = clases[0]

            # Como hog se trata desde una lista extraída del archivo, tengo que cambiar si tengo la lista, sino le paso
            # un vector vacío
            if self.bool_metodos[2]:
                # Si es el primer frame que genere la cabecera, lo hago acá porque tengo que saber el largo de los
                # vectores de características
                if nro_frame == 1:
                    am.CabeceraArff(nombre_video, lbp_range, hop_range, len(hog[nro_frame - 1]), len(AUs), clases,
                                    self.zonas)
                # Agrego la fila con los vectores concatenados por método
                am.FilaArff(nombre_video, lbp_hist, hop_hist, hog[nro_frame - 1], AUs, etiqueta)
            else:
                if nro_frame == 1:
                    am.CabeceraArff(nombre_video, lbp_range, hop_range, len(hog), len(AUs), clases, self.zonas)
                am.FilaArff(nombre_video, lbp_hist, hop_hist, hog, AUs, etiqueta)

            nro_frame = nro_frame + 1
        print("Tiempo en extraer caracteristicas: ", time.time() - start, " segundos")


class VideoEnParte:
    def __init__(self, binarizar_etiquetas, zonas, metodos):
        self.binarizar_etiquetas = binarizar_etiquetas

        # Defino las zonas donde quiero calcular lbp y hop, las opciones son las de abajo
        # cejas, cejaizq, cejader, ojos, ojoizq, ojoder, cara, nariz, boca
        # zonas = np.array(['cejaizq', 'cejader', 'ojoizq', 'ojoder', 'boca'])

        self.zonas = zonas

        # Defino cuales caracteristicas quiero utilizarr
        # metodos = np.array['LBP','HOP','HOG','AU']
        # Creo un vector booleano donde cada posicion significa si un metodo va a ser o no tenido en cuenta
        self.bool_metodos = np.zeros(4, dtype=bool)
        # El switch define que metodo se relaciona con que posicion del vector booleano
        switcher = {
            'LBP': 0,
            'HOP': 1,
            'HOG': 2,
            'AU': 3
        }
        # Si encuentro el metodo que use el switch para definir la posicion del vector que pasa a ser verdadera
        for i in metodos:
            self.bool_metodos[switcher.get(i)] = True

    def __call__(self, persona, etapa):
        # persona = '01'
        # etapa = '1'

        # Defino los nombres de la clase segun si binarizo
        clases = np.array(['Estresado', 'No-Estresado'])
        if not self.binarizar_etiquetas:
            clases = np.array(['N', 'B', 'M', 'A'])

        nro_zonas = len(self.zonas)

        # Uso openface para obtener los landmarks y las intensidades de las AUs
        # Aca solo lo inicializo
        ruta_bd = 'Base de datos'
        op_fa = met.OpenFace(False, True, True, True)

        # Inicializo las clases de los metodos de extraccion
        lbp = met.OriginalLBP()
        # hop = met.HistogramOfPhase(False, False)

        # Cargo el archivo con las etiquetas
        arch_etiquetas = hrm.leeCSV('EtiquetadoConTiempo.csv')
        partes = 7
        if etapa == 2:
            partes = 6

        for j in range(0, partes):
            start = time.time()
            nombre = 'Sujeto_' + persona + '_' + etapa + '_r' + str(j + 1)

            # Ejecuto open face
            op_fa(persona, etapa, parte=str(j + 1))

            # Cargo el video a analizar y obtengo su numero de frames
            path = ruta_bd + os.sep + 'Sujeto ' + persona + os.sep + 'Etapa ' + etapa + os.sep + nombre + ".mp4"
            if not os.path.exists(path):
                print("Ruta de archivo incorrecta o no valida")
                return
            video = cv.VideoCapture(path)

            archivo = hrm.leeCSV('Procesado' + os.sep + nombre + '.csv')

            # Del 0 al 67 son los landmarks, guardo los indices de inicio y fin de cada coordenada de estos
            LimLandmarksX1 = archivo[0].index('x_0')
            LimLandmarksX2 = archivo[0].index('x_67')
            LimLandmarksY1 = archivo[0].index('y_0')
            dif_landmarks = LimLandmarksX2 - LimLandmarksX1

            if self.bool_metodos[2]:
                # Cargo el archivo con las caracteristicas hog
                hog, inds_hog = hrm.leeHOG('Procesado' + os.sep + nombre + '.hog')
            else:
                hog = np.array([])

            AUs = np.array([])

            if self.bool_metodos[3]:
                # Lo mismo con las intensidades de los AUs
                LimIntAUs1 = archivo[0].index('AU01_r')
                LimIntAUs2 = archivo[0].index('AU45_r')

            # Leo la etiqueta correspondiente
            etiqueta = hrm.leeEtiqueta(arch_etiquetas, persona, etapa, str(j + 1))
            if self.binarizar_etiquetas:
                # Binarizacion
                if etiqueta == 'N':
                    etiqueta = clases[1]
                else:
                    etiqueta = clases[0]

            # Inicializo los rangos donde indican el inicio y fin de las caracteristicas en cada zona segun el metodo
            # Esto sirve para darles el nombre de zonas al guardar las caracteristicas en los arff
            lbp_range = list([0])
            hop_range = list([0])

            # Numero de cuadro que va recorriendo
            nro_frame = 1

            # Comienzo a recorrer el video por cada cuadro
            while video.isOpened():
                ret, frame = video.read()
                if ret == 0:
                    break

                # Obtengo los landmarks del archivo
                lm_x = archivo[nro_frame][LimLandmarksX1:LimLandmarksX1 + dif_landmarks]
                lm_y = archivo[nro_frame][LimLandmarksY1:LimLandmarksY1 + dif_landmarks]

                # Inicializo los vectores donde se van a ir concatenando las caracteristicas de todas las zonas
                lbp_hist = np.array([])
                hop_hist = np.array([])
                # Por cada zona repito
                for i in range(0, nro_zonas):
                    # Recorto las roi, las expando y aplico un resize para que tengan tamaño constante en todos los frames
                    roi = hrm.ROI(frame, lm_x, lm_y, self.zonas[i], True, True)

                    if self.bool_metodos[0]:
                        # Obtengo los patrones locales binarios y sus histogramas
                        aux_lbp = np.array(hrm.Histograma(lbp(roi)))
                        if nro_frame == 1:
                            # A partir del anterior, le voy sumando el tamaño de este
                            lbp_range.append(lbp_range[len(lbp_range) - 1] + len(aux_lbp))
                        lbp_hist = np.concatenate([lbp_hist, aux_lbp])

                    if self.bool_metodos[1]:
                        # Obtengo los histogramas de fase, ravel lo uso para que quede en una sola fila
                        # DATASO: Al agregar mas regiones para analizar con HOP, aunque estan impliquen menor tamaño que tomar una region mas grande, demora mas
                        # start2 = time.time()
                        # Obtengo los patrones locales binarios y sus histogramas
                        aux_lbp = np.array(hrm.Histograma(lbp(roi)))
                        if nro_frame == 1:
                            # A partir del anterior, le voy sumando el tamaño de este
                            lbp_range.append(lbp_range[len(lbp_range) - 1] + len(aux_lbp))
                        lbp_hist = np.concatenate([lbp_hist, aux_lbp])
                        # print("Tiempo HOP " + zonas[i] + ' ', time.time() - start2)

                if self.bool_metodos[3]:
                    # Obtengo las intensidades de las AUs de openface
                    AUs = archivo[nro_frame][LimIntAUs1:LimIntAUs2]

                # Como hog se trata desde una lista extraida del archivo, tengo que cambiar si tengo la lista, sino le paso un vector vacio
                if self.bool_metodos[2]:
                    # Si es el primer frame que genere la cabecera, lo hago aca porque tengo que saber el largo de los vectores de caracteristicas
                    if nro_frame == 1:
                        am.CabeceraArff(nombre, lbp_range, hop_range, len(hog[nro_frame - 1]), len(AUs), clases,
                                        self.zonas)
                    # Agrego la fila con los vectores concatenados por metodo
                    am.FilaArff(nombre, lbp_hist, hop_hist, hog[nro_frame - 1], AUs, etiqueta)
                else:
                    if nro_frame == 1:
                        am.CabeceraArff(nombre, lbp_range, hop_range, len(hog), len(AUs), clases, self.zonas)
                    am.FilaArff(nombre, lbp_hist, hop_hist, hog, AUs, etiqueta)

                # print(nro_frame)
                nro_frame = nro_frame + 1
            print("Tiempo en extraer caracteristicas parte " + str(j + 1) + " : ", time.time() - start, " segundos")

# ===================================================================================================================


class Audio:
    def __init__(self, binarizar_etiquetas):
        self.binarizar_etiquetas = binarizar_etiquetas

    def __call__(self, persona, etapa):
        # Defino los nombres de la clase segun si binarizo
        clases = np.array(['Estresado', 'No-Estresado'])
        if not self.binarizar_etiquetas:
            clases = np.array(['N', 'B', 'M', 'A'])

        # Definicion de rutas e inicializaciones de los metodos
        ruta_bd = 'Base de datos'

        ruta_ffmpeg = 'Librerias' + os.sep + 'ffmpeg' + os.sep + 'bin'
        ffmpeg = met.FFMPEG(ruta_bd, ruta_ffmpeg)

        ruta_os = 'Librerias' + os.sep + 'opensmile'
        open_smile = met.OpenSmile(False, True, ruta_os, 'IS09_emotion.conf')

        arch_etiquetas = hrm.leeCSV('EtiquetadoConTiempo.csv')

        # Según la etapa, distinta cantidad de partes
        partes = 7
        if etapa == 2:
            partes = 6

        for j in range(0, partes):
            # Me fijo si existe el archivo
            nombre = 'Sujeto_' + persona + '_' + etapa + '_r' + str(j + 1)
            path = ruta_bd + os.sep + 'Sujeto ' + persona + os.sep + 'Etapa ' + etapa + os.sep + nombre + ".mp4"
            if not os.path.exists(path):
                print("Ruta de archivo incorrecta o no valida")
                return

            # Leo la etiqueta correspondiente
            etiqueta = hrm.leeEtiqueta(arch_etiquetas, persona, etapa, str(j + 1))

            # Ejecuto los metodos para extraer el wav del video y luego el extractor de caracteristicas
            ffmpeg(persona, etapa, str(j + 1))
            open_smile(persona, etapa, str(j + 1), paso_ventaneo='0.3')

            if self.binarizar_etiquetas:
                # Binarizacion
                if etiqueta == 'N':
                    etiqueta = clases[1]
                else:
                    etiqueta = clases[0]

            # Modifico el arff devuelto por opensmile para agregarle la etiqueta a toda la respuesta
            am.AgregaEtiqueta(nombre + '.wav', clases, etiqueta)
