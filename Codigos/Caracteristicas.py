import os
import ArffManager as am
import numpy as np
import Metodos
import Herramientas as hrm
import cv2 as cv
import time


class VideoEntero:
    def __init__(self, binarizar_etiquetas, zonas):
        self.binarizar_etiquetas = binarizar_etiquetas

        # Defino las zonas donde quiero calcular lbp y hop, las opciones son las de abajo
        # cejas, cejaizq, cejader, ojos, ojoizq, ojoder, cara, nariz, boca
        # zonas = np.array(['cejaizq', 'cejader', 'ojoizq', 'ojoder', 'boca'])
        self.zonas = zonas

    def __call__(self, persona, etapa):
        # persona = '01'
        # etapa = '1'
        nombre = 'Sujeto_' + persona + '_' + etapa
        start = time.time()

        # Defino los nombres de la clase segun si binarizo
        clases = np.array(['Estresado', 'No-Estresado'])
        if not self.binarizar_etiquetas:
            clases = np.array(['N', 'B', 'M', 'A'])

        nro_zonas = len(self.zonas)

        # Uso openface para obtener los landmarks y las intensidades de las AUs
        # ruta_bd = 'D:' + os.sep + 'Descargas' + os.sep + 'Proyecto Final de Carrera' + os.sep + 'Bd propia' + os.sep + 'Conejitos de india'
        ruta_bd = 'D:' + os.sep + 'Google Drive' + os.sep + 'Proyecto Final de Carrera' + os.sep + 'Base de datos'
        # ruta_of = 'D:' + os.sep + 'Descargas' + os.sep + 'OpenFace_2.2.0_win_x64'
        # op_fa = Metodos.OpenFace(False, True, True, True, ruta_bd, ruta_of)
        # op_fa(persona, etapa)

        # Cargo el archivo con las caracteristicas hog
        hog, inds_hog = hrm.leeHOG('Procesado' + os.sep + nombre + '.hog')

        # Cargo el video a analizar y obtengo su numero de frames
        archivo = hrm.leeCSV('Procesado' + os.sep + nombre + '.csv')
        path = ruta_bd + os.sep + 'Sujeto ' + persona + os.sep + 'Etapa ' + etapa + os.sep + nombre + ".mp4"
        video = cv.VideoCapture(path)
        frames_totales = int(video.get(cv.CAP_PROP_FRAME_COUNT))

        # Cargo el archivo con las etiquetas
        arch_etiquetas = hrm.leeCSV('EtiquetadoConTiempo.csv')
        partes = 7
        if etapa == 2:
            partes = 6

        # Cargo los tiempos donde termina cada respuesta, para saber en que intervalos va cada etiqueta, esto esta en segundos
        tiempos = np.zeros(partes)
        for i in range(0, partes):
            tiempos[i] = hrm.leeTiemposRespuesta(arch_etiquetas, persona, etapa, i + 1)
        # Obtengo los fps para que al multiplicarlos por los tiempos sepa en cuadro voy del video
        fps = frames_totales / tiempos[partes - 1]

        # Del 0 al 67 son los landmarks, guardo los indices de inicio y fin de cada coordenada de estos
        LimLandmarksX1 = archivo[0].index('x_0')
        LimLandmarksX2 = archivo[0].index('x_67')
        LimLandmarksY1 = archivo[0].index('y_0')
        dif_landmarks = LimLandmarksX2 - LimLandmarksX1

        # Lo mismo con las intensidades de los AUs
        LimIntAUs1 = archivo[0].index('AU01_r')
        LimIntAUs2 = archivo[0].index('AU45_r')

        # Inicializo las clases de los metodos de extraccion
        lbp = Metodos.OriginalLBP()
        hop = Metodos.HistogramOfPhase(False, False)

        # Leo la etiqueta correspondiente a la primer parte para empezar
        etiqueta = hrm.leeEtiqueta(arch_etiquetas, persona, etapa, 1)
        if self.binarizar_etiquetas:
            # Binarizacion
            if etiqueta == 'N':
                etiqueta = 'No-Estresado'
            else:
                etiqueta = 'Estresado'

        # Permite saber en que respuesta voy para saber cuando cambiar la etiqueta
        nro_intervalo = 1
        # Numero de cuadro que va recorriendo
        nro_frame = 1
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

                # Obtengo los patrones locales binarios y sus histogramas
                lbp_hist = np.concatenate([lbp_hist, np.array(hrm.Histograma(lbp(roi)))])

                # # Obtengo los histogramas de fase, ravel lo uso para que quede en una sola fila
                # DATASO: Al agregar mas regiones para analizar con HOP, aunque estan impliquen menor tamaño que tomar una region mas grande, demora mas
                # start2 = time.time()
                hop_hist = np.concatenate([hop_hist, np.ravel(hop(roi))])
                # print("Tiempo HOP " + zonas[i] + ' ', time.time() - start2)

            # Obtengo las intensidades de las AUs de openface
            AUsInt = archivo[nro_frame][LimIntAUs1:LimIntAUs2]

            # Si es el primer frame que genere la cabecera, lo hago aca porque tengo que saber el largo de los vectores de caracteristicas
            if nro_frame == 1:
                am.CabeceraArff(nombre, len(lbp_hist), len(hop_hist), len(hog[nro_frame - 1]), len(AUsInt), clases)

            # Si paso el tiempo donde termina la respuesta, leo la siguiente etiqueta
            # Me fijo tambien si el nro de intervalo no es el ultimo, en ese caso debe etiquetarse hasta el final
            # por esa razon no debe cambiar mas de etiqueta, esta verificacion esta por si el error numerico al calcular
            # los fps y detecte un cambio de etiqueta unos cuadros antes de la ultima etiqueta, lo que provocaria que
            # quiera leer la etiqueta de un nro de intervalo que no existe
            if nro_frame >= tiempos[nro_intervalo - 1] * fps and nro_intervalo != -1:
                nro_intervalo = nro_intervalo + 1
                etiqueta = hrm.leeEtiqueta(arch_etiquetas, persona, etapa, nro_intervalo)
                print(etiqueta)
                print(nro_frame)
                print("Tiempo: ", time.time() - start)
                # Paso a usar nro_intervalo como bandera por si es la ultima etiqueta de la ultima parte
                if nro_intervalo == partes:
                    nro_intervalo = -1
                if self.binarizar_etiquetas:
                    # Binarizacion
                    if etiqueta == 'N':
                        etiqueta = 'No-Estresado'
                    else:
                        etiqueta = 'Estresado'

            # Agrego la fila con los vectores concatenados por metodo
            am.FilaArff(nombre, lbp_hist, hop_hist, hog[nro_frame - 1], AUsInt, etiqueta)

            # print(nro_frame)
            nro_frame = nro_frame + 1
        print("Tiempo en extraer caracteristicas: ", time.time() - start, " segundos")


class VideoEnParte:
    def __init__(self, binarizar_etiquetas, zonas):
        self.binarizar_etiquetas = binarizar_etiquetas

        # Defino las zonas donde quiero calcular lbp y hop, las opciones son las de abajo
        # cejas, cejaizq, cejader, ojos, ojoizq, ojoder, cara, nariz, boca
        # zonas = np.array(['cejaizq', 'cejader', 'ojoizq', 'ojoder', 'boca'])
        self.zonas = zonas

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
        ruta_bd = 'D:' + os.sep + 'Google Drive' + os.sep + 'Proyecto Final de Carrera' + os.sep + 'Base de datos'
        ruta_of = 'D:' + os.sep + 'Descargas' + os.sep + 'OpenFace_2.2.0_win_x64'
        op_fa = Metodos.OpenFace(False, True, True, True, ruta_bd, ruta_of)

        # Inicializo las clases de los metodos de extraccion
        lbp = Metodos.OriginalLBP()
        hop = Metodos.HistogramOfPhase(False, False)

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

            # Cargo el archivo con las caracteristicas hog
            hog, inds_hog = hrm.leeHOG('Procesado' + os.sep + nombre + '.hog')

            # Cargo el video a analizar y obtengo su numero de frames
            archivo = hrm.leeCSV('Procesado' + os.sep + nombre + '.csv')
            path = ruta_bd + os.sep + 'Sujeto ' + persona + os.sep + 'Etapa ' + etapa + os.sep + nombre + ".mp4"
            video = cv.VideoCapture(path)
            # frames_totales = int(video.get(cv.CAP_PROP_FRAME_COUNT))

            # Del 0 al 67 son los landmarks, guardo los indices de inicio y fin de cada coordenada de estos
            LimLandmarksX1 = archivo[0].index('x_0')
            LimLandmarksX2 = archivo[0].index('x_67')
            LimLandmarksY1 = archivo[0].index('y_0')
            dif_landmarks = LimLandmarksX2 - LimLandmarksX1

            # Lo mismo con las intensidades de los AUs
            LimIntAUs1 = archivo[0].index('AU01_r')
            LimIntAUs2 = archivo[0].index('AU45_r')

            # Leo la etiqueta correspondiente
            etiqueta = hrm.leeEtiqueta(arch_etiquetas, persona, etapa, str(j + 1))
            if self.binarizar_etiquetas:
                # Binarizacion
                if etiqueta == 'N':
                    etiqueta = 'No-Estresado'
                else:
                    etiqueta = 'Estresado'

            # Numero de cuadro que va recorriendo
            nro_frame = 1
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

                    # Obtengo los patrones locales binarios y sus histogramas
                    lbp_hist = np.concatenate([lbp_hist, np.array(hrm.Histograma(lbp(roi)))])

                    # # Obtengo los histogramas de fase, ravel lo uso para que quede en una sola fila
                    # DATASO: Al agregar mas regiones para analizar con HOP, aunque estan impliquen menor tamaño que tomar una region mas grande, demora mas
                    # start2 = time.time()
                    hop_hist = np.concatenate([hop_hist, np.ravel(hop(roi))])
                    # print("Tiempo HOP " + zonas[i] + ' ', time.time() - start2)

                # Obtengo las intensidades de las AUs de openface
                AUsInt = archivo[nro_frame][LimIntAUs1:LimIntAUs2]

                # Si es el primer frame que genere la cabecera, lo hago aca porque tengo que saber el largo de los vectores de caracteristicas
                if nro_frame == 1:
                    am.CabeceraArff(nombre, len(lbp_hist), len(hop_hist), len(hog[nro_frame - 1]), len(AUsInt), clases)

                # Agrego la fila con los vectores concatenados por metodo
                am.FilaArff(nombre, lbp_hist, hop_hist, hog[nro_frame - 1], AUsInt, etiqueta)

                # print(nro_frame)
                nro_frame = nro_frame + 1
            print("Tiempo en extraer caracteristicas parte " + str(j + 1) + " : ", time.time() - start, " segundos")
