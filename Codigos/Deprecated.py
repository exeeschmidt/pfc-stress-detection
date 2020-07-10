import os
import numpy as np
from scipy.signal import convolve2d
import Codigos.Datos as datos
# import Codigos.Herramientas as hrm


# =================================================== Metodos ==========================================================
# Metodo parte de elimina silencios
def _metodo2(energia, cruces, nro_ref, nro_muestras, tam_ventana, muestreo):
    """
    Toma las primeras 5 ventanas como silencio para tener números de referencia acerca de los cruces por cero y
    la energía .
    """
    nro_iteraciones = int(nro_muestras / (tam_ventana * muestreo))
    ref_sil_c = sum(cruces[0:nro_ref]) / nro_ref
    ref_sil_e = sum(energia[0:nro_ref]) / nro_ref
    vector = np.zeros([nro_muestras])

    # Para cada ventana compara con los números de referencia y define el vector binario
    for i in range(0, nro_iteraciones):
        if energia[i] >= ref_sil_e and cruces[i] < ref_sil_c:
            inicia = int(i * tam_ventana * muestreo)
            termina = int(inicia + tam_ventana * muestreo)
            vector[inicia:termina] = np.ones([int(tam_ventana * muestreo)])
    return vector


class LocalDescriptor(object):
    def __init__(self, neighbors):
        self._neighbors = neighbors

    def __call__(self, X):
        raise NotImplementedError("Every LBPOperator must implement the __call__ method.")

    @property
    def neighbors(self):
        return self._neighbors

    def __repr__(self):
        return "LBPOperator (neighbors=%s)" % self._neighbors


class ExtendedLBP(LocalDescriptor):
    def __init__(self, radius=1, neighbors=8):
        LocalDescriptor.__init__(self, neighbors=neighbors)
        self._radius = radius

    def __call__(self, X):
        X = np.asanyarray(X)
        ysize, xsize = X.shape
        # define circle
        angles = 2 * np.pi / self._neighbors
        theta = np.arange(0, 2 * np.pi, angles)
        # calculate sample points on circle with radius
        sample_points = np.array([-np.sin(theta), np.cos(theta)]).T
        sample_points *= self._radius
        # find boundaries of the sample points
        miny = min(sample_points[:, 0])
        maxy = max(sample_points[:, 0])
        minx = min(sample_points[:, 1])
        maxx = max(sample_points[:, 1])
        # calculate block size, each LBP code is computed within a block of size bsizey*bsizex
        blocksizey = np.ceil(max(maxy, 0)) - np.floor(min(miny, 0)) + 1
        blocksizex = np.ceil(max(maxx, 0)) - np.floor(min(minx, 0)) + 1
        # coordinates of origin (0,0) in the block
        origy = int(0 - np.floor(min(miny, 0)))
        origx = int(0 - np.floor(min(minx, 0)))
        # calculate output image size
        dx = int(xsize - blocksizex + 1)
        dy = int(ysize - blocksizey + 1)
        # get center points
        C = np.asarray(X[origy:origy + dy, origx:origx + dx], dtype=np.uint8)
        result = np.zeros((dy, dx), dtype=np.uint32)
        for i, p in enumerate(sample_points):
            # get coordinate in the block
            y, x = p + (origy, origx)
            # Calculate floors, ceils and rounds for the x and y.
            fx = int(np.floor(x))
            fy = int(np.floor(y))
            cx = int(np.ceil(x))
            cy = int(np.ceil(y))
            # calculate fractional part
            ty = y - fy
            tx = x - fx
            # calculate interpolation weights
            w1 = (1 - tx) * (1 - ty)
            w2 = tx * (1 - ty)
            w3 = (1 - tx) * ty
            w4 = tx * ty
            # calculate interpolated image
            N = w1 * X[fy:fy + dy, fx:fx + dx]
            np.add(N, w2 * X[fy:fy + dy, cx:cx + dx], out=N, casting="unsafe")
            np.add(N, w3 * X[cy:cy + dy, fx:fx + dx], out=N, casting="unsafe")
            np.add(N, w4 * X[cy:cy + dy, cx:cx + dx], out=N, casting="unsafe")
            # update LBP codes
            D = N >= C
            np.add(result, (1 << i) * D, out=result, casting="unsafe")
        return result

    @property
    def radius(self):
        return self._radius

    def __repr__(self):
        return "ExtendedLBP (neighbors=%s, radius=%s)" % (self._neighbors, self._radius)


class VarLBP(LocalDescriptor):
    def __init__(self, radius=1, neighbors=8):
        LocalDescriptor.__init__(self, neighbors=neighbors)
        self._radius = radius

    def __call__(self, X):
        X = np.asanyarray(X)
        ysize, xsize = X.shape
        # define circle
        angles = 2 * np.pi / self._neighbors
        theta = np.arange(0, 2 * np.pi, angles)
        # calculate sample points on circle with radius
        sample_points = np.array([-np.sin(theta), np.cos(theta)]).T
        sample_points *= self._radius
        # find boundaries of the sample points
        miny = min(sample_points[:, 0])
        maxy = max(sample_points[:, 0])
        minx = min(sample_points[:, 1])
        maxx = max(sample_points[:, 1])
        # calculate block size, each LBP code is computed within a block of size bsizey*bsizex
        blocksizey = np.ceil(max(maxy, 0)) - np.floor(min(miny, 0)) + 1
        blocksizex = np.ceil(max(maxx, 0)) - np.floor(min(minx, 0)) + 1
        # coordinates of origin (0,0) in the block
        origy = 0 - np.floor(min(miny, 0))
        origx = 0 - np.floor(min(minx, 0))
        # Calculate output image size:
        dx = int(xsize - blocksizex + 1)
        dy = int(ysize - blocksizey + 1)
        # Allocate memory for online variance calculation:
        mean = np.zeros((dy, dx), dtype=np.float32)
        # delta = np.zeros((dy, dx), dtype=np.float32)
        m2 = np.zeros((dy, dx), dtype=np.float32)
        # Holds the resulting variance matrix:
        # result = np.zeros((dy, dx), dtype=np.float32)
        for i, p in enumerate(sample_points):
            # Get coordinate in the block:
            y, x = p + (origy, origx)
            # Calculate floors, ceils and rounds for the x and y:
            fx = int(np.floor(x))
            fy = int(np.floor(y))
            cx = int(np.ceil(x))
            cy = int(np.ceil(y))
            # Calculate fractional part:
            ty = y - fy
            tx = x - fx
            # Calculate interpolation weights:
            w1 = (1 - tx) * (1 - ty)
            w2 = tx * (1 - ty)
            w3 = (1 - tx) * ty
            w4 = tx * ty
            # Calculate interpolated image:
            N = w1 * X[fy:fy + dy, fx:fx + dx]
            np.add(N, w2 * X[fy:fy + dy, cx:cx + dx], out=N, casting="unsafe")
            np.add(N, w3 * X[cy:cy + dy, fx:fx + dx], out=N, casting="unsafe")
            np.add(N, w4 * X[cy:cy + dy, cx:cx + dx], out=N, casting="unsafe")
            # Update the matrices for Online Variance calculation
            # (http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm):
            delta = N - mean
            mean = mean + delta / float(i + 1)
            m2 = m2 + delta * (N - mean)
        # Optional estimate for variance is m2/self._neighbors:
        result = m2 / (self._neighbors - 1)
        return result

    @property
    def radius(self):
        return self._radius

    def __repr__(self):
        return "VarLBP (neighbors=%s, radius=%s)" % (self._neighbors, self._radius)


class LPQ(LocalDescriptor):
    """
    This implementation of Local Phase Quantization (LPQ) is a 1:1 adaption of the original implementation by Ojansivu
    V & Heikkilä J, which is available at: http://www.cse.oulu.fi/CMV/Downloads/LPQMatlab. So all credit goes to them.
    Reference:
        Ojansivu V & Heikkilä J (2008) Blur insensitive texture classification
        using local phase quantization. Proc. Image and Signal Processing
        (ICISP 2008), Cherbourg-Octeville, France, 5099:236-243.
        Copyright 2008 by Heikkilä & Ojansivu
    """

    def __init__(self, radius=3):
        LocalDescriptor.__init__(self, neighbors=8)
        self._radius = radius

    @staticmethod
    def euc_dist(X):
        Y = X = X.astype(np.float)
        XX = np.sum(X * X, axis=1)[:, np.newaxis]
        YY = XX.T
        distances = np.dot(X, Y.T)
        distances *= -2
        distances += XX
        distances += YY
        np.maximum(distances, 0, distances)
        distances.flat[::distances.shape[0] + 1] = 0.0
        return np.sqrt(distances)

    def __call__(self, X):
        f = 1.0
        x = np.arange(-self._radius, self._radius + 1)
        n = len(x)
        rho = 0.95
        [xp, yp] = np.meshgrid(np.arange(1, (n + 1)), np.arange(1, (n + 1)))
        pp = np.concatenate((xp, yp)).reshape(2, -1)
        dd = self.euc_dist(pp.T)  # squareform(pdist(...)) would do the job, too...
        C = np.power(rho, dd)

        w0 = (x * 0.0 + 1.0)
        w1 = np.exp(-2 * np.pi * 1j * x * f / n)
        w2 = np.conj(w1)

        q1 = w0.reshape(-1, 1) * w1
        q2 = w1.reshape(-1, 1) * w0
        q3 = w1.reshape(-1, 1) * w1
        q4 = w1.reshape(-1, 1) * w2

        u1 = np.real(q1)
        u2 = np.imag(q1)
        u3 = np.real(q2)
        u4 = np.imag(q2)
        u5 = np.real(q3)
        u6 = np.imag(q3)
        u7 = np.real(q4)
        u8 = np.imag(q4)

        M = np.array(
            [u1.flatten(), u2.flatten(), u3.flatten(), u4.flatten(), u5.flatten(), u6.flatten(), u7.flatten(),
             u8.flatten()])

        D = np.dot(np.dot(M, C), M.T)
        U, S, V = np.linalg.svd(D)

        Qa = convolve2d(convolve2d(X, w0.reshape(-1, 1), mode='same'), w1.reshape(1, -1), mode='same')
        Qb = convolve2d(convolve2d(X, w1.reshape(-1, 1), mode='same'), w0.reshape(1, -1), mode='same')
        Qc = convolve2d(convolve2d(X, w1.reshape(-1, 1), mode='same'), w1.reshape(1, -1), mode='same')
        Qd = convolve2d(convolve2d(X, w1.reshape(-1, 1), mode='same'), w2.reshape(1, -1), mode='same')

        Fa = np.real(Qa)
        Ga = np.imag(Qa)
        Fb = np.real(Qb)
        Gb = np.imag(Qb)
        Fc = np.real(Qc)
        Gc = np.imag(Qc)
        Fd = np.real(Qd)
        Gd = np.imag(Qd)

        # REEMPLACE matrix(..) por array(...) y flatten(1) por flatten()
        F = np.array(
            [Fa.flatten(), Ga.flatten(), Fb.flatten(), Gb.flatten(), Fc.flatten(), Gc.flatten(), Fd.flatten(),
             Gd.flatten()])
        G = np.dot(V.T, F)

        t = 0

        # Calculate the LPQ Patterns:
        B = (G[0, :] >= t) * 1 + (G[1, :] >= t) * 2 + (G[2, :] >= t) * 4 + (G[3, :] >= t) * 8 + (
                    G[4, :] >= t) * 16 + (
                    G[5, :] >= t) * 32 + (G[6, :] >= t) * 64 + (G[7, :] >= t) * 128

        return np.reshape(B, np.shape(Fa))

    @property
    def radius(self):
        return self._radius

    def __repr__(self):
        return "LPQ (neighbors=%s, radius=%s)" % (self._neighbors, self._radius)


# ================================================ Herramientas ========================================================
def segmentaResumen(resu_1, resu_2):
    """
    Algoritmo para segmentar como en Lefter - Recognizing stress using semantics and modulation of speech and gestures.
    A partir de dos resumenes de predicciones (suponiendo que pueden ser de distinto tamaño) devuelvo los dos conjuntos
    en un solo resumen con la misma segmentación, conservando las etiquetas que se tenían. Esta nueva segmentación
    cuenta con segmentos de tamaño variable, por lo que de cada segmento se guarda su etiqueta y el porcentaje del total
    que representa.
    """

    # Número de métodos en cada modalidad
    num_metodos_1 = resu_1.shape[1] - 1
    num_metodos_2 = resu_2.shape[1] - 1

    # Cantidad de segmentos de cada modalidad más cabecera
    tam_pre_1 = resu_1.shape[0]
    tam_pre_2 = resu_2.shape[0]

    # Saco el porcentaje inicial que representa cada segmento constante en los conjuntos originales
    tam_segmento_1 = 1 / (tam_pre_1 - 1)
    tam_segmento_2 = 1 / (tam_pre_2 - 1)

    # Inicializo el nuevo vector que todavía no sabemos el número de segmentos, pero tendrá los métodos aplicados a
    # audio como a video, más la etiqueta, más una columna con los porcentajes que representan cada segmento
    new_resu = np.empty((0, num_metodos_1 + num_metodos_2 + 2))

    # Armo la cabecera, extraigo los métodos usados en cada resumen
    new_resu = np.append(new_resu, np.array([np.append(np.array(['Porcentaje', 'Etiqueta']),
                                                       np.append(resu_1[0, 1:], resu_2[0, 1:]))]), axis=0)

    # Las porciones que queden de segmento, inicialmente son igual al tamaño entero de segmento
    porc_1 = tam_segmento_1
    porc_2 = tam_segmento_2

    if porc_1 < porc_2:
        avance = porc_1
    else:
        avance = porc_2

    # Índices en los conjuntos iniciales
    ind1 = 1
    ind2 = 1
    while ind1 < tam_pre_1 and ind2 < tam_pre_2:
        # Depende que porción sea más chica, avanzo unicamente esa cantidad
        # Al avanzar la cantidad más chica tengo que reducir el tamaño de la otra porción ya que estaría cortando un
        # segmento. Al indicar la porcion más chica es porque termino ese segmento, por lo que tengo que avanzar en el
        # índice de los conjuntos.
        # En caso de ser iguales el avance es el mismo tanto en porcentaje como para los índices de los conjuntos.

        # Recorro cada método de cada modalidad y formo una fila por modalidad
        # De la primera ya agrego el porcentaje y luego del primer método de la primer modalidad la etiqueta
        fila_1 = np.array([avance, resu_1[ind1, 0]])
        for i in range(1, num_metodos_1 + 1):
            fila_1 = np.append(fila_1, np.array([resu_1[ind1, i]]))

        fila_2 = np.empty(0)
        for i in range(1, num_metodos_2 + 1):
            fila_2 = np.append(fila_2, np.array([resu_2[ind2, i]]))

        # Agrego cada fila al vector general correspondiente
        new_resu = np.append(new_resu, np.array([np.concatenate([fila_1, fila_2])]), axis=0)

        if porc_1 < porc_2:
            avance = porc_1
            ind1 = ind1 + 1
            porc_2 = porc_2 - avance
            porc_1 = tam_segmento_1
        elif porc_2 < porc_1:
            avance = porc_2
            ind2 = ind2 + 1
            porc_1 = porc_1 - avance
            porc_2 = tam_segmento_2
        else:
            avance = porc_1
            ind1 = ind1 + 1
            ind2 = ind2 + 1
            porc_1 = tam_segmento_1
            porc_2 = tam_segmento_2
    return new_resu


def segmentaPrediccion(predi_1, predi_2):
    """
    Algoritmo para segmentar como en Lefter - Recognizing stress using semantics and modulation of speech and gestures.
    A partir de dos conjuntos de etiquetas, con distinto tamaño, devuelvo los dos conjuntos con las misma segmentación
    conservando las etiquetas que se tenían. Esta nueva segmentación cuenta con segmentos de tamaño variable, por lo
    que de cada segmento se guarda su etiqueta, y el porcentaje del total que representa.
    Recibe dos vectores de matrices (uno con los resultados de múltiples clasificaciones de video y otro con los de
    audio). Devuele una matriz por modalidad, donde las filas son los segmentos, la primer columna el porcentaje y luego
    tiene una columna por las etiquetas de cada método de clasificación.
    """

    # Número de métodos en cada modalidad
    num_metodos_1 = predi_1.shape[0]
    num_metodos_2 = predi_2.shape[0]

    # Cantidad de segmentos de cada modalidad más cabecera
    tam_pre_1 = predi_1.shape[1]
    tam_pre_2 = predi_2.shape[1]

    # Saco el porcentaje inicial que representa cada segmento constante en los conjuntos originales
    tam_segmento_1 = 1 / (tam_pre_1 - 1)
    tam_segmento_2 = 1 / (tam_pre_2 - 1)

    # Inicializo ambos vectores vacíos con el primer número de fila y las columnas apropiadas (etiqueta y porcentaje)
    new_predi_1 = np.empty((0, num_metodos_1 + 1))
    new_predi_2 = np.empty((0, num_metodos_2 + 1))

    # Las porciones que queden de segmento, inicialmente son igual al tamaño entero de segmento
    porc_1 = tam_segmento_1
    porc_2 = tam_segmento_2

    if porc_1 < porc_2:
        avance = porc_1
    else:
        avance = porc_2

    # Índices en los conjuntos iniciales
    ind1 = 0
    ind2 = 0
    while ind1 < tam_pre_1 and ind2 < tam_pre_2:
        # Depende que porción sea más chica, avanzo unicamente esa cantidad
        # Al avanzar la cantidad más chica tengo que reducir el tamaño de la otra porción ya que estaría cortando un
        # segmento. Al indicar la porcion más chica es porque termino ese segmento, por lo que tengo que avanzar en el
        # índice de los conjuntos.
        # En caso de ser iguales el avance es el mismo tanto en porcentaje como para los índices de los conjuntos.

        # Recorro cada método de cada modalidad y formo una fila por modalidad
        fila_1 = np.array([avance])
        for i in range(0, num_metodos_1):
            fila_1 = np.append(fila_1, predi_1[i, ind1, 2], axis=0)

        fila_2 = np.array([avance])
        for i in range(0, num_metodos_2):
            fila_2 = np.append(fila_2, predi_2[i, ind2, 2], axis=0)

        # Agrego cada fila al vector general correspondiente
        new_predi_1 = np.append(new_predi_1, np.array([fila_1]), axis=0)
        new_predi_2 = np.append(new_predi_2, np.array([fila_2]), axis=0)

        if porc_1 < porc_2:
            avance = porc_1
            ind1 = ind1 + 1
            porc_2 = porc_2 - avance
            porc_1 = tam_segmento_1
        elif porc_2 < porc_1:
            avance = porc_2
            ind2 = ind2 + 1
            porc_1 = porc_1 - avance
            porc_2 = tam_segmento_2
        else:
            avance = porc_1
            ind1 = ind1 + 1
            ind2 = ind2 + 1
            porc_1 = tam_segmento_1
            porc_2 = tam_segmento_2
    return new_predi_1, new_predi_2


def BinarizoPorPersonas(sujetos, etapas):
    for i in sujetos:
        for j in etapas:
            path_final = os.path.join(datos.PATH_CARACTERISTICAS, datos.buildVideoName(sujetos[i], etapas[j]))
            hrm.BinarizoEtiquetas(path_final)


# ================================================= ArffManager ========================================================
def FilaArff(nombre, lbp_feat, hop_feat, hog_feat, au_feat, etiqueta):
    """
    A partir de varios vectores de caracteristicas, los va agregando a una fila de un arff
    """

    # Abro el archivo con cabecera, la bandera 'a' permite anexar el texto
    file = open(os.path.join(datos.PATH_CARACTERISTICAS, nombre + '.arff'), 'a')

    # Fila de características
    fila = ''

    # Extraigo el largo de cada vector
    lbp_range = np.size(lbp_feat)
    hop_range = np.size(hop_feat)
    hog_range = np.size(hog_feat)
    au_range = np.size(au_feat)

    # Concateno cada vector a la misma fila
    for i in range(0, lbp_range):
        fila = fila + str(lbp_feat[i]) + ','

    for i in range(0, hop_range):
        fila = fila + str(hop_feat[i]) + ','

    for i in range(0, hog_range):
        fila = fila + str(hog_feat[i]) + ','

    for i in range(0, au_range):
        fila = fila + str(au_feat[i]) + ','

    fila = fila + etiqueta

    file.write(fila + '\n')
    file.close()


def CabeceraArff(nombre, lbp_range, hop_range, hog_range, au_range, clases, zonas):
    """
    A partir de los largos de los vectores de caracteristicas por separado crea lo nombres de los atributos del arff
    A estos atributos los divide por zonas y agrega las clases
    """

    # Crea el archivo si no existe, en modo escritura
    file = open(os.path.join(datos.PATH_CARACTERISTICAS, nombre + '.arff'), 'w')
    # La primera línea no se para que sirve pero lo vi en otros arff
    file.write('@relation VideoFeatures' + os.linesep)

    # Recorro dentro de los intervalos pasados por el rango seleccionando la zona correspondiente
    for j in range(0, len(lbp_range) - 1):
        cont = 1
        for i in range(lbp_range[j], lbp_range[j + 1]):
            file.write('@attribute lbp_hist_' + zonas[j] + '[' + str(cont) + '] numeric' + '\n')
            cont = cont + 1

    # Recorro dentro de los intervalos pasados por el rango seleccionando la zona correspondiente
    for j in range(0, len(hop_range) - 1):
        cont = 1
        for i in range(hop_range[j], hop_range[j + 1]):
            file.write('@attribute hop_hist_' + zonas[j] + '[' + str(cont) + '] numeric' + '\n')
            cont = cont + 1

    # Recorro dentro de los intervalos pasados por el rango seleccionando la zona correspondiente
    for j in range(0, len(hog_range) - 1):
        cont = 1
        for i in range(hog_range[j], hog_range[j + 1]):
            file.write('@attribute hog_hist_' + zonas[j] + '[' + str(cont) + '] numeric' + '\n')
            cont = cont + 1

    for i in range(0, au_range):
        file.write('@attribute au_intensity[' + str(i + 1) + '] numeric' + '\n')

    linea_clase = '@attribute class {' + clases[0]
    # file.write('@attribute class {Estresado, No-Estresado}' + os.linesep)
    for i in range(1, len(clases)):
        linea_clase = linea_clase + ', ' + clases[i]
    linea_clase = linea_clase + '}' + os.linesep

    file.write(linea_clase)
    file.write('@data' + os.linesep)
    file.close()


def FilaArffv2(nombre, feat, etiqueta):
    """
    Escribe el vector de caracteristicas en una fila del arff, a diferencia de la v1 este recibe un solo vector
    """
    # Abro el archivo con cabecera, la bandera 'a' permite anexar el texto
    file = open(os.path.join(datos.PATH_CARACTERISTICAS, nombre + '.arff'), 'a')

    # Fila de características
    fila = ''

    for i in feat:
        fila = fila + str(i) + ','

    fila = fila + etiqueta
    file.write(fila + '\n')
    file.close()


def ConcatenaArff(nombre_salida, sujetos, etapas, partes=0, bool_wav=False, rangos_audibles=None):
    """
    Algoritmo para unificar en un solo arff los creados por audio o por video para cada persona, respuesta, parte o subparte
    Los primeros dos parametros tienen que ser np.array de números.

    El parametro partes tiene 3 modos: si es -1 equivale a que no se evaluen las partes, si es 0 se evaluan, y si es
    un entero mayor a cero se analiza solo esa parte
    """

    if partes > 0:
        bool_audible = True
    else:
        bool_audible = False

    extension = '.arff'
    if bool_wav:
        extension = '.wav.arff'

    # Creo el archivo que va a ser la salida de la concatenación
    salida = open(os.path.join(datos.PATH_CARACTERISTICAS, nombre_salida + '.arff'), 'w')

    # Cambio la ruta del primer archivo a leer según si considero o no las partes
    if partes == 0:
        path_primero = os.path.join(datos.PATH_CARACTERISTICAS, datos.buildVideoName(str(sujetos[0]), str(etapas[0]),
                                                                                     str(1)))
    elif partes == -1:
        path_primero = os.path.join(datos.PATH_CARACTERISTICAS, datos.buildVideoName(str(sujetos[0]), str(etapas[0])))
    else:
        path_primero = os.path.join(datos.PATH_CARACTERISTICAS, datos.buildVideoName(str(sujetos[0]), str(etapas[0]),
                                                                                     str(partes))) + '_1'

    # Tomo el primer archivo para crear la cabecera
    archivo = open(path_primero + extension, 'r')
    linea = archivo.readline()
    # Al encontrar @data paro, pero tengo que guardar igual esta línea y otra más en blanco antes de los datos puros
    while linea[0:5] != '@data':
        salida.write(linea)
        linea = archivo.readline()
    salida.write(linea)
    linea = archivo.readline()
    salida.write(linea)
    # Guardo la posición en bytes de donde termina la cabecera, al tener todos los archivos la misma cabecera comienzo
    # leyendo siempre del mismo lugar
    data_pos = archivo.tell()
    archivo.close()

    instancias = 0
    for i in sujetos:
        for j in etapas:
            # Las partes serían si se dividen en respuestas
            if partes == -1:
                # En caso que no se analicen partes simplemente va de 0 a 1 sin cambiar nada el for
                fin_partes = 1
                ini_partes = 0
            elif partes == 0:
                # Si se tienen en cuenta las partes, el for empieza desde el principio y segun la etapa recorre las 6
                # o 7 respuestas que tenga este
                fin_partes = 8
                ini_partes = 1
                if j == '2':
                    fin_partes = 7
            else:
                # En caso de una repuesta en particular el for toma el valor por unica vez de esa parte
                fin_partes = partes + 1
                ini_partes = partes

            for k in range(ini_partes, fin_partes):
                # print('Concatenando parte: ', k)
                base_path = os.path.join(datos.PATH_CARACTERISTICAS, datos.buildVideoName(str(i), str(j)))
                # Las subpartes son si eliminas los silencios, donde cada respuesta a su vez se vuelve a segmentar
                subpartes = 1
                if bool_audible:
                    subpartes = rangos_audibles.shape[0]

                for n in range(0, subpartes):
                    if partes >= 0:
                        parte_path = '_r' + str(k)
                        if bool_audible:
                            parte_path = parte_path + '_' + str(n + 1)
                    else:
                        parte_path = ''
                    path_final = base_path + parte_path + extension
                    archivo = open(path_final, 'r')
                    # Salto donde termina la cabecera y comienzan los datos
                    archivo.seek(data_pos, 0)
                    linea = archivo.readline()
                    # Cuando termina el archivo linea devuelve ""
                    while linea != "":
                        salida.write(linea)
                        linea = archivo.readline()
                        instancias = instancias + 1
                    archivo.close()
                    # Borra el archivo luego de usarlo en la concatenacion
                    # os.remove(base_path + parte_path + extension)
    salida.close()
    return instancias


def ConcatenaArffv2(nombre_salida, nombre_archivo1, nombre_archivo2):
    """
    Algoritmo para unificar los resultados de audio y video en un solo arff.
    Ya previamente concatenados por la otra version de concatena
    """

    # Creo el archivo que va a ser la salida de la concatenación
    salida = open(os.path.join(datos.PATH_CARACTERISTICAS, nombre_salida + '.arff'), 'w')

    # Cargo los archivos que se van a concatenar
    arch1 = open(os.path.join(datos.PATH_CARACTERISTICAS, nombre_archivo1 + '.arff'), 'r')
    arch2 = open(os.path.join(datos.PATH_CARACTERISTICAS, nombre_archivo2 + '.arff'), 'r')

    # Creo manualmente las dos primeras lineas que describen el arff y el salto de linea
    salida.write('@relation AudiovisualFeatures\n')
    salida.write('\n')

    # Esta variable me permite guardan las posiciones, en bytes en cada archivo, en donde comienzan los datos
    pos_data = np.zeros(2)
    # Leo ambas cabeceras, salteando las dos primeras lineas que traen el relation y una linea en blanco.
    # Llego hasta data

    linea = ""
    for i in range(0, 3):
        linea = arch1.readline()
    # Busco hasta que encuentro el atributo de clase, despues salteo esa linea mas un blanco mas data mas otro blanco
    while linea[0:16] != '@attribute class':
        salida.write(linea)
        linea = arch1.readline()
    arch1.readline()
    arch1.readline()
    arch1.readline()
    pos_data[0] = arch1.tell()

    for i in range(0, 3):
        linea = arch2.readline()
    # Busco donde comienza data recien, tomo el atributo clase de aca
    while linea[0:5] != '@data':
        salida.write(linea)
        linea = arch2.readline()
    arch2.readline()
    pos_data[1] = arch2.tell()
    # Cuando termino con la cabecera del ultimo, recien escribo la linea de data y el salto
    salida.write('@data\n')
    salida.write('\n')

    # Ahora recorro simultaneamente las lineas de los dos archivos a la vez, estas tienen que unificarse
    arch1.seek(pos_data[0], 0)
    arch2.seek(pos_data[1], 0)
    linea1 = arch1.readline()
    linea2 = arch2.readline()
    instancias = 0
    while linea1 != "" and linea2 != "":
        # Recorto en cada linea del primer archivo los ultimos digitos correspondiente a la etiqueta
        nueva_linea = linea1[0:len(linea1) - 2] + linea2
        salida.write(nueva_linea)
        linea1 = arch1.readline()
        linea2 = arch2.readline()
        instancias = instancias + 1

    arch1.close()
    arch2.close()
    salida.close()
    return instancias


def AgregaEtiqueta(nombre, clases, etiqueta):
    """
    Permite agregar la etiqueta a los arff ya creados por open smile
    """
    # Abro el archivo para lectura y escritura
    archivo = open(os.path.join(datos.PATH_CARACTERISTICAS, nombre + '.arff'), 'r+')

    # Creo la línea como deberían ser las clases
    linea_clases = '@attribute class {' + clases[0]
    for i in range(1, len(clases)):
        linea_clases = linea_clases + ',' + clases[i]
    linea_clases = linea_clases + '}'

    # Recorro todas las líneas del archivo
    lineas = archivo.readlines()
    nuevas_lineas = list()
    for linea in lineas:
        # Elimino la linea con el atributo string innecesario
        if linea != '@attribute name string\n':
            # Si encuentro la línea donde está definida el atributo clase, la reemplazo por la línea creada antes
            if linea == '@attribute class numeric\n':
                aux = linea_clases + '\n'
            # Busco las líneas de datos (no están en blanco y no tienen el '@' de atributo), corto las últimas 3 (?\n)
            # y agrego la etiqueta más el salto nuevamente
            # Empiezo en la posicion 10 para saltear el primer atributo
            elif linea[0] != '\n' and linea[0] != '@':
                aux = linea[10:len(linea) - 2] + etiqueta + '\n'
            else:
                aux = linea
            nuevas_lineas.append(aux)
    # Borro, llevo el puntero al principio y escribo las líneas ya modificadas
    archivo.truncate(0)
    archivo.seek(0)
    archivo.writelines(nuevas_lineas)
    archivo.close()


def NormalizaArff(nombre_archivo1, nombre_archivo2):
    # Cargo los archivos que se van a normalizar
    arch1 = open(os.path.join(datos.PATH_CARACTERISTICAS, nombre_archivo1 + '.arff'), 'r+')
    arch2 = open(os.path.join(datos.PATH_CARACTERISTICAS, nombre_archivo2 + '.arff'), 'r+')

    # Esta variable me permite guardan las posiciones en donde comienzan los datos
    pos_data = np.ones(2)
    for j in range(0, 2):
        if j == 0:
            arch = arch1
        else:
            arch = arch2
        arch.seek(0)
        linea = arch.readline()
        # Busco donde comienza data recien
        while linea != '@data\n':
            linea = arch.readline()
            pos_data[j] = pos_data[j] + 1
        arch2.readline()
        pos_data[j] = pos_data[j] + 2

    arch1.seek(0)
    arch2.seek(0)
    lineas1 = arch1.readlines()
    lineas2 = arch2.readlines()
    # Verifico quien tiene mas instancias, en caso de tener mas tiene se recorta
    instancias = 0
    if len(lineas1) - pos_data[0] < len(lineas2) - pos_data[1]:
        instancias = len(lineas1) - pos_data[0] + 1
        lineas2 = lineas2[0:int(pos_data[1] + instancias - 1)]
        arch2.truncate(0)
        arch2.seek(0)
        arch2.writelines(lineas2)
    elif len(lineas1) - pos_data[0] > len(lineas2) - pos_data[1]:
        instancias = len(lineas2) - pos_data[1] + 1
        lineas1 = lineas1[0:int(pos_data[0] + instancias - 1)]
        arch1.truncate(0)
        arch1.seek(0)
        arch1.writelines(lineas1)
    arch1.close()
    arch2.close()
    return int(instancias)
