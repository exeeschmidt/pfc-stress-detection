import histogramofphase
import imagesc
import os
import cv2 as cv
import numpy as np
import matlab
import subprocess
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
from scipy.signal import convolve2d
import Codigos.Datos as datos

# Contiene todos los métodos a utilizar, donde las implementaciones de cada uno comienzan con un bookmark.

# Nota importante:
#   Al ejecutar por primera vez siempre hacerlo con OpenFace y luego OpenSmile, explicado en el apartado de OpenSmile


# ===================================================== OPEN FACE ======================================================

class OpenFace:
    """
    Extrae las carasterísticas a partir de OpenFace, guardando estos en una carpeta "Procesado" dentro del proyecto. 
    Toma los videos de la ruta de la base de datos.

    Ejemplo:
        ruta_of = 'D:/Descargas/OpenFace_2.2.0_win_x64'
        ruta_bd = 'D:/Descargas/Proyecto Final de Carrera/BD propia/Conejitos de india'
        op_fa = Metodos.OpenFace(False, True, True, True, ruta_bd, ruta_of)
        persona = 'Sujeto 01a'
        parte = 'a'
        op_fa(persona, parte)
    """
    
    def __init__(self, cara, hog, landmarks, aus):
        # Los 4 parámetros son banderas para extraer características
        self._cara = cara
        self._hog = hog
        self._landmarks = landmarks
        self._aus = aus

    def __call__(self, path_video):
        path_save = datos.PATH_PROCESADO

        # Comando base para ejecutar OpenFace
        comando = ['FeatureExtraction.exe', '-f', path_video, '-out_dir', path_save]

        # Según las banderas se le agregan parámetros al comando
        if self._cara:
            comando.append('-simalign')
        if self._hog:
            comando.append('-hogalign')
        if self._landmarks:
            comando.append('-2Dfp')
        if self._aus:
            comando.append('-aus')

        # Cambio al directorio de OpenFace, ejecuto el comando y luego vuelvo al directorio donde están los códigos
        os.chdir(datos.PATH_OPENFACE)
        subprocess.run(comando, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        os.chdir(datos.PATH_CODIGOS)


"""
LISTA DE PARAMETROS APLICABLES A OPEN FACE 2.0
CADA UNO DEBE AGREGARSE COMO PARTE DEL VECTOR A EJECUTAR POR SUBPROCESS

-verbose visualise the processing steps live: tracked face with gaze, action units, similarity aligned face, and HOG
  features (not visualized by default), this flag turns all of them on, below flags allow for more fine-grained
  control. Visualizing these outputs will reduce processing speeds, potentially by a significant amount.
-vis-track visualise the tracked face
-vis-hog visualise the HOG features
-vis-align visualise similarity aligned faces
-vis-aus visualise Action Units
-simscale <float> scale of the face for similarity alignment (default 0.7)
-simsize <int> width and height of image in pixels when similarity aligned (default 112)
-format_aligned <format> output image format for aligned faces (e.g. png or jpg), any format supported by OpenCV
-format_vis_image <format> output image format for visualized images (e.g. png or jpg), any format supported by
  OpenCV. Only applicable to FaceLandmarkImg
-nomask forces the aligned face output images to not be masked out
-g output images should be grayscale (for saving space)
By default the executable will output all features (tracked videos, HOG files, similarity aligned images and a .csv
file with landmarks, action units and gaze). You might not always want to extract all the output features, you can
specify the desired output using the following flags:
-2Dfp output 2D landmarks in pixels
-3Dfp output 3D landmarks in milimeters
-pdmparams output rigid and non-rigid shape parameters
-pose output head pose (location and rotation)
-aus output the Facial Action Units
-gaze output gaze and related features (2D and 3D locations of eye landmarks)
-hogalign output extracted HOG feaure file
-simalign output similarity aligned images of the tracked faces
-nobadaligned if outputting similarity aligned images, do not output from frames where detection failed or is
  unreliable (thus saving some disk space)
"""


# ===================================================== OPEN SMILE =====================================================

class OpenSmile:
    """
    Extrae las carasteristicas a partir de opensmile, guardando estos en una carpeta "Procesado" del proyecto. Lee 
    los wav desde la carpeta procesado, que ya deberian haberse generado con FFMPEG.
    IMPORTANTE: a diferencia de OpenFace, si la carpeta "Procesado" no esta creada de antemano, falla.

    Ejemplo:
        ruta_os = 'D:/Descargas/opensmile/opensmile-2.3.0'
        ruta_bd = 'D:/Descargas/Proyecto Final de Carrera/Bd propia/Conejitos de india'
        config_file = 'IS09_emotion.conf'
        op_sm = Metodos.OpenSmile(False, False, ruta_os, ruta_bd, config_file)
        persona = 'Sujeto 01a'
        parte = 'a'
        op_sm(persona, parte)
    """
    
    def __init__(self, salida_csv, ventaneo, config_file):
        # Bandera para salida como csv, sino sale como arff
        self._salida_csv = salida_csv

        # Bandera para definir si se ventanea. Si es True deben incluirse los parametros adicionales en el __call__
        # Si no se llama el ventaneo se define cada 0.5s y el shift inicial en 0
        self._ventaneo = ventaneo

        # Nombre del archivo de configuracion utilizado
        # config_file = 'IS09_emotion.conf'
        self._config_file = config_file

    def __call__(self, nombre_archivo, paso_ventaneo='0.125', shift_ini_ventaneo='0'):
        # Comando base de OpenSmile
        comando = [os.path.join('bin', 'Win32', 'SMILExtract_Release.exe'), '-C',
                   os.path.join('config', self._config_file), '-I',
                   os.path.join(datos.PATH_PROCESADO, nombre_archivo)]

        # Según las banderas se le agregan parámetros al comando
        if self._salida_csv:
            comando.append('-appendcsv')
            comando.append('0')
            comando.append('-csvoutput')
            comando.append(os.path.join(datos.PATH_CARACTERISTICAS, nombre_archivo + '.csv'))
        else:
            comando.append('-appendarff')
            comando.append('0')
            comando.append('-output')
            comando.append(os.path.join(datos.PATH_CARACTERISTICAS, nombre_archivo + '.arff'))

        # En caso de ventaneo se utiliza el config_file que se permite escribir desde la función archivo_ventaneo
        if self._ventaneo:
            path_ventaneo = os.path.join('config', 'shared', 'FrameModeFunctionalsVentana.conf.inc')
            self._archivo_ventaneo(os.path.join(datos.PATH_OPENSMILE, path_ventaneo), paso_ventaneo, shift_ini_ventaneo)
            comando.append('-frameModeFunctionalsConf')
            # En caso de modificar el paso y el solapamiento modificar ese archivo
            comando.append(path_ventaneo)

        # Cambio al directorio de OpenSmile, ejecuto el comando y luego vuelvo al directorio donde están los códigos
        os.chdir(datos.PATH_OPENSMILE)
        subprocess.run(comando, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        os.chdir(datos.PATH_CODIGOS)

    @staticmethod
    def _archivo_ventaneo(path, paso, shift_ini):
        f = open(path, 'w')
        f.write('frameMode = fixed\n')
        f.write('frameSize = ' + paso + '\n')
        f.write('frameStep = ' + shift_ini + '\n')
        f.write('frameCenterSpecial = left\n')


"""
The following options are available for controlling the output data formats (for configurations
which provide feature summaries via statistical functionals, such as all INTERSPEECH and
AVEC challenge sets):
-----------------------------
-instname <string> Usually the input filename, saved in first column in CSV and ARFF output. Default is "unknown".
-----------------------------
-lldcsvoutput, -D <filename> Enables LLD frame-wise output to CSV.
-appendcsvlld <0/1> Set to 1 to append to existing CSV output file. Default is overwrite (0).
-timestampcsvlld <0/1> Set to 0 to disable timestamp output to CSV in second column. Default is 1.
-headercsvlld <0/1> Set to 0 to disable header output (1st line) to CSV. Default is 1 (enabled)
-----------------------------
-lldhtkoutput <filename> Enables LLD frame-wise output to HTK format.
-----------------------------
-lldarffoutput, -D <filename> Enables LLD frame-wise output to ARFF.
-appendarfflld <0/1> Set to 1 to append to existing ARFF output file. Default is overwrite (0).
-timestamparfflld <0/1> Set to 0 to disable timestamp output to ARFF in second column. Default 1.
-lldarfftargetsfile <file> Specify the configuration include, that defines the target fields (classes) Default: shared/arff_targets_conf.inc
-----------------------------
-output, -O <filename> The default output option. To ARFF file format, for feature summaries.
-appendarff <0/1> Set to 0 to not append to existing ARFF output file. Default is append (1).
-timestamparff <0/1> Set to 1 to enable timestamp output to ARFF in second column. Default 0.
-arfftargetsfile <file> Specify the configuration include, that defines the target fields (classes) Default: shared/arff_targets_conf.inc
-----------------------------
-csvoutput <filename> The default output option. To CSV file format, for feature summaries.
-appendcsv <0/1> Set to 0 to not append to existing CSV output file. Default is append (1).
-timestampcsv <0/1> Set to 0 to disable timestamp output to CSV in second column. Default 1.
-headercsv <0/1> Set to 0 to disable header output (1st line) to CSV. Default is 1 (enabled)
-----------------------------
-htkoutput <filename> Enables output to HTK format of feature summaries (functionals)

For configurations which provide Low-Level-Descriptor (LLD) features only (i.e. which do
not summarise features by means of statistical functionals over time), the following output
options are available:
-----------------------------
-csvoutput <filename> The default output option. To CSV file format, for frame-wise LLD.
-appendcsv <0/1> Set to 1 to append to existing CSV output file. Default is overwrite (0).
-timestampcsv <0/1> Set to 0 to disable timestamp output to CSV in second column. Default 1.
-headercsv <0/1> Set to 0 to disable header output (1st line) to CSV. Default is 1 (enabled)
-----------------------------
-output, -O <filename> Default output to HTK format of feature summaries (functionals).
-----------------------------
-arffoutput <filename> The default output option. To ARFF file format, for frame-wise LLD.
-appendarff <0/1> Set to 0 to not append to existing ARFF output file. Default is append (1).
-timestamparff <0/1> Set to 0 to disable timestamp output to ARFF in second column. Default 1.
-arfftargetsfile <file> Specify the configuration include, that defines the target fields (classes)
  Default: shared/arff_targets_conf.inc
"""


# ======================================================== LBP =========================================================

"""
Copyright (c) Philipp Wagner. All rights reserved.
Licensed under the BSD license. See LICENSE file in the project root for full license information.

Ejemplo:
    lbp_original = Metodos.OriginalLBP()
    lbp = lbp_original(imagen_en_escala_de_grises)

NOTA: Además de pasar la imagen en escala de grises, se debe castear a np.uint8 para ser visualizada
"""


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


class OriginalLBP(LocalDescriptor):
    def __init__(self):
        LocalDescriptor.__init__(self, neighbors=8)

    def __call__(self, X):
        X = np.asarray(X)
        X = (1 << 7) * (X[0:-2, 0:-2] >= X[1:-1, 1:-1]) \
            + (1 << 6) * (X[0:-2, 1:-1] >= X[1:-1, 1:-1]) \
            + (1 << 5) * (X[0:-2, 2:] >= X[1:-1, 1:-1]) \
            + (1 << 4) * (X[1:-1, 2:] >= X[1:-1, 1:-1]) \
            + (1 << 3) * (X[2:, 2:] >= X[1:-1, 1:-1]) \
            + (1 << 2) * (X[2:, 1:-1] >= X[1:-1, 1:-1]) \
            + (1 << 1) * (X[2:, :-2] >= X[1:-1, 1:-1]) \
            + (1 << 0) * (X[1:-1, :-2] >= X[1:-1, 1:-1])
        return X

    def __repr__(self):
        return "OriginalLBP (neighbors=%s)" % self._neighbors


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


# ======================================================== HOP =========================================================

class HistogramOfPhase:
    """
    A partir de un algoritmo hecho en MATLAB por un investigador devuelve HOP y la congruencia de fase
    
    Ejemplo:
        img = cv.imread("Frame.jpg")
        hop = Metodos.HistogramOfPhase(False, False)
        [feat, pc] = hop(img)
    """
    
    def __init__(self, plotear, resize):
        self._plotear = plotear
        self._resize = resize

    def __call__(self, imagen):
        image = np.copy(imagen)

        # Se convierte a escala de grises por si no lo está
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Convierto de tipo np a uint8 de matlab para poder ser pasado a la librería
        image = matlab.uint8(image.tolist())

        hop = histogramofphase.initialize()
        # Devuelve los histogramas de HOP concatenados y la congruencia de fase
        features, pc = hop.mainHOP(image, self._resize, nargout=2)
        hop.terminate()

        # print(pc.size)
        # print(features)
        if self._plotear:
            array_pc = np.array(pc)
            imagesc.clean(array_pc)
        # Solo retorno las características
        return features


# ================================================= ELIMINA SILENCIOS ==================================================

class EliminaSilencios:
    """
    A través de los cruces por cero y energía calcula los silencios. Devuelve el audio con igual longitud que el 
    original pero con valor 0 donde se presentan silencios según lo calculado. Lee los wav desde la carpeta Procesado, 
    que ya debería haberse generado con FFMPEG.
    
    Ejemplo:
        ruta_bd = 'D:/Descargas/Proyecto Final de Carrera/Bd propia/Conejitos de india'
        el_si = Metodos.EliminaSilencios(ruta_bd, True)
        persona = 'Sujeto 01a'
        parte = 'a'
        el_si(persona, parte)
    """
    def __init__(self, plotear):
        # tam_ventana = 0.2
        self._plotear = plotear

    def __call__(self, path, tam_ventana=0.125, umbral=0.008):
        # Extraigo la posición donde está la extensión para después eliminarla en el nombre del archivo de salida
        nro_extension = path.index('.wav')

        # Lectura, casteo y extracción de características básicas de la señal
        muestreo, sonido = waves.read(path)
        sonido = sonido.astype(np.int64)
        nro_muestras = sonido.shape[0]
        canales = sonido.shape[1]

        # Inicializaciones
        nro_iteraciones = int(nro_muestras / (tam_ventana * muestreo))
        energia = np.zeros([nro_iteraciones, canales])
        cruces = np.zeros([nro_iteraciones, canales])

        # Cálculo de las energías y cruces por cero en todas las ventanas de todos los canales de sonido
        for i in range(0, canales):
            energia[:, i], cruces[:, i] = self._calcula_caract(sonido[:, i], nro_muestras, tam_ventana, muestreo)

        # Devolución del vector de booleanos, indicando con verdadero cada porción que sea audible
        vector_audible = self._metodo1(energia, cruces, umbral, nro_muestras, tam_ventana, muestreo, canales)

        if self._plotear:
            plt.plot(sonido)
            plt.plot(vector_audible * np.max(sonido), 'r')
            plt.show()

        # Guardo los rangos donde se encuentran los segmentos audibles como tuplas
        # Aprovecho también para cortar el audio y guardarlos en wavs por separado
        activo = False
        rangos_audibles = np.empty((0, 2))
        # Recorto el nombre para borrarle la extensión y que no me quede como parte de los archivos de salida
        path = path[0:nro_extension]
        for i in range(0, vector_audible.size):
            # Si es audible y no estaba activada la bandera, comienza un tramo
            if vector_audible[i] is True and activo is False:
                activo = True
                comienzo = i
            # Si estaba activo el rango y deja de ser audible, o si sigue siendo audible pero llego al final del vector,
            # guardo el comienzo y el fin del rango
            elif (vector_audible[i] is False and activo is True) or \
                    (vector_audible[i] is True and i == vector_audible.size - 1):
                activo = False
                rangos_audibles = np.append(rangos_audibles, np.array([np.array([comienzo, i])]), axis=0)
                # Según la cantidad de canales recorto uno solo o los recorto por separado para luego unirlos
                if canales == 1:
                    recortado = sonido[comienzo:i, 0]
                else:
                    # aux1 = sonido[comienzo:i, 0]
                    # aux2 = sonido[comienzo:i, 1]
                    # recortado = np.empty((0, i-comienzo))
                    # recortado = np.append(recortado, np.array([aux1]), axis=0)
                    # recortado = np.append(recortado, np.array([aux2]), axis=0)
                    recortado = sonido[comienzo:i]
                # Guardo el wav
                recortado = recortado.astype(np.int16)
                waves.write(path + '_' + str(rangos_audibles.shape[0]) + '.wav', muestreo, recortado)

        # Como solo me interesa devolver los porcentajes del total en los segmentos para recortar el video
        rangos_audibles = rangos_audibles / nro_muestras

        return rangos_audibles

    def _calcula_caract(self, sonido, nro_muestras, tam_ventana, muestreo):
        nro_iteraciones = int(nro_muestras / (tam_ventana * muestreo))
        energia = np.zeros([nro_iteraciones])
        cruces = np.zeros([nro_iteraciones])

        # Calcula la energía y cruces en todas las ventanas
        for i in range(0, nro_iteraciones):
            inicia = int(i * tam_ventana * muestreo)
            termina = int(inicia + tam_ventana * muestreo)
            energia[i] = self._energy(sonido[inicia:termina])
            cruces[i] = self._cruces_por_cero(sonido[inicia:termina])
        return energia, cruces

    @staticmethod
    def _metodo1(energia, cruces, umbral, nro_muestras, tam_ventana, muestreo, canales):
        """ 
        El método se basa en calcular un puntaje con todas las características para cada muestra y compararlo con un
        umbral. En caso de ser mayor que el umbral se descarta. Los puntajes por muestra se calculan en base a la
        energía y los cruces por cero. Al tener múltiples canales para cada número de muestra se promedia el puntaje de
        todos los canales en ese número de muestra.
        """

        nro_iteraciones = int(nro_muestras / (tam_ventana * muestreo))
        puntajes = np.zeros([nro_iteraciones, canales])
        vector = np.zeros([nro_muestras])

        for j in range(0, canales):
            # Normaliza los valores de energía y cruces por cero
            energia[:, j] = energia[:, j] / max(energia[:, j])
            cruces[:, j] = cruces[:, j] / max(cruces[:, j])
            # Calcula puntajes según: Energía * (1 - Cruces por cero)
            puntajes[:, j] = energia[:, j] * (1 - cruces[:, j])
            # Normaliza los puntajes
            puntajes[:, j] = puntajes[:, j] / max(puntajes[:, j])

        # Según un umbral calculado empiricamente se diferencia las ventanas de silencio
        for i in range(0, nro_iteraciones):
            # Combinación: tomo las puntuaciones de cada canal y las promedio para asi calcular directamente el vector
            # final
            prom = 0
            for j in range(0, canales):
                prom = prom + puntajes[i, j]
            prom = prom / (canales + 1)
            if prom > umbral:
                inicia = int(i * tam_ventana * muestreo)
                termina = int(inicia + tam_ventana * muestreo)
                vector[inicia:termina] = np.ones([int(tam_ventana * muestreo)])
        return vector

    @staticmethod
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

    @staticmethod
    def _cruces_por_cero(data):
        """ 
        Calcula los cruces por cero. Hay voz si la cantidad de cruces por cero son bajas 
        """
        tam = len(data)
        cantidad = 0
        for i in range(0, tam - 1):
            if data[i] * data[i + 1] < 0:
                cantidad = cantidad + 1
        rate = cantidad / tam
        return rate

    @staticmethod
    def _energy(data):
        return sum(data * data) / len(data)


# ====================================================== FFMPEG ========================================================

class FFMPEG:
    def __init__(self):
        return

    def __call__(self, persona, etapa, parte):
        # archivo = 'Sujeto 01'
        # parte = '1'
        nombre_video = datos.buildVideoName(persona, etapa, parte, extension=False)

        # Comando base
        comando = ['.' + os.sep + 'ffmpeg', '-y', '-i', datos.buildPathVideo(persona, etapa, nombre_video, extension=True),
                   '-ab', '195k', '-ac', '2', '-ar', '48000', '-vn',
                   os.path.join(datos.PATH_PROCESADO, nombre_video + '.wav')]
        # comando = ['.' + os.sep + 'ffmpeg', '-version']

        # Cambio al directorio de ffmpeg, ejecuto el comando y luego vuelvo al directorio donde están los códigos
        os.chdir(datos.PATH_FFMPEG)
        subprocess.run(comando, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        os.chdir(datos.PATH_CODIGOS)
