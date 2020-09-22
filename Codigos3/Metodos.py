# import histogramoforientedphase
# import matlab
import imagesc
import os
import cv2 as cv
import numpy as np
import subprocess
import Datos
import Herramientas as Hrm

# Contiene todos los métodos a utilizar, donde las implementaciones de cada uno comienzan con un bookmark.

# Nota importante:
#   Al ejecutar por primera vez siempre hacerlo con OpenFace y luego OpenSmile, explicado en el apartado de OpenSmile


# ===================================================== OPEN FACE ======================================================

class OpenFace:
    """
    Extrae las carasterísticas a partir de OpenFace, guardando estos en una carpeta "Procesado" dentro del proyecto. 
    Toma los videos de la ruta de la base de datos.
    """
    
    def __init__(self, face, hog, landmarks, aus):
        # Los 4 parámetros son banderas para extraer características
        self.face = face
        self.hog = hog
        self.landmarks = landmarks
        self.aus = aus

    def __call__(self, video_path):
        save_path = Datos.PATH_PROCESADO

        # Comando base para ejecutar OpenFace
        command = ['FeatureExtraction.exe', '-f', video_path, '-out_dir', save_path]

        # Según las banderas se le agregan parámetros al comando
        if self.face:
            command.append('-simalign')
        if self.hog:
            command.append('-hogalign')
        if self.landmarks:
            command.append('-2Dfp')
        if self.aus:
            command.append('-aus')

        # Cambio al directorio de OpenFace, ejecuto el comando y luego vuelvo al directorio donde están los códigos
        os.chdir(Datos.PATH_OPENFACE)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # results = subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # print("The exit code was: %d" % results.returncode)
        os.chdir(Datos.PATH_CODIGOS)


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
    """
    
    def __init__(self, window):
        # Bandera para definir si se ventanea. Si es True deben incluirse los parametros adicionales en el __call__
        # Si no se llama el ventaneo se define cada 0.5s y el shift inicial en 0
        self.window = window

    def __call__(self, audio_name, audio_path, window_size='0.125'):
        # Comando base de OpenSmile
        command = ['SMILExtract_Release', '-C', Datos.PATH_CONFIG_FILE, '-I', audio_path, '-appendarff', '0',
                   '-output', Hrm.buildOpenSmileFilePath(audio_name)]

        # En caso de ventaneo se utiliza el config_file que se permite escribir desde la función archivo_ventaneo
        if self.window:
            window_conf_path = os.path.join('config', 'shared', 'FrameModeFunctionalsVentana.conf.inc')
            self.createWindowFile(os.path.join(Datos.PATH_OPENSMILE, window_conf_path), window_size)
            command.append('-frameModeFunctionalsConf')
            # En caso de modificar el paso y el solapamiento modificar ese archivo
            command.append(window_conf_path)

        # Cambio al directorio de OpenSmile, ejecuto el comando y luego vuelvo al directorio donde están los códigos
        os.chdir(Datos.PATH_OPENSMILE)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # results = subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # print("The exit code was: %d" % results.returncode)
        os.chdir(Datos.PATH_CODIGOS)

    @staticmethod
    def createWindowFile(path, paso):
        f = open(path, 'w')
        f.write('frameMode = fixed\n')
        f.write('frameSize = ' + paso + '\n')
        f.write('frameStep = 0' + '\n')
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
-lldarfftargetsfile <file> Specify the configuration include, that defines the target fields (classes) Default:
                                                                                    shared/arff_targets_conf.inc
-----------------------------
-output, -O <filename> The default output option. To ARFF file format, for feature summaries.
-appendarff <0/1> Set to 0 to not append to existing ARFF output file. Default is append (1).
-timestamparff <0/1> Set to 1 to enable timestamp output to ARFF in second column. Default 0.
-arfftargetsfile <file> Specify the configuration include, that defines the target fields (classes) Default: 
                                                                                    shared/arff_targets_conf.inc
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

    def __call__(self, x):
        raise NotImplementedError("Every LBPOperator must implement the __call__ method.")

    @property
    def neighbors(self):
        return self._neighbors

    def __repr__(self):
        return "LBPOperator (neighbors=%s)" % self._neighbors


class OriginalLBP(LocalDescriptor):
    def __init__(self):
        LocalDescriptor.__init__(self, neighbors=8)

    def __call__(self, x):
        x = np.asarray(x)
        x = (1 << 7) * (x[0:-2, 0:-2] >= x[1:-1, 1:-1]) \
            + (1 << 6) * (x[0:-2, 1:-1] >= x[1:-1, 1:-1]) \
            + (1 << 5) * (x[0:-2, 2:] >= x[1:-1, 1:-1]) \
            + (1 << 4) * (x[1:-1, 2:] >= x[1:-1, 1:-1]) \
            + (1 << 3) * (x[2:, 2:] >= x[1:-1, 1:-1]) \
            + (1 << 2) * (x[2:, 1:-1] >= x[1:-1, 1:-1]) \
            + (1 << 1) * (x[2:, :-2] >= x[1:-1, 1:-1]) \
            + (1 << 0) * (x[1:-1, :-2] >= x[1:-1, 1:-1])
        return x

    def __repr__(self):
        return "OriginalLBP (neighbors=%s)" % self._neighbors


# ======================================================== HOP =========================================================

class HistogramOfPhase:
    """
    A partir de un algoritmo hecho en MATLAB por un investigador devuelve HOP y la congruencia de fase

    Ejemplo:
        img = cv.imread("Frame.jpg")
        hop = Metodos.HistogramOfPhase(False, False)
        [feat, pc] = hop(img)
    """

    def __init__(self, plot, resize):
        self.plot = plot
        self.resize = resize

    def __call__(self, image):
        # img = np.copy(image)
        #
        # # Se convierte a escala de grises por si no lo está
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #
        # # Convierto de tipo np a uint8 de matlab para poder ser pasado a la librería
        # img = matlab.uint8(img.tolist())
        #
        # hop = histogramoforientedphase.initialize()
        # # Devuelve los histogramas de HOP concatenados y la congruencia de fase
        # features, pc = hop.execute(img, False, nargout=2)
        # hop.terminate()
        #
        # # print(pc.size)
        # # print(features)
        # if self.plot:
        #     array_pc = np.array(pc)
        #     imagesc.clean(array_pc)
        # # Solo retorno las características
        # return features
        return None

# ====================================================== FFMPEG ========================================================


class FFMPEG:
    def __init__(self):
        return

    def __call__(self, video_name, video_path):
        # Comando base
        command = ['.' + os.sep + 'ffmpeg', '-y', '-i', video_path,
                   '-ab', '195k', '-ac', '2', '-ar', '48000', '-vn',
                   Hrm.buildOutputPathFFMPEG(video_name)]
        # comando = ['.' + os.sep + 'ffmpeg', '-version']

        # Cambio al directorio de ffmpeg, ejecuto el comando y luego vuelvo al directorio donde están los códigos
        os.chdir(Datos.PATH_FFMPEG)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # results = subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # print("The exit code was: %d" % results.returncode)
        os.chdir(Datos.PATH_CODIGOS)

    def lowFpsVideo(self, video_name, video_path):
        output_path = os.path.join(Datos.PATH_PROCESADO, video_name + Datos.EXTENSION_VIDEO)
        command = ['.' + os.sep + 'ffmpeg', '-y', '-i', video_path, '-r', str(Datos.LIMITE_FPS),
                   '-c:v', 'libx264', '-b:v', '3M', '-strict', '-2', '-movflags', 'faststart', output_path]
        # Cambio al directorio de ffmpeg, ejecuto el comando y luego vuelvo al directorio donde están los códigos
        os.chdir(Datos.PATH_FFMPEG)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # results = subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # print("The exit code was: %d" % results.returncode)
        os.chdir(Datos.PATH_CODIGOS)
        return output_path