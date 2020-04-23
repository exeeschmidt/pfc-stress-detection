import os
import numpy as np
import Codigos.Datos as datos


# Ejemplo:
#   import ArffManager as am
#   import numpy as np
#   from random import randrange
#
#   n_lbp = 10
#   n_hop = 5
#   n_au = 7
#
#   lbp = np.zeros(n_lbp)
#   hop = np.zeros(n_hop)
#   au = np.zeros(n_au)
#
#   for i in range(0, n_lbp):
#     lbp[i] = randrange(256)
#   for i in range(0, n_hop):
#     hop[i] = randrange(100) / 100
#   for i in range(0, n_au):
#     au[i] = randrange(5)
#
#   clases = np.array(['Estresado','No-Estresado'])
#
#   am.CrearCabeceraArff('Prueba', n_lbp, n_hop, n_au, clases)
#   am.AgregarFilaArff('Prueba', lbp, hop, au, 'Estresado')

def CabeceraArff(nombre, lbp_range, hop_range, hog_range, au_range, clases, zonas):
    """
    Se ingresa el largo de cada característica (las columnas).
    Tiene en cuenta dos clases: Estresado y No-Estresado.
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

    for i in range(0, hog_range):
        file.write('@attribute hog_hist[' + str(i + 1) + '] numeric' + '\n')

    for i in range(0, au_range):
        file.write('@attribute au_intensity[' + str(i + 1) + '] numeric' + '\n')

    linea_clase = '@attribute class {' + clases[0]
    # file.write('@attribute class {Estresado, No-Estresado}' + os.linesep)
    for i in range(1, len(clases)):
        linea_clase = linea_clase + ', ' + clases[i]
    linea_clase = linea_clase + '}' + os.linesep

    file.write(linea_clase)
    file.write('@data' + os.linesep)


def FilaArff(nombre, lbp_feat, hop_feat, hog_feat, au_feat, etiqueta):
    """
    Se ingresan las características extraídas de un cuadro y la etiqueta de clase (Estresado o No-Estresado).
    """

    # Abro el archivo con cabecera, la bandera 'a' permite anexar el texto
    file = open(os.path.join(datos.PATH_CARACTERISTICAS, nombre + '.arff'), 'a')

    # Fila de características
    fila = ''

    # Extraigo el largo de cada vector
    lbp_range = np.size(lbp_feat)
    hop_range = np.size(hop_feat)
    hog_range = len(hog_feat)
    # hog_range = np.size(hog_feat)[0]
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


def ConcatenaArff(nombre_salida, sujetos, etapas, bool_partes=True, bool_audio=False, rangos_audibles=None):
    """
    Los primeros dos parametros tienen que ser np.array de números.
    """

    if rangos_audibles is None:
        bool_audible = False
    else:
        bool_audible = True

    extension = '.arff'
    if bool_audio:
        extension = '.wav.arff'
        bool_partes = True

    # Creo el archivo que va a ser la salida de la concatenación
    salida = open(os.path.join(datos.PATH_CARACTERISTICAS, nombre_salida + '.arff'), 'w')

    # Cambio la ruta del primer archivo a leer según si considero o no las partes
    if bool_partes:
        path = os.path.join(datos.PATH_CARACTERISTICAS, datos.buildVideoName(str(sujetos[0]), str(etapas[0]), 1))
    else:
        path = os.path.join(datos.PATH_CARACTERISTICAS, datos.buildVideoName(str(sujetos[0]), str(etapas[0])))

    # Tomo el primer archivo para crear la cabecera
    archivo = open(path + extension, 'r')
    linea = archivo.readline()
    # Al encontrar @data paro, pero tengo que guardar igual esta línea y otra más en blanco antes de los datos puros
    while linea != '@data\n':
        salida.write(linea)
        linea = archivo.readline()
    salida.write(linea)
    linea = archivo.readline()
    salida.write(linea)
    # Guardo la posición en bytes de donde termina la cabecera, al tener todos los archivos la misma cabecera comienzo
    # leyendo siempre del mismo lugar
    data_pos = archivo.tell()
    archivo.close()

    for i in sujetos:
        for j in etapas:
            # Las partes serían si se dividen en respuestas, las subpartes en caso de eliminar los silencios donde cada
            # respuesta a su vez se vuelve a segmentar
            partes = 1
            subpartes = 1
            if bool_partes:
                partes = 7
                if j == '2':
                    partes = 6

            for k in range(0, partes):
                # print('Concatenando parte: ', k)
                base = os.path.join(datos.PATH_CARACTERISTICAS, datos.buildVideoName(str(i), str(j)))

                if bool_audible:
                    subpartes = rangos_audibles[k].shape[0]

                for n in range(0, subpartes):
                    if bool_partes:
                        parte_path = '_r' + str(k + 1)
                        if bool_audible:
                            subpartes = rangos_audibles[k].shape[0]
                            parte_path = parte_path + '_' + str(n + 1)
                    else:
                        parte_path = ''
                    archivo = open(base + parte_path + extension, 'r')
                    # Salto donde termina la cabecera y comienzan los datos
                    archivo.seek(data_pos, 0)
                    linea = archivo.readline()
                    # Cuando termina el archivo linea devuelve ""
                    while linea != "":
                        salida.write(linea)
                        linea = archivo.readline()
                    archivo.close()
    salida.close()


def AgregaEtiqueta(nombre, clases, etiqueta):
    # Abro el archivo para lectura y escritura
    archivo = open(os.path.join(datos.PATH_CARACTERISTICAS, nombre + '.arff'), 'r+')

    # Creo la línea como deberían ser las clases
    linea_clases = '@attribute class {' + clases[0]
    for i in range(1, len(clases)):
        linea_clases = linea_clases + ',' + clases[i]
    linea_clases = linea_clases + '}'

    # Recorro todas las líneas del archivo
    lineas = archivo.readlines()
    ind = 0
    for linea in lineas:
        # Si encuentro la línea donde está definida el atributo clase, la reemplazo por la línea creada antes
        if linea == '@attribute class numeric\n':
            lineas[ind] = linea_clases + '\n'
        # Busco las líneas de datos (no están en blanco y no tienen el '@' de atributo), corto las últimas 3 (?\n)
        # y agrego la etiqueta más el salto nuevamente
        elif linea[0] != '\n' and linea[0] != '@':
            lineas[ind] = linea[0:len(linea)-2] + etiqueta + '\n'
        ind = ind + 1
    # Llevo el puntero al principio y escribo las líneas ya modificadas
    archivo.seek(0)
    archivo.writelines(lineas)
    archivo.close()
