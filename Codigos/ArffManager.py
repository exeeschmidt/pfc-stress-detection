import os
import numpy as np


# Ejemplo
# import ArffManager as am
# import numpy as np
# from random import randrange
#
# n_lbp = 10
# n_hop = 5
# n_au = 7
#
# lbp = np.zeros(n_lbp)
# hop = np.zeros(n_hop)
# au = np.zeros(n_au)
#
# for i in range(0, n_lbp):
#     lbp[i] = randrange(256)
# for i in range(0, n_hop):
#     hop[i] = randrange(100) / 100
# for i in range(0, n_au):
#     au[i] = randrange(5)
#
# am.CrearCabeceraArff('Prueba', n_lbp, n_hop, n_au)
# am.AgregarFilaArff('Prueba', lbp, hop, au, 'Estresado')

def CabeceraArff(nombre, lbp_range, hop_range, hog_range, au_range, clases):
    # Se ingresan el largo de cada caracteristica (las columnas)
    # Tiene en cuenta dos clases : Estresado y No-Estresado

    # Crea el archivo si no existe, modo escritura
    file = open('Caracteristicas' + os.sep + nombre + '.arff', 'w')
    # La primer linea no se para que sirve pero lo vi en otros arff
    file.write('@relation VideoFeatures' + os.linesep)

    for i in range(0, lbp_range):
        file.write('@attribute lbp_hist[' + str(i + 1) + '] numeric' + '\n')

    for i in range(0, hop_range):
        file.write('@attribute hop_hist[' + str(i + 1) + '] numeric' + '\n')

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
    # Se ingresan las caracteristicas extraidas de un cuadro y la etiqueta de clase (Estresado o No-Estresado)

    # Abro el archivo con cabecera, la bandera 'a' permite anexar el texto
    file = open('Caracteristicas' + os.sep + nombre + '.arff', 'a')

    # Fila de caracteristicas
    fila = ''

    # Extraigo la cantidad de vector
    lbp_range = np.size(lbp_feat)
    hop_range = np.size(hop_feat)
    hog_range = np.size(hog_feat)[0]
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


def ConcatenaArff(nombre_salida, sujetos, etapas, bool_partes):
    # Los primeros dos parametros tienen que ser np.array de numeros, el tercero un booleano si se unen partes o no

    # Creo el archivo que va a ser la salida de la concatenacion
    salida = open('Caracteristicas' + os.sep + nombre_salida + '.arff', 'w')

    # Cambio la ruta del primer archivo a leer segun si considero o no las partes
    if bool_partes:
        path = 'Caracteristicas' + os.sep + 'Sujeto_' + str(sujetos[0]) + '_' + str(etapas[0]) + '_r1'
    else:
        path = 'Caracteristicas' + os.sep + 'Sujeto_' + str(sujetos[0]) + '_' + str(etapas[0])

    # Tomo el primer archivo para crear la cabecera
    archivo = open(path + '.arff', 'r')
    linea = archivo.readline()
    # Al encontrar @data paro, pero tengo que guardar igual esta linea y otra mas en blanco antes de los datos puros
    while linea != '@data\n':
        salida.write(linea)
        linea = archivo.readline()
    salida.write(linea)
    linea = archivo.readline()
    salida.write(linea)
    # Guardo la posicion en bytes de donde termina la cabecera, al tener todos los archivos la misma cabecera comienzo
    # leyendo siempre del mismo lugar
    data_pos = archivo.tell()
    archivo.close()

    for i in sujetos:
        for j in etapas:
            if bool_partes:
                partes = 7
                if j == '2':
                    partes = 6
                for k in range(1, partes + 1):
                    print('Concatenando parte: ', k)
                    archivo = open(
                        'Caracteristicas' + os.sep + 'Sujeto_' + str(i) + '_' + str(j) + '_r' + str(k) + '.arff', 'r')
                    # Salto donde termina la cabecera y comienzan los datos
                    archivo.seek(data_pos, 0)
                    linea = archivo.readline()
                    # Cuando termina el archivo linea devuelve ""
                    while linea != "":
                        salida.write(linea)
                        linea = archivo.readline()
                    archivo.close()
            else:
                archivo = open('Caracteristicas' + os.sep + 'Sujeto_' + str(i) + '_' + str(j) + '.arff', 'r')
                archivo.seek(data_pos, 0)
                linea = archivo.readline()
                # Cuando termina el archivo linea devuelve ""
                while linea != "":
                    salida.write(linea)
                    linea = archivo.readline()
                archivo.close()
    salida.close()