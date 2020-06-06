import Codigos.Datos as datos
import os
import datetime


def crea():
    # Fecha y hora
    x = datetime.datetime.now()
    nombre = str(x.day) + '-' + str(x.month) + '-' + str(x.year) + ' ' + str(x.hour) + '_' + str(x.minute)
    datos.defineCarpetaLog(nombre)
    try:
        os.stat(datos.PATH_LOGS)
    except (ValueError, Exception):
        os.mkdir(datos.PATH_LOGS)
    archivo = open(os.path.join(datos.PATH_LOGS, 'log.txt'), 'w', encoding="utf-8")
    archivo.write('Experimento: ' + datos.EXPERIMENTO + '\n\n')

    if datos.OUT == -1:
        archivo.write('Leave-X-out inactivo' + '\n')
    else:
        archivo.write('Leave-' + str(datos.OUT) + ' -out' + '\n')
    archivo.write('Binarización de etiquetas: ' + str(datos.BINARIZO_ETIQUETA) + '\n')
    archivo.write('Eliminación de silencios: ' + str(datos.ELIMINA_SILENCIOS) + '\n')

    archivo.write('Instancias por periodos: ' + str(datos.INSTANCIAS_POR_PERIODOS) + '\n')
    archivo.write('Votos entre mejores ' + str(datos.VOTO_MEJORES_X) + '\n')
    archivo.write('Atributos seleccionados PCA: ' + str(datos.ATRIBS_PCA) + '\n')
    archivo.write('Atributos seleccionados PSO: ' + str(datos.ATRIBS_PSO) + '\n')
    archivo.write('Atributos seleccionados BF: ' + str(datos.ATRIBS_BF) + '\n')
    archivo.write('Atributos finales: ' + str(datos.ATRIBS_FINALES) + '\n')
    archivo.write('Tiempo de microexpresion(ms): ' + str(datos.TIEMPO_MICROEXPRESION) + '\n\n')

    archivo.write('Personas: ' + '\n')
    for i in range(0, datos.PERSONAS.size - 1):
        archivo.write(datos.PERSONAS[i] + '-')
    archivo.write(datos.PERSONAS[datos.PERSONAS.size - 1] + '\n\n')

    archivo.write('Etapas: ' + '\n')
    for i in range(0, datos.ETAPAS.size):
        archivo.write(datos.ETAPAS[i] + ' ')
    archivo.write('\n\n')

    switcher_zonas = {
        'ojoizq': 'Ojo izquierdo',
        'ojoder': 'Ojo derecho',
        'cejaizq': 'Ceja izquierdo',
        'cejader': 'Ceja derecha',
        'boca': 'Boca',
        'nariz': 'Nariz',
        'cejas': 'Cejas',
        'ojos': 'Ojos'
    }

    archivo.write('Zonas: ' + '\n')
    for i in range(0, datos.ZONAS.size):
        archivo.write('..' + switcher_zonas.get(datos.ZONAS[i]) + '\n')
    archivo.write('\n')

    archivo.write('Métodos de extraccion de características' + '\n')
    for i in range(0, datos.MET_EXTRACCION.size):
        if datos.MET_EXTRACCION[i] == '':
            archivo.write('..' + 'Sin selección (solo para métodos basados en árboles)' + '\n')
        else:
            archivo.write('..' + datos.MET_EXTRACCION[i] + '\n')
    archivo.write('\n')

    archivo.write('Métodos de seleccion de características' + '\n')
    for i in range(0, datos.MET_SELECCION.size):
        archivo.write('..' + datos.MET_SELECCION[i] + '\n')
    archivo.write('\n')

    archivo.write('Clasificadores' + '\n')
    for i in range(0, datos.MET_CLASIFICACION.size):
        archivo.write('..' + datos.MET_CLASIFICACION[i] + '\n')
    archivo.write('\n')

    archivo.write('\n')
    archivo.write('--------------------------------------------------------------------------------------------------')
    archivo.write('\n')
    archivo.close()


def agrega(dato):
    archivo = open(os.path.join(datos.PATH_LOGS, 'log.txt'), 'a+', encoding="utf-8")
    archivo.write(str(dato) + '\n')
    archivo.close()
