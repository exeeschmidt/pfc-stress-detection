import Datos
import os
import datetime


def create():
    # Fecha y hora
    x = datetime.datetime.now()
    folder_name = '{0}-{1}-{2} {3}_{4} {5}'.format(str(x.day), str(x.month), str(x.year), str(x.hour), str(x.minute),
                                                   str(x.microsecond))
    Datos.defineLogFolder(folder_name)
    try:
        os.stat(Datos.PATH_LOGS)
    except (ValueError, Exception):
        os.makedirs(Datos.PATH_LOGS)
    tables_file = open(os.path.join(Datos.PATH_LOGS, 'tablas.txt'), 'w', encoding="utf-8")
    log_file = open(os.path.join(Datos.PATH_LOGS, 'log.txt'), 'w', encoding="utf-8")
    log_file.write('Experimento: ' + Datos.EXPERIMENTO + '\n\n')

    if Datos.TEST == -1:
        log_file.write('Leave-X-out inactivo' + '\n')
    else:
        log_file.write('Leave-' + str(Datos.TEST) + ' -out' + '\n')
        log_file.write('Validación con ' + str(Datos.VAL) + ' personas' + '\n')
    log_file.write('Binarización de etiquetas: ' + str(Datos.BINARIZO_ETIQUETA) + '\n')
    log_file.write('Eliminación de silencios: ' + str(Datos.ELIMINA_SILENCIOS) + '\n')

    log_file.write('Instancias por periodos: ' + str(Datos.INSTANCIAS_POR_PERIODOS) + '\n')
    log_file.write('Votos entre mejores ' + str(Datos.VOTO_MEJORES_X) + '\n')
    log_file.write('Porcentaje de atributos seleccionados PCA: ' + str(Datos.PORC_ATRIBS_PCA) + '\n')
    log_file.write('Porcentaje de atributos seleccionados PSO: ' + str(Datos.PORC_ATRIBS_PSO) + '\n')
    log_file.write('Porcentaje de atributos seleccionados BF: ' + str(Datos.PORC_ATRIBS_BF) + '\n')
    log_file.write('Porcentaje de atributos finales: ' + str(Datos.PORC_ATRIBS_FINALES) + '\n')
    log_file.write('Tiempo de microexpresion(s): ' + str(Datos.TIEMPO_MICROEXPRESION) + '\n\n')

    log_file.write('Personas: ' + '\n')
    for i in range(0, Datos.PERSONAS.size - 1):
        log_file.write(Datos.PERSONAS[i] + '-')
    log_file.write(Datos.PERSONAS[Datos.PERSONAS.size - 1] + '\n\n')

    log_file.write('Etapas: ' + '\n')
    for i in range(0, Datos.ETAPAS.size):
        log_file.write(Datos.ETAPAS[i] + ' ')
    log_file.write('\n\n')

    zones_translate = {
        'ojoizq': 'Ojo izquierdo',
        'ojoder': 'Ojo derecho',
        'cejaizq': 'Ceja izquierdo',
        'cejader': 'Ceja derecha',
        'boca': 'Boca',
        'nariz': 'Nariz',
        'cejas': 'Cejas',
        'ojos': 'Ojos'
    }

    log_file.write('Zonas: ' + '\n')
    for i in range(0, Datos.ZONAS.size):
        log_file.write('..' + zones_translate.get(Datos.ZONAS[i]) + '\n')
    log_file.write('\n')

    log_file.write('Métodos de extraccion de características' + '\n')
    for i in range(0, Datos.MET_EXTRACCION.size):
        if Datos.MET_EXTRACCION[i] == '':
            log_file.write('..' + 'Sin selección (solo para métodos basados en árboles)' + '\n')
        else:
            log_file.write('..' + Datos.MET_EXTRACCION[i] + '\n')
    log_file.write('\n')

    log_file.write('Métodos de seleccion de características' + '\n')
    for i in range(0, Datos.MET_SELECCION.size):
        log_file.write('..' + Datos.MET_SELECCION[i] + '\n')
    log_file.write('\n')

    log_file.write('Clasificadores' + '\n')
    for i in range(0, Datos.MET_CLASIFICACION.size):
        log_file.write('..' + Datos.MET_CLASIFICACION[i] + '\n')
    log_file.write('\n')

    log_file.write('\n')
    log_file.write('--------------------------------------------------------------------------------------------------')
    log_file.write('\n')
    log_file.close()
    tables_file.write('Tablas de resultados')
    tables_file.write('\n')
    tables_file.write(
        '--------------------------------------------------------------------------------------------------')
    tables_file.write('\n')


def add(data):
    file = open(os.path.join(Datos.PATH_LOGS, 'log.txt'), 'a+', encoding="utf-8")
    file.write(str(data) + '\n')
    file.close()


def addToTable(data):
    file = open(os.path.join(Datos.PATH_LOGS, 'tablas.txt'), 'a+', encoding="utf-8")
    file.write(str(data) + '\n')
    file.close()
