import os
import numpy as np
import Codigos.Datos as datos

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


def MezclaInstanciasArff(nombre_archivo, orden):
    arch = open(os.path.join(datos.PATH_CARACTERISTICAS, nombre_archivo + '.arff'), 'r+')

    pos_data = 2
    linea = arch.readline()
    # Busco donde comienza data recien
    while linea[0:5] != '@data':
        linea = arch.readline()
        # Esta variable guarda en que numero de linea empiezan los datoa
        pos_data = pos_data + 1
    arch.readline()

    arch.seek(0)
    # Leo todas las lineas para usarlo como vector
    lineas = arch.readlines()
    # Extraigo las correspondientes a los datos
    lineas_datos = np.array(lineas[pos_data:len(lineas)])
    # A la ultima linea actual, que no tiene salto de carro, le agrego por si deja de quedar ultima
    # lineas_datos[len(lineas_datos) - 1] = lineas_datos[len(lineas_datos) - 1] + '\n'
    # Mezclo las lineas
    lineas_datos = lineas_datos[orden]
    # Uno al resto de las lineas las nuevas instancias mezcladas
    lineas = lineas[0:pos_data] + list(lineas_datos)
    # A la ultima nueva linea le extraigo el salto de carro
    ultimo_indice = len(lineas)-1
    lineas[ultimo_indice] = lineas[ultimo_indice][0:len(lineas[ultimo_indice])-1]
    # Limpio el archivo, voy al inicio y escribo las lineas nuevas
    arch.truncate(0)
    arch.seek(0)
    arch.writelines(lineas)
    arch.close()