import os
import numpy as np
import Codigos.Datos as datos
from weka.core.converters import Loader, Saver
from weka.filters import Filter
from weka.core.dataset import Instances, Attribute, Instance
import Codigos.Herramientas as hrm


def CargaYFiltrado(path):
    # Cargo los datos
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file(path)

    # Creo un filtro para eliminar atributos de tipo string en caso que se presenten
    remove = Filter(classname="weka.filters.unsupervised.attribute.RemoveType", options=["-T", "string"])
    remove.inputformat(data)
    data = remove.filter(data)

    # Indico que la clase es el último atributo
    data.class_is_last()
    return data


def Guarda(persona, etapa, sub, data):
    path = hrm.buildPathSub(persona, etapa, sub)
    saver = Saver()
    saver.save_file(data, path)


def Guardav2(path, data):
    saver = Saver()
    saver.save_file(data, path)


def Une(data_vec):
    # Une varios datasets que pueden tener o no diferentes atributos pero igual numero de instancias
    data = Instances.copy_instances(data_vec[0])
    for i in range(1, data_vec.size):
        data = Instances.merge_instances(data, data_vec[i])
    return data


def Unev2(data_vec):
    # Une varios datasets con los mismos atributos pero con distinto numero de instancias
    data = Instances.copy_instances(data_vec[0])
    for i in range(1, data_vec.size):
        data = Instances.append_instances(data, data_vec[i])
    return data


def Concatena(personas, etapas, sub, sub2=''):
    # Levanta y une los dataset de multiples personas y etapas
    # Sub cambia segun el conjunto de caracteristicas, si se presenta sub2 es por si hay que concatenar el audio y video
    # entre sí también
    data_vec = np.empty(0)
    data_vec2 = np.empty(0)
    for i in personas:
        for j in etapas:
            path = hrm.buildPathSub(i, j, sub)
            data = CargaYFiltrado(path)
            data_vec = np.append(data_vec, data)
            if sub2 != '':
                path = hrm.buildPathSub(i, j, sub2)
                data = CargaYFiltrado(path)
                data_vec2 = np.append(data_vec2, data)
    if sub2 != '':
        data_sub1 = Unev2(data_vec)
        data_sub2 = Unev2(data_vec2)
        data_vec_f = Normaliza(np.array([data_sub1, data_sub2]))
        data_final = Une(data_vec_f)
    else:
        data_final = Unev2(data_vec)
    return data_final


def Normaliza(data_vec):
    # Deja todos los dataset presentes en el vector con el numero de instancias del menor
    instancias = np.empty(0)
    for i in range(0, data_vec.size):
        instancias = np.append(instancias, NroInstancias(data_vec[i]))

    for i in range(0, data_vec.size):
        if instancias[i] > min(instancias):
            data_vec[i] = Instances.copy_instances(data_vec[i], 0, min(instancias))

    return data_vec


def NuevaData(data_raiz):
    # Crea un nuevo dataset apartir de los atributos de otro
    data_new = Instances.template_instances(data_raiz, 0)
    return data_new


def Cabecera(nombre_atrib, range_atrib, zonas):
    # Crea la cabecera del arff con los nombres, la cantidad segun cada metodo y las distintas zonas del video
    atrib = list()
    # Recorro dentro de los intervalos pasados por el rango seleccionando la zona correspondiente
    if type(range_atrib) == list:
        for j in range(0, len(range_atrib) - 1):
            cont = 1
            for i in range(range_atrib[j], range_atrib[j + 1]):
                atrib.append(Attribute.create_numeric(nombre_atrib + '_' + zonas[j] + '[' + str(cont) + ']'))
                cont = cont + 1
    else:
        for j in range(0, range_atrib):
            atrib.append(Attribute.create_numeric(nombre_atrib + '[' + str(j + 1) + ']'))

    data = Instances.create_instances(nombre_atrib + "features", atrib, 0)
    return data


def AgregaInstancia(data_t, valores):
    data = Instances.copy_instances(data_t)
    inst = Instance.create_instance(valores)
    data.add_instance(inst)
    return data


def AgregaInstanciaClase(data_t, valores, indice_clase):
    data = Instances.copy_instances(data_t)
    inst = Instance.create_instance(np.append(valores, indice_clase))
    data.add_instance(inst)
    return data


def getValores(data_t, indice):
    data = Instances.copy_instances(data_t)
    # Si aparecen faltantes devuelvo los valores como vacios
    if data.get_instance(indice).has_missing():
        valores = np.empty(0)
    else:
        valores = np.array(data.get_instance(indice).values)
        # Le saco la clase
        valores = valores[0:valores.size - 1]
    return valores


def AgregaAtributoClase(data_t, clases):
    data = Instances.copy_instances(data_t)
    atrib_class = Attribute.create_nominal('class', clases)
    if data.has_class:
        data.no_class()
        data.delete_last_attribute()
    data.insert_attribute(atrib_class, data.num_attributes)
    data.class_is_last()
    return data


def AgregaEtiqueta(data_t, indice, indice_etiqueta):
    data = Instances.copy_instances(data_t)
    inst = data.get_instance(indice)
    inst.set_value(inst.class_index, indice_etiqueta)
    # inst = Instance.create_instance(np.append(data.get_instance(indice).values, indice_etiqueta))
    data.set_instance(indice, inst)
    return data


def FiltraZonas(data_t, zonas):
    # Se le pasa un vector con las zonas, si en el atributo no detecta que pertenezca a ninguna de las zonas, lo elimina
    zonas = np.append(zonas, 'AUs')
    data = Instances.copy_instances(data_t)
    for i in range(data.num_attributes - 1, -1, -1):
        elimina = True
        for j in range(0, zonas.size):
            if data.attribute(i).name.find(zonas[j]) != -1:
                elimina = False
        if elimina:
            data.delete_attribute(i)
    return data


def RangoAUs(data):
    comienzo = 0
    fin = 0
    for i in range(0, data.num_attributes):
        if data.attribute(i).name.find('AUs') != -1:
            if comienzo == 0:
                comienzo = i
        elif comienzo != 0:
            fin = i - 1
            break
    if fin == 0:
        fin = data.num_attributes - 1
    return comienzo, fin


def CambiarRelationName(data_t, nombre):
    data = Instances.copy_instances(data_t)
    data.relationname = nombre
    return data


def NroInstancias(data):
    return data.num_instances


def valorFaltante():
    return Instance.missing_value()


def BinarizoEtiquetas(path_archivo):
    # Abro el archivo para lectura y escritura
    archivo = open(os.path.join(datos.PATH_CARACTERISTICAS, path_archivo + '.arff'), 'r+')

    # Recorro todas las líneas del archivo
    lineas = archivo.readlines()
    nuevas_lineas = list()
    for linea in lineas:
        # Si encuentro la línea donde está definida el atributo clase, la reemplazo por la línea creada antes
        if linea == '@attribute class {N, B, M, A}\n':
            aux = '@attribute class {N, E}' + '\n'
        elif linea[0] != '@' and linea[0] != '\n':
            aux = linea.replace('B', 'E')
            aux = aux.replace('M', 'E')
            aux = aux.replace('A', 'E')
        else:
            aux = linea
        nuevas_lineas.append(aux)
    # Borro, llevo el puntero al principio y escribo las líneas ya modificadas
    archivo.truncate(0)
    archivo.seek(0)
    archivo.writelines(nuevas_lineas)
    archivo.close()


def MezclaInstancias(data, orden):
    path = os.path.join(datos.PATH_CARACTERISTICAS, 'Resultado' + '.arff')
    Guardav2(path, data)
    arch = open(path, 'r+')

    pos_data = 2
    linea = arch.readline()
    # Busco donde comienza data recien
    while linea[0:5] != '@data':
        linea = arch.readline()
        # Esta variable guarda en que numero de linea empiezan los datoa
        pos_data = pos_data + 1
    arch.readline()
    pos_data = pos_data - 1

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
    ultimo_indice = len(lineas) - 1
    lineas[ultimo_indice] = lineas[ultimo_indice][0:len(lineas[ultimo_indice]) - 1]
    # Limpio el archivo, voy al inicio y escribo las lineas nuevas
    arch.truncate(0)
    arch.seek(0)
    arch.writelines(lineas)
    arch.close()

    data_f = CargaYFiltrado(path)
    return data_f


def GeneraOrdenInstancias(data, instancias_intervalos):
    numero_intervalos = int(NroInstancias(data) / instancias_intervalos)
    orden_intervalos = np.array(range(0, numero_intervalos))
    np.random.shuffle(orden_intervalos)
    orden_instancias = np.empty(0, dtype=np.int)
    for i in orden_intervalos:
        for j in range(0, instancias_intervalos):
            orden_instancias = np.append(orden_instancias, i * instancias_intervalos + j)
    for i in range(numero_intervalos * instancias_intervalos, NroInstancias(data)):
        orden_instancias = np.append(orden_instancias, i)
    return orden_instancias
