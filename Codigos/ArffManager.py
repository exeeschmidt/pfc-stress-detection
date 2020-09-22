import os
import numpy as np
import Datos
import Herramientas as Hrm
from weka.core.converters import Loader, Saver
from weka.filters import Filter
from weka.core.dataset import Instances, Attribute, Instance


def loadAndFiltered(path):
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


def saveInSubfolder(file_name, sub_name, data):
    path = Hrm.buildSubFilePath(file_name, sub_name)
    saver = Saver()
    saver.save_file(data, path)


def save(path, data):
    saver = Saver()
    saver.save_file(data, path)


def joinDatasetByInstances(data_vec):
    # Une varios datasets que pueden tener o no diferentes atributos pero igual numero de instancias
    data = Instances.copy_instances(data_vec[0])
    for i in range(1, data_vec.size):
        data = Instances.merge_instances(data, data_vec[i])
    data.class_is_last()
    return data


def joinDatasetByAttributes(data_vec):
    # Une varios datasets con los mismos atributos pero con distinto numero de instancias
    data = Instances.copy_instances(data_vec[0])
    for i in range(1, data_vec.size):
        data = Instances.append_instances(data, data_vec[i])
    data.class_is_last()
    return data


def joinPersonStageData(persons, stages, sub, optional_sub=None, join=False, answer_limits_list=None):
    # Levanta y une los dataset de multiples personas y etapas
    # Sub cambia segun el conjunto de caracteristicas, si se presenta sub2 es por si hay que concatenar el audio y video
    # entre sí también
    data_vec1 = np.empty(0)
    data_vec2 = np.empty(0)

    for i in persons:
        for j in stages:
            file_name = Hrm.buildFileName(i, j)
            path = Hrm.buildSubFilePath(file_name, sub)
            data1 = loadAndFiltered(path)
            if optional_sub is not None:
                path = Hrm.buildSubFilePath(file_name, optional_sub)
                data2 = loadAndFiltered(path)
                data_vec_norm = normalizeDatasets(np.array([data1, data2]))
                data_vec1 = np.append(data_vec1, data_vec_norm[0])
                data_vec2 = np.append(data_vec2, data_vec_norm[1])
                if answer_limits_list is not None:
                    # El nuevo limite lo establece el numero de instancias luego de la normalizacion
                    new_limit = instancesNumber(data_vec_norm[0])
                    actual_index = (int(i) - 1) * 2 + int(j) - 1
                    new_limit_index = 0
                    # Busco donde tengo un limite mayor al nuevo limite, esto por si recorta un intervalo mayor al
                    # del ultimo limite (no deberia pasar nunca, y si lo hace igual seria bastante malo)
                    for k in range(0, len(answer_limits_list[actual_index])):
                        if answer_limits_list[actual_index][k] >= new_limit:
                            new_limit_index = k
                            break
                    # Recorto por si es necesario, que tampoco deberia recortarse si anda bien
                    answer_limits_list[actual_index] = answer_limits_list[actual_index][0:new_limit_index + 1]
                    # Defino el nuevo limite reemplazando el ultimo limite puesto
                    answer_limits_list[actual_index][new_limit_index] = new_limit
            else:
                data_vec1 = np.append(data_vec1, data1)
    data_sub1 = joinDatasetByAttributes(data_vec1)
    if optional_sub is not None:
        data_sub2 = joinDatasetByAttributes(data_vec2)
        if join:
            data_sub1.no_class()
            data_sub1.delete_last_attribute()
            data_final = joinDatasetByInstances(np.array([data_sub1, data_sub2]))
            return data_final, answer_limits_list
        else:
            return data_sub1, data_sub2, answer_limits_list
    else:
        return data_sub1


def joinListData(file_list):

    data_vec1 = np.empty(0)
    data_vec2 = np.empty(0)
    answer_limits_list = list()
    for row_file in file_list:
        file_name = row_file[0]
        path = Hrm.buildSubFilePath(file_name, 'VCompFus')
        data1 = loadAndFiltered(path)
        path = Hrm.buildSubFilePath(file_name, 'AComp')
        data2 = loadAndFiltered(path)
        data_vec_norm = normalizeDatasets(np.array([data1, data2]))
        data_vec1 = np.append(data_vec1, data_vec_norm[0])
        data_vec2 = np.append(data_vec2, data_vec_norm[1])

        if len(answer_limits_list) == 0:
            answer_limits_list.append(instancesNumber(data_vec_norm[0]))
        else:
            answer_limits_list.append(answer_limits_list[len(answer_limits_list) - 1] +
                                      instancesNumber(data_vec_norm[0]))
    data_sub1 = joinDatasetByAttributes(data_vec1)
    data_sub2 = joinDatasetByAttributes(data_vec2)
    data_sub1.no_class()
    data_sub1.delete_last_attribute()
    data_final = joinDatasetByInstances(np.array([data_sub1, data_sub2]))
    return data_final, answer_limits_list


def normalizeDatasets(data_vec):
    # Deja todos los dataset presentes en el vector con el numero de instancias del menor
    instances = np.empty(0)
    for i in range(0, data_vec.size):
        instances = np.append(instances, instancesNumber(data_vec[i]))

    for i in range(0, data_vec.size):
        if instances[i] > min(instances):
            data_vec[i] = Instances.copy_instances(data_vec[i], 0, min(instances))

    return data_vec


def newDataset(root_data):
    # Crea un nuevo dataset apartir de los atributos de otro
    new_data = Instances.template_instances(root_data, 0)
    return new_data


def createHeader(attrib_name, attrib_length, zones):
    # Crea la cabecera del arff con los nombres, la cantidad segun cada metodo y las distintas zonas del video
    atrib = list()
    # Recorro dentro de los intervalos pasados por el rango seleccionando la zona correspondiente
    if type(attrib_length) == list:
        for j in range(0, len(attrib_length) - 1):
            cont = 1
            for i in range(attrib_length[j], attrib_length[j + 1]):
                atrib.append(Attribute.create_numeric(attrib_name + '_' + zones[j] + '[' + str(cont) + ']'))
                cont = cont + 1
    else:
        for j in range(0, attrib_length):
            atrib.append(Attribute.create_numeric(attrib_name + '[' + str(j + 1) + ']'))

    data = Instances.create_instances(attrib_name + "features", atrib, 0)
    return data


def addInstance(data_t, values):
    data = Instances.copy_instances(data_t)
    inst = Instance.create_instance(values)
    data.add_instance(inst)
    return data


def addInstanceWithLabel(data_t, values, class_index):
    data = Instances.copy_instances(data_t)
    inst = Instance.create_instance(np.append(values, class_index))
    data.add_instance(inst)
    return data


def addLabel(data_t, index, class_index):
    data = Instances.copy_instances(data_t)
    inst = data.get_instance(index)
    inst.set_value(inst.class_index, class_index)
    data.set_instance(index, inst)
    return data


def addClassAttribute(data_t, classes):
    data = Instances.copy_instances(data_t)
    class_attrib = Attribute.create_nominal('class', classes)
    if data.has_class:
        data.no_class()
        data.delete_last_attribute()
    data.insert_attribute(class_attrib, data.num_attributes)
    data.class_is_last()
    return data


def getValues(data_t, index):
    data = Instances.copy_instances(data_t)
    # Si aparecen faltantes devuelvo los valores como vacios
    if data.get_instance(index).has_missing():
        values = np.empty(0)
    else:
        values = np.array(data.get_instance(index).values)
        # Le saco la clase
        values = values[0:values.size - 1]
    return values


def filterZones(data_t, zones):
    # Se le pasa un vector con las zonas, si en el atributo no detecta que pertenezca a ninguna de las zonas, lo elimina
    zones = np.append(zones, 'AUs')
    data = Instances.copy_instances(data_t)
    for i in range(data.num_attributes - 1, -1, -1):
        delete = True
        for j in range(0, zones.size):
            if data.attribute(i).name.find(zones[j]) != -1:
                delete = False
        if delete:
            data.delete_attribute(i)
    return data


def ausRange(data):
    begin = 0
    end = 0
    for i in range(0, data.num_attributes):
        if data.attribute(i).name.find('AUs') != -1:
            if begin == 0:
                begin = i
        elif begin != 0:
            end = i - 1
            break
    if end == 0:
        end = data.num_attributes - 1
    return begin, end


def changeRelationName(data_t, name):
    data = Instances.copy_instances(data_t)
    data.relationname = name
    return data


def instancesNumber(data):
    return data.num_instances


def missingValue():
    return Instance.missing_value()


def binarizeLabels(file_path):
    # Abro el archivo para lectura y escritura
    file = open(os.path.join(Datos.PATH_CARACTERISTICAS, file_path + '.arff'), 'r+')

    # Recorro todas las líneas del archivo
    lines = file.readlines()
    new_lines = list()
    for line in lines:
        # Si encuentro la línea donde está definida el atributo clase, la reemplazo por la línea creada antes
        if line == '@attribute class {N, B, M, A}\n':
            aux = '@attribute class {N, E}' + '\n'
        elif line[0] != '@' and line[0] != '\n':
            aux = line.replace('B', 'E')
            aux = aux.replace('M', 'E')
            aux = aux.replace('A', 'E')
        else:
            aux = line
        new_lines.append(aux)
    # Borro, llevo el puntero al principio y escribo las líneas ya modificadas
    file.truncate(0)
    file.seek(0)
    file.writelines(new_lines)
    file.close()


def mixInstances(data, order):
    path = os.path.join(Datos.PATH_CARACTERISTICAS, 'Resultado' + '.arff')
    save(path, data)
    file = open(path, 'r+')

    data_begin_position = 2
    line = file.readline()
    # Busco donde comienza data recien
    while line[0:5] != '@data':
        line = file.readline()
        # Esta variable guarda en que numero de linea empiezan los datos
        data_begin_position = data_begin_position + 1
    file.readline()
    data_begin_position = data_begin_position - 1

    file.seek(0)
    # Leo todas las lineas para usarlo como vector
    lines = file.readlines()
    # Extraigo las correspondientes a los datos
    data_lines = np.array(lines[data_begin_position:len(lines)])
    # A la ultima linea actual, que no tiene salto de carro, le agrego por si deja de quedar ultima
    # lineas_datos[len(lineas_datos) - 1] = lineas_datos[len(lineas_datos) - 1] + '\n'
    # Mezclo las lineas
    data_lines = data_lines[order]
    # Uno al resto de las lineas las nuevas instancias mezcladas
    lines = lines[0:data_begin_position] + list(data_lines)
    # A la ultima nueva linea le extraigo el salto de carro
    last_index = len(lines) - 1
    lines[last_index] = lines[last_index][0:len(lines[last_index]) - 1]
    # Limpio el archivo, voy al inicio y escribo las lineas nuevas
    file.truncate(0)
    file.seek(0)
    file.writelines(lines)
    file.close()

    data_f = loadAndFiltered(path)
    return data_f


def generateInstancesOrder(data, instances_intervals):
    interval_number = int(instancesNumber(data) / instances_intervals)
    interval_order = np.array(range(0, interval_number))
    np.random.shuffle(interval_order)
    instance_order = np.empty(0, dtype=np.int)
    for i in interval_order:
        for j in range(0, instances_intervals):
            instance_order = np.append(instance_order, i * instances_intervals + j)
    for i in range(interval_number * instances_intervals, instancesNumber(data)):
        instance_order = np.append(instance_order, i)
    return instance_order
