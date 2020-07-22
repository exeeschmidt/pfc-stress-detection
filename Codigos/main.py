from weka.core import jvm

import Codigos.Experimentos as exp
import Codigos.Datos as datos
import Codigos.LogManager as log

# NOTA: si se rompe la maquina virtual de java al usar HOP, no detener la maquina virtual de java dentro de experimentos


def main():
    # exp.ExtractorDeCaracteristicas()
    # if datos.EXPERIMENTO == 'Unimodal':
    #     exp.Unimodal()
    # elif datos.EXPERIMENTO == 'Primer multimodal':
    #     exp.PrimerMultimodal()
    # else:
    #     exp.SegundoMultimodal()

    jvm.start(max_heap_size="9G", packages=True)
    datos.parametrosSeleccion()
    for i in range(0, 2):
        if i == 1:
            datos.parametrosClasificacion()
        datos.EXPERIMENTO = 'Unimodal'
        exp.Unimodal()
        datos.EXPERIMENTO = 'Primer multimodal'
        exp.PrimerMultimodal()
        datos.EXPERIMENTO = 'Segundo multimodal'
        exp.SegundoMultimodal()

    print('Fin de ejecucion')
    log.agrega('Fin de ejecucion')
    jvm.stop()


if __name__ == '__main__':
    main()
