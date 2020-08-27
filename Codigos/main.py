import os

from weka.core import jvm, packages

import Experimentos as exp
import Datos as datos
import LogManager as log

# NOTA: si se rompe la maquina virtual de java al usar HOP, no detener la maquina virtual de java dentro de experimentos


def main():
    # exp.ExtractorDeCaracteristicas()

    jvm.start(max_heap_size="32G", packages=True)
    # packages.install_package('LibSVM')
    # packages.install_package('PSOSearch')
    # datos.parametrosSeleccion()
    # for i in range(0, 2):
    #     if i == 1:
    #         datos.parametrosClasificacion()
    #     datos.EXPERIMENTO = 'Unimodal'
    #     exp.Unimodal()
    #     datos.EXPERIMENTO = 'Primer multimodal'
    #     exp.PrimerMultimodal()
    #     datos.EXPERIMENTO = 'Segundo multimodal'
    #     exp.SegundoMultimodal()

    datos.EXPERIMENTO = 'Unimodal'
    exp.Unimodal()
    datos.EXPERIMENTO = 'Primer multimodal'
    exp.PrimerMultimodal()
    datos.EXPERIMENTO = 'Segundo multimodal'
    exp.SegundoMultimodal()

    print('Fin de ejecucion')
    log.add('Fin de ejecucion')
    jvm.stop()
    # Para los casos que matlab crea su propia JVM, para que se detengan todas
    # os._exit(0)


if __name__ == '__main__':
    main()
