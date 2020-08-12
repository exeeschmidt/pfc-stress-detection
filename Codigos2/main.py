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

    # datos.EXPERIMENTO = 'Unimodal'
    # exp.Unimodal()
    datos.EXPERIMENTO = 'Primer multimodal'
    exp.PrimerMultimodal()
    # datos.EXPERIMENTO = 'Segundo multimodal'
    # exp.SegundoMultimodal()

    print('Fin de ejecucion')
    log.agrega('Fin de ejecucion')
    jvm.stop()


if __name__ == '__main__':
    main()
