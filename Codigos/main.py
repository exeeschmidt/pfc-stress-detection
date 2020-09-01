import os

from weka.core import jvm, packages

import Experimentos as Exp
import Datos
import LogManager as Log

# NOTA: si se rompe la maquina virtual de java al usar HOP, no detener la maquina virtual de java dentro de experimentos


def main():
    # Exp.ExtractorDeCaracteristicas()
    Exp.pruebaMSP()
    # jvm.start(max_heap_size="32G", packages=True)
    # packages.install_package('LibSVM')
    # packages.install_package('PSOSearch')
    # Datos.parametrosSeleccion()
    # for i in range(0, 2):
    #     if i == 1:
    #         Datos.parametrosClasificacion()
    #     Datos.EXPERIMENTO = 'Unimodal'
    #     Exp.Unimodal()
    #     Datos.EXPERIMENTO = 'Primer multimodal'
    #     Exp.PrimerMultimodal()
    #     Datos.EXPERIMENTO = 'Segundo multimodal'
    #     Exp.SegundoMultimodal()

    # Datos.EXPERIMENTO = 'Unimodal'
    # Exp.Unimodal()
    # Datos.EXPERIMENTO = 'Primer multimodal'
    # Exp.PrimerMultimodal()
    # Datos.EXPERIMENTO = 'Segundo multimodal'
    # Exp.SegundoMultimodal()
    #
    # print('Fin de ejecucion')
    # Log.add('Fin de ejecucion')
    # jvm.stop()
    # Para los casos que matlab crea su propia JVM, para que se detengan todas
    # os._exit(0)


if __name__ == '__main__':
    main()
