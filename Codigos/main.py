import Codigos.Experimentos as exp
import Codigos.Datos as datos
import Codigos.LogManager as log

# NOTA: si se rompe la maquina virtual de java al usar HOP, no detener la maquina virtual de java dentro de experimentos


def main():
    # exp.ExtractorDeCaracteristicas()

    log.crea()
    if datos.EXPERIMENTO == 'Unimodal':
        exp.Unimodal()
    elif datos.EXPERIMENTO == 'Primer multimodal':
        exp.PrimerMultimodalCompleto()
    else:
        exp.SegundoMultimodalCompleto()
    print('Fin de ejecucion')
    log.agrega('Fin de ejecucion')


if __name__ == '__main__':
    main()
