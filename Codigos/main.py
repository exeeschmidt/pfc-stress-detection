import numpy as np
from tabulate import tabulate
import Codigos.Experimentos as exp

# NOTA: si se rompe la maquina virtual de java al usar HOP, no detener la maquina virtual de java dentro de experimentos
def main():
    personas = np.array(['03'])
    etapas = np.array(['1'])
    # zonas = np.array(['ojoizq', 'ojoder', 'cejaizq', 'cejader', 'boca', 'nariz'])
    zonas = np.array(['cejader'])
    met_caracteristicas = np.array(['LBP', 'AU', 'HOG'])
    met_seleccion = np.array(['PSO', 'PCA'])
    # met_seleccion = np.array([])
    met_clasificacion = np.array(['RForest', 'J48', 'SVM'])

    # resultados = exp.Unimodal(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion)
    # resultados = exp.PrimerMultimodalCompleto(personas, etapas, zonas, met_caracteristicas, met_seleccion,
    #                                           met_clasificacion, elimino_silencios=True)
    resultados = exp.SegundoMultimodalCompleto(personas, etapas, zonas, met_caracteristicas, met_seleccion,
                                               met_clasificacion)
    headers = resultados[0, :]
    table = tabulate(resultados[1:, :], headers, tablefmt="fancy_grid")
    print(table)

    print('Fin de ejecucion')

if __name__ == '__main__':
    main()
