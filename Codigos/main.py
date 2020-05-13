import numpy as np
import Codigos.Experimentos as exp

# NOTA: si se rompe la maquina virtual de java al usar HOP, no detener la maquina virtual de java dentro de experimentos
def main():
    # personas = np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
    #                      '17', '18', '19', '20', '21'])
    personas = np.array(['01', '02', '21'])
    # personas = np.array(['06'])
    etapas = np.array(['1', '2'])
    zonas = np.array(['ojoizq', 'ojoder', 'cejaizq', 'cejader', 'boca', 'nariz'])
    # zonas = np.array(['cejaizq', 'cejader', 'boca'])
    met_caracteristicas = np.array(['LBP', 'AU', 'HOG'])
    met_seleccion = np.array(['', 'PSO', 'PCA'])
    # met_seleccion = np.array([''])
    met_clasificacion = np.array(['RForest', 'SVM', 'J48'])

    exp.Unimodal(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion, folds=-1)
    #exp.PrimerMultimodalCompleto(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion)
    #exp.SegundoMultimodalCompleto(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion)
    print('Fin de ejecucion')


if __name__ == '__main__':
    main()
