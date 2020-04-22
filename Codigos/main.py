import Codigos.Experimentos as exp
import numpy as np
from tabulate import tabulate

personas = np.array(['01'])
etapas = np.array(['1'])
zonas = np.array(['ojoizq', 'ojoder', 'boca', 'nariz'])
met_caracteristicas = np.array(['LBP', 'AU', 'HOG'])
# met_seleccion = np.array(['Firsts', 'PCA'])
met_seleccion = np.array([])
met_clasificacion = np.array(['RForest', 'J48', 'SVM'])

resultados = exp.Unimodal(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion)
# resultados = exp.MultimodalCompleto(personas, etapas, zonas, met_caracteristicas, met_seleccion, met_clasificacion)
headers = resultados[0, :]
table = tabulate(resultados[1:, :], headers, tablefmt="fancy_grid")
print(table)

print('Fin de ejecucion')