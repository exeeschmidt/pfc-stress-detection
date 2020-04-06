import Codigos.Caracteristicas as carac
import Codigos.Weka as wek
import numpy as np
import Codigos.ArffManager as am
import os

# features = carac.VideoEntero(False, np.array(['ojoizq', 'ojoder', 'boca']), np.array(['LBP', 'AU', 'HOP']))
# features('03', '1')
# features = carac.VideoEnParte(False, np.array(['cejaizq', 'cejader', 'ojoizq', 'ojoder', 'boca']), np.array(['LBP', 'AU', 'HOP']))
# features('03', '1')
features = carac.Audio(False)
features('01', '1')

# am.ConcatenaArff('Resultado Video', np.array(['01']), np.array(['1']), True, False)
# am.ConcatenaArff('Resultado Audio', np.array(['01']), np.array(['1']), True, True)
#
# path = 'Caracteristicas' + os.sep + 'Resultado Audio.arff'
# aux = wek.MetodosWeka('MLP', 'Firsts')
# aux(path, True)
