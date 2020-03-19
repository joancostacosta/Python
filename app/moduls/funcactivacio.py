import numpy as np
# funcions d'activació [0] i derivades [1]
sigm = (lambda x: 1 / (1 + np.e ** (-x)), 
        lambda x: (1 / (1 + np.e ** (-x))) * (1 - (1 / (1 + np.e ** (-x)))) )

relu = (lambda x: np.maximum(0, x),
        lambda x: np.where(x > 0, 1, 0))
