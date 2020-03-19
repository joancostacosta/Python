import numpy as np
# funció de cost de la dieta D
def cost(D):
    #processem tots els elements de la dieta (D)
    cost = 0
    with np.nditer(D, flags=['multi_index']) as it:
        while not it.finished:
            idx = it.multi_index
            cost = cost + D[idx]
            #print(idx[2], D[idx])
            it.iternext()
    return cost