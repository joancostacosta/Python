import numpy as np
# imprimeix nom i dimensions de una matriu
def imprimeix_dim(nom, matriu):
    d = matriu.ndim
    print(nom + " : " + str(d) + " dimensions ")
    for x in range(0, d, 1):
        print("dim "+ str(x+1) +" : "+ str(np.size(matriu,x)))