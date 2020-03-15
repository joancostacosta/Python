import numpy as np

def imprimeix_dim(nom, matriu):
    d = matriu.ndim
    print(nom + " : " + str(d) + " dimensions ")
    for x in range(0, d, 1):
        print("dim "+ str(x+1) +" : "+ str(np.size(matriu,x)))


D = np.ones((7, 4, 5))  # DIETA jornades, menjars, aliments
W = np.random.randint(0, 10, (7, 4, 5))  # PESOS jornades, menjars, aliments

print(W)
print(D)
#WD = np.dot(W, D)
WD = np.einsum('ijn,ijn->ij', W, D)

imprimeix_dim("WD", WD)
print(WD)