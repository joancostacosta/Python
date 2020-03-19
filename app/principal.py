# CALCUL DE DIETES
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

import moduls.restriccions as rst
import moduls.funcactivacio as fac
import moduls.utils as utl
import moduls.funcost as cst

# definim variables i estructures globals
JR = 7   # nro. de jornades de la dieta
MJ = 4   # nro. d'àpats per jornada

Aliments = []       # llista d'aliments
Elements = []       # llista d'elements components dels aliments
Composicio = [[]]   # composició d'elements de cada aliment

# inicialitzam variables i estructures
def inicialitza():
    Aliments.extend(["Poma", "Bistec", "Amanida", "Iogurt", "Café"])
    Elements.extend(["Calories", "LDL", "HDL", "Sucres", "Proteines", "Vitamines"])

    Composicio.append([10.5, 3.4, 1.2, 7.6, 0.6, 0.7])      # Poma
    Composicio.append([78.0, 1.1, 4.7, 2.2, 0.6, 0.7])      # Bistec
    Composicio.append([12.0, 2.1, 1.2, 4.3, 0.65, 0.75])    # Amanida
    Composicio.append([21.4, 3.9, 5.2, 6.6, 0.5, 0.5])      # Iogurt
    Composicio.append([15.5, 0.0, 1.3, 8.7, 1.65, 1.75])    # Café

    #print(Aliments)
    #print(Elements)
    #print(Composicio)

# PROGRAMA PRINCIPAL
def main():
    # carreguem aliments i composició
    inicialitza()

    # muntem la matriu dieta (D) de quantitats d'aliment: 
    # jornades x menjars x aliments
    D = np.zeros( (JR, MJ, len(Aliments)))

    with np.nditer(D, flags=['multi_index'], op_flags=['readwrite']) as it:
        while not it.finished:
            idx = it.multi_index
            # PENDENT: cal fer random entre els valors max i min d'aliment per àpat, normatizats: minim = 0 i maxim = 1
            D[idx] = rnd.random()   
            it.iternext()

    utl.imprimeix_dim("D", D)
    print(D)
    print("cost de D: " + str(cst.cost(D)))

    # creem les capes d'ENTRADA i de SORTIDA de la XN
    # matriu de pesos capa d'ENTRADA "We" (3 dimensions: j x m x a)
    We = np.random.rand(JR, MJ, len(Aliments)) * 2 - 1
    # matriu de bias capa d'ENTRADA "Be" (2 dimensions: j x m)
    Be = np.random.rand(JR, MJ) * 2 - 1
    # funció d'activació capa d'ENTRADA "fe"
    fe = fac.sigm

    #print(We)
    #print(Be)

    # matriu de pesos capa de SORTIDA "Ws" (5 dimensions: (j x m x a) x (j x m))
    Ws = np.random.rand(JR, MJ, len(Aliments), JR, MJ) * 2 - 1
    # matriu de bias capa de SORTIDA "Bs" (3 dimensions: j x m x a)
    Bs = np.random.rand(JR, MJ, len(Aliments)) * 2 - 1
    # funció d'activació capa de SORTIDA "fs"
    fs = fac.relu

    # forward pass
    # capa d'ENTRADA
    Ze = np.einsum('ijn,ijn->ij', We, D) + Be
    utl.imprimeix_dim("Ze", Ze)
    #print(Ze)
    Se = fe[0](Ze)
    utl.imprimeix_dim("Se", Se)
    #print(Se)
    # capa de SORTIDA
    Zs = np.einsum('ijklm,lm->ijk', Ws, Se) + Bs
    utl.imprimeix_dim("Zs", Zs)
    #print(Zs)
    Ss = fs[0](Zs)
    utl.imprimeix_dim("Ss", Ss)
    print(Ss)
    print("cost de Ss: " + str(cst.cost(Ss)))
    # calcul de la funcio de cost de la sortida de la darrera capa "Ss"

    # backpropagation per obtenir les derivades parcials de la funcio de cost respecte als parametres de les neurones

    # aplicar el gradient descent i ajustar els parametres de les neurones

    #entrena(xn, D, D, Cost, 0.5)
    #print(Cost(D))

# CRIDEM AL PROGRAMA PRINCIPAL
if __name__=='__main__':
    main()