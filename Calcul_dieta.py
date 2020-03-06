# CALCUL DE DIETES
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

# funció de cost
def Cost():
    #processem tots els elements de la dieta (D)
    with np.nditer(D, flags=['multi_index']) as it:
        while not it.finished:
            idx = it.multi_index
            print(idx[2], D[idx])
            it.iternext()
    return JR * AP

JR = 7   # nro. de jornades de la dieta
AP = 4   # nro. d'àpats per jornada

Aliments = ["Poma", "Bistec", "Amanida", "Iogurt", "Café"]
Elements = ["Calories", "LDL", "HDL", "Sucres", "Proteines", "Vitamines"]

# Composició d'aliments (CMP): matriu de nA x nE elements
CMP = np.zeros( (len(Aliments), len(Elements)))

#poma
CMP[0][0]=10.5
CMP[0][1]=3.4
CMP[0][2]=1.2
CMP[0][3]=7.6
CMP[0][4]=0.6
CMP[0][5]=0.7
#bistec
CMP[1][0]=78
CMP[1][1]=1.1
CMP[1][2]=4.7
CMP[1][3]=2.2
CMP[1][4]=0.6
CMP[1][5]=0.7
#amanida
CMP[2][0]=12
CMP[2][1]=2.1
CMP[2][2]=1.2
CMP[2][3]=4.3
CMP[2][4]=0.65
CMP[2][5]=0.75
#Iogurt
CMP[3][0]=21.4
CMP[3][1]=3.9
CMP[3][2]=5.2
CMP[3][3]=6.6
CMP[3][4]=0.5
CMP[3][5]=0.5
#cafe
CMP[4][0]=15.5
CMP[4][1]=0.0
CMP[4][2]=1.3
CMP[4][3]=8.7
CMP[4][4]=1.65
CMP[4][5]=1.75

print(CMP)

# muntem la matriu dieta (D) de quantitats d'aliment: 
# jornades x àpats x aliments
#D = np.random.rand(JR, AP, len(Aliments))
#print(D)

D = np.zeros( (JR, AP, len(Aliments)))

with np.nditer(D, flags=['multi_index'], op_flags=['readwrite']) as it:
    while not it.finished:
        idx = it.multi_index
        D[idx] = rnd.random() * 10
        it.iternext()

print(D)

print(Cost())