# CALCUL DE DIETES
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

JR = 7   # nro. de jornades de la dieta
AP = 4   # nro. d'àpats per jornada

# capa de neurones
class capa_neural():

    # parametres: nro. connexions d'entrada a cada neurona, nro de neurones, funcio d'activacio
    def __init__(self, n_con_entr, n_neurones, f_activ):
        # inicialitzem vector bias entre -1 i 1
        self.b = np.random.rand(1, n_neurones) * 2 - 1    
        # inicialitzem matriu W entre -1 i 1
        self.W = np.random.rand(n_con_entr, n_neurones) * 2 - 1
        # inicialitzem funcio d'activació
        self.f_activ = f_activ

# funcions d'activació [0] i derivades [1]
sigm = (lambda x: 1 / (1 + np.e ** (-x)), 
        lambda x: (1 / (1 + np.e ** (-x))) * (1 - (1 / (1 + np.e ** (-x)))) )

relu = (lambda x: np.maximum(0, x),
        lambda x: np.where(x > 0, 1, 0))

# AUXILIAR: mostrar funcio
#_x = np.linspace(-5, 5, 100)
#plt.plot(_x, relu[1](_x))
#plt.show()

# funció de cost
def Cost(D):
    #processem tots els elements de la dieta (D)
    with np.nditer(D, flags=['multi_index']) as it:
        while not it.finished:
            idx = it.multi_index
            #print(idx[2], D[idx])
            it.iternext()
    return JR * AP

# funció d'entrenament
def entrena(xn, De, Ds, f_cost, lr=0.5):

    # forward pass
    # capa d'entrada
    _ze = De @ xn[0].W + xn[0].b
    _ae = xn[0].f_activ[0](_ze)
    # capa de sortida
    _zs = _ae @ xn[1].W + xn[1].b
    _as = xn[1].f_activ[0](_zs)  

    print(_as)
    # calcul de la funcio de cost de la sortida de la darrera capa

    # backpropagation per obtenir les derivades parcials de la funcio de cost respecte als parametres de les neurones

    # aplicar el gradient descent i ajustar els parametres de les neurones


# PROGRAMA PRINCIPAL
def main():

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

    #print(CMP)

    # muntem la matriu dieta (D) de quantitats d'aliment: 
    # jornades x àpats x aliments

    D = np.zeros( (JR, AP, len(Aliments)))

    with np.nditer(D, flags=['multi_index'], op_flags=['readwrite']) as it:
        while not it.finished:
            idx = it.multi_index
            D[idx] = rnd.random() * 10
            it.iternext()

    print(D)

    # creem les capes d'entrada i de sortida de la XN
    ce = capa_neural(len(Aliments), JR*AP, relu)
    cs = capa_neural(JR*AP, JR*AP*len(Aliments), relu)

    # creem la XN
    xn = []
    xn.append(ce)
    xn.append(cs)
    
    entrena(xn, D, D, Cost, 0.5)
    #print(Cost(D))

# CRIDEM AL PROGRAMA PRINCIPAL
if __name__=='__main__':
    main()