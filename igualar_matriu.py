# muntar XN per igualar la matriu a la matriu objectiu
import numpy
import math
import json
from os import system
import matplotlib.pyplot as plt

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj): # pylint: disable=E0202
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, numpy.float32) or isinstance(obj, numpy.float64): 
            return float(obj)
        return json.JSONEncoder.default(self, obj)

# subclass JSONEncoder
#class Encoder(JSONEncoder):
#        def default(self, o): # pylint: disable=method-hidden
#            return o.__dict__

# constants
dimx = 2
dimy = 5

lr = 0.01           # learning rate
maxiter = 3000      # maxim d'iteracions
maxdesv = 0.001     # quoficient maxim d'error per aturar aprenentatge

# funcions d'activació [0] i derivades [1]
sigm = (lambda x: 1 / (1 + numpy.e ** (-x)), 
        lambda x: (1 / (1 + numpy.e ** (-x))) * (1 - (1 / (1 + numpy.e ** (-x)))) )

relu = (lambda x: numpy.maximum(0, x),
        lambda x: numpy.where(x > 0, 1, 0))

def fcost():
    i = j = 0
    cost = 0
    while i < dimx:
        j = 0
        while j < dimy:
            c = (matriu[i][j] - matriu_obj[i][j]) ** 2
            #print("c "+ str(i) + " " + str(j) + "=" + str(c))
            cost = cost + c
            j = j + 1
        i = i + 1
    return math.sqrt(cost)      

def dfcost():
    i = j = 0
    cost = fcost()
    dcost = numpy.zeros((dimx, dimy))
    while i < dimx:
        j = 0
        while j < dimy:
            dcost[i][j] = (matriu[i][j] - matriu_obj[i][j]) / cost
            j = j + 1
        i = i + 1
    return dcost    

matriu_obj = numpy.random.rand(dimx,dimy)
matriu = numpy.random.rand(dimx, dimy)

#matriu_obj = numpy.array([[0.26611883, 0.81601518, 0.57543956, 0.84941003, 0.14785238], [0.09620474, 0.58006093, 0.44382366, 0.4122605, 0.02058356]])
#matriu = numpy.array([[0.53146912, 0.38142962, 0.20635783, 0.96782322, 0.81159195], [0.95660215, 0.52107673, 0.303842, 0.22514365, 0.69744771]])

#matriu_obj = numpy.array([[0.266, 0.816, 0.575, 0.849, 0.147], [0.096, 0.580, 0.443, 0.412, 0.020]])
#matriu = numpy.array([[0.531, 0.381, 0.206, 0.967, 0.811], [0.956, 0.521, 0.303, 0.225, 0.697]])

'''print("matriu actual:")
print(matriu)
print()
print("matriu objectiu:")
print(matriu_obj)
print()
print("cost = " + str(fcost()))
#print("cost numpy = " + str(numpy.linalg.norm(matriu-matriu_obj)))
print()
print("matriu derivada del cost:")
print(dfcost())'''

# matriu de pesos capa d'ENTRADA "We" (2 dimensions: m x a)
We = numpy.random.rand(dimx, dimy) * 2 - 1
# matriu de bias capa d'ENTRADA "Be" (1 dimensions: m)
Be = numpy.random.rand(dimx) * 2 - 1
# funció d'activació capa d'ENTRADA "fe"
fe = sigm   
#print(We) 
#print(Be)

# matriu de pesos capa de SORTIDA "Ws" (3 dimensions: (m x a) x m)
Ws = numpy.random.rand(dimx, dimy, dimx) * 2 - 1
# matriu de bias capa de SORTIDA "Bs" (2 dimensions: m x a)
Bs = numpy.random.rand(dimx, dimy) * 2 - 1
# funció d'activació capa de SORTIDA "fs"
fs = sigm   # relu
#print(Ws) 
#print(Bs)

# <- AQUI COMENÇA EL BUCLE
iteracions = 0
punts = []
while True:

    # FORWARD PASS
    # capa d'ENTRADA
    Ze = numpy.einsum('ij,ij->i', We, matriu) + Be
    #utl.imprimeix_dim("Ze", Ze)
    #print(Ze)
    #print()

    Se = fe[0](Ze)
    #utl.imprimeix_dim("Se", Se)
    #print(Se)
    #print()

    # capa de SORTIDA
    Zs = numpy.einsum('ijk,k->ij', Ws, Se) + Bs
    #utl.imprimeix_dim("Zs", Zs)
    #print(Zs)
    #print()

    Ss = fs[0](Zs)
    #utl.imprimeix_dim("Ss", Ss)
    #print(Ss)
    #print()

    # actualitzem la dieta amb la sortida de la XN
    matriu_ant = matriu
    matriu = Ss


    # calculem la funció de cost de D com la suma de les desviacions ponderades de cadascuna de les restriccions
    cost = fcost()
    punts.append(cost)
    print("cost de matriu: " + str(cost))
    #print('***********************************************')

    # <->SORTIR DEL BUCLE SI COST 0 O NRO. ITERACIONS COMPLERT
    if cost <= maxdesv or iteracions == maxiter: break
    iteracions = iteracions + 1

    # Neteja pantalla output. En Win (os.name=='nt') neteja la pantalla print. En Mac i Linux (os.name=='posix'): system('clear')
    system('cls') 

    # BACKPROPAGATION per obtenir les derivades parcials de la funcio de cost respecte als parametres de les neurones
    print("BACKPROPAGATION " + str(iteracions))
        
    # dC/dSs = variacio de cost_D per cada element de D: matriu (m x a) 
    dCdSs = dfcost()
    #utl.imprimeix_dim("dCdSs = cost_D_d() ", dCdSs)
    #print()

    # dSs/dZs = fs[1](Zs) derivada de la funció d'activació de sortida respecte a Zs: matriu (m x a)
    dSsdZs = fs[1](Zs)
    #utl.imprimeix_dim("dSsdZs = fs[1](Zs) ", dSsdZs)
    #print()

    # calcular delta de capa de sortida: deltaS (derivada de cost_D respecte a cada element de Zs) = dC/dSs · dSs/dZs : matriu (m x a)
    deltaS = numpy.einsum('ij,ij->ij', dCdSs, dSsdZs)       
    #utl.imprimeix_dim("deltaS", deltaS) 
    #print()

    # dSe/dZe = fe[1](Ze) derivada de la funció d'activació d'entrada respecte a Ze: vector (m x 1)
    dSedZe = fe[1](Ze)
    #utl.imprimeix_dim("dSedZe = fe[1](Ze)", dSedZe)
    #print()
        
    # calcular delta de capa d'entrada a partir de la de sortida: deltaE = deltaS · Ws · dSe/dZe : (m x 1) = (m x a) · (m x a x m) · (m x 1)    
    deltaE = numpy.einsum('ij,ijk->k', deltaS, Ws) 
    deltaE = numpy.einsum('i,j->j', deltaE, dSedZe)
    #utl.imprimeix_dim("deltaE", deltaE)
    #print(deltaE)
    #print()

    # aplicar el GRADIENT DESCENT i ajustar els parametres de les neurones

    # ajustem dinamicament el lr
    lr = cost #* 0.5

    # capa de sortida
    #Ws = Ws - Se @ deltaS * lr 
    Ws = Ws - numpy.einsum('i,jk->jki', Se, deltaS) * lr
    #utl.imprimeix_dim("Ws nou", Ws)
    Bs = Bs - deltaS * lr
    #utl.imprimeix_dim("Bs nou", Bs)
    #print()

    # capa d'entrada
    #We = We - Dant @ deltaE * lr
    #We = We - np.einsum('i,ij->ij', deltaE, Dant) * lr 
    We = We - numpy.einsum('ij,i->ij', matriu_ant, deltaE) * lr 
    #utl.imprimeix_dim("We nou", We)
    Be = Be - deltaE * lr
    #utl.imprimeix_dim("Be nou", Be)
    #print()

# -> AQUI ACABA EL BUCLE

print("matriu actual:")
print(matriu)
print()
print("matriu objectiu:")
print(matriu_obj)
print()

# Imprimim costos 
maxx = len(punts)
maxy = numpy.amax(punts)
x = numpy.linspace(0, maxx-1, num=maxx)
plt.plot(x, [punts[int(i)] for i in x])
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.xlim(0, maxx)
plt.ylim(0, maxy)
plt.show()

# Hem entrenat la XN perque aproximi qualsevol matriu a la matriu objectiu
# provem la configuració obtinguda de la XN per un altra matriu inicial
matriu = numpy.random.rand(dimx, dimy)
print("nova matriu inicial:")
print(matriu)
print()

cost = fcost()
print("cost de nova matriu inicial: " + str(cost))
print()

# FORWARD PASS
# capa d'ENTRADA
Ze = numpy.einsum('ij,ij->i', We, matriu) + Be
Se = fe[0](Ze)

# capa de SORTIDA
Zs = numpy.einsum('ijk,k->ij', Ws, Se) + Bs
Ss = fs[0](Zs)

matriu_ant = matriu
matriu = Ss

print("nova matriu processada:")
print(matriu)
print()

# calculem la funció de cost de D com la suma de les desviacions ponderades de cadascuna de les restriccions
cost = fcost()
print("cost de nova matriu processada: " + str(cost))

# Guardem la matriu objectiu i la configuració de la XN entrenada per aproximar matrius a ella
#dumped = json.dumps({'matriu_obj': matriu_obj, 'We': We, 'Be': Be, 'fe': 'sigma', 'Ws': Ws, 'Bs': Bs, 'fs': 'sigma'}, indent=4, cls=NumpyEncoder, ensure_ascii=False)  
dumped = json.dumps({'matriu_obj': matriu_obj, 'We': We, 'Be': Be, 'fe': 'sigma', 'Ws': Ws, 'Bs': Bs, 'fs': 'sigma'}, cls=NumpyEncoder)  
# FALTA 'fe': fe, 'fs': fs
# jsonpickle is a Python library for serialization and deserialization of complex Python objects to and from JSON.
with open("xn_save.json", 'w') as fp:
    json.dump(dumped, fp)
    fp.close()