# https://stackoverflow.com/questions/15535954/multiplication-of-3-dimensional-matrix-in-numpy
import numpy as np
import time

def imprimeix_dim(nom, matriu):
    d = matriu.ndim
    print(nom + " : " + str(d) + " dimensions ")
    for x in range(0, d, 1):
        print("dim "+ str(x+1) +" : "+ str(np.size(matriu,x)))

A = np.random.random_sample((28, 28, 300))
B = np.random.random_sample((28, 28, 300))
#C = np.random.random_sample((2,2,3))
start = time.time()
C1 = np.empty((28, 28, 300))
for i in range(300):
    C1[:, :, i] = np.dot(A[:, :, i], B[:, :, i])
vector_add_time = time.time() - start
print("C1 took for % seconds" % vector_add_time)

start = time.time()
C2 = np.einsum('ijn,jkn->ikn', A, B)
vector_add_time = time.time() - start
print("C2 took for % seconds" % vector_add_time)
# np.allclose(C1, C2) per saber si son iguals

imprimeix_dim("A", A)
print(A)
imprimeix_dim("B", B)
print(B)
#imprimeix_dim("C1", C1)
#print(C1)
imprimeix_dim("C2", C2)
print(C2)