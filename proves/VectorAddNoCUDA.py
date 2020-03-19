import numpy as np
import time

def VectorAdd(a, b, c):
    for i in range(a.size):
        c[i] = a[i] + b[i]

def main():
    N = 32000000  # nro d'elements del array

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.zeros(N, dtype=np.float32)

    start = time.time()
    VectorAdd(A, B, C)
    vector_add_time = time.time() - start

    print("C[:5] = " + str(C[:5])) 
    print("C[-5:] = " + str(C[-5:]))

    print("VectorAdd took for %s seconds" % vector_add_time)

if __name__=='__main__':
    main()