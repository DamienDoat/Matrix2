import numpy as np
import matplotlib.pyplot as plt

###QUESTION E1###

def force_term(xsi):
    if xsi <= 0.2:
        return -1
    elif xsi >= 0.8:
        return 2
    else:
        return 5*xsi - 2

n = 101
h = 1/(n-1)

def construct_matrices(n):
    A = np.zeros((n,n))
    b = np.zeros(n)
    for i in range(1,n-1):
        b[i] = force_term(h*i)
        A[i][i-1] = 1
        A[i][i] = -2
        A[i][i+1] = 1
    A = (1/h**2)*A

    #imposing initial conditions
    b[0] = 0
    b[-1] = 0
    A[0][0] = 1
    A[n-1][n-1] = 1

    return A, b

A, b = construct_matrices(n)

plt.figure("Sparse")
plt.spy(A, marker='o', color='r', markersize=0.6)
plt.show()


###QUESTION E2###
#bonjour les amis
