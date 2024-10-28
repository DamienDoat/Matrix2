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

def norm (x):
    to_return = 0
    for i in range(len(x)):
        to_return += x[i]**2
    return to_return**(1/2)

def matrix_product(A,x):
    m, n = np.shape(A) 
    to_return = np.zeros(m)
    for i in range(m):
        for j in range(n):
            to_return[i] += A[i][j]*x[j]
    return to_return

def dot(x,y):
    to_return = 0
    for i in range(len(x)):
        to_return += x[i]*y[i]
    return to_return

def Arnoldi(A, b, order):
    H_return = []
    H = np.zeros((order[-1]+1, order[-1]))
    Q = np.zeros((len(b), order[-1]+1))
    q = b/norm(b)
    beta = 0
    Q[:,0] = q
    l = 0
    proj = 0
    for i in range(1, order[-1]+1):
        v = matrix_product(A,q)
        for j in range(i):
            proj = dot(v,Q[:,j])
            H[j,i-1] = proj
            v -= proj*Q[:,j]
        beta = norm(v)
        v = v/beta
        H[i,i-1] = beta
        q = v
        if i in order:
            H_return.append(H[0:order[l],0:order[l]+1])
            l += 1
    return H
            
        

A, b = construct_matrices(n)
r = [10, 20, 30, 40, 50]
Arnoldi_result = Arnoldi(A, b, r)
u = np.zeros((5,101))
print(Arnoldi_result)
for i in range(len(r)):
    H = Arnoldi_result[i]
    u[i] = np.linalg.solve(H.T@H, norm(b)*H.T[:,0])


plt.figure("Sparse")
plt.spy(A, marker='o', color='r', markersize=0.6)
plt.show()


###QUESTION E2###
