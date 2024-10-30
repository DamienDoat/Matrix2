import numpy as np
import numba as nb
import matplotlib.pyplot as plt

#@nb.jit(nopython=True,parallel=True)
def norm (x):
    to_return = 0
    for i in range(len(x)):
        to_return += x[i]**2
    return to_return**(1/2)

#@nb.jit(nopython=True, parallel=True)
def matrix_product(A,x):
    m, n = np.shape(A) 
    to_return = np.zeros(m)
    for i in range(m):
        for j in range(n):
            to_return[i] += A[i][j]*x[j]
    return to_return

#@nb.jit(nopython=True, parallel=True)
def dot(x,y):
    to_return = 0
    for i in range(len(x)):
        to_return += x[i]*y[i]
    return to_return

#@nb.jit(nopython=True, parallel=True)
def A_B(A,B):
    m, n = np.shape(A)
    o, p = np.shape(B)
    to_return = np.zeros((m,p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                to_return[i,j] += A[i][k]*B[k][j]
    return to_return

#@nb.jit(nopython=True, parallel=True)
def Arnoldi(A, b, order):
    H_return = []
    Q_return = []
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
        if (beta == 0):
            print("Le r est trop grand on a la dimension max atteinte")
            return
        v = v/beta
        H[i,i-1] = beta
        q = v
        Q[:,i] = q
        if i in order:
            H_return.append(H[0:order[l]+1,0:order[l]])              
            Q_return.append(Q[:,0:order[l]])          
            l += 1
    return H_return, Q_return

def on_key(event):
    if event.key == 'escape':
        plt.close(event.canvas.figure)
