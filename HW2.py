import numpy as np
import matplotlib.pyplot as plt
import numba as nb

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

def soluce(xsi):
    A = 0.08
    C = 0.18
    D = -0.0066
    E = -1.42
    F = 0.42
    if xsi <= 0.2:
        return -(xsi**2)/2 + A*xsi
    elif xsi >= 0.8:
        return (xsi**2) + E*xsi + F
    else:
        return (5/6)*(xsi**3) - xsi**2 + C*xsi + D

def construct_matrices(n):
    A = np.zeros((n,n))
    b = np.zeros(n)
    for i in range(1,n-1):
        b[i] = force_term(h*i)
        A[i][i-1] = 1
        A[i][i] = -2
        A[i][i+1] = 1
    A = (1/(h**2))*A

    #imposing initial conditions
    b[0] = 0
    b[-1] = 0
    A[0][0] = 1
    A[n-1][n-1] = 1

    return A, b

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
        v = v/beta
        H[i,i-1] = beta
        q = v
        Q[:,i] = q
        if i in order:
            H_return.append(H[0:order[l]+1,0:order[l]])              #### attention aux deep et shallow copy ici
            Q_return.append(Q[:,0:order[l]])
            l += 1
    return H_return, Q_return
            
def on_key(event):
    if event.key == 'escape':
        plt.close(event.canvas.figure)
        

A, b = construct_matrices(n)
r = [10, 20, 30, 40, 50, 60, 70, 80]
Arnoldi_result_H, Arnoldi_result_Q = Arnoldi(A, b, r)                 
np.savetxt("test.out", Arnoldi_result_H[0])
u = np.zeros((len(r),101))
for i in range(len(r)):
    H = Arnoldi_result_H[i]
    print(np.shape(H))
    Q = Arnoldi_result_Q[i]
    y = np.linalg.solve(H.T@H, norm(b)*H.T[:,0])
    print(len(y))
    print(np.shape(Q))
    u[i] = matrix_product(Q, y)

#print(u[0])
colors = ['blue', 'red', 'green', 'cyan', 'purple', 'salmon', 'gold', 'indigo', 'darkviolet']
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', on_key)
xsi = np.linspace(0,1,101)
sol = np.zeros(n)
for i in range(n):
    sol[i] = soluce(xsi[i])
plt.plot(xsi, sol, color='black', label="Exact soluce")
for i in range(len(r)):
    plt.plot(xsi,u[i], color=colors[i], label="r =" + str(r[i]))
plt.title("Poisson's equation")
plt.xlabel('xsi')
plt.ylabel('u')
plt.grid('True')
plt.legend()
plt.show()

#plt.figure("Sparse")
#plt.spy(A, marker='o', color='r', markersize=0.6)
#plt.show()


###QUESTION E2###
