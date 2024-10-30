import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from tools import *


#######################################
###QUESTION E1###
#######################################


def force_term(xsi):
    if xsi <= 0.2:
        return -1
    elif xsi >= 0.8:
        return 2
    else:
        return 5*xsi - 2

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

def construct_matricesE1(n,h):
    A = np.zeros((n,n))
    b = np.zeros(n)
    for i in range(1,n-1):
        b[i] = force_term(h*i)
        A[i][i-1] = 1
        A[i][i] = -2
        A[i][i+1] = 1
    #imposing initial conditions
    b[0] = 0
    b[-1] = 0
    A[0][0] = 1
    A[n-1][n-1] = 1
    A = (1/(h**2))*A
    return A, b

def questionE1():

    n = 101
    h = 1/(n-1)
    A, b = construct_matricesE1(n,h)
    r = [10, 20, 30, 40, 50]#, 60, 70, 80]
    Arnoldi_result_H, Arnoldi_result_Q = Arnoldi(A, b, r)                 
    u = np.zeros((len(r),101))
    for i in range(len(r)):
        H = Arnoldi_result_H[i]
        Q = Arnoldi_result_Q[i]
        y = np.linalg.solve((H.T)@H, norm(b)*H.T[:,0])
        u[i] = matrix_product(Q, y)

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

##########################################################
###QUESTION E2###
##########################################################

def potential_V(xsi):
    if xsi <= -1:
        return 100
    elif xsi >= 1:
        return 100
    else:
        return 0.0

def construct_matricesE2(n,h):
    A_hat = np.zeros((n,n))
    v = np.zeros(n)
    for i in range(n):
        if i == 0:
            A_hat[i,i+1] = 1
        elif i == (n-1):
            A_hat[i,i-1] = 1
        else:
            A_hat[i,i-1] = 1
            A_hat[i,i+1] = 1
        A_hat[i][i] = -2
        v[i] = potential_V(-2 + h*i)
    A_hat = (1/(h**2))*A_hat
    A = -A_hat + np.diag(v)
    b = np.linspace(1,101, 101)
    return A, b

def questionE2():
    n = 101
    h = 4/(n-1)
    A, b = construct_matricesE2(n,h)
    r = [50]
    Arnoldi_result_H, Arnoldi_result_Q = Arnoldi(A, b, r)
    H_tilde = Arnoldi_result_H[0]
    Q = Arnoldi_result_Q[0]
    H = H_tilde[:-1,:]
    beta = H_tilde[-1:-1]
    lambd, y = np.linalg.eig(H)
    print(np.sort(lambd))
    small = np.argsort(lambd)
    small_eigvals = np.zeros(5)
    small_eigvects = np.zeros((n,5))
    for i in range(5):
        small_eigvals[i] = lambd[small[i]]
        small_eigvects[:,i] = Q@y[:,small[i]]
    print(small_eigvals)

    fig = plt.figure()
    colors = ['b', 'r', 'g', '#ff7f0e', '#9467bd']
    for i in range(5):
        plt.scatter(np.linspace(1,101,101), (small_eigvects.T)[i], c = colors[i], label='Eigvect n°' + str(i+1), s=10) 
    plt.xlabel("ième component")
    plt.ylabel("value")
    plt.title("Approximate eigvects of A associated to 5 smallest eigvals of H")
    plt.legend()
    plt.show()

#    print("Voici mes valeurs propres: \n", np.sort(lambd))
questionE1()
questionE2()
