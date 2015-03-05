from numpy import *
from mpi.wrappyfftw import *
import numpy as np
import matplotlib.pyplot as plt

params = {
    'M': 8,
    'temporal': 'RK4',
    'plot_result': 10,         # Show an image every..
    'nu': 0.001,
    'dt': 0.05,
    'T': 50.0
}

IC = 'Taylor-Green'
vars().update(params)

def rfft2_mpi(u, fu):
    fu[:] = rfft2(u, axes=(0,1))
    return fu    

def irfft2_mpi(fu, u):
    u[:] = irfft2(fu, axes=(0,1))
    return u

N = 2**M
L = 2 * pi
dx = L / N
X = mgrid[:N, :N].astype(float)*L/N

Nf = N/2+1

kx = fftfreq(N, 1./N)
ky = kx[:Nf].copy(); ky[-1] *= -1
K = array(meshgrid(kx, ky, indexing='ij'), dtype=int)
K2 = sum(K*K, 0)
K_over_K2 = array(K, dtype=float) / where(K2==0, 1, K2)


kmax = 2./3.*(N/2+1)
dealias = array((abs(K[0]) < kmax)*(abs(K[1]) < kmax), dtype=bool)


U     = empty((2, N, N))
U_hat = empty((2, N, Nf), dtype="complex")

if IC =='Taylor-Green':
    U[0] = sin(X[0])*cos(X[1])
    U[1] =-cos(X[0])*sin(X[1])

elif IC == 'vortices':
    w = exp(-((X[0]-pi)**2+(X[1]-pi+pi/4)**2)/(0.2))+exp(-((X[0]-pi)**2+(X[1]-pi-pi/4)**2)/(0.2))-0.5*exp(-((X[0]-pi-pi/4)**2+(X[1]-pi-pi/4)**2)/(0.4))
    w_hat = U_hat[0].copy()
    w_hat = rfft2_mpi(w, w_hat)
    U[0] = irfft2_mpi(1j*K_over_K2[1]*w_hat, U[0])
    U[1] = irfft2_mpi(-1j*K_over_K2[0]*w_hat, U[1])

# Transform initial data
U_hat[0] = rfft2_mpi(U[0], U_hat[0])
U_hat[1] = rfft2_mpi(U[1], U_hat[1])

imag_part = np.max(np.abs(U))*np.max(np.abs(dealias*K), 0)
real_part = - nu * K2
lamda = real_part + 1j*imag_part
lamda = lamda.ravel()

plt.plot(np.real(lamda), np.imag(lamda), 'or')
