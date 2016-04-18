spectralDNS
=======

[![Build Status](https://travis-ci.org/spectralDNS/spectralDNS.svg?branch=master)](https://travis-ci.org/spectralDNS/spectralDNS)
[![Anaconda-Server Badge](https://anaconda.org/spectraldns/spectraldns/badges/build.svg)](https://anaconda.org/spectraldns/spectraldns/builds)

spectralDNS contains a classical high-performance pseudo-spectral Navier-Stokes DNS solver for triply periodic domains. The most notable feature of this solver is that it's written entirely in Python using NumPy, MPI for Python (mpi4py) and pyFFTW. MPI decomposition is performed using either the "slab" or the "pencil" approach and, stripping away unnecessary pre- and post-processing steps, the slab solver is no more than 100 lines long, including the MPI. The code has been found to scale very well in tests on the Shaheen Blue Gene/P supercomputer at KAUST Supercomputing Laboratory. Results of both weak and strong scaling tests are shown below. In addition to incompressible Navier-Stokes there are also solvers for MHD and Navier-Stokes or MHD with variable density through a Boussinesq approximation. The solver is described more thoroughly in this paper:

M. Mortensen and H. P. Langtangen "High performance Python for direct numerical simulations of turbulent flows", in press, Computer Physics Communications (2016) http://arxiv.org/pdf/1602.03638v1.pdf

The efficiency of the pure NumPy/mpi4py solver has been enhanced using Cython for certain routines. The strong scaling results on Shaheen shown below have used the optimized Python/Cython solver, which is found to be faster than a pure C++ implementation of the same solver.

A channel flow solver is implemented using the Shen basis (Jie Shen, SIAM Journal on Scientific Computing, 16, 74-87, 1995) for the scheme described by Kim, Moin and Moser (J. Fluid Mechanics, Vol 177, 133-166, 1987).

See the demo folder for usage.

<p align="center">
    <img src="https://www.dropbox.com/s/pi4f25c0pyluxz0/weak_scaling_shaheen_numpy_noopt.png?dl=1" width="600" height="400" alt="Weak scaling of pure numpy/mpi4py solver on Shaheen BlueGene/P"/>
</p>
<p align="center">
    Weak scaling of pure numpy/mpi4py solver on Shaheen BlueGene/P. The C++ solver uses slab decomposition and MPI communication is performed by the FFTW library.
</p>

<p align="center">
    <img src="https://www.dropbox.com/s/p7uapi7eaqjmham/strong_scaling_shaheen_512.png?dl=1" width="600" height="400" alt="Strong scaling of optimized Python/Cython solver on Shaheen BlueGene/P"/>
</p>
<p align="center">
    Strong scaling of optimized Python/Cython solver on Shaheen BlueGene/P. The C++ solver uses slab decomposition and MPI communication is performed by the FFTW library.
</p>

<p align="center">
    <img src="https://www.dropbox.com/s/ynhicrl87cvwhzz/weak_scaling_avg.png?dl=1" width="600" height="400" alt="Weak scaling on Abel cluster"/>
</p>

<p align="center">
    Weak scaling on the Abel cluster.
</p>

<p align="center">
    <img src="https://www.dropbox.com/s/8oayxts0ix359hi/KHmovie2.gif?dl=1" width="600" height="400" alt="Channel flow"/>
</p>
<p align="center">
    Turbulent channel flow at Re_tau = 395. Simulations are performed using 128 cores on Shaheen II with the KMM channel flow solver (solvers/ShenKMM.py) using 256^3 points in real physical space.
</p>

<p align="center">
    <img src="https://www.dropbox.com/s/8oayxts0ix359hi/KHmovie2.gif?dl=1" width="600" height="400" alt="Kelvin Helmholtz instability"/>
</p>

<p align="center">
    Evolution of vorticity. Two-dimensional simulation of Kelvin Helmholtz shear instability using a Boussinesq formulation (solvers/spectralDNS2D_Boussinesq.py)
</p>

