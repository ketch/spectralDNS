spectralDNS
=======
spectralDNS is a classical pseudo-spectral direct Navier-Stokes solver for triply periodic domains. The only unique feature is that it's written entirely in python using numpy, mpi4py and pyfftw and, stripping away unnecessary pre- and post-processing steps, the solver is approximately 100 lines long, including the MPI. The code scales very well in preliminary tests. On the Abel linux cluster at the University of Oslo it scales well up to 512 cores. For a cube of size 512**3, using 512 cores, the code runs at 1.5 seconds per time step. The code scales weakly up to 2048 cores on the Shaheen BlueGene/P supercomputer at KAUST. MPI decomposition is performed using either the "slab" or "pencil" approach. There is also a solver implemented for MHD.
<p align="center">
    <img src="https://www.dropbox.com/s/ynhicrl87cvwhzz/weak_scaling_avg.png?dl=1" width="600" height="400" alt="Weak scaling on Abel cluster"/>
</p>

<p align="center">
    Weak scaling on the Abel cluster.
</p>

<p align="center">
    <img src="https://www.dropbox.com/s/jqjyrfmn835ychu/weak_scaling_shaheen.png?dl=1" width="600" height="400" alt="Weak scaling on Shaheen BlueGene/P"/>
</p>
<p align="center">
    Weak scaling on Shaheen BlueGene/P.
</p>

