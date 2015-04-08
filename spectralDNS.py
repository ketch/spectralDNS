__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from MPI_knee import mpi_import, MPI, imp
with mpi_import():
    import time
    import importlib
    import config
    import importlib
    t0 = time.time()
    import sys, cProfile
    from numpy import *
    from src import *

# Parse parameters from the command line and update config
commandline_kwargs = parse_command_line(sys.argv[1:])
config.update(commandline_kwargs)

# Import problem specific methods and solver methods specific to either slab or pencil decomposition
with mpi_import():
    from src.mpi import setup, ifftn_mpi, fftn_mpi
    from src.maths import *

comm = MPI.COMM_WORLD
comm.barrier()
num_processes = comm.Get_size()
rank = comm.Get_rank()
if comm.Get_rank()==0: print "Import time ", time.time()-t0

# Set types based on configuration
float, complex, mpitype = {"single": (float32, complex64, MPI.F_FLOAT_COMPLEX),
                           "double": (float64, complex128, MPI.F_DOUBLE_COMPLEX)}[config.precision]

# Apply correct precision and set mesh size
dt = float(config.dt)
nu = float(config.nu)
N = 2**config.M
L = float(2*pi)
dx = float(L/N)

hdf5file = HDF5Writer(comm, dt, N, vars(config), float)
if config.make_profile: profiler = cProfile.Profile()

# Set up solver using wither slab or decomposition
vars().update(setup(**vars()))

def standardConvection(c):
    """c_i = u_j du_i/dx_j"""
    for i in range(3):
        for j in range(3):
            U_tmp[j] = ifftn_mpi(1j*K[j]*U_hat[i], U_tmp[j])
        c[i] = fftn_mpi(sum(U*U_tmp, 0), c[i])
    return c

def divergenceConvection(c, add=False):
    """c_i = div(u_i u_j)"""
    if not add: c.fill(0)
    for i in range(3):
        F_tmp[i] = fftn_mpi(U[0]*U[i], F_tmp[i])
    c[0] += 1j*sum(K*F_tmp, 0)
    c[1] += 1j*K[0]*F_tmp[1]
    c[2] += 1j*K[0]*F_tmp[2]
    F_tmp[0] = fftn_mpi(U[1]*U[1], F_tmp[0])
    F_tmp[1] = fftn_mpi(U[1]*U[2], F_tmp[1])
    F_tmp[2] = fftn_mpi(U[2]*U[2], F_tmp[2])
    c[1] += (1j*K[1]*F_tmp[0] + 1j*K[2]*F_tmp[1])
    c[2] += (1j*K[1]*F_tmp[1] + 1j*K[2]*F_tmp[2])
    return c

def Cross(a, b, c):
    """c_k = F_k(a x b)"""
    U_tmp[:] = cross1(U_tmp, a, b)
    c[0] = fftn_mpi(U_tmp[0], c[0])
    c[1] = fftn_mpi(U_tmp[1], c[1])
    c[2] = fftn_mpi(U_tmp[2], c[2])
    return c

def Curl(a, c):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    F_tmp[:] = cross2(F_tmp, K, a)
    c[0] = ifftn_mpi(F_tmp[0], c[0])
    c[1] = ifftn_mpi(F_tmp[1], c[1])
    c[2] = ifftn_mpi(F_tmp[2], c[2])    
    return c

def getConvection(convection):
    """Return function used to compute convection"""
    if convection == "Standard":
        
        def Conv(dU):
            dU = standardConvection(dU)
            return dU
        
    elif convection == "Divergence":
        
        def Conv(dU):
            dU = divergenceConvection(dU, False)
            return dU
        
    elif convection == "Skewed":
        
        def Conv(dU):
            dU = standardConvection(dU)
            dU = divergenceConvection(dU, True)        
            dU *= 0.5
            return dU
        
    elif convection == "Vortex":
        
        def Conv(dU):
            curl[:] = Curl(U_hat, curl)
            dU = Cross(U, curl, dU)
            return dU
        
    return Conv           

conv = getConvection(config.convection)

@optimizer
def add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, nu):
    """Add contributions from pressure and diffusion to the rhs"""
    
    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat = sum(dU*K_over_K2, 0, out=P_hat)
        
    # Subtract pressure gradient
    dU -= P_hat*K
    
    # Subtract contribution from diffusion
    dU -= nu*K2*U_hat
    
    return dU

def ComputeRHS(dU, rk):
    """Compute and return entire rhs contribution"""
    
    if rk > 0: # For rk=0 the correct values are already in U
        for i in range(3):
            U[i] = ifftn_mpi(U_hat[i], U[i])
    
    # Compute convective term and place in dU
    dU = conv(dU)
    
    dU = dealias_rhs(dU, dealias)
    
    dU = add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, nu)
        
    return dU


# Transform initial data
for i in range(3):
   U_hat[i] = fftn_mpi(U[i], U_hat[i])

# Set up function to perform temporal integration (using config.integrator parameter)
integrate = getintegrator(**vars())

def update(**kwargs):
    pass

def initialize(**kwargs):
    pass

def solve():
    t = 0.0
    tstep = 0
    fastest_time = 1e8
    slowest_time = 0.0
    tic = t0 = time.time()
    while t < config.T-1e-8:
        t += dt; tstep += 1
        
        U_hat[:] = integrate(t, tstep, dt)

        for i in range(3):
            U[i] = ifftn_mpi(U_hat[i], U[i])
                 
        globals().update(locals())
        update(**globals())
        
        tt = time.time()-t0
        t0 = time.time()
        if tstep > 1:
            fastest_time = min(tt, fastest_time)
            slowest_time = max(tt, slowest_time)
            
        if tstep == 1 and config.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()

    toc = time.time()-tic

    # Get min/max of fastest and slowest process
    fast = (comm.reduce(fastest_time, op=MPI.MIN, root=0),
            comm.reduce(slowest_time, op=MPI.MIN, root=0))
    slow = (comm.reduce(fastest_time, op=MPI.MAX, root=0),
            comm.reduce(slowest_time, op=MPI.MAX, root=0))

    if rank == 0:
        print "Time = ", toc
        print "Fastest = ", fast
        print "Slowest = ", slow    
        
    if config.make_profile:
        results = create_profile(**vars())
        
    hdf5file.close()
