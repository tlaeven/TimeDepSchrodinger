import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as splalg


# Input parameters
N  = 1000          # Number of mesh points
dt = 0.1            # Time step
v  = 300.0          # Wave packet initial velocity

sigma = 3*np.sqrt(N)# sigma: inv. proportional to gaussian packet width 
x0 = 0.2            # Gaussian wave packet center position

Vwall = 1
wall_position = N/2
###############################################################################

# Construct mesh
x = np.linspace(0,1,N)


# Construct Laplacian:
laplace1d = sp.diags([1,-2,1], offsets = [-1,0,1], shape=(N,N))

# Construct potential:
V = np.zeros(N)
V[wall_position] = Vwall

# Build Hamiltonian
H = -laplace1d + sp.diags(V, offsets=0)
H[0,-1] = -1 # Periodic boundary conditions
H[-1,0] = -1 # Periodic boundary conditions

# Construct Crank-Nicholson matrices
CN_back = sp.eye(N) - .5j*dt*H
CN_forw = sp.eye(N) + .5j*dt*H

# Initial wavefunction
sigma2 = sigma**2
psi_zero = np.exp(1.0j*v*x) * np.exp(-sigma2*(x-x0)**2)
psi_zero = psi_zero/np.linalg.norm(psi_zero)
psi_zero[V==Vwall] = 0 # set wavefunction to zero on high potential barriers


###############################################################################

# Initial setup
step_forward = psi_zero
lines = plt.plot(x,step_forward, x, V/sigma)
ylim = N/sigma
plt.ylim((0, ylim))
plt.pause(0.001)

for i in range(100000):
    # Actual calculation
    step_back = CN_back.dot(step_forward)
    step_forward = splalg.bicgstab(CN_forw, step_back)[0]
    
    if i%50 == 0:
        p = np.abs(step_forward)**2
        Energy = np.real(np.dot(
                    np.conjugate(step_forward),
                    H.dot(step_forward)))

        y_max = np.max(p)
        lines[0].set_ydata(p)
        plt.pause(0.001)

        print("Norm of wavefunction: %.5f" % np.sum(p),
              "Time: %.0i" % i,
              "<E>: %.5f" % Energy)
        
        if (ylim < y_max) | (y_max < ylim/20.0):
            lines[1].set_ydata(y_max* V/Vwall)
            ylim = 1.4*y_max
            plt.ylim((0, ylim))