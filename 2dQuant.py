import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as splalg

# Input parameters
N   = 128           # Number of mesh points
dt  = 0.1           # Time step

vx   = 300.0        # Wave packet initial velocity x-direction
vy   = 0.0          #                     velocity y-direction

s2x = N/1.0         # sigma_x^2: inv. proportional to gaussian packet width 
s2y = N/8.0         # sigma_y^2:        ||         ||
x0 = 0.2           # Gaussian wave packet center x-coordinate
y0 = 0.5           # Gaussian wave packet center y-coordinate

Vwall = 1000
wall_position = round(N/2)
N_holes = 8

###############################################################################

# Construct mesh
N2 = N*N
x = np.linspace(0.0,1.0,N)
y = np.linspace(0.0,1.0,N)
(M_xmesh,M_ymesh) = np.meshgrid(x,y) # mesh in Matrix form
v_xmesh = M_xmesh.flatten() #mesh in vector form
v_ymesh = M_ymesh.flatten()

index_mapping = np.reshape(np.arange(0,N2), (N,N)) # Easy way to select columns in row-major form
    # Ex. N=5 :
    #     0  1  2  3  4
    #     5  6  7  8  9
    #    10 11 12 13 14
    #    15 16 17 18 19
    #    20 21 22 23 24 


# Construct Laplacian:
main_diag  = np.ones(N2)
vert_diags = np.ones(N2-N)
hori_diags = np.ones(N2-1)
hori_diags[np.arange(1,N2)%N==0] = 0
laplace2d = sp.diags((vert_diags,
                        hori_diags,
                            main_diag, 
                        hori_diags,
                      vert_diags),
    offsets = (-N,-1,0,1,N), shape = (N2,N2))


# Construct potential:
V = np.zeros(N2)
V[index_mapping[:, wall_position]] = Vwall

pos_holes = round(N/(N_holes+1)) * np.arange(1, N_holes+1)
V[index_mapping[pos_holes, wall_position]] = 0


# Build Hamiltonian
H = -laplace2d + sp.diags(V, offsets = 0)


# Crank-Nicholson matrices
CN_back = sp.eye(N2) - .5j*dt*H
CN_forw = sp.eye(N2) + .5j*dt*H


# Initial wavefunction
psi_zero = (np.exp(1.0j*(vx*v_xmesh + vy*v_ymesh))* #velocity part
                                np.exp(-(s2x*(v_xmesh - x0)**2 + 
                                         s2y*(v_ymesh - y0)**2)))
psi_zero[V==Vwall] = 0 # set wavefunction to zero on high potential barriers
psi_zero = psi_zero/np.linalg.norm(psi_zero) 

###############################################################################

# Initial setup
step_forward = psi_zero
im = plt.imshow(np.reshape(np.abs(psi_zero), (N,N)), interpolation='none')
plt.pause(0.001)

for i in range(100000):
    # Actual calculation
    step_back   = CN_back.dot(step_forward)
    step_forward= splalg.bicgstab(CN_forw, step_back)[0]
    
    # Draw to screen
    if i%10 == 0:
        p = np.abs(step_forward)**2
        Energy = np.real(
            np.dot(np.conjugate(step_forward),
                     H.dot(step_forward)))
        
        im.set_data(np.reshape(p, (N,N)))
        im.set_clim(vmin = np.min(p),
                    vmax = np.max(p)/5) #Lower max color limit to make fringes more visible
        plt.pause(0.001)

        print("Norm of wavefunction: %.5f" % np.sum(p),
              "Time: %.0i" % i*dt,
              "<E>: %.5f" % Energy)
