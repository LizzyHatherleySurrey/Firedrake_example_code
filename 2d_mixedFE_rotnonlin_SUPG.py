# 1d split Hamiltonian LSFEM
# Author: W. Bauer
# Date  :  


from firedrake import *
from pyop2.mpi import MPI
from firedrake.output import VTKFile

file = VTKFile("SUPGoutput.pvd")

# ens_comm = Ensemble(COMM_WORLD, 6)

import numpy as np
from scipy.optimize import minimize
import sys

# To visualise the output, we also need to import matplotlib.pyplot to display
# the visual output ::

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

# We then set the parameters for the scheme. ::
dt = 0.05 # crashed for 512 and H = 0.75
dt = 0.025 #

# dt = 0.1
# ~ dt = 0.0005
Dt = Constant(dt)

## Initialization:

from split_initializations import *

print(H)
# n0 = 16  # Spatial resolution
n0 = 32        # Spatial resolution
# n0 = 64  # Spatial resolution
# n0 = 128        # Spatial resolution
# n0 = 256        # Spatial resolution
# n0 = 512        # Spatial resolution

problem = "quads"
# ~ problem = "tris"
problem = "trisBDM"
modeldeg = 1

if problem == "quads":
    mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly, quadrilateral=True)
    Vh = FunctionSpace(mesh, "CG", modeldeg + 0)
    VuE = FunctionSpace(mesh, "RTCE", modeldeg + 0)
    VD = FunctionSpace(mesh, "DG", modeldeg - 1 + 0)
    VuF = FunctionSpace(mesh, "RTCF", modeldeg + 0 + 0)
elif problem == "tris":
    mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly)
    VuE = FunctionSpace(mesh, "N1curl", modeldeg + 0)
    VuF = FunctionSpace(mesh, "RT", modeldeg)
    VD = FunctionSpace(mesh, "DG", modeldeg - 1 + 1)
    Vh = FunctionSpace(mesh, "CG", modeldeg + 0 + 1)
elif problem == "trisBDM":
    mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly, comm=COMM_WORLD)
    VuF = FunctionSpace(mesh, "BDM", modeldeg)
    VuE = FunctionSpace(mesh, "N2curl", modeldeg)
    VD = FunctionSpace(mesh, "DG", modeldeg - 1)
    Vh = FunctionSpace(mesh, "CG", modeldeg + 1)
else:
    error('select method')

### Select initial conditions!!!
x = SpatialCoordinate(mesh)  # define (x[0], x[1])

## 1) vortex pair interaction:
# ~ Dn0 = Function(VD).interpolate(vortex_pair_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
# ~ hn0 = Function(Vh).project(Dn0)
hn0 = Function(Vh).interpolate(vortex_pair_elevation(x[0], x[1], H, DeltaH, Lx, Ly))
Dn0 = Function(VD).project(hn0)

## 2) shear flow:
# H = 1.076
# DeltaH = 0.03
# hn0 = Function(Vh).interpolate(shear_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
# Dn0 = Function(VD).project(hn0)
# Dn0 = Function(VD).interpolate(shear_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
# hn0 = Function(Vh).project(Dn0)

# ~ hn0 = Function(Vh).interpolate(vortex_single_elevation_pos(x[0],x[1],H,DeltaH,Lx,Ly))
# ~ Dn0 = Function(VD).project(hn0)
# Dn0 = Function(VD).interpolate(vortex_single_elevation_pos(x[0],x[1],H,DeltaH,Lx,Ly))
# hn0 = Function(Vh).project(Dn0)


## calculate velocity field via geostrophic balance
un0 = project(perp(grad(hn0)), VuF)  # vn = -g/f*Dtan*h_vtx; % in km/day
un0 *= g / f
# ~ un0 *=0

vn0 = project(un0, VuE)

# ~ plot(hn0)
# ~ plot(Dn0)
# ~ plot(un0)
# ~ plot(vn0)
# ~ plt.show()
# ~ sys.exit('interrupt here')


# solve better inital conditions!


#################################


W = MixedFunctionSpace((VuF, VD, VuF, Vh))
U_n = Function(W)
u_n, D_n, F_n, q_n = U_n.subfunctions

D_n.assign(Dn0)
u_n.project(un0)
F_n.project(un0 * Dn0)

U_p = Function(W)
u_p, D_p, F_h, q_h = split(U_p)
U_p.assign(U_n)

w, phi, v, gamma = TestFunctions(W)
u_h = 0.5 * (u_n + u_p)
D_h = 0.5 * (D_n + D_p)

method = 'energy'
# ~ method = 'CN'

if method == 'energy':
    print('Poisson integrator')
    K_h = inner(u_n, u_n) / 3 + inner(u_n, u_p) / 3 + inner(u_p, u_p) / 3
    F_h_rhs = u_p * D_p / 3 + u_n * D_p / 6 + u_p * D_n / 6 + u_n * D_n / 3
elif method == 'CN':
    print('implicit midpoint')
    F_h_rhs = u_h * D_h
    K_h = inner(u_h, u_h)
else:
    sys.exit('choose energy method')

# APVM:
# tau = 0.5 * dt
tau = (0.5 * dt)/10

Qh = - tau * dot(u_h, grad(q_h)) * F_h

Ueqn = (
        inner(w, u_p - u_n) * dx
        + dt * inner(w, perp(F_h) * q_h + perp(Qh)) * dx
        - dt * div(w) * (0.5 * K_h + g * D_h) * dx
        + phi * (D_p - D_n + dt * div(F_h)) * dx
        + inner(v, F_h - F_h_rhs) * dx
        + gamma * (q_h * D_h) * dx - gamma * f * dx + inner(perp(grad(gamma)), u_h) * dx
)
gmres_solver_parameters = {'mat_type': 'aij',
                           'snes_linesearch_type': 'basic',
                           "snes_monitor": None,
                           'ksp_type': 'gmres',
                           # 'ksp_gmres_restart': 100,
                           'ksp_monitor': None,
                           # 'ksp_converged_reason': None,
                           # 'ksp_monitor_true_residual': True,
                           # 'ksp_view': True,
                           # 'snes_type': 'test'
                           'pc_type': 'lu',
                           'pc_factor_mat_solver_package': 'mumps'
                           }
Uproblem = NonlinearVariationalProblem(Ueqn, U_p)
Usolver = NonlinearVariationalSolver(Uproblem,
                                     solver_parameters=gmres_solver_parameters
                                     )
Usolver.solve()

#################################


e_tot_0 = assemble(0.5 * inner(u_n, D_n * u_n) * dx + 0.5 * g * (D_n ** 2) * dx)
q0tw_0 = assemble(0.5 * q_h * q_h * D_h * dx)
all_e_tot = []
all_e_tot.append(e_tot_0)
all_q0tw = []
all_q0tw.append(q0tw_0)

# name = "mixed_supg_64_10d_075H"
# name = "mixed_supg_512_10d"
# name = "mixed_supg_512_10d_075H_v1"
# name = "mixed_supg_64_10d_shear"
# name = "mixed_supg_64_10d_shear_notau"
# name = "mixed_supg_64_10d_shear_0dt05"
# name = "mixed_supg_64_10d_shear_0dt05_0tau1"
# name = "mixed_supg_64_10d_shear_0dt025_0tau4"

name = "mixed_supg_512_10d_dual_0dt025_05tau250"
name = "mixed_supg_256_10d_dual_0dt025_05tau250"
name = "mixed_supg_128_10d_dual_0dt025_05tau250"
name = "mixed_supg_128_10d_dual_0dt025_05tau100"
name = "mixed_supg_128_10d_dual_0dt025_05tau10"
name = "mixed_supg_256_10d_dual_0dt025_05tau10"


u_n.rename('Velocity')
D_n.rename('Depth')
q_n.project(q_h)
q_n.rename('Potential Vorticity')
# ~ Qh = Function(VD).project(qh)

tmax = 5  # days
t = 0.
dt1 = 1
dumpfreq = 10
dumpcount = dumpfreq
ufile = VTKFile("SUPGoutput/" + name + ".pvd")


def dump():
    global dumpcount
    dumpcount += 1
    if (dumpcount > dumpfreq):
        ufile.write(u_n, D_n, q_n, time=t)
        dumpcount -= dumpfreq


dump()

# fig = plt.gcf()
fig = plt.figure(1)
# fig.show()
# fig.canvas.draw()

while t < tmax / dt - dt1 / 2:
    t += dt1
    print("t= ", t * dt)

    Usolver.solve()
    # ~ solve(dF == 0,U_p)
    U_n.assign(U_p)

    # ~ Qh.project(qh)

    # used dump here instead of write!
    dump()
    # ufile.write(u_n, D_n, q_n, time=t)

    e_tot_t = assemble(0.5 * inner(u_n, D_n * u_n) * dx + 0.5 * g * (D_n ** 2) * dx)
    q0tw_t = assemble(0.5 * q_h * q_h * D_h * dx)
    all_e_tot.append(e_tot_t / e_tot_0 - 1)
    all_q0tw.append(q0tw_t / q0tw_0 - 1)

    print(e_tot_t / e_tot_0 - 1, 'Energy')
    print(q0tw_t / q0tw_0 - 1, 'Enstrophy')

    # plot field in run time
    plt.clf()
    plt.subplot(211)
    plt.plot(all_e_tot[1:])
    plt.subplot(212)
    plt.plot(all_q0tw[1:])
    fig.canvas.draw()
    plt.pause(0.1)

# plt.figure(2)
# plt.subplot(211)
# plt.plot(all_e_tot[1:])
# plt.subplot(212)
# plt.plot(all_q0tw[1:])
# plt.show()
