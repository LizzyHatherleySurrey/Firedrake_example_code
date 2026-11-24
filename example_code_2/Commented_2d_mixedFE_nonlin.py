# 1d split Hamiltonian LSFEM
# Author: W. Bauer
# Date  :  

from firedrake import *
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
dt = 0.05 #timestep
# dt = 0.01
# ~ dt = 0.0005
Dt = Constant(dt)

## Initialization:
from gfd_initializations import * #defines physical parameters e.g. Lx, Ly, H, DeltaH, g, f

# ~ n0 = 16  # Spatial resolution (number of cells in x and y direction)
# ~ n0 = 32  # Spatial resolution
n0 = 64  # Spatial resolution
# ~ n0 = 128        # Spatial resolution
# ~ n0 = 256        # Spatial resolution

problem = "quads" #quadrilateral cells with RTCE/RTCF elements.
# ~ problem = "tris" # triangular cells with N1curl/RT elements.
# ~ problem = "trisBDM" # triangular cells with BDM/N2curl elements.
modeldeg = 1 #polynomial degree of the finite element spaces.

# chosing function spaces is not arbitrary but see generalised below

# CG for height field (continuous and differentiable), RTCE for velocity field (edge-based)(vector field), DG for depth field (conserved quantity), and RTCF for flux field (face-based) on quadrilateral meshes (vector field).
if problem == "quads": 
    mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly, quadrilateral=True)
    Vh = FunctionSpace(mesh, "CG", modeldeg + 0) #continuous Galerkin space for height field
    VuE = FunctionSpace(mesh, "RTCE", modeldeg + 0) #Raviart-Thomas space for velocity field (edge-based)
    VD = FunctionSpace(mesh, "DG", modeldeg - 1 + 0) #discontinuous Galerkin space for depth field
    VuF = FunctionSpace(mesh, "RTCF", modeldeg + 0 + 0) #Raviart-Thomas space for flux field (face-based)
elif problem == "tris":
    mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly)
    VuE = FunctionSpace(mesh, "N1curl", modeldeg + 0) #Nedelec space for velocity field (edge-based)
    VuF = FunctionSpace(mesh, "RT", modeldeg) #Raviart-Thomas space for flux field (face-based)
    VD = FunctionSpace(mesh, "DG", modeldeg - 1 + 1) #discontinuous Galerkin space for depth field
    Vh = FunctionSpace(mesh, "CG", modeldeg + 0 + 1) #continuous Galerkin space for height field
elif problem == "trisBDM":
    mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly)
    VuF = FunctionSpace(mesh, "BDM", modeldeg) #Brezzi-Douglas-Marini space for flux field (face-based)
    VuE = FunctionSpace(mesh, "N2curl", modeldeg)  #Nedelec space for velocity field (edge-based)
    VD = FunctionSpace(mesh, "DG", modeldeg - 1) #discontinuous Galerkin space for depth field
    Vh = FunctionSpace(mesh, "CG", modeldeg + 1) #continuous Galerkin space for height field
else:
    error('select method')

### Select initial conditions!!!
x = SpatialCoordinate(mesh)  # define (x[0], x[1]) # creates a vector of spatial coordinates for the mesh, which can be used to define initial conditions based on spatial position.
#possible initial conditions  
## 1) vortex pair interaction:
# ~ Dn0 = Function(VD).interpolate(vortex_pair_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
# ~ hn0 = Function(Vh).project(Dn0)
# ~ hn0 = Function(Vh).interpolate(vortex_pair_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
# ~ Dn0 = Function(VD).project(hn0)

## 2) shear flow:
# hn0 = Function(Vh).interpolate(shear_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
# Dn0 = Function(VD).project(hn0)
# Dn0 = Function(VD).interpolate(shear_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
# hn0 = Function(Vh).project(Dn0)

## 3) single vortex flow:
# ~ hn0 = Function(Vh).interpolate(vortex_single_elevation_pos(x[0],x[1],H,DeltaH,Lx,Ly))
# ~ Dn0 = Function(VD).project(hn0)
Dn0 = Function(VD).interpolate(vortex_single_elevation_pos(x[0], x[1], H, DeltaH, Lx, Ly)) # Dn0 creates new firedrake funtion in the depth space Vd, interpolate evaluates expression (vortex...) pointwise in the finite element space, hn0 = projects DG depth field into CG height space vh
hn0 = Function(Vh).project(Dn0)

## calculate velocity field via geostrophic balance
un0 = project(perp(grad(hn0)), VuF)  # vn = -g/f*Dtan*h_vtx; % in km/day #perp grad for geostrophic velocity at 90 degrees to height gradient, project into flux space VuF
un0 *= g / f # Note that the units of g and f are such that this scaling is necessary to ensure that the velocity field is in the correct units of km/day. The factor of g/f arises from the geostrophic balance equation, which relates the pressure gradient force to the Coriolis force in a rotating fluid system. By multiplying by g/f, we ensure that the velocity field is consistent with the physical parameters of the system and accurately represents the geostrophic flow.
un0 *= 0 #vel everywhere is zero, so that we can test the energy conservation of the scheme without any dynamics. This is a common approach in numerical simulations to isolate and evaluate specific aspects of the numerical method, such as energy conservation, without the influence of complex flow dynamics. By setting the velocity field to zero, we can focus on the behavior of the numerical scheme itself and assess its performance in terms of energy conservation.

vn0 = project(un0, VuE) #projects same vel fiend into edge based space

# ~ plot(hn0)
# ~ plot(Dn0)
# ~ plot(un0)
# ~ plot(vn0)
# ~ plt.show()
# ~ sys.exit('interrupt here')


#################################


W = MixedFunctionSpace((VuF, VD, VuF)) #creates mixed functin space W
U_n = Function(W) #state of step at time n
u_n, D_n, F_n = U_n.subfunctions #extracts subfunctions corresponding to velocity, depth, and flux from the mixed function space W

D_n.assign(Dn0) #copies initial depth field into mixed state
u_n.project(un0) #project initial velocity field into mixed velocity component
F_n.project(un0 * Dn0) #sets initial flux field as product of velocity and depth fields, projected into mixed flux component

U_p = Function(W) #another mixed funtion for new timestep
u_p, D_p, F_h = split(U_p) #splits new mixed function into its components for velocity, depth, and flux with F_h the symbolic representation of the flux field at the new timestep. This allows for the formulation of the variational problem in terms of these components, facilitating the implementation of the numerical scheme.
U_p.assign(U_n) #initialises new state with old state
#split for buildfing variational problem and assignfor initial guesses in nonlinear problems

w, phi, v = TestFunctions(W) #test functions for the variational formulation, corresponding to the velocity, depth, and flux components of the mixed function space W. These test functions are used to construct the weak form of the governing equations, allowing for the application of the finite element method to solve the problem numerically.
u_h = 0.5 * (u_n + u_p) #midpoint velocity between time steps n and p
D_h = 0.5 * (D_n + D_p) #midpoint depth between time steps n and p #we know to write these for time - centred schemes such as CN and energy conserivng methods

method = 'energy'
# method = 'CN'

if method == 'energy':
    print('Poisson integrator')
    F_h_rhs = u_p * D_p / 3 + u_n * D_p / 6 + u_p * D_n / 6 + u_n * D_n / 3
    K_h = inner(u_n, u_n) / 3 + inner(u_n, u_p) / 3 + inner(u_p, u_p) / 3
# ~ K_h = project(K_h,VD)
elif method == 'CN':
    print('implicit midpoint')
    F_h_rhs = u_h * D_h
    K_h = inner(u_h, u_h)
    K_h = project(K_h, VD)

else:
    sys.exit('choose energy method')

nn = FacetNormal(mesh)
Ueqn = (
        inner(w, u_p - u_n) * dx
        # + dt * inner(w,perp(F_h)*q_h) * dx
        - dt * div(w) * (0.5 * K_h + g * D_h) * dx
        # ~ - dt * dot(avg(w),jump(0.5*K_h + g* D_h,nn )) * dS
        # ~ + dt * dot(w,grad(0.5*K_h + g* D_h)) * dx
        + phi * (D_p - D_n + dt * div(F_h)) * dx
        + inner(v, F_h - F_h_rhs) * dx
        # + gamma_st * (qh_st * Dh_st) * dx - gamma_st * f * dx + inner(perp(grad(gamma_st)), uh_st) * dx
)
gmres_solver_parameters = {'mat_type': 'aij',
                           'snes_linesearch_type': 'basic',
                           "snes_monitor": True,
                           'ksp_type': 'gmres',
                           # 'ksp_gmres_restart': 100,
                           # 'ksp_converged_reason': True,
                           # 'ksp_monitor_true_residual': True,
                           # 'ksp_view': True,
                           # 'snes_type': 'test'
                           'pc_type': 'lu',
                           'pc_factor_mat_solver_package': 'mumps'
                           }
lu_solver_parameters = {'ksp_type': 'preonly',
                        'mat_type': 'aij',
                        'pc_type': 'lu',
                        'pc_factor_mat_solver_package': 'mumps'}
Uproblem = NonlinearVariationalProblem(Ueqn, U_p)
Usolver = NonlinearVariationalSolver(Uproblem,
                                     # ~ solver_parameters= gmres_solver_parameters
                                     solver_parameters=lu_solver_parameters
                                     )
Usolver.solve()

#################################


e_tot_0 = assemble(0.5 * inner(u_n, D_n * u_n) * dx + 0.5 * g * (D_n ** 2) * dx)
# ~ q0tw_0  = assemble(0.5*qh*qh*h1tw_h*dx)
all_e_tot = []
# ~ all_e_tot.append(e_tot_0)
all_q0tw = []
# ~ all_q0tw.append(q0tw_0)

name = "mixed_nonrot"

# ~ Qh = Function(VD).project(qh)

tmax = 1  # days
t = 0.
dt1 = 1
dumpfreq = 10
dumpcount = dumpfreq
ufile = VTKFile("output_data/output_determ/" + name + ".pvd")


def dump():
    global dumpcount
    dumpcount += 1
    if (dumpcount > dumpfreq):
        ufile.write(u_n, D_n, time=t)
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

    ufile.write(u_n, D_n, time=t)

    e_tot_t = assemble(0.5 * inner(u_n, D_n * u_n) * dx + 0.5 * g * (D_n ** 2) * dx)
    # ~ q0tw_t = assemble(0.5 * qh * qh * h1tw_h * dx)
    all_e_tot.append(e_tot_t / e_tot_0 - 1)
    # ~ all_q0tw.append(q0tw_t/q0tw_0 - 1)

    print(e_tot_t / e_tot_0 - 1, 'Energy')
    # ~ print(q0tw_t/q0tw_0 - 1, 'Enstrophy')

    # plot field in run time
    plt.clf()
    plt.subplot(111)
    plt.plot(all_e_tot[1:])
    # ~ plt.subplot(212)
    # ~ plt.plot(all_q0tw[1:])
    fig.canvas.draw()
    plt.pause(0.1)

# plt.figure(2)
# plt.subplot(211)
# plt.plot(all_e_tot)
# plt.subplot(212)
# plt.plot(all_q0tw)
# plt.show()
