# 1d split Hamiltonian LSFEM
# Author: W. Bauer
# Date  :  

from firedrake import *
import numpy as np
from scipy.optimize import minimize
import sys

from split_initializations import *
from firedrake.output import VTKFile

file = VTKFile("output.pvd")

# To visualise the output, we also need to import matplotlib.pyplot to display
# the visual output ::
try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

Dt = 0.05
Dt = 0.1
dt = Constant(Dt)

## Initialization:
n0 = 16  # Spatial resolution
n0 = 32        # Spatial resolution
# n0 = 64  # Spatial resolution
# ~ n0 = 128        # Spatial resolution
# ~ n0 = 256        # Spatial resolution

problem = "quads"
# ~ problem = "tris"
# ~ problem = "trisBDM"
modeldeg = 2

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
    mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly)
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
# vn0 = project(un0, VuE)

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

nn = FacetNormal(mesh)
Ueqn = (
        inner(w, u_p - u_n) * dx
        + dt * inner(w, perp(F_h) * q_h) * dx
        + phi * (D_p - D_n + dt * div(F_h)) * dx
        + inner(v, F_h - F_h_rhs) * dx
        + gamma * (q_h * D_h) * dx - gamma * f * dx + inner(perp(grad(gamma)), u_h) * dx
        # 1): weak term:
        # - dt * div(w) * (0.5 * K_h + g * D_h) * dx
        # 2): strong term, set modeldeg>=2
        # - dt * dot(avg(w), jump(0.5 * K_h + g * D_h, nn)) * dS
        - dt * jump(w*(0.5 * K_h + g * D_h), nn) * dS
        + dt * dot(w, grad(0.5 * K_h + g * D_h)) * dx
)

### Todo: don't forget to take into account perp_nabla!!!

gmres_solver_parameters = {'mat_type': 'aij',
                           'snes_linesearch_type': 'basic',
                           # "snes_monitor": True,
                           'ksp_type': 'gmres',
                           # 'ksp_gmres_restart': 100,
                           # 'ksp_converged_reason': True,
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
q_n.project(q_h)

#################################


e_tot_0 = assemble(0.5 * inner(u_n, D_n * u_n) * dx + 0.5 * g * (D_n ** 2) * dx)
q0tw_0 = assemble(0.5 * q_h * q_h * D_h * dx)
all_e_tot = []
all_e_tot.append(e_tot_0)
all_q0tw = []
all_q0tw.append(q0tw_0)

name = "mixed_nl_rot_v2"

u_n.rename('Velocity')
D_n.rename('Depth')
q_n.rename('Potential Vorticity')

tmax = 2  # days
t = 0.
dt1 = 1
dumpfreq = 10
dumpcount = dumpfreq
ufile = VTKFile("output/" + name + ".pvd")


def dump():
    global dumpcount
    dumpcount += 1
    if (dumpcount > dumpfreq):
        ufile.write(u_n, D_n, q_n, time=t)
        dumpcount -= dumpfreq


dump()

# fig = plt.gcf()
fig = plt.figure(1)
fig.show()
fig.canvas.draw()

while t < tmax / Dt - dt1 / 2:
    t += dt1
    print("t= ", t * Dt)

    Usolver.solve()
    # ~ solve(dF == 0,U_p)
    U_n.assign(U_p)

    # ~ Qh.project(qh)

    ufile.write(u_n, D_n, q_n, time=t)

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
    # ~ plt.pause(0.1)

plt.figure(2)
plt.subplot(211)
plt.plot(all_e_tot[1:])
plt.subplot(212)
plt.plot(all_q0tw[1:])
plt.show()


# Golo Code upwinding:
# n = FacetNormal(mesh)
# s = lambda u: 0.5*(sign(dot(u, n)) + 1)
# uw = lambda u, v: (s(u)('+')*v('+') + s(u)('-')*v('-'))
# # Density
# if rho_upw:
#     rho_ad_form = jump(phi*un, n)*uw(un, rhon)*dS_
#     if u_fs_degree > 1:
#         rho_ad_form -= inner(grad(phi), un*rhon)*dx
# else:
#     rho_ad_form = phi*div(fields.F)*dx
# rho_eqn = r_*phi*dx + rho_ad_form
# rho_ad_p = LinearVariationalProblem(lhs(rho_eqn), rhs(rho_eqn), fields.rho_ad)
# rho_solver = LinearVariationalSolver(rho_ad_p, solver_parameters=prms)
