# 2d split shallow-water equations
# Pseudo-inverse Ekin version
# Author: W. Bauer
# Date  : 07.01.2017


from firedrake import *

# To visualise the output, we also need to import matplotlib.pyplot to display
# the visual output ::

try:
  import matplotlib.pyplot as plt
except:
  warning("Matplotlib not imported")

from firedrake.output import VTKFile

file = VTKFile("output.pvd")
# We then set the parameters for the scheme. ::
dt = 0.0001
Dt = Constant(dt)

# These are set with type :class:`~.Constant` so that the values can be
# changed without needing to regenerate code.
#
# We use a :func:`periodic mesh <.PeriodicIntervalMesh>` of width 40
# with 100 cells, ::

n  = 1000
#Lx = Constant(1000.0)

#BEGIN mesh definition!
#refinements = 3
#mesh = UnitIcosahedralSphereMesh(refinements)
#mesh = PeriodicIntervalMesh(n, Lx)

from split_initializations import *

#Lx   = 5000                                 # Zonal length
#Ly   = 5000                                   # Meridonal length
n0   =   50                                    # Spatial resolution
#mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly,  direction="x", quadrilateral=True) # 50x50 2D mesh [0,2 pi]  ,
mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly,  quadrilateral=True)		# without "direction", double periodic boundaries!

#refinements = 3
#mesh = UnitCubedSphereMesh(refinements)

#END mesh definition


# and build a :class:`mixed function space <.MixedFunctionSpace>` for the
# two variables. ::

#BEGIN werner
L0   = FunctionSpace(mesh, "CG", 2)
L1   = FunctionSpace(mesh, "RTCE", 2)
L2   = FunctionSpace(mesh, "DG", 1)
L0tw   = FunctionSpace(mesh, "CG", 1)
L1tw   = FunctionSpace(mesh, "RTCF", 1)
L2tw   = FunctionSpace(mesh, "DG", 0)

#END werner

#from numpy import linalg as LA


l1trial    = TrialFunction(L1)
l1test     = TestFunction(L1)
l1twtrial  = TrialFunction(L1tw)
l1twtest   = TestFunction(L1tw)

l1matrix   = assemble(inner(l1trial,l1test)*dx)

l1twmatrix = assemble(inner(l1twtrial,l1twtest)*dx)

l0trial = TrialFunction(L0)
l0test   = TestFunction(L0)
l0matrix = assemble(l0trial*l0test*dx)

l0twtrial = TrialFunction(L0tw)
l0twtest   = TestFunction(L0tw)
l0twmatrix = assemble(l0twtrial*l0twtest*dx)
 
l2twtrial = TrialFunction(L2tw)
l2twtest   = TestFunction(L2tw)
l2twmatrix = assemble(l2twtrial*l2twtest*dx) 




x = SpatialCoordinate(mesh)							# define (x[0], x[1])


# We construct a :class:`~.Function` to store the two variables at time
# level ``n``, and :meth:`~.Function.split` it so that we can
# interpolate the initial condition into the two components. ::


##BEGIN:
hn0  = Function(L0).interpolate(vortex_single_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
#hn0  = Function(L0).interpolate(vortex_pair_elevation(x[0],x[1],H,DeltaH,Lx,Ly))

#un0  = project(Expression(["0", 0]), L1)
#Calculate geostrophic balanced initialization:
perp_vel = lambda k: as_vector((-k[1],k[0]))

#n = FacetNormal(mesh)
#n_perp  = perp_vel(n)
#un0 = -g/f*dot(perp_vel(grad(hn0)),n_perp)


un0  = project(perp_vel(grad(hn0)), L1,name = 'vel')     #vn = -g/f*Dtan*h_vtx; % in km/day
un0 *= -g/f
#qn0  =  assign(0) #project(f*perp_vel(un0), L1) 
#kn0  =  assign(0)# project(0.5*un0**2,L0)

### recalculate hn0 such that it is geostrophically balanced
# linearly balanced hn0
q_cell = project(curl(un0) + f,L0tw)
k_cell = project(0.5*inner(un0,un0) ,L0tw)
vt =     project(-perp_vel(un0),L1tw)


dlt_hn0bal     = TestFunction(L0tw)
trial_hn0bal   = TrialFunction(L0tw)

hn0bal         = Function(L0tw,name='h_balanced')

L_init = g*inner(grad(trial_hn0bal),grad(dlt_hn0bal))*dx
R_init = div(vt*q_cell)*dlt_hn0bal*dx
#L_init = L_init - R_init

solve (L_init == R_init, hn0bal)
hn0bal += H



#BEGIN: alternative initialization
#h0tw_init  = Function(L0tw).interpolate(vortex_pair_elevation(x[0],x[1],H,DeltaH,Lx,Ly))
h0tw_init  = Function(L0tw).interpolate(vortex_single_elevation(x[0],x[1],H,DeltaH,Lx,Ly))

u1tw_init  = project(perp_vel(grad(h0tw_init)), L1tw,name = 'Uinit')     #vn = -g/f*Dtan*h_vtx; % in km/day
u1tw_init *= -g/f

q_cell = project(curl(u1tw_init) + f,L0tw)
k_cell = project(0.5*inner(u1tw_init,u1tw_init) ,L0tw)
vt =     project(-perp_vel(u1tw_init),L1tw)


dlt_hn0bal     = TestFunction(L0tw)
trial_hn0bal   = TrialFunction(L0tw)

hn0bal1         = Function(L0tw,name='h_balanced 1')

L_init = g*inner(grad(trial_hn0bal),grad(dlt_hn0bal))*dx
R_init = div(vt*q_cell)*dlt_hn0bal*dx
#L_init = L_init - R_init

solve (L_init == R_init, hn0bal1)
hn0bal1 += H






# Project from L2 to L0
We              = MixedFunctionSpace((L2tw,L1tw))
we0             = Function(We)
he0,ue0         = we0.subfunctions

#he0 = project(hn0bal,L2tw)
he0 = project(hn0,L2tw)

ue0 = project(un0,L1tw)



##END

# Next we build the weak form of the timestepping algorithm. This is expressed
# as a mixed nonlinear problem, which must be written as a bilinear form
# that is a function of the output :class:`~.Function` ``w1``. ::


#BEGIN Write 2d split shallow-water solver!!!




#Wfull    = MixedFunctionSpace((L2tw , L1, L0, L1tw ,   L0 , L2tw))
Wfull    = MixedFunctionSpace((L2tw , L1, L0, L1tw ,    L2tw))

w0full   = Function(Wfull)

#h2tw_0, u1st_0, h0st_0, F1tw_0 ,   k0st_0 , phi2tw_0  = w0full.split()
h2tw_0, u1st_0, h0st_0, F1tw_0 ,    phi2tw_0  = w0full.subfunctions


#write initial condition into full vector
h2tw_0.assign(he0)
u1st_0.assign(un0)
#u1st_0 = project(un0,L1)
#h0st_0.assign(project(hn0bal,L0))
h0st_0.assign(hn0)

#h0st_0 = project(hn0,L0)

F1tw_0 =project(h2tw_0*u1st_0,L1tw)   #FIXME


#Q1st_0.assign(un0)
#k0st_0.assign(un0)


w1full                   = Function(Wfull)
w1full.assign(w0full)

#h2tw_1,u1st_1, h0st_1, F1tw_1 ,  k0st_1 , phi2tw_1 = split(w1full)
h2tw_1,u1st_1, h0st_1, F1tw_1 ,  phi2tw_1 = split(w1full)


#set up trial and test functions:
#dlt_h2tw   , dlt_u1st   , dlt_h0st   , dlt_F1tw   ,    dlt_k0st , dlt_phi2tw  = TestFunctions(Wfull)
dlt_h2tw   , dlt_u1st   , dlt_h0st   , dlt_F1tw  ,  dlt_phi2tw  = TestFunctions(Wfull)



perp_vel = lambda k: as_vector((-k[1],k[0]))


#BEGIN Nonlinear 2D RSW

########################



L_top_u = (         inner(u1st_1                             , dlt_u1st)  -        inner(u1st_0                            , dlt_u1st)  
           - 0.5*Dt*inner((curl(u1st_1)+f)*(F1tw_1)  , dlt_u1st)  - 0.5*Dt*inner((curl(u1st_0)+f)*(F1tw_0) , dlt_u1st) 
           + 0.5*Dt*inner(g*grad(h0st_1)                     , dlt_u1st)  + 0.5*Dt*inner(g*grad(h0st_0)                    , dlt_u1st) 
          # + Dt*inner(  grad(k0st_1)                         , dlt_u1st)  
          )*dx





L_met_u = (  inner(F1tw_1,dlt_F1tw) - inner(h0st_1*(u1st_1)   , dlt_F1tw) #- 0.0*inner(h0st_0*perp_vel(u1st_0)   , dlt_F1tw) 
          # + k0st_1*dlt_k0st     -  0.25*inner(u1st_1,u1st_1)*dlt_k0st  -  0.25*inner(u1st_0,u1st_0)*dlt_k0st        
            

          #+ k0st_1*dlt_phi2tw      - k2tw_1*dlt_phi2tw 
          #+ k0st_1*dlt_k0st        - phi2tw_1*dlt_k0st
          #+ k2tw_1*dlt_k2tw        - 0.5*inner(u1st_1,u1tw_1)*dlt_k2tw   - 0.5*inner(u1st_0,u1tw_0)*dlt_k2tw       
           )*dx


        
L_top_h = (              h2tw_1*dlt_h2tw  -            h2tw_0*dlt_h2tw 
           + 0.5*Dt*div(F1tw_1)*dlt_h2tw + 0.5*Dt*div(F1tw_0)*dlt_h2tw
           )*dx


L_met_h = (#h0st_1*dlt_h0st        - h2tw_1*dlt_h0st
           h0st_1*dlt_phi2tw      - h2tw_1*dlt_phi2tw 
         + h0st_1*dlt_h0st        - phi2tw_1*dlt_h0st
           )*dx





############

#L_top_u = (         inner(u1st_1              , dlt_u1tw)  -        inner(u1st_0              , dlt_u1tw)  
           #- 0.5*Dt*inner(q2tw_1*h0st_1*perp_vel(u1tw_1)         , dlt_u1tw)  - 0.5*Dt*inner(q2tw_0*h0st_0*perp_vel(u1tw_0)       , dlt_u1tw) 
           #+ 0.5*Dt*inner(g*grad(h0st_1)      , dlt_u1tw)  + 0.5*Dt*inner(g*grad(h0st_0)      , dlt_u1tw) 
           #+ Dt*inner(  grad(k0st_1)      , dlt_u1tw)  
          ##+ 0.5*Dt*inner(  grad(k0st_0)      , dlt_u1tw)       
          #)*dx


#L_met_u = (  inner(u1st_1,dlt_u1st) - inner(u1tw_1   , dlt_u1st)           
          ## + inner(u1st_1,dlt_u1tw)  - inner(u1phi_1   ,dlt_u1tw)
          
         ## + inner(Q1st_1,dlt_Q1st) - inner(q2tw_1*h0st_1*perp_vel(u1tw_1),dlt_Q1st) 
          #+ q2tw_1*h0st_1*dlt_q2tw - (f + curl(u1st_1))*dlt_q2tw
          
          #+ k0st_1*dlt_k0st       -  0.25*inner(u1st_1,u1tw_1)*dlt_k0st  -  0.25*inner(u1st_0,u1tw_0)*dlt_k0st        

          ##+ k0st_1*dlt_phi2tw      - k2tw_1*dlt_phi2tw 
          ##+ k0st_1*dlt_k0st        - phi2tw_1*dlt_k0st
          ##+ k2tw_1*dlt_k2tw        - 0.5*inner(u1st_1,u1tw_1)*dlt_k2tw   - 0.5*inner(u1st_0,u1tw_0)*dlt_k2tw       
           #)*dx

        
#L_top_h = (                     
           #h2tw_1*dlt_h2tw -                    h2tw_0*dlt_h2tw 
           ##h0st_1*dlt_h2tw -                    h0st_0*dlt_h2tw 
           #+ 0.5*Dt*div(h0st_1*u1tw_1)*dlt_h2tw + 0.5*Dt*div(h0st_0*u1tw_0)*dlt_h2tw
           #)*dx


#L_met_h = (h0st_1*dlt_h0st - h2tw_1*dlt_h0st)*dx

#######################


L = L_top_u + L_met_u +L_top_h +  L_met_h
#END


#Define full solver:
FullProb                 = NonlinearVariationalProblem(L, w1full)
lu_solver_parameters    = {'ksp_type':'preonly',
                            'mat_type': 'aij',
                            'pc_type':'lu',
                            'pc_factor_mat_solver_package': 'mumps',
                            'snes_linesearch_type': 'bt',
                             'snes_monitor': None,
                            'snes_converged_reason': None,
                            'ksp_converged_reason': None}

FullSolver               = NonlinearVariationalSolver(FullProb, 
                                  solver_parameters = lu_solver_parameters)

#END 





# Note the use of :func:`split(w1) <ufl.split_functions.split>` here, which splits up a
# :class:`~.Function` so that it may be inserted into a UFL
# expression. ::

# Since we are in one dimension, we use a direct solver for the linear
# system within the Newton algorithm. To do this, we assemble a monolithic
# rather than blocked system. ::

# Next we use the other form of :meth:`~.Function.split`, ``w0.split()``,
# which is the way to split up a Function in order to access its data
# e.g. for output. ::



#h2tw_0,u1st_0, h0st_0, u1tw_0 ,   k0st_0  , phi2tw_0 = w0full.split()
#h2tw_1,u1st_1, h0st_1, u1tw_1 ,   k0st_1  , phi2tw_1 = w1full.split()

h2tw_0,u1st_0, h0st_0, F1tw_0 ,    phi2tw_0 = w0full.subfunctions
h2tw_1,u1st_1, h0st_1, F1tw_1 ,    phi2tw_1 = w1full.subfunctions

# calculate Vorticity
#q2st_1 = project(curl(u1st_1) +f,L2,name = "Vorticity")

# We choose a final time, and initialise a :class:`~.File` object for
# storing ``u``. as well as an array for storing the function to be visualised::



#BEGIN Werner
T =  10
#ufile = File('output_snglvrtx_2drotnonlinSWE.pvd')
#ufile = File('output_2dRSE_NONLdirectEkin_vtxpair_q2tw.pvd')
ufile = VTKFile('output_2dnonlinRSW_vtxpair.pvd')

t = 0.0
ufile.write(h2tw_1,u1st_1,h0st_1,F1tw_1 , time=t)
all_hs = []
all_us = []
all_ps = []
#END



k0st_1 = project(0.5*inner(u1st_1,u1st_1),L0)
#h0st_1 = project(h0st_1,L0)


# We also initialise a dump counter so we only dump every 10 timesteps. ::


ndump = 1
dumpn = 0

# Now we enter the timeloop. ::

while (t < T - 0.5*dt):
   t += dt

# The energy can be computed and checked. ::

#
   ##E = assemble((u0*u0 + alphasq*u0.dx(0)*u0.dx(0))*dx)
   ##print "t = ", t, "E = ", E

   print; "t = ", t   

   #print assemble(q0tw_1**2*h2tw_1*dx)
   #print assemble(k0st_init*dx - k0st_1*dx)
   #print assemble(0.5*g*(h0st_init-H)**2*dx - 0.5*g*(h0st_1-H)**2*dx)
   #print assemble(k0st_init*dx - k0st_1*dx + 0.5*g*(h0st_init-H)**2*dx - 0.5*g*(h0st_1-H)**2*dx)
   print; "Ekin = ", assemble(h0st_1*k0st_1*dx)
   print; "Epot = ", assemble(            0.5*g*(h0st_1 - H)*(h2tw_1 - H)*dx)
   print; "Etot = ", assemble(k0st_1*dx + 0.5*g*(h0st_1 - H)*(h2tw_1 - H)*dx)
   

# To implement the timestepping algorithm, we just call the solver, and assign
# ``w1`` to ``w0``. ::

#
   ##usolver.solve()
   ##w0.assign(w1)
   
   #BEGIN Werner
   FullSolver.solve()
   w0full.assign(w1full)   
   #END Werner
   
   #KinSolver.solve()
   #w0kin.assign(w1kin)  

   
   
   #zeta2st_1.assign(project(curl(u1st_1),L2))
   

# Finally, we check if it is time to dump the data. The function will be appended
# to the array of functions to be plotted later::

   dumpn += 1
   if dumpn == ndump:
       dumpn -= ndump
      ##ufile.write(u1, time=t)
      ##all_us.append(Function(u1))
      
      #BEGIN Werner
       ufile.write(h2tw_1,u1st_1,h0st_1,F1tw_1 , time=t)
       all_hs.append(Function(h0st_1))
       all_us.append(Function(F1tw_1))
      #END Werner
       #plt.ion()
       #plot(h2tw_1) #,'interactive') 
       #plt.show()
       #plot(h0st_1) #,'interactive') 
       #plt.draw()
      
   
# This solution leads to emergent peakons (peaked solitons); the left
# peakon is travelling faster than the right peakon, so they collide and
# momentum is transferred to the right peakon.
#
# At last, we call the :func:`plot(all_us) <firedrake.plot.plot>` to plot the
# image::

#try:
  #plot(all_hs[::-1]) #[::-1] calls the last entry in array
  #plot(all_us[::-1]) #[::-1] calls the last entry in array  
  #plot(all_ps[::-1]) #[::-1] calls the last entry in array
#except Exception as e:
  #warning("Cannot plot figure. Error msg: '%s'" % e.message)

# And finally show the figure::

#try:
  #plt.show()
#except Exception as e:
  #warning("Cannot show figure. Error msg: '%s'" % e.message)



## BEGIN NOTE: use filter, warp-scalar for paraview



