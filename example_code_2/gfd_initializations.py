#initializations
from firedrake import *

Lx   = 5000 # in km                           # Zonal length
Ly   = 4330 # in km                           # Meridonal length

# H       = 10 # in km
H       = 0.75 # in km
DeltaH  = 0.075

H = 1.076
DeltaH = 0.03

theta   = 25./360.*2.*pi
f       = 2.*sin(theta)*2.*pi  # in 1/days
g       = 9.81*(24.*3600.)**2./1000. # in  km/days^2

def vortex_pair_elevation(x,y,H,DeltaH,Lx,Ly):
    o   = 0.1
    xc1 = (1./2.-o)*Lx
    yc1 = (1./2.-o)*Ly
    xc2 = (1./2.+o)*Lx
    yc2 = (1./2.+o)*Ly
    sx = 1.5*Lx/20.
    sy = 1.5*Ly/20.
    
    xp1 = Lx*sin(pi*(x-xc1)/Lx)/sx/pi
    yp1 = Ly*sin(pi*(y-yc1)/Ly)/sy/pi
    xp2 = Lx*sin(pi*(x-xc2)/Lx)/sx/pi
    yp2 = Ly*sin(pi*(y-yc2)/Ly)/sy/pi
    
    h = H - DeltaH*(exp(-1./2.*(xp1**2.+yp1**2.)) + exp(-1./2.*(xp2**2.+yp2**2.)) - 4.*pi*sx*sy/Lx/Ly )
    return h

def vortex_single_elevation(x,y,H,DeltaH,Lx,Ly):
    o   = 0.1
    xc1 = (1./2.-o)*Lx
    yc1 = (1./2.-o)*Ly
    xc2 = (1./2.+o)*Lx
    yc2 = (1./2.+o)*Ly
    sx = 1.5*Lx/20.
    sy = 1.5*Ly/20.
    
    xp1 = Lx*sin(pi*(x-xc1)/Lx)/sx/pi
    yp1 = Ly*sin(pi*(y-yc1)/Ly)/sy/pi
    xp2 = 0*Lx*sin(pi*(x-xc2)/Lx)/sx/pi
    yp2 = 0*Ly*sin(pi*(y-yc2)/Ly)/sy/pi
    
    h = H - DeltaH*(exp(-1./2.*(xp1**2.+yp1**2.)) + exp(-1./2.*(xp2**2.+yp2**2.)) - 4.*pi*sx*sy/Lx/Ly )
    return h

def vortex_single_elevation_pos(x,y,H,DeltaH,Lx,Ly):
    o   = 0.1
    xc1 = (1./2.-o)*Lx
    yc1 = (1./2.-o)*Ly
    xc2 = (1./2.+o)*Lx
    yc2 = (1./2.+o)*Ly
    sx = 1.5*Lx/20.
    sy = 1.5*Ly/20.
    
    xp1 = Lx*sin(pi*(x-xc1)/Lx)/sx/pi
    yp1 = Ly*sin(pi*(y-yc1)/Ly)/sy/pi
    xp2 = 0*Lx*sin(pi*(x-xc2)/Lx)/sx/pi
    yp2 = 0*Ly*sin(pi*(y-yc2)/Ly)/sy/pi
    
    h = H  + DeltaH*(exp(-1./2.*(xp1**2.+yp1**2.)) + exp(-1./2.*(xp2**2.+yp2**2.)) - 4.*pi*sx*sy/Lx/Ly )
    return h

def isolated_vortex(x,y,H,DeltaH,Lx,Ly):
    o = 1
    return o
    
def vct_cnst(x,y,H,DeltaH,Lx,Ly):
    c = H  + 0*(x + y)
    return c



def nullmode(x,y,Lx,Ly):
    h = sin(2*pi*x/Lx)*sin(2*pi*y/Ly)
    return h

def shear_elevation(x,y,H,DeltaH,Lx,Ly):
    H       = 1.076 # in km
    DeltaH  = 0.03
    lx = 0.5
    sy = 1./12.
    k   = 0.1

    xp = x/Lx
    yp = (y-Ly/2.)/Ly
    yp1 = 1./pi*sin(pi*yp)
    yp2 = 1./(2.*pi)*sin(2.*pi*yp)

    h = H - DeltaH*(yp2/sy)*(exp(-yp1**2./(2.*sy**2.)+1./2.)*(1.+k*sin(2.*pi*xp/lx)))
    return h
