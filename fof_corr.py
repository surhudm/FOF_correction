import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--Lbox", help="Box size", type=float, default=250.0)
parser.add_argument("--Npart", help="Particle number should be Npart^3", type=float, default=2048)
parser.add_argument("--b", help="Linking length", type=float, default=0.2)
parser.add_argument("--Om0", help="Omegam", type=float, default=0.3)
parser.add_argument("--mDelta", help="SO halo mass", type=float, default=1.E15)
parser.add_argument("--cDelta", help="Concentration", type=float, default=4.0)
parser.add_argument("--Delta", help="Overdensity with respect to mean", type=float, default=200.0)

args = parser.parse_args()

# Critical percolation threshold
nc = 0.652960

# These routines will output the FOF halo mass

# Get particle mass from Lbox, Npart and Omega0
def getMpart(Lbox, Npart, Om0):
    gee = 4.2994e-9
    mp = Om0 * 3.E4/(8.*np.pi*gee) * (Lbox/Npart)**3
    return mp

#
# Get RDelta/(lbar)
# Mdelta = 4/3 * pi * Rdelta^3 rhobar Delta
# Mdelta/mpart = 4/3 * pi * Rdelta^3 nbar Delta
# Mdelta/mpart = 4/3 * pi * (Rdelta/lbar)^3  Delta
def getRDelta_by_lbar(MDelta, Delta, Lbox, Npart, Om0):
    return (MDelta/getMpart(Lbox, Npart, Om0)/(4./3.*np.pi*Delta))**(1./3.)

# Scale all radii by the scale radius of the halo
#
# Density of the halo:
#
# rho(x)/rhobar = Delta/3 c^3/mu(c) 1/x/(1+x)**2

# NFW mass factor
def mu(x):
    return np.log(1.+x) - x/(1.+x)

# NFW density scaled by the average density of the Universe
def nfw_rho(x, c, Delta):
    return Delta/3*c**3/mu(c)/x/(1+x)**2

# Mass ratio at x=r/rs
def mass_ratio(x, c):
    return mu(x)/mu(c)

# Get the zero of this function to get the x to which you percolate at
# infinite resolution: Equation 5 in arxiv:1103.0005
def resid(x, b, c, Delta):
    return nc - b**3 * nfw_rho(x, c, Delta)

# Poisson probability for a point at x to be joined: Eq. A1
def poisson_prob(x, c, Delta, b):
    return 1.0 - np.exp(-1./6.*np.pi*nfw_rho(x, c, Delta)*b**3)

# Poisson probability at the percolation threshold: Eq. A2
def poisson_prob_threshold():
    return 1.0 - np.exp(-1./6.*np.pi*nc)

# To correct FOF mass using Eq. B11
def get_dlnm_dp(c, Delta, b):
    x = np.logspace(-2.0, 1.0, 50)
    px = poisson_prob(x, c, Delta, b)
    lnmx = np.log(mu(x))
    from scipy.interpolate import UnivariateSpline, interp1d
    #spl = UnivariateSpline(px[::-1], lnmx[::-1])
    spl = interp1d(px[::-1], lnmx[::-1])

    #return spl(poisson_prob_threshold(), nu=1)
    loc_der = poisson_prob_threshold()
    return np.absolute(spl(loc_der*1.05)-spl(loc_der*0.95))/(0.02)

# Given the b work out the x to which you percolate, and the corresponding mass
from scipy.optimize import brentq
xfofinf = brentq(resid, 1.E-5, 10*args.cDelta, args=(args.b, args.cDelta, args.Delta))
mfofinf = args.mDelta * mass_ratio(xfofinf, args.cDelta)

# Compute Lsize based on B10 first equality
# Lsize = 2 Rfofinf/ (b lbar) = 2 * xfofinf *  RDelta/cDelta /b/lbar
Lsize = 2 * xfofinf * getRDelta_by_lbar(args.mDelta, args.Delta, args.Lbox, args.Npart, args.Om0)/args.cDelta/args.b

# Find the fof mass for a given Lsize: Eq. B11
alpha = (1.+2*xfofinf/(1.+xfofinf))
nu = 4./3.
mfof = mfofinf*(1.+0.22*alpha*Lsize**(-1/nu)*get_dlnm_dp(args.cDelta, args.Delta, args.b))

print args.mDelta, args.cDelta, mfof, mfofinf, mfof/mfofinf
