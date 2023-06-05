# Script to run unify and run all mps algs

import sys 
import matplotlib.pyplot as plt

from scipy import stats
from canon_forms import *
from hamiltonians import *
from calc_scf import *
from calc_dcf import *
from vumps import vumps
from excitations import excitations

model = str(sys.argv[1])
d = int(sys.argv[2])
D = int(sys.argv[3])
x = float(sys.argv[4])
y = float(sys.argv[5])
z = float(sys.argv[6])
mu = float(sys.argv[7])
N = int(sys.argv[8])
gamma = float(sys.argv[9])

if model == 'halfXXZ':
    h = XYZ_half(x, y, z, mu, size='two')

if model == 'TFI':
    h = TFI(x, g, size='two')

if model == 'oneXXZ':
    h = XYZ_one(x, y, z, size='two')

if model == 'tV':
    if x == y:
        t = x
        h = tV(t, z, mu)
    else:
        exit('x and y not equal')

if model == 'tVV2':
    h = tVV2(x, y, z, mu)

if model == 'tt2tc':
    h = tt2tc(x, y, z, mu)

if d == 2:
    sx = np.array([[0, 1],[1, 0]])
    sy = np.array([[0, -1j],[1j, 0]]) 
    sz = np.array([[1, 0],[0, -1]])

if d == 3:
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) 
    sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]) # no 1/2

sp = 0.5 * (sx + 1.0j * sy)
sm = 0.5 * (sx - 1.0j * sy)
n = 0.5 * (sz + np.eye(d))

tol = stol = 1e-12

mps = mixed_canon_mps(d, D, tol, stol)
gauge_checks(*mps)

site = 'two'

gs_mps, gse = vumps(*mps, h, tol, stol, site, 0, 0)

qm, nk = calc_momentum_dist(*gs_mps, sp, sm, -sz)
qs, sk = calc_stat_struc_fact(*gs_mps, n, n, None)

mom_vec = np.linspace(-1, 1, 5) * np.pi

disp = excitations(*gs_mps, h, site, mom_vec, tol, N)

freq_min = disp[0].min() - (3 * gamma)
freq_max = disp[0].max() + (3 * gamma)
num = int(np.ceil(5 * ((freq_max - freq_min) / gamma)))

freq_vec = np.linspace(freq_min, freq_max, num)

dsf = calc_dsf(*gs_mps, *disp, mom_vec, freq_vec, gamma, n)
specfxn = calc_specfxn(*gs_mps, *disp, mom_vec, freq_vec, gamma, sp, -sz)


















