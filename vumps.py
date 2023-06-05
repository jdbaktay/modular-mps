import numpy as np
import ncon as nc
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import functools

from canon_forms import *
from mps_tools import HeffTerms_two, HeffTerms_three

def dynamic_expansion(AL, AR, C, Hl, Hr, h, delta_D):
    Al = AL.reshape(d * D, D)
    Ar = AR.transpose(1, 0, 2).reshape(D, d * D)

    def calcnullspace(n):
        u, s, vh = spla.svd(n, full_matrices=True)

        right_null = vh.conj().T[:,D:]
        left_null = u.conj().T[D:,:]

        return left_null, right_null

    _, Nl = calcnullspace(Al.T.conj())
    Nr, _ = calcnullspace(Ar.T.conj())

    def eff_ham(X):
        X = X.reshape(d, D, d, D)

        tensors = [AL, X, h, AL.conj()]
        indices = [(4,1,2), (5,2,-3,-4), (3,-1,4,5),(3,1,-2)]
        contord = [1,2,3,4,5]
        H1 = nc.ncon(tensors,indices,contord)

        tensors = [X, h]
        indices = [(1,-2,2,-4), (-1,-3,1,2)]
        contord = [1,2]
        H2 = nc.ncon(tensors,indices,contord)

        tensors = [X, AR, h, AR.conj()]
        indices = [(-1,-2,4,2), (5,2,1), (-3,3,4,5), (3,-4,1)]
        contord = [1,2,3,4,5]
        H3 = nc.ncon(tensors,indices,contord)

        tensors = [Hl, X]
        indices = [(-2,1), (-1,1,-3,-4)]
        H4 = nc.ncon(tensors,indices)

        tensors = [X, Hr]
        indices = [(-1,-2,-3,1), (1,-4)]
        H5 = nc.ncon(tensors,indices)
        return H1 + H2 + H3 + H4 + H5

    A_two_site = nc.ncon([AL, C, AR], [(-1, -2, 1), (1, 2), (-3, 2, -4)])
    A_two_site = eff_ham(A_two_site).reshape(d * D, d * D)

    t = Nl.conj().T @ A_two_site @ Nr.conj().T
    u, s, vh = spla.svd(t, full_matrices=True)
    print('deltaD svals', s[:delta_D])
    print('>deltaD svals', s[delta_D:])

    u = u[:, :delta_D]
    vh = vh[:delta_D, :]

    if delta_D > D:
        expand_left = (Nl @ u).reshape(d, D, D)
        expand_right = (vh @ Nr).reshape(D, d, D).transpose(1, 0, 2)
        t = delta_D - D
    else:
        expand_left = (Nl @ u).reshape(d, D, delta_D)
        expand_right = (vh @ Nr).reshape(delta_D, d, D).transpose(1, 0, 2)
        t = 0

    AL_new = np.concatenate((AL, expand_left), axis=2)
    AR_new = np.concatenate((AR, expand_right), axis=1)

    AL, AR = [], []

    for i in range(AL_new.shape[0]):
        AL.append(np.pad(AL_new[i,:,:], pad_width=((0, delta_D), (0, t)), 
                                        mode='constant')
        )

    for i in range(AR_new.shape[0]):
        AR.append(np.pad(AR_new[i,:,:], pad_width=((0, t), (0, delta_D)), 
                                        mode='constant')
        )

    C = np.pad(C, pad_width=((0, delta_D), (0, delta_D)), mode='constant')
    Hl = np.pad(Hl, pad_width=((0, delta_D), (0, delta_D)), mode='minimum')
    Hr = np.pad(Hr, pad_width=((0, delta_D), (0, delta_D)), mode='minimum')
    return np.array(AL), np.array(AR), C, Hl, Hr

def Apply_HC_two(AL, AR, Hl, Hr, h, D, d, X):
    X = X.reshape(D, D)

    t = AL.reshape(d * D, D) @ X @ AR.transpose(1, 0, 2).reshape(D, d * D)
    t = (h.reshape(d**2, d**2) 
       @ t.reshape(d, D, d, D).transpose(0, 2, 1, 3).reshape(d**2, D * D)
       )

    t = t.reshape(d, d, D, D).transpose(0, 2, 1, 3).reshape(d * D, d * D)

    H1 = (AL.conj().transpose(2, 0, 1).reshape(D, d * D) 
        @ t @ AR.conj().transpose(0, 2, 1).reshape(d * D, D)
        )

    H2 = Hl @ X
    H3 = X @ Hr
    return (H1 + H2 + H3).ravel()

def Apply_HAC_two(hL_mid, hR_mid, Hl, Hr, D, d, X):
    X = X.reshape(D, d, D)

    t = hL_mid.reshape(D * d, D * d) @ X.reshape(D * d, D)
    H1 = t.reshape(D, d, D)

    t = X.reshape(D, d * D) @ hR_mid.reshape(d * D, d * D).transpose(1, 0)
    H2 = t.reshape(D, d, D)

    t = Hl @ X.reshape(D, d * D)
    H3 = t.reshape(D, d, D)

    t = X.reshape(D * d, D ) @ Hr
    H4 = t.reshape(D, d, D)
    return (H1 + H2 + H3 + H4).ravel()

def Apply_HC_three(hl_mid, hr_mid, AL, AR, Hl, Hr, h, D, d, X):
    X = X.reshape(D, D)

    t = hl_mid.transpose(0, 1, 3, 2).reshape(D * d * d, D) @ X
    t = t.reshape(D * d, d * D) @ AR.reshape(d * D, D)
    H1 = t.reshape(D, d * D) @ AR.conj().transpose(0, 2, 1).reshape(d * D, D)

    t = X @ hr_mid.transpose(2, 0, 1, 3).reshape(D, d * D * d)
    t = t.reshape(D, d, D, d).transpose(3, 0, 1, 2).reshape(d * D, d * D)
    t = AL.transpose(1, 0, 2).reshape(D, d * D) @ t
    H2 = AL.conj().transpose(2, 1, 0).reshape(D, D * d) @ t.reshape(D * d, D)

    H3 = Hl @ X
    H4 = X @ Hr
    return (H1 + H2 + H3 + H4).ravel()

def Apply_HAC_three(hl_mid, hr_mid, AL, AR, Hl, Hr, h, D, d, X):
    X = X.reshape(D, d, D)

    t = hl_mid.reshape(D * d, D * d) @ X.reshape(D * d, D)
    H1 = t.reshape(D, d, D)

    t = AL.reshape(d * D, D) @ X.reshape(D, d * D)
    t = t.reshape(d * D * d, D) @ AR.transpose(1, 0, 2).reshape(D, d * D)
    t = t.reshape(d, D, d, d, D).transpose(1, 4, 0, 2, 3).reshape(D * D, d * d * d)
    t = t @ h.reshape(d * d * d, d * d * d).transpose(1, 0)
    t = t.reshape(D, D, d, d, d).transpose(0, 2, 3, 1, 4).reshape(D * d, d * D * d)
    t = AL.conj().transpose(2, 1, 0).reshape(D, D * d) @ t 
    t = t.reshape(D * d, D * d) @ AR.conj().transpose(2, 0, 1).reshape(D * d, D)
    H2 = t.reshape(D, d, D)

    t = X.reshape(D, d * D) @ hr_mid.transpose(3, 2, 0, 1).reshape(d * D,d * D)
    H3 = t.reshape(D, d, D)

    t = Hl @ X.reshape(D, d * D)
    H4 = t.reshape(D, d, D)

    t = X.reshape(D * d, D) @ Hr
    H5 = t.reshape(D, d, D)
    return (H1 + H2 + H3 + H4 + H5).ravel()

def calc_new_A(AL, AR, AC, C):
    d = AL.shape[0] 
    D = C.shape[0]

    Al = AL.reshape(d * D, D)
    Ar = AR.transpose(1, 0, 2).reshape(D, d * D)

    def calcnullspace(n):
        u, s, vh = spla.svd(n, full_matrices=True)

        right_null = vh.conj().T[:,D:]
        left_null = u.conj().T[D:,:]

        return left_null, right_null

    _, Al_right_null = calcnullspace(Al.T.conj())
    Ar_left_null, _  = calcnullspace(Ar.T.conj())

    Bl = Al_right_null.T.conj() @ AC.transpose(1, 0, 2).reshape(d * D, D)
    Br = AC.reshape(D, d * D) @ Ar_left_null.T.conj()

    epl = spla.norm(Bl)
    epr = spla.norm(Br)

    s = spla.svdvals(C)
    print('first svals', s[:5])
    print('last svals', s[-5:])

    ulAC, plAC = spla.polar(AC.reshape(D * d, D), side='right')
    urAC, prAC = spla.polar(AC.reshape(D, d * D), side='left')

    ulC, plC = spla.polar(C, side='right')
    urC, prC = spla.polar(C, side='left')

    AL = (ulAC @ ulC.T.conj()).reshape(D, d, D).transpose(1, 0, 2)
    AR = (urC.T.conj() @ urAC).reshape(D, d, D).transpose(1, 0, 2)
    return epl, epr, AL, AR

def vumps_two(AL, AR, C, Hl, Hr, h, ep):
    d = AL.shape[0] 
    D = C.shape[0]

    h = h.reshape(d, d, d, d)

    AC = np.tensordot(C, AR, axes=(1, 1))

    Hl, Hr, e = HeffTerms_two(AL, AR, C, Hl, Hr, h, ep)

    tensors = [AL, h, AL.conj()]
    indices = [(2, 1, -3), (3, -2, 2, -4), (3, 1, -1)]
    contord = [1, 2, 3]
    hL_mid = nc.ncon(tensors, indices, contord)

    tensors = [AR, h, AR.conj()]
    indices = [(2, -4, 1), (-1, 3, -3, 2), (3, -2, 1)]
    contord = [1, 2, 3]
    hR_mid = nc.ncon(tensors, indices, contord)    

    f = functools.partial(Apply_HC_two, AL, AR, Hl, Hr, h, D, d)
    g = functools.partial(Apply_HAC_two, hL_mid, hR_mid, Hl, Hr, D, d)

    H = spspla.LinearOperator((D * D, D * D), matvec=f)
    w, v = spspla.eigsh(H, k=1, which='SA', v0=C.ravel(), tol=ep/100)
    C = v[:,0].reshape(D, D)
    print('C_eval', w[0], C.shape)

    H = spspla.LinearOperator((D * d * D, D * d * D), matvec=g)
    w, v = spspla.eigsh(H, k=1, which='SA', v0=AC.ravel(), tol=ep/100)
    AC = v[:,0].reshape(D, d, D)
    print('AC_eval', w[0], AC.shape)

    epl, epr, AL, AR = calc_new_A(AL, AR, AC, C)
    return AL, AR, C, Hl, Hr, e, epl, epr

def vumps_three(AL, AR, C, Hl, Hr, h, ep):
    d = AL.shape[0] 
    D = C.shape[0]

    h = h.reshape(d, d, d, d, d, d)

    AC = np.tensordot(C, AR, axes=(1, 1))

    Hl, Hr, e = HeffTerms_three(AL, AR, C, Hl, Hr, h, ep)

    tensors = [AL, AL, h, AL.conj(), AL.conj()]
    indices = [(4, 7, 8), (5, 8, -3), (1, 2, -2, 4, 5, -4), (1, 7, 9), (2, 9, -1)]
    contord = [7, 8, 9, 1, 2, 4, 5]
    hl_mid = nc.ncon(tensors,indices,contord)

    tensors = [AR, AR, h, AR.conj(), AR.conj()]
    indices = [(5, -3, 8), (6, 8, 7), (-1, 2, 3, -4, 5, 6), (2, -2, 9), (3, 9,7 )]
    contord = [7, 8, 9, 2, 3, 5, 6]
    hr_mid = nc.ncon(tensors,indices,contord)

    f = functools.partial(Apply_HC_three, hl_mid, hr_mid, AL, AR, Hl, Hr, h, D, d)
    g = functools.partial(Apply_HAC_three, hl_mid, hr_mid, AL, AR, Hl, Hr, h, D, d)

    H = spspla.LinearOperator((D * D, D * D), matvec=f)
    w, v = spspla.eigsh(H, k=1, which='SA', v0=C.ravel(), tol=ep/100)
    C = v[:,0].reshape(D, D)
    print('C_eval', w[0], C.shape)

    H = spspla.LinearOperator((D * d * D, D * d * D), matvec=g)
    w, v = spspla.eigsh(H, k=1, which='SA', v0=AC.ravel(), tol=ep/100)
    AC = v[:,0].reshape(D, d, D)
    print('AC_eval', w[0], AC.shape)

    epl, epr, AL, AR = calc_new_A(AL, AR, AC, C)
    return AL, AR, C, Hl, Hr, e, epl, epr

def vumps(AL, AR, C, h, tol, stol, size, Dmax, delta_D):
    D = C.shape[0]
    d = AL.shape[1]

    AL = AL.transpose(1, 0, 2)
    AR = AR.transpose(1, 0, 2)

    energy, error = [], []

    count, ep = 0, 1e-2

    Hl, Hr = np.eye(D, dtype=AL.dtype), np.eye(D, dtype=AR.dtype)

    if size == 'two':
        AL, AR, C, Hl, Hr, *_ = vumps_two(AL, AR, C, Hl, Hr, h, ep)
    if size == 'three':
        AL, AR, C, Hl, Hr, *_ = vumps_three(AL, AR, C, Hl, Hr, h, ep)

    AL, C = left_gauge(AR, C, tol / 100, stol)
    AR, C = right_gauge(AL, C, tol / 100, stol)

    while (ep > tol or D < Dmax) and count < 5000:
        print(count)
        print('AL', AL.shape)
        print('AR', AR.shape)
        print('C', C.shape)

        if ep < tol and delta_D != 0:
            AL, AR, C, Hl, Hr = dynamic_expansion(AL, AR, C, Hl, Hr, h, delta_D)

            D = D + delta_D

            print('AL new', AL.shape)
            print('AR new', AR.shape)
            print('C new', C.shape)
            print('Hl new', Hl.shape)
            print('Hr new', Hr.shape)

        if size == 'two':
            AL, AR, C, Hl, Hr, e, epl, epr = vumps_two(AL, AR, C, Hl, Hr, h, ep)
        if size == 'three':
            AL, AR, C, Hl, Hr, e, epl, epr = vumps_three(AL, AR, C, Hl, Hr, h, ep)

        gauge_checks(AL.transpose(1, 0, 2), AR.transpose(1, 0, 2), C)
        print('energy', e)
        print('epl', epl)
        print('epr', epr)

        ep = np.maximum(epl, epr)

        print('ep ', ep)
        print()

        energy.append(e)
        error.append(ep)

        count += 1
        
    gs_mps = (AL.transpose(1, 0, 2), AR.transpose(1, 0, 2), C)
    gs_energy = min(energy)
    return gs_mps, gs_energy




