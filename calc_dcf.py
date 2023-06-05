import numpy as np
import ncon as nc
import scipy.linalg as spla
import scipy.sparse.linalg as spspla

def lorentzian(x, x0, gamma):
    return (1 / np.pi) * ((0.5 * gamma)/((x - x0)**2 + (0.5 * gamma)**2))

def op_transfer_matrix(A, B, o3):
    D = A.shape[0]

    def left_transfer_op(X):
        tensors = [X.reshape(D, D), A, o3, B.conj()]
        indices = [(4, 5), (5, 2, -2), (1, 2), (4, 1, -1)]
        contord = [4, 5, 2, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    E = spspla.LinearOperator((D * D, D * D), matvec=left_transfer_op)
    wl, lfp_AB = spspla.eigs(E, k=1, which='LM', tol=1e-14)

    lfp_AB = lfp_AB.reshape(D, D)
    return lfp_AB

def calc_dsf(AL, AR, C, 
             excit_energy, excit_states, 
             mom_vec, freq_vec, gamma, O):

    D = AL.shape[0]
    d = AL.shape[1]

    AC = np.tensordot(AL, C, axes=(2, 0))

    VL = spla.null_space(AL.conj().reshape(D * d, D).T)
    VL = VL.reshape(D, d, (d - 1) * D)

    def right_env(X):
        X = X.reshape(D, D)

        tensors = [AL, X, AL.conj()]
        indices = [(-1, 1, 2), (2, 3), (-2, 1, 3)]
        contord = [2, 3, 1]
        XT = nc.ncon(tensors, indices, contord)

        if p == 0:
            XL = np.trace(X) * (C @ C.T.conj())
            return (X - np.exp(+1.0j * p) * (XT - XL)).ravel()
        else:
            return (X - np.exp(+1.0j * p) * XT).ravel()

    print('<o>', nc.ncon([AC, O, AC.conj()], [[1, 3, 4], [2, 3], [1, 2, 4]]))

    O = (O 
         - nc.ncon([AC, O, AC.conj()], [[1, 3, 4], [2, 3], [1, 2, 4]]) 
           * np.eye(d)
           )

    dsf = []
    for i in range(mom_vec.size):
        p = mom_vec[i]
        print('p', p)

        dsf_p = np.zeros(freq_vec.size)
        for j in range(excit_states.shape[2]):
            X = excit_states[i,:,j].reshape((d - 1) * D, D)
            B = np.tensordot(VL, X, axes=(2, 0))

            tensors = [B, AC.conj()]
            indices = [(-1, 2, 1), (-2, 2, 1)]
            contord = [1, 2]
            right_vec = nc.ncon(tensors, indices, contord)

            rand_init = np.random.rand(D, D) - 0.5

            right_env_op = spspla.LinearOperator((D * D, D * D), 
                                                 matvec=right_env
                                                 )

            RB = spspla.gmres(right_env_op, right_vec.ravel(), 
                                            x0=rand_init.ravel(), 
                                            tol=1e-14, 
                                            atol=1e-14
                                            )[0].reshape(D, D)

            tensors = [B, O, AC.conj()]
            indices = [(3, 2, 4), (1, 2), (3, 1, 4)]
            contord = [3, 4, 1, 2]
            t1 = nc.ncon(tensors, indices, contord)

            tensors = [AL, O, AL.conj(), RB]
            indices = [(3, 2, 4), (1, 2), (3, 1, 5), (4, 5)]
            contord = [4, 5, 3, 1, 2]
            t2 = nc.ncon(tensors, indices, contord)

            spec_weight = np.abs(t1 + np.exp(+1j * p) * t2)

            lorentz_j = (2 * np.pi 
                         * lorentzian(freq_vec, excit_energy[i,j], gamma)
                         * spec_weight**2
                         )

            dsf_p += lorentz_j
        dsf.append(dsf_p)
    return np.array(dsf).reshape(mom_vec.size, freq_vec.size).T

def calc_specfxn(AL, AR, C, 
                 excit_energy, excit_states, 
                 mom_vec, freq_vec, gamma, O, o3):
    D = AL.shape[0]
    d = AL.shape[1]

    AC = np.tensordot(AL, C, axes=(2, 0))

    VL = spla.null_space(AL.conj().reshape(D * d, D).T)
    VL = VL.reshape(D, d, (d - 1) * D)

    def left_env(X):
        X = X.reshape(D, D)

        tensors = [X, AR, o3, AR.conj()]
        indices = [(3, 4), (4, 2, -2), (1, 2), (3, 1, -1)]
        contord = [3, 4, 1, 2]
        XT = nc.ncon(tensors, indices, contord)
        return (X - np.exp(-1.0j * p) * XT).ravel()

    def right_env(X):
        X = X.reshape(D, D)

        tensors = [AL, X, AL.conj()]
        indices = [(-1, 1, 2), (2, 3), (-2, 1, 3)]
        contord = [2, 3, 1]
        XT = nc.ncon(tensors, indices, contord)

        if p == 0:
            XL = np.trace(X) * (C @ C.T.conj())
            return (X - np.exp(+1.0j * p) * (XT - XL)).ravel()
        else:
            return (X - np.exp(+1.0j * p) * XT).ravel()

    lz = op_transfer_matrix(AL, AL, o3)

    specfxn = []
    for i in range(mom_vec.size):
        p = mom_vec[i]
        print('p', p)

        specfxn_p = np.zeros(freq_vec.size)
        for j in range(excit_states.shape[2]):
            X = excit_states[i,:,j].reshape((d - 1) * D, D)
            B = np.tensordot(VL, X, axes=(2, 0))

            tensors = [lz, B, o3, AC.conj()]
            indices = [(3, 4), (4, 2, -2), (1, 2), (3, 1, -1)]
            contord = [3, 4, 1, 2]
            left_vec = nc.ncon(tensors, indices, contord)

            tensors = [B, AC.conj()]
            indices = [(-1, 2, 1), (-2, 2, 1)]
            contord = [1, 2]
            right_vec = nc.ncon(tensors, indices, contord)

            rand_init = np.random.rand(D, D) - 0.5

            left_env_op = spspla.LinearOperator((D * D, D * D),
                                                matvec=left_env
                                                )

            right_env_op = spspla.LinearOperator((D * D, D * D), 
                                                 matvec=right_env
                                                 )
            
            LB = spspla.gmres(left_env_op, left_vec.ravel(),
                                           x0=rand_init.ravel(), 
                                           tol=1e-14, 
                                           atol=1e-14
                                           )[0].reshape(D, D)

            RB = spspla.gmres(right_env_op, right_vec.ravel(), 
                                            x0=rand_init.ravel(), 
                                            tol=1e-14, 
                                            atol=1e-14
                                            )[0].reshape(D, D)

            tensors = [lz, B, O, AC.conj()]
            indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 5)]
            contord = [3, 4, 5, 1, 2]
            t1 = nc.ncon(tensors, indices, contord)

            tensors = [lz, AL, O, AL.conj(), RB]
            indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 6), (5, 6)]
            contord = [3, 4, 5, 6, 1, 2]
            t2 = nc.ncon(tensors, indices, contord)

            tensors = [LB, AR, O, AR.conj()]
            indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 5)]
            contord = [3, 4, 5, 1, 2]
            t3 = nc.ncon(tensors, indices, contord)

            spec_weight = np.abs(t1 
                               + np.exp(+1j * p) * t2
                               + np.exp(-1j * p) * t3
                               )

            lorentz_j = (2 * np.pi 
                         * lorentzian(freq_vec, excit_energy[i,j], gamma)
                         * spec_weight**2
                         )

            specfxn_p += lorentz_j
        specfxn.append(specfxn_p)
    return np.array(specfxn).reshape(mom_vec.size, freq_vec.size).T








