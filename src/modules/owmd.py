import numpy as np


def sinkhorn_knopp(a, b, D, lambda1, lambda2,
                   sigma, max_iter, verbose=False, log=False):
    a = np.asarray(a, dtype=np.float64)  # alpha in paper
    b = np.asarray(b, dtype=np.float64)  # beta in paper
    D = np.asarray(D, dtype=np.float64)
    if len(a) == 0:
        a = np.ones((D.shape[0],), dtype=np.float64) / D.shape[0]
    if len(b) == 0:
        b = np.ones((D.shape[1],), dtype=np.float64) / D.shape[1]
    Nini = len(a)  # N in paper
    Nfin = len(b)  # M in paper
    if log:
        log = {'err': []}
    u = np.ones(Nini) / Nini  # k1 in paper
    v = np.ones(Nfin) / Nfin  # k2 in paper
    K = np.zeros(D.shape)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            lij = np.abs(float(i) / Nini - float(j) / Nfin) / \
                np.sqrt(1 / float(Nini)**2 + 1 / float(Nfin)**2)
            p = np.exp(-np.square(lij) / (2 * np.square(sigma))) / \
                (sigma * np.sqrt(2 * np.pi))
            s = float(lambda1) / \
                (np.square(float(i) / Nini - float(j) / Nfin) + 1)
            d = D[i][j]
            K[i][j] = p * np.exp((s - d) / float(lambda2))
    iter = 0
    err = 1
    while (iter < max_iter):
        uprev = u
        vprev = v
        u = np.divide(a, np.dot(K, v))
        v = np.divide(b, np.dot(K.T, u))
        if (np.any(np.dot(K.T, u) == 0) or
                np.any(np.isnan(u)) or np.any(np.isnan(v)) or
                np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            # print('Warning: numerical errors at iteration', iter)
            u = uprev
            v = vprev
            break
        if iter % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = np.dot(np.dot(np.diag(u), K), np.diag(v))
            err = np.linalg.norm((np.sum(transp, axis=0) - b))**2
            if log:
                log['err'].append(err)

            if verbose:
                if iter % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(iter, err))
        iter = iter + 1
    if log:
        log['u'] = u
        log['v'] = v
    T = np.dot(np.dot(np.diag(u), K), np.diag(v))
    if log:
        return T, log
    else:
        return T
