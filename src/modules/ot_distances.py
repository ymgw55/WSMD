import numpy as np


def get_WMD_cost(t1: np.ndarray, t2: np.ndarray):
    n = len(t1)
    m = len(t2)

    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            C[i][j] = np.linalg.norm(t1[i] - t2[j])
    return C


def matrix_elemnt_mean(C):
    n = len(C)
    m = len(C[0])
    m_ = float(np.sum(C) / (n*m))
    return m_


def matrices_elemnt_mean(C1, C2):
    c1 = len(C1)
    c2 = len(C2)
    tot = 0
    for i1 in range(c1):
        for i2 in range(c1):
            C1ii = C1[i1, i2]
            tot += np.sum(np.abs(C1ii-C2)**2)
    m = float(tot / (c1*c1*c2*c2))
    return m


def get_WRD_cost(t1: np.ndarray, t2: np.ndarray):
    n = len(t1)
    m = len(t2)

    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            C[i][j] = max(0, 1.0 - np.dot(t1[i], t2[j]) /
                          (np.linalg.norm(t1[i]) * np.linalg.norm(t2[j])))
    return C
