import numpy as np


def lev_dist(a: str, b: str) -> int:
    '''
    :a: primera cadena de caracteres
    :b: segunda cadena de caracteres
    '''
    n = len(a)
    m = len(b)
    d = np.empty((n+1, m+1))
    for i in range(n+1):
        d[i, 0] = (i)
    for j in range(m+1):
        d[0, j] = (j)
    for i in np.arange(1, n+1):
        for j in np.arange(1, m+1):
            ind = 1*(a[i-1] != b[j-1])
            d[i, j] = min(d[i-1, j] + 1,
                          d[i, j-1] + 1,
                          d[i-1, j-1] + ind
                          )
    return (d[n, m])

lev_dist('inaki','isabel')