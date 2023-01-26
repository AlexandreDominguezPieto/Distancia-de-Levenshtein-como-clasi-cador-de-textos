import numpy as np
import string

alphabet = list(string.ascii_lowercase) + [' ', '@'] + list(string.digits)



def dam_lev_dist(a: str, b: str, alf=alphabet) -> int:
    '''
    :a: first string
    :b: second string
    :alf: list of all the elements of the source alphabet for the strings 
    '''
    n = len(a)
    m = len(b)
    alflen = len(alf)
    da = np.zeros(alflen)
    d = np.zeros((n+2, m+2))
    max_dist = n+m
    d[0, 0] = max_dist
    for i in range(n+2):
        d[i, 0] = max_dist
    for i in np.arange(1, n+2):
        d[i, 1] = i-1
    for j in range(m+2):
        d[0, j] = max_dist
    for j in np.arange(1, m+2):
        d[1, j] = j-1
    for i in np.arange(2, n+2):
        db = 0
        for j in np.arange(2, m+2):
            bj = alf.index(b[j-2])
            k = int(da[bj])
            t = db
            if (a[i-2] == b[j-2]):
                ind = 0
                db = j-2
            else:
                ind = 1
                db = 0
            d[i, j] = min(d[i-1, j] + 1,
                          d[i, j-1] + 1,
                          d[i-1, j-1] + ind,
                          d[k, t] + (i-2)-k-1 + (j-2)-t-1 + 1
                          )
        ai = alf.index(a[i-2])
        da[ai] = i-2
    # print(d)
    return (d[n+1, m+1])


dam_lev_dist('aclaro', 'calvos')
