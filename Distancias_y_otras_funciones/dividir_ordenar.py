import numpy as np


def Div_and_sort(s: str) -> str:
    '''
    :s: string we want to divide by blank spaces and sort alphabetically before returning it as a string
    :type s: string
    '''
    L = s.split(' ')
    L.sort()
    t = L[0]
    for i in np.arange(1, len(L)):
        t = t+' '+L[i]
    return (t)
