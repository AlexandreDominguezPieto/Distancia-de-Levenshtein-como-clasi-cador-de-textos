import numpy as np
import pandas as pd
from Distancias_y_otras_funciones.dividir_ordenar import Div_and_sort
from Distancias_y_otras_funciones.Wagner_Fisher import lev_dist
from Distancias_y_otras_funciones.Dameru_Levesnshtein import dam_lev_dist


def clas_direct(data: pd.DataFrame, column_names: tuple, K: float, dist):
    '''
    :param data: dataset to classifie
    :type data: dataframe
    :param column_names: list with the names of the columns we want to classify
    :type column_ind: list
    :param K: parameter used for clasification
    :type K: float    
    :param dist: distance we want to apply
    :type dist: function
    '''
    Tags = []
    assigned_tag = []
    for row in data.iterrows():
        if row[0] == 0:
            Tags = [row[1][column_names[0]]]
            assigned_tag = [row[1][column_names[0]]]
            continue
        d = []
        for j in range(len(Tags)):
            if ' ' in row[1][column_names[0]] or ' ' in Tags[j]:
                d.append(dist(Div_and_sort(row[1][column_names[0]]), Div_and_sort(
                    Tags[j]))/max(len(row[1][column_names[0]]), len(Tags[j])))
            else:
                d.append(dist(row[1][column_names[0]], Tags[j])/(
                    (max(len(row[1][column_names[0]]),  len(Tags[j])))))
        if min(d) > K:
            Tags.append(row[1][column_names[0]])
            assigned_tag.append(row[1][column_names[0]])
        else:
            assigned_tag.append(Tags[np.argmin(d)])
    return (Tags)
