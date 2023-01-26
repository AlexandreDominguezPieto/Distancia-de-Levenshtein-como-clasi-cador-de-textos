import numpy as np
import pandas as pd
from Distancias_y_otras_funciones.dividir_ordenar import Div_and_sort
from Distancias_y_otras_funciones.Wagner_Fisher import lev_dist
from Distancias_y_otras_funciones.Dameru_Levesnshtein import dam_lev_dist


def classifier(train_data: pd.DataFrame, validation_data: pd.DataFrame, column_names: tuple, K: float, dist) -> float:
    '''
    :param train_data: dataset used for training
    :type train_data: dataframe
    :param validation_data: dataset used for validation
    :type validation_data: dataframe
    :param column_names: list with the names of the columns we want to classify and evaluate
    :type column_ind: list
    :param K: parameter used for clasification
    :type K: float    
    :param dist: distance we want to apply
    :type dist: function
    '''
    Tags = list(set(list(train_data[column_names[1]])))
    assigned_tag = []
    for row in validation_data.iterrows():
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

    validation_data['assigned tag'] = assigned_tag
    Score = []
    for row in validation_data.iterrows():
        if ' ' in row[1]['assigned tag'] or ' ' in row[1][column_names[1]]:
            Score.append(Div_and_sort(row[1]['assigned tag']) == Div_and_sort(
                row[1][column_names[1]]))
        else:
            Score.append(row[1]['assigned tag'] ==
                         row[1][column_names[1]])
    validation_data['Score'] = Score
    return ([K, sum(Score)/len(validation_data)])
