import numpy as np
import pandas as pd
import time
import plotly.express as px
from Data_cleaners.data_cleaner_3 import data_cleaner3
from Distancias_y_otras_funciones.dividir_ordenar import Div_and_sort
from Distancias_y_otras_funciones.Wagner_Fisher import lev_dist
from Distancias_y_otras_funciones.Dameru_Levesnshtein import dam_lev_dist
from Clasificadores.clasificador_train_val import classifier


def cross_validation(data: pd.DataFrame, column_names: tuple, CV: int, interval: tuple, nsearch: int, dist):
    '''
    :param data: dataset where the cross validation is going to be applied
    :type column_ind: dataframe
    :param column_names: list with the names of the columns we want to clean
    :type column_ind: list
    :param CV: number of sections in which we want to divide the data
    :type CV: integer    
    :param interval: real interval where we are looking for de best k
    :type interval: tuple
    :param nsearch: number of values in the interval we are going to explore
    :type nsearch: integer
    :param dist: distance we want to apply
    :type dist: function
    '''
    L = len(data)
    p = int(L/CV)

    data=data.sample(n=L,random_state=27122)
    data=data.reset_index()

    step = (interval[1]-interval[0])/nsearch
    posk = [interval[0]]
    for i in range(1, nsearch):
        posk.append(posk[i-1]+step)

    
    df=pd.DataFrame({'k-fold':[],'K':[],'Score':[]})
    for i in range(CV):
        if i == CV-1:
            validation_data = data.iloc[i*p:]
        else:
            validation_data = data.iloc[i*p:(i+1)*p]
        train_data = data.drop(validation_data.index)
        validation_data = validation_data.reset_index()
        train_data = train_data.reset_index()
        for K in posk:
            results=classifier(train_data,validation_data,column_names,K,dist)
            results.insert(0,int(i+1))
            df.loc[len(df.index)]=results
    return(df)