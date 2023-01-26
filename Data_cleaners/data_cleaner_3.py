from array import array
import pandas as pd
import numpy as np
from unidecode import unidecode
import os.path


def parse_string(input_str):
    input_str = input_str.lower()
    input_str = unidecode(input_str)
    input_str = input_str.replace('-', ' ')
    input_str = input_str.replace(',', ' ')
    input_str = input_str.replace(';', ' ')
    input_str = input_str.replace(':', ' ')
    input_str = input_str.replace('.', ' ')
    input_str = input_str.replace('_', ' ')
    input_str = input_str.replace(')', ' ')
    input_str = input_str.replace('(', ' ')
    input_str = input_str.replace('{', ' ')
    input_str = input_str.replace('}', ' ')
    input_str = input_str.replace('[', ' ')
    input_str = input_str.replace(']', ' ')
    return input_str


def data_cleaner3(data_file: str, col_names: list) -> pd.DataFrame:
    '''
    :param data_file: name of the .csv which we want to clean
    :type data_file: string
    :param col_names: list with the names of the columns we want to clean
    :type col_ind: list
    '''
    data = pd.read_csv(os.path.join('Datasets_y_resultados',  data_file))
    data = data.dropna(axis=1, how='all')
    data = data.dropna(axis=0, how='any')
    for i in col_names:
        data[i] = data[i].apply(lambda x: parse_string(x))
    return (data)


data = pd.read_csv(os.path.join('Datasets_y_resultados',  'Damerau_fixed.csv'))
L=len(data)
sum(data['fixed'])