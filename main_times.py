from Clasificadores.clasificador import clas_direct
from Data_cleaners.data_cleaner_3 import data_cleaner3
from Distancias_y_otras_funciones.Dameru_Levesnshtein import dam_lev_dist
from Distancias_y_otras_funciones.Wagner_Fisher import lev_dist
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import time
import numpy as np

data = data_cleaner3('client account owner - client owner account.csv',
                     ['current account owner'])
# data = data_cleaner3('fleet account owner - fleet owner account.csv',
#                     ['current account owner'])


for K in [0.69, 0.6]:
    if K == 0.69:
        distance = lev_dist
    else:
        distance = dam_lev_dist
    times = []
    for i in np.arange(50):
        datanew = data.sample(frac=1, random_state=27122+(i+1), replace=False)
        datanew = datanew.reset_index()
        t0 = time.time()
        clas_direct(datanew, ['current account owner'], K, distance)
        t1 = time.time()
        times.append(t1-t0)
    if K == 0.69:
        dl = pd.DataFrame({'iter': np.arange(50)+1, 'time': times})
    else:
        ddl = pd.DataFrame({'iter': np.arange(50)+1, 'time': times})

fig1 = px.line(x=dl["iter"], y=dl["time"],
               color=px.Constant("Distancia de Levenshtein"))
fig1.add_trace(go.Scatter(
    x=ddl['iter'], y=ddl['time'], name='Distancia de Damerau-Levenshtein'))
fig1.update_layout(
    title={'text': "Comparación de tiempos de ejecución sobre CAO"})
fig1.show()
