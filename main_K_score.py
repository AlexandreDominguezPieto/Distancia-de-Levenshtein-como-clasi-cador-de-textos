from Clasificadores.clasificador_train_val import classifier
from Data_cleaners.data_cleaner_3 import data_cleaner3
from Distancias_y_otras_funciones.Dameru_Levesnshtein import dam_lev_dist
from Distancias_y_otras_funciones.Wagner_Fisher import lev_dist
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import time
import numpy as np

data = data_cleaner3('fleet account owner - fleet owner account.csv',
                     ['current account owner', 'good account owner'])

Kscore = []
times = []
df = pd.DataFrame({'K': [], 'Score': []})
for K in [0.6, 0.62, 0.63, 0.64]:
    for i in np.arange(100):
        train_data = data.sample(
            frac=0.8, random_state=27122+(i+1), replace=False)
        train_index_real = train_data.index
        train_data = train_data.reset_index()
        train_index = train_data.index
        validation_data = data.drop(train_index_real)
        t0 = time.time()
        Kscore = classifier(train_data, validation_data, [
            'current account owner', 'good account owner'], K, dam_lev_dist)
        t1 = time.time()
        times.append(t1-t0)
        df.loc[len(df.index)] = Kscore


fig = go.Figure()
fig.add_trace(go.Box(x=df['K'], y=df['Score'], boxpoints='all', boxmean=True))
fig.update_layout(title={'text': "Damerau-Levenshtein distance"})
fig.show()
