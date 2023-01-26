from Clasificadores.clasificador_train_val import classifier
from Data_cleaners.data_cleaner_3 import data_cleaner3
from Distancias_y_otras_funciones.Dameru_Levesnshtein import dam_lev_dist
from Distancias_y_otras_funciones.Wagner_Fisher import lev_dist
from Distancias_y_otras_funciones.cross_val import cross_validation
import plotly.express as px
import plotly.graph_objects as go


data = data_cleaner3('fleet account owner - fleet owner account.csv',
                     ['current account owner', 'good account owner'])


results = cross_validation(data, ['current account owner',
                                  'good account owner'], 10, [0.6, 0.7], 10, dam_lev_dist)


fig = go.Figure()
fig.add_trace(
    go.Box(x=results['K'], y=results['Score'], boxpoints='all', boxmean=True))
fig.update_layout(title={'text': "Levenshtein distance"})
fig.show()


