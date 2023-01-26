from dash.dependencies import Input, Output
from dash import Dash, dcc, html
from Data_cleaners.data_cleaner_3 import data_cleaner3
from Distancias_y_otras_funciones.Dameru_Levesnshtein import dam_lev_dist
from Distancias_y_otras_funciones.Wagner_Fisher import lev_dist
from Distancias_y_otras_funciones.dividir_ordenar import Div_and_sort
from Clasificadores.clasificador import clas_direct
import numpy as np

data = data_cleaner3('client account owner - client owner account.csv',
                     ['current account owner'])

data = data.sample(frac=1, random_state=27122, replace=False)
data = data.reset_index()

K = 0.69
tags = clas_direct(data, ['current account owner'], K, lev_dist)
print(tags)  # 39

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [html.I(
        "Some Text"),
     html.Br(),
     dcc.Input(id="input1", type="text", placeholder="",
               style={'marginRight': '10px'}),
     html.Div(id="output1"),
     html.Div(id="output2"),
     html.Div(id="output3"),]
)


@app.callback(
    [Output("output1", "children"),
     Output("output2", "children"),
     Output("output3", "children")],
    [Input("input1", "value")],
)
def update_output(input1):
    if input1 is None:
        return ['', '', '']
    else:
        tag1, tag2, tag3 = function_that_returns_three_values(input1)
        return ([u' {} '.format(tag1), u' {} '.format(tag2), u' {} '.format(tag3)])


def function_that_returns_three_values(input1):
    d = []
    for i in range(len(tags)):
        if ' ' in input1 or ' ' in tags[i]:
            d.append(lev_dist(Div_and_sort(tags[i]), Div_and_sort(
                input1))/max(len(input1), len(tags[i])))
        else:
            d.append(lev_dist(tags[i], input1)/max(len(input1), len(tags[i])))
    if min(d) > K:
        Tag1 = ''
        Tag2 = ''
        Tag3 = ''
    else:
        i1 = np.argmin(d)
        Tag1 = tags[i1]
        tags.remove(Tag1)
        d.remove(d[i1])
        if min(d) > K:
            Tag2 = ''
            Tag3 = ''
        else:
            i2 = np.argmin(d)
            Tag2 = tags[i2]
            tags.remove(Tag2)
            d.remove(d[i2])
            if min(d) > K:
                Tag3 = ''
            else:
                i3 = np.argmin(d)
                Tag3 = tags[i3]
                tags.remove(Tag3)
                d.remove(d[i3])

    return (Tag1, Tag2, Tag3)


if __name__ == "__main__":
    app.run_server(debug=True)
