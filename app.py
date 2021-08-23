import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import pickle
import plotly.graph_objs as go
from dash.dependencies import Input, Output

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

app = dash.Dash()

server = app.server

df = pd.read_csv('cleaned_churn_data.csv')
df.drop(columns=['Unnamed: 0'], axis = 1, inplace=True)
X = df.drop(columns=['churn'], axis = 1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
loaded_model = pickle.load(open('dtree.sav', 'rb'))

app.layout = html.Div([
    # Setting the main title of the Dashboard
    html.H1("Chrun Data Data Analysis", style={"textAlign": "center"}), html.Div([
                html.H1("Decision Tree Modeling Confusion Matrix", 
                        style={'textAlign': 'center'}),
                # Adding the first dropdown menu and the subsequent time-series graph
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Decision Tree', 'value': 'dtree_default'},
                                      {'label': 'kNN','value': 'kNN'}], 
                             multi=False,value=['dtree_default'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='conf_matrix'),
                html.H1("Decision Tree Modeling Feature Importance", style={'textAlign': 'center'}),
                # Adding the second dropdown menu and the subsequent time-series graph
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Decision Tree', 'value': 'dtree_default'},
                                      {'label': 'kNN','value': 'kNN'}], 
                             multi=False,value=['dtree_default'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='feat_imp')
            ], className="container"),
        ])
        
@app.callback(Output('conf_matrix', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    title = 'Decision Tree Confusion Matrix'
    cm = list(confusion_matrix(y_test, loaded_model.predict(X_test)))
    labels = ['Retained', 'Churned']
    data = go.Heatmap(z=cm, y=labels, x=labels)
    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            annotations.append(
                {
                    "x": labels[i],
                    "y": labels[j],
                    "font": {"color": "white"},
                    "text": str(value),
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False
                }
            )
    layout = {
        "title": title,
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
        "annotations": annotations
    }
    fig = go.Figure(data=data, layout=layout)
    return fig
    
@app.callback(Output('feat_imp', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    title = 'Decision Tree Feature Importances'
    feature_list = list(X_train.columns)
    importances = loaded_model.feature_importances_
    data = go.Bar(x = importances, y = feature_list, orientation = 'h')
    fig = go.Figure(data = data)
    return fig
    
if __name__ == '__main__':
    app.run_server()