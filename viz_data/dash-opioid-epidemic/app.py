import os
import pathlib
import re
  
import base64
from io import BytesIO as _BytesIO

import dash
import json
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from PIL import Image
from dash.dependencies import Input, Output, State
import plotly.express as px



# Image utility functions
def pil_to_b64(im, enc_format='png', verbose=False, **kwargs):
    """
    Converts a PIL Image into base64 string for HTML displaying
    :param im: PIL Image object
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :return: base64 encoding
    """

    buff = _BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded

def numpy_to_b64(np_array, enc_format='png', scalar=True, **kwargs):
    """
    Converts a numpy image into base 64 string for HTML displaying
    :param np_array:
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :param scalar:
    :return:
    """
    # Convert from 0-1 to 0-255
    if scalar:
        np_array = np.uint8(255 * np_array)
    else:
        np_array = np.uint8(np_array)

    im_pil = Image.fromarray(np_array)

    return pil_to_b64(im_pil, enc_format, **kwargs)


styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

# Initialize app
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ]
)
server = app.server

#### Load data ####
APP_PATH = str(pathlib.Path(__file__).parent.resolve())

# load pca plot information
pcs = pd.read_csv(
    os.path.join(APP_PATH, os.path.join("data", "pcadata.csv"))
)
for col, dtype in zip(pcs.columns, pcs.dtypes):
    if dtype == "O":
        pcs[col] = pcs[col].fillna("")

pcs = pcs.infer_objects()
pcs["size"] = 1
#del pcs["size"]
fig = px.scatter(data_frame = pcs,
                x = '0.00',
                y = '1.00',
                color = 'phenotypes',
                hover_data = ['phenotypes', 'age', 'sex', 'race', 'hospitalization', 'lung_weight_grams', 'treated', 'days_of_disease', 'days_in_hospital', 'sample', 'roi'],
                opacity = 0.7,
                labels = {'0.00':'PC1', '1.00':'PC2'},
                template="plotly_dark")
fig.update_traces(marker=dict(size=12))

fig.update_layout(clickmode='event+select')


# load options for dropdown box
channel_options = []
channel_info = pd.read_csv(
    os.path.join(APP_PATH, os.path.join("data", "channel_info.csv"))
)
for value, label in enumerate(channel_info["channel"]):
    channel_options.append({"label":label, "value":value})

color_options = []

for label in pcs.columns[63:-1]:
    color_options.append({"label":label, "value":label})

# App layout
app.layout = html.Div(
    id="root",
    children=[
        html.Div(
            id="header",
            children=[
                #html.Img(id="logo", src=app.get_asset_url("dash-logo.png")),
                html.H4(children="COVID"),
                html.P(
                    id="description",
                    children="Interactive Plot to explore results from \
                    The spatio-temporal landscape of lung pathology in SARS-CoV-2 infection,\
                    (https://www.medrxiv.org/content/10.1101/2020.10.26.20219584v1) Figure 3",
                ),
            ],
        ),
        html.Div(
            id="app-container",
            children=[
                html.Div(
                    id="left-column",
                    children=[
                        html.Div(
                            id="pcaplot-container",
                            children=[
                                html.H5(
                                    "PCA Plot",
                                    id="pcaplot-title",
                                ),
                                dcc.Graph(
                                    id='live-update-graph',
                                    figure=fig,
                                    style={"height":"100%"}
                                )
                            ],
                            style={'height': '100%', "padding":"10px"}
                        )
                    ],
                    style={'width': '50%'}
                ),
            
                html.Div(
                    id="right-column",
                    children=[
                        html.Div([
                            html.H5("Color by:"),
                            dcc.Dropdown(
                                options=color_options,
                                id="color",
                                value="phenotypes"
                            )
                        ], style = {"padding":"10px", "width":"30%"}
                        ),
                        html.Div([
                            html.H5("Markers:"),
                            html.H5("RED:", style={"color":"red",'margin-left':'25px'}),
                            dcc.Dropdown(
                                options=channel_options,
                                id="red",
                                value=40,
                                style={"width":"150px", "background-color":"red",'margin-left':'10px','margin-right':'10px'}
                            ),
                            html.H5("GREEN:", style={"color":"green",'margin-left':'25px'}),
                            dcc.Dropdown(
                                options=channel_options,
                                id="green",
                                value=41,
                                style={"width":"150px", "background-color":"green",'margin-left':'10px','margin-right':'10px'}
                            ),
                            html.H5("BLUE:", style={"color":"blue",'margin-left':'25px'}),
                            dcc.Dropdown(
                                options=channel_options,
                                id="blue",
                                value=1,
                                style={"width":"150px", "background-color":"blue",'margin-left':'10px','margin-right':'10px'}
                            )
                        ],
                        style={"display": "flex", "flexWrap": "wrap", "padding":"10px"}
                    ),
                    html.Div(
                        id="image",
                        children=[
                            html.Div(
                                id="image-panel",
                                children=[
                                    html.Div([
                                        html.H5(
                                            "ROI Viewer (Click on the points to Load Images)"
                                        ),
                                        html.Pre(id="image-file-data")
                                    ]),
                                    html.Div([
                                        dcc.Markdown("""
                                            **Hover Data**
                                                Mouse over values in the graph.
                                        """),
                                        html.Pre(id='hover-data')
                                    ]),
                                    html.Div([
                                        dcc.Markdown("""
                                            **Click Data**
                                            Click on points in the graph.
                                            """),
                                        html.Pre(id='click-data'),
                                    ])
                                ]
                            )
                        ], style = {"padding":"10px"}
                    )
                ],style={'width': '50%'}
                )
            ]
        )    
    ],
    style={"width":"95%"}
)


@app.callback(
    Output('hover-data', 'children'),
    [Input('live-update-graph', 'hoverData')])
def display_hover_data(hoverData):
    if hoverData == None:
        return None
    return hoverData['points'][0]['customdata'][-1] + ".png"

@app.callback(
    Output('click-data', 'children'),
    [Input('live-update-graph', 'clickData')])
def display_click_data(clickData):
    if clickData == None:
        return None
    return clickData['points'][0]['customdata'][-1]


@app.callback(
    Output('image-file-data', 'children'),
    [Input('live-update-graph', 'clickData'),
    Input('red', 'value'),
    Input('green', 'value'),
    Input('blue', 'value')])
def display_image_data(clickData, redValue, greenValue, blueValue):
    if clickData == None:
        return None
    img_name = clickData['points'][0]['customdata'][-1] + ".npy"
    red_channel = redValue
    green_channel = greenValue
    blue_channel = blueValue
    ret = [img_name, red_channel, green_channel, blue_channel]
    img_names = os.listdir("data/images/")

    if img_name in img_names:
        img = np.load("data/images/" + img_name)
        output = np.zeros(shape=(3, img.shape[1], img.shape[2]))
        output[0] = img[red_channel]
        output[1] = img[green_channel]
        output[2] = img[blue_channel]

        encoded_image = numpy_to_b64(output.transpose((-1, 1, 0)))
        del output
        return html.Img(src='data:image/png;base64,{}'.format(encoded_image))
        
    return '\t'.join(["Image Does not exist. Metadata:"] + [str(x) for x in ret])

# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              [Input('color', 'value')])
def update_graph_live(value):
    fig = px.scatter(data_frame = pcs,
                x = '0.00',
                y = '1.00',
                color = str(value),
                hover_data = ['phenotypes', 'age', 'sex', 'race', 'hospitalization', 'lung_weight_grams', 'treated', 'days_of_disease', 'days_in_hospital', 'sample', 'roi'],
                opacity = 0.7,
                labels = {'0.00':'PC1', '1.00':'PC2'},
                template="plotly_dark")
    fig.update_traces(marker=dict(size=12))
    return fig
if __name__ == "__main__":
    app.run_server(debug = True)
