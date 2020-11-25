import os
import pathlib
import re
  
import base64
from io import BytesIO as _BytesIO
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from PIL import Image
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

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

# get ellipse around centroid for each groups
def ellipse(x, y, n_std = 2, N = 100):
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    cos45 = np.cos(np.pi/4)
    sin45 = np.sin(np.pi/4)    
    R = np.array([[cos45, -sin45], [sin45, cos45]])

    t = np.linspace(0, 2*np.pi, N)

    xs = np.sqrt(1 + pearson) * np.cos(t)
    ys = np.sqrt(1 - pearson) * np.sin(t)

    xp, yp = np.dot(R, [xs, ys])
    x = xp * scale_x + mean_x 
    y = yp * scale_y + mean_y
    return x, y


hoverinfo = ['phenotypes', 'age', 'sex', 'race', 'hospitalization', 'lung_weight_grams', 'treated', 'days_of_disease', 'days_in_hospital', 'sample', 'roi']
custom_hovertemplate = "<br>".join(["" + col + ":%{customdata[" + str(i) + "]}" for i, col in enumerate(hoverinfo)])

# generate pca plot from pca df
def plot_from_data(df, label = "phenotypes", draw_centroid = True, draw_ellipse = True, pca_loadings = None):
    
    fig = go.Figure()

    if df[label].dtype == "object":
        i = 0
        for target, group in df.groupby(label):
            # skip missing values
            if target == "":
                continue

            fig.add_scatter(
                x = group["0.00"],
                y = group["1.00"],
                mode = "markers",
                name = str(target),
                legendgroup = target,
                marker_size = 12,
                opacity = 0.7,
                customdata = group[['phenotypes', 'age', 'sex', 'race', 'hospitalization', 'lung_weight_grams', 'treated', 'days_of_disease', 'days_in_hospital', 'sample', 'roi']],
                hovertemplate = custom_hovertemplate,
                marker_color = colors[i])

            # add center
            fig.add_scatter(
                x = [group.mean()["0.00"]],
                y = [group.mean()["1.00"]],
                marker_symbol = "cross",
                mode = "markers",
                legendgroup = target,
                customdata = None,
                hoverinfo='skip',
                marker_color = "white",
                marker_size = 8,
                opacity = 0.7,
                showlegend = False)

            x, y = ellipse(x = group["0.00"], y = group["1.00"])
            fig.add_scatter(x = x,
                y = y,
                mode = "lines",
                legendgroup = target,
                customdata = None,
                hoverinfo='skip',
                marker_color = colors[i],
                opacity = 0.15,
                showlegend = False)
            i = i + 1

        # add missing values
        group = df[df[label] == ""]
        target = "Unlabelled"
        fig.add_scatter(
            x = group["0.00"],
            y = group["1.00"],
            mode = "markers",
            name = str(target),
            legendgroup = target,
            marker_size = 12,
            opacity = 0.7,
            customdata = group[['phenotypes', 'age', 'sex', 'race', 'hospitalization', 'lung_weight_grams', 'treated', 'days_of_disease', 'days_in_hospital', 'sample', 'roi']],
            hovertemplate = custom_hovertemplate,
            marker_color = colors[i])

        # add center
        fig.add_scatter(
            x = [group.mean()["0.00"]],
            y = [group.mean()["1.00"]],
            marker_symbol = "cross",
            mode = "markers",
            legendgroup = target,
            customdata = None,
            hoverinfo='skip',
            marker_color = "white",
            marker_size = 8,
            opacity = 0.7,
            showlegend = False)

        x, y = ellipse(x = group["0.00"], y = group["1.00"])
        fig.add_scatter(x = x,
            y = y,
            mode = "lines",
            legendgroup = target,
            customdata = None,
            hoverinfo='skip',
            marker_color = colors[i],
            opacity = 0.15,
            showlegend = False)

    else:
        fig.add_scatter(
            x = df["0.00"],
            y = df["1.00"],
            mode = "markers",
            opacity = 0.7,
            customdata = df[['phenotypes', 'age', 'sex', 'race', 'hospitalization', 'lung_weight_grams', 'treated', 'days_of_disease', 'days_in_hospital', 'sample', 'roi']],
            hovertemplate = custom_hovertemplate,
            marker = dict(
                size = 12,
                color=df[label],
                colorbar=dict(title="Colorbar"),
                colorscale="Viridis")
            )

    if len(pca_loadings):
        for component in pca_loadings.index[:15]:
            fig.add_scatter(
                    x = [0, pca_loadings.loc[component]["0"]],
                    y = [0, pca_loadings.loc[component]["1"]],
                    mode = "lines + markers",
                    marker_color = "#e9fae3",
                    #hoverinfo='skip',
                    customdata = ["-".join(component.split("-")[1:])],
                    name = component,
                    text = ["Origin", component],
                    hovertemplate = "%{text}",
                    opacity = 0.2,
                    line = {"dash":"dash"},
                    textposition="bottom center",
                    showlegend = False
                )

    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2", template = "plotly_dark")

    return fig



# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "40rem",
    "padding": "2rem 1rem",
    "background-color": "#111111"#"
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "44rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


# Initialize app
app = dash.Dash(__name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}]
)
server = app.server

#### Load data ####
APP_PATH = str(pathlib.Path(__file__).parent.resolve())

# load pca plot information
pcs = pd.read_csv(
    os.path.join(APP_PATH, os.path.join("data", "pcadata.csv"))
)

loadings = pd.read_csv(
    os.path.join(APP_PATH, os.path.join("data", "loadings.csv")), index_col=0
)

for col, dtype in zip(pcs.columns, pcs.dtypes):
    if dtype == "O":
        pcs[col] = pcs[col].fillna("")

pcs = pcs.infer_objects()

colors = px.colors.qualitative.D3 + px.colors.qualitative.G10 + px.colors.qualitative.T10 + px.colors.qualitative.Plotly

fig = plot_from_data(pcs, pca_loadings = loadings)
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

sidebar = html.Div(
    [
        html.H2("Image Navigator", className="display-4", style = {'margin-left':'5px'}),
        html.Hr(),        
        html.Div([
            html.Div([
                html.H5("Label Datapoints by:"),
                dcc.Dropdown(
                    options=color_options,
                    id="color",
                    value="phenotypes"
                )], style = {"padding":"10px"}
            ),
            html.Hr(),
            html.Div([
                html.H5("IMC Image Markers:"),
                html.H5("RED:", style={"color":"red",'margin-left':'5px'}),
                dcc.Dropdown(
                    options=channel_options,
                    id="red",
                    value=40,
                ),
                html.H5("GREEN:", style={"color":"green",'margin-left':'5px'}),
                dcc.Dropdown(
                    options=channel_options,
                    id="green",
                    value=41
                ),
                html.H5("BLUE:", style={"color":"blue",'margin-left':'5px'}),
                dcc.Dropdown(
                    options=channel_options,
                    id="blue",
                    value=1
                )
            ], style = {"padding":"10px"})
        ])
    
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(
    id="root",
    children=[
        html.Div(
            id="header",
            children=[
                #html.Img(id="logo", src=app.get_asset_url("dash-logo.png")),
                html.H2(children="COVID", className="display-4"),
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
                                    "Principal Component Plot",
                                    id="pcaplot-title",
                                ),
                                html.Hr(),
                                dcc.Graph(
                                    id='live-update-graph',
                                    figure=fig,
                                    style={"width":"inherit", "height":"inherit"}
                                )
                            ], style={'height': '100%', 'overflow': 'contain'}
                        )
                    ], style={'max-width': '50%', "width":"50%", "height":"100%", "max-height":"100%", "padding":"10px"}
                ),

                html.Div(
                    id="right-column",
                    children=[
                        html.Div(
                            id="image",
                            children=[
                                html.Div(
                                    id="image-panel",
                                    children=[
                                        html.H5(
                                            "ROI Viewer (Click on the points to Load Images)"
                                        ),
                                        html.Hr(),
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
                                        ]),
                                        html.P([
                                            html.Pre(id="image-file-data")
                                        ], style={"text-align":"center"})
                                    ], style = {"overflow":"contain"}
                                )
                            ]
                        )
                    ], style={'max-width': '50%', "width":"50%", "height":"100%", "max-height":"100%", "padding":"10px"}
                )
            ]
        )    
    ], style=CONTENT_STYLE
)

# App layout
app.layout = html.Div([sidebar, content])

# Callback Functions for the App
@app.callback(
    Output('hover-data', 'children'),
    [Input('live-update-graph', 'hoverData')])
def display_hover_data(hoverData):
    if hoverData == None:
        return None
    elif 'customdata' in hoverData['points'][0]:
        return hoverData['points'][0]['customdata'][-1] + ".npy"
    return 

@app.callback(
    Output('click-data', 'children'),
    [Input('live-update-graph', 'clickData')])
def display_click_data(clickData):
    if clickData == None:
        return None
    elif "customdata" in clickData['points'][0]:
        return clickData['points'][0]['customdata'][-1]
    return 


@app.callback(
    Output('image-file-data', 'children'),
    [Input('live-update-graph', 'clickData'),
    Input('red', 'value'),
    Input('green', 'value'),
    Input('blue', 'value')])
def display_image_data(clickData, redValue, greenValue, blueValue):
    if clickData == None:
        return None
    if 'customdata' not in clickData['points'][0]:
        return 

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

        output = output.transpose((-1, 1, 0))
        if output.shape[0] > output.shape[1]:
            output = output.transpose((1, 0, 2))
        encoded_image = numpy_to_b64(output)
        del output
        htmlimg = html.Img(src='data:image/png;base64,{}'.format(encoded_image),
            style = {'height': '100%', 'width': '50%', 'object-fit':'contain'})
        return htmlimg
        
    return '\t'.join(["Image Does not exist. Metadata:"] + [str(x) for x in ret])

# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              [Input('color', 'value')])
def update_graph_live(value):
    fig = plot_from_data(pcs, label = value, pca_loadings = loadings)
    return fig

if __name__ == "__main__":
    app.run_server(debug = True)
