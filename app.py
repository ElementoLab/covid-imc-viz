# to dos
# 
# Change Color Scheme
# Caching for images

import os
import pathlib
import re

from io import BytesIO as _BytesIO
import requests

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np
from PIL import Image

import plotly.express as px
import plotly.graph_objects as go

import json

from flask_caching import Cache
import datetime

# get ellipse around centroid for each groups
def ellipse(x, y, n_std=2, N=100):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    cos45 = np.cos(np.pi / 4)
    sin45 = np.sin(np.pi / 4)
    R = np.array([[cos45, -sin45], [sin45, cos45]])

    t = np.linspace(0, 2 * np.pi, N)

    xs = np.sqrt(1 + pearson) * np.cos(t)
    ys = np.sqrt(1 - pearson) * np.sin(t)

    xp, yp = np.dot(R, [xs, ys])
    x = xp * scale_x + mean_x
    y = yp * scale_y + mean_y
    return x, y


hoverinfo = [
    "Disease Group",
    "age",
    "sex",
    "race",
    "lung_weight_grams",
    "days_of_disease",
    "days_in_hospital",
    "roi",
]

col_interest = [
    'roi',
    '0',
    '1',
    'age',
    'sex',
    'race',
    'smoker',
    'disease',
    'Disease Group',
    'classification',
    'cause_of_death',
    'lung_weight_grams',
    'comorbidities',
    'treatment',
    'days_intubated',
    'days_of_disease',
    'days_in_hospital',
    'fever_temperature_celsius',
    'cough',
    'shortness_of_breath',
    'Other lung lesions',
    'PLT/mL',
    'D-dimer (mg/L)',
    'WBC',
    'LY%',
    'PMN %'
]

custom_hovertemplate = "<br>".join(
    ["" + col.capitalize() + ": %{customdata[" + str(i) + "]}" for i, col in enumerate(hoverinfo)]
)

# generate pca plot from pca df
def plot_from_data(
    df, label="Disease Group", draw_centroid=True, draw_ellipse=True, pca_loadings=None
):

    fig = go.Figure()

    if df[label].dtype == "object":
        i = 0
        for target, group in df.groupby(label):
            # skip missing values
            if target == "":
                continue

            customdata = group[hoverinfo]
            numeric_col = customdata.select_dtypes("float64").columns
            customdata.loc[:, numeric_col] = customdata.select_dtypes("float64").astype(
                str
            )
            customdata = customdata.replace(["", "nan"], "NA", regex=False)

            fig.add_scatter(
                x=group['0'],
                y=group['1'],
                mode="markers",
                name=str(target),
                legendgroup=target,
                marker_size=12,
                opacity=0.7,
                customdata=customdata,
                hovertemplate=custom_hovertemplate,
                marker_color=colors[i],
            )

            # add center
            fig.add_scatter(
                x=[group.mean()['0']],
                y=[group.mean()['1']],
                marker_symbol="cross",
                mode="markers",
                legendgroup=target,
                customdata=None,
                hoverinfo="skip",
                marker_color="white",
                marker_size = 8,
                opacity = 0.7,
                showlegend=False,
            )

            x, y = ellipse(x=group['0'], y=group['1'])
            fig.add_scatter(
                x=x,
                y=y,
                mode="lines",
                legendgroup=target,
                customdata=None,
                hoverinfo="skip",
                marker_color=colors[i],
                opacity = 0.15,
                showlegend=False,
            )
            i = i + 1

        # add missing values
        group = df[df[label] == ""]
        target = "Unlabelled"
        fig.add_scatter(
            x=group['0'],
            y=group['1'],
            mode="markers",
            name=str(target),
            legendgroup=target,
            marker_size = 12,
            opacity = 0.7,
            customdata=customdata,
            hovertemplate=custom_hovertemplate,
            marker_color=colors[i],
        )

        # add center
        fig.add_scatter(
            x=[group.mean()['0']],
            y=[group.mean()['1']],
            marker_symbol="cross",
            mode="markers",
            legendgroup=target,
            customdata=None,
            hoverinfo="skip",
            marker_color="white",
            marker_size = 8,
            opacity = 0.7,
            showlegend=False,
        )

        x, y = ellipse(x=group['0'], y=group['1'])
        fig.add_scatter(
            x=x,
            y=y,
            mode="lines",
            legendgroup=target,
            customdata=None,
            hoverinfo="skip",
            marker_color=colors[i],
            opacity = 0.15,
            showlegend=False,
        )

    else:
        customdata = df[hoverinfo]
        numeric_col = customdata.select_dtypes("float64").columns
        customdata.loc[:, numeric_col] = customdata.select_dtypes("float64").astype(str)
        customdata = customdata.replace(["", "nan"], "NA", regex=False)

        fig.add_scatter(
            x=df['0'],
            y=df['1'],
            mode="markers",
            opacity=0.7,
            customdata=customdata,
            hovertemplate=custom_hovertemplate,
            showlegend=False,
            marker=dict(
                size=12,
                color=df[label],
                colorbar=dict(title="Colorbar"),
                colorscale="Viridis",
            ),
        )

    if pca_loadings is not None:
        for component in pca_loadings.index[:15]:
            fig.add_scatter(
                x=[0, pca_loadings.loc[component]["0"]],
                y=[0, pca_loadings.loc[component]["1"]],
                mode="lines + markers",
                marker_color="#e9fae3",
                # hoverinfo='skip',
                customdata=["-".join(component.split("-")[1:])],
                name=component,
                text=["Origin", component],
                hovertemplate="%{text}",
                opacity=0.2,
                line={"dash": "dash"},
                textposition="bottom center",
                showlegend=False,
            )

    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2", template="plotly_dark")

    return fig


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "27rem",
    "padding": "2rem 1rem",
    "background-color": "#111111",  # "
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "30rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

empty_fig = {
    "data": [],
    "layout": go.Layout(
        xaxis={
            "showticklabels": False,
            "ticks": "",
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "showticklabels": False,
            "ticks": "",
            "showgrid": False,
            "zeroline": False,
        },
        template="plotly_dark",
    ),
}


# Initialize app
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server
cache = Cache(server, config={
    'CACHE_TYPE': 'simple',

    # should be equal to maximum number of users on the app at a single time
    # higher numbers will store more data in the filesystem / redis cache
    'CACHE_THRESHOLD': 25,
    "CACHE_DEFAULT_TIMEOUT": 300
})

#### Load data ####
APP_PATH = str(pathlib.Path(__file__).parent.resolve())

# load pca plot information
pcs = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "pcadata.csv")))
pcs = pcs[col_interest]

loadings = pd.read_csv(
    os.path.join(APP_PATH, os.path.join("data", "loadings.csv")), index_col=0
)

for col, dtype in zip(pcs.columns, pcs.dtypes):
    if dtype == "O":
        pcs[col] = pcs[col].fillna("")

pcs = pcs.infer_objects()

colors = (
    px.colors.qualitative.D3
    + px.colors.qualitative.G10
    + px.colors.qualitative.T10
    + px.colors.qualitative.Plotly
)

fig = plot_from_data(pcs, pca_loadings=loadings)
fig.update_layout(clickmode="event+select")

# load options for dropdown box

color_options = []
for label in col_interest[3:]:
    color_options.append({"label": label.capitalize().replace("_"," "), "value": label})

with open("metadata/processed_stack.upload_info.json") as f:
    roi2url = json.load(f)

with open("metadata/processed_stack.channel_info.json") as f:
    roi2channel = json.load(f)

channel_options = []
for value, label in enumerate(roi2channel):
    channel_options.append({"label": label, "value": value})


sidebar = html.Div(
    [
        html.H2("Image Navigator", className="display-4", style={"margin-left": "5px"}),
        html.Hr(),
        html.Div(
            [
                html.Div(
                    [
                        html.H5("Label Datapoints by:"),
                        dcc.Dropdown(
                            options=color_options, id="color", value="Disease Group"
                        ),
                    ],
                    style={"padding": "10px"},
                ),
                html.Hr(),
                html.Div(
                    [
                        html.H5("IMC Image Markers:"),
                        html.H5("RED:", style={"color": "red", "margin-left": "5px"}),
                        dcc.Dropdown(options=channel_options, id="red", value = 13),
                        html.H5(
                            "GREEN:", style={"color": "green", "margin-left": "5px"}
                        ),
                        dcc.Dropdown(options=channel_options, id="green", value = 28),
                        html.H5("BLUE:", style={"color": "blue", "margin-left": "5px"}),
                        dcc.Dropdown(options=channel_options, id="blue", value = 10),
                    ],
                    style={"padding": "10px"},
                ),
            ]
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(
    id="root",
    children=[
        html.Div(
            id="header",
            children=[
                # html.Img(id="logo", src=app.get_asset_url("dash-logo.png")),
                html.H2(children="SARS-CoV-2 (COVID-19) IMC Explorer", className="display-4"),
                html.P(
                    id="description",
                    children="Interactive Plot to explore results from \
                    The spatio-temporal landscape of lung pathology in SARS-CoV-2 infection,\
                    (https://www.medrxiv.org/content/10.1101/2020.10.26.20219584v1) Figure 3.\
                    Produced by: Elemento Lab, Weill Cornell Medicine",
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
                                    id="live-update-graph",
                                    figure=fig,
                                    style={"width": "inherit", "height": "inherit"},
                                ),
                            ],
                            style={"height": "100%", "overflow": "contain"},
                        )
                    ],
                    style={
                        "max-width": "48%",
                        "width": "48%",
                        "height": "100%",
                        "max-height": "100%",
                        "padding": "10px",
                    },
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
                                        # html.P([
                                        #    html.Pre(id="image-file-data")
                                        # ], style={"text-align":"center"})
                                        dcc.Graph(
                                            id="image-file-data",
                                            figure=empty_fig,
                                            style={
                                                "width": "inherit",
                                                "height": "50vh",
                                                "align": "center",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                dcc.Markdown(
                                                    """
                                                **Hover Data**
                                                    Mouse over values in the graph.
                                            """
                                                ),
                                                html.Pre(id="hover-data"),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                dcc.Markdown(
                                                    """
                                                **Click Data**
                                                Click on points in the graph.
                                                """
                                                ),
                                                html.Pre(id="click-data"),
                                            ]
                                        ),
                                    ],
                                    style={"height": "100%", "overflow": "contain"},
                                )
                            ],
                        )
                    ],
                    style={
                        "max-width": "48%",
                        "width": "48%",
                        "height": "100%",
                        "max-height": "100%",
                        "padding": "10px",
                    },
                ),
            ],
        ),
    ],
    style=CONTENT_STYLE,
)

# App layout
app.layout = html.Div([sidebar, content])

@cache.memoize(timeout=300)
def fetch_array(sample_name):
    if sample_name in roi2url:

        req_url = roi2url[sample_name]["shared_download_url"]
        print("Downloading: {} at {}".format(sample_name, datetime.datetime.now().strftime('%H:%M:%S')))
        
        res = requests.get(req_url)
        print("Download Completed: {} at {}".format(sample_name, datetime.datetime.now().strftime('%H:%M:%S')))
        
        res.raise_for_status()

        img = np.load(_BytesIO(res.content))["stack"]

        return img
    return None

# Callback Functions for the App
@app.callback(
    Output("hover-data", "children"), [Input("live-update-graph", "hoverData")]
)
def display_hover_data(hoverData):
    if hoverData == None:
        return None
    elif "customdata" in hoverData["points"][0]:
        return hoverData["points"][0]["customdata"][-1]
    return


@app.callback(
    Output("click-data", "children"), [Input("live-update-graph", "clickData")]
)
def display_click_data(clickData):
    if clickData == None:
        return None
    elif "customdata" in clickData["points"][0]:
        return "LOADING: {}".format(clickData["points"][0]["customdata"][-1])
    return


@app.callback(
    Output("image-file-data", "figure"),
    [
        Input("live-update-graph", "clickData"),
        Input("red", "value"),
        Input("green", "value"),
        Input("blue", "value"),
    ],
)
def display_image_data(clickData, redValue, greenValue, blueValue):
    # img_dir = "assets/"
    if clickData == None:
        return empty_fig
    if "customdata" not in clickData["points"][0]:
        return empty_fig

    img_name = clickData["points"][0]["customdata"][-1]
    # print(img_name)
    red_channel = redValue
    green_channel = greenValue
    blue_channel = blueValue
    ret = [img_name, red_channel, green_channel, blue_channel]
    # img_names = os.listdir(img_dir)

    img = fetch_array(img_name)
    if img is None:
        print(img_name, "not found")
        return empty_fig
    
    output = np.zeros(shape=(3, img.shape[1], img.shape[2]))
    output[0] = img[red_channel]
    output[1] = img[green_channel]
    output[2] = img[blue_channel]

    output = output.transpose((-1, 1, 0))
    if output.shape[0] > output.shape[1]:
        output = output.transpose((1, 0, 2))

    # print(output.shape)
    fig2 = px.imshow(output)
    fig2.update_xaxes(showticklabels=False, showgrid=False)
    fig2.update_yaxes(showticklabels=False, showgrid=False)
    fig2.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_dark"
    )

    return fig2
'''
    if img_name in roi2url:

        req_url = roi2url[img_name]["shared_download_url"]
        res = requests.get(req_url)

        res.raise_for_status()

        print("Downloading: " + img_name)
        img = np.load(_BytesIO(res.content))["stack"]

        output = np.zeros(shape=(3, img.shape[1], img.shape[2]))
        output[0] = img[red_channel]
        output[1] = img[green_channel]
        output[2] = img[blue_channel]
'''
# Multiple components can update everytime interval gets fired.
@app.callback(Output("live-update-graph", "figure"), [Input("color", "value")])
def update_graph_live(value):
    fig = plot_from_data(pcs, label=value, pca_loadings=loadings)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
