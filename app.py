# to dos
#
# Domain Rerouting
# Image Redirection

import os
from pathlib import Path
import re
import json
import datetime
import tempfile
from typing import (
    Tuple,
    List,
    Mapping,
    Union,
    Optional,
    Literal,
    overload,
)

from io import BytesIO as _BytesIO
import requests
import parmap

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np
from PIL import Image

import plotly
import plotly.express as px
import plotly.graph_objects as go

from flask_caching import Cache

pd.options.mode.chained_assignment = None


# Some custom types
DataFrame = Union[pd.DataFrame]
Array = Union[np.ndarray]
Fig = Union[plotly.graph_objects.Figure]

# Config and defaults
APP_PATH = Path(__file__).parent.resolve()
CONFIG_FILE = "config.json"
config = json.load(open(APP_PATH / CONFIG_FILE, "r"))
# DEFAULT_FIGURE = (
#     config["DEFAULT_ROI"],
#     config["DEFAULT_CHANNELS"],
# )  # change to (None, None) to display empty
DEFAULT_FIGURE = (None, None)

# Set up cache
cache = Cache(
    config=dict(
        **config["CACHE_CONFIG"],
        **(
            {"CACHE_DIR": tempfile.TemporaryDirectory().name}
            if config["CACHE_CONFIG"]["CACHE_TYPE"] == "filesystem"
            else {}
        ),
    )
)


@overload
def now(format: Literal[True]) -> str:
    ...


@overload
def now(format: Literal[False]) -> datetime.datetime:
    ...


def now(format: bool = True) -> Union[str, datetime.datetime]:
    n = datetime.datetime.now()
    return n.strftime("%H:%M:%S") if format else n


def ellipse(
    x: float, y: float, n_std: int = 2, N: int = 100
) -> Tuple[float, float]:
    """
    Get ellipse around centroid for each group.
    """
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


def plot_from_data(
    df: DataFrame,
    color_map: Mapping[str, List[str]],
    label: str = "Disease Group",
    draw_centroid: bool = True,
    draw_ellipse: bool = True,
    pca_loadings: DataFrame = None,
) -> Fig:
    """
    Generate pca plot from pca dataframe with metadata.
    """
    # create figure layout
    fig = go.Figure()

    # if label is categorical
    if df[label].dtype.name == "category":
        i = 0

        # get color map
        if label in color_map:
            colors = color_map[label]
        else:
            colors = color_map["default"]

        # iterate through each category
        for target, group in df.groupby(label):

            # skip missing values
            if target == "":
                continue

            # retrieve hoverinformation / roi information and fill empty values with empty string
            customdata = group[config["HOVERINFO"]]
            numeric_col = customdata.select_dtypes("float64").columns
            customdata.loc[:, numeric_col] = customdata.select_dtypes(
                "float64"
            ).astype(str)
            customdata = customdata.replace(["", "nan"], "NA", regex=False)

            # add pca scatter plot
            fig.add_scatter(
                x=group["0"],
                y=group["1"],
                mode="markers",
                name=str(target),
                legendgroup=target,
                marker_size=12,
                opacity=0.7,
                customdata=customdata,
                hovertemplate=custom_hovertemplate,
                marker_color=colors[i],
            )

            # add centroids
            fig.add_scatter(
                x=[group.mean()["0"]],
                y=[group.mean()["1"]],
                marker_symbol="cross",
                mode="markers",
                legendgroup=target,
                customdata=None,
                hoverinfo="skip",
                marker_color="white",
                marker_size=8,
                opacity=0.7,
                showlegend=False,
            )

            # add ellipse denoting 95% percent confidence covariance
            x, y = ellipse(x=group["0"], y=group["1"])
            fig.add_scatter(
                x=x,
                y=y,
                mode="lines",
                legendgroup=target,
                customdata=None,
                hoverinfo="skip",
                marker_color=colors[i],
                opacity=0.15,
                showlegend=False,
            )
            i = i + 1

        # add missing values at the end
        group = df[df[label] == ""]

        if len(group) > 0:
            target = "Unlabelled"
            fig.add_scatter(
                x=group["0"],
                y=group["1"],
                mode="markers",
                name=str(target),
                legendgroup=target,
                marker_size=12,
                opacity=0.7,
                customdata=customdata,
                hovertemplate=custom_hovertemplate,
                marker_color="grey",
            )

            # add missing value centroid
            fig.add_scatter(
                x=[group.mean()["0"]],
                y=[group.mean()["1"]],
                marker_symbol="cross",
                mode="markers",
                legendgroup=target,
                customdata=None,
                hoverinfo="skip",
                marker_color="white",
                marker_size=8,
                opacity=0.7,
                showlegend=False,
            )

            # add ellipse denoting 95% percent confidence covariance for missing values
            x, y = ellipse(x=group["0"], y=group["1"])
            fig.add_scatter(
                x=x,
                y=y,
                mode="lines",
                legendgroup=target,
                customdata=None,
                hoverinfo="skip",
                marker_color="grey",
                opacity=0.15,
                showlegend=False,
            )

    else:
        # if selected label is numeric

        # get hoverinformation
        customdata = df[config["HOVERINFO"]]
        numeric_col = customdata.select_dtypes("float64").columns
        customdata.loc[:, numeric_col] = customdata.select_dtypes(
            "float64"
        ).astype(str)
        customdata = customdata.replace(["", "nan"], "NA", regex=False)

        # add scatterplot and colorbar for values
        fig.add_scatter(
            x=df["0"],
            y=df["1"],
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

    # add pca loadings if provided
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

    # add figure axis and theme
    fig.update_layout(
        xaxis_title="PC1", yaxis_title="PC2", template="plotly_dark"
    )

    return fig


@cache.memoize(timeout=300)
def get_cxy_array(img_name: str, channels: List[int]) -> Array:
    """Get CXY image array in a cached manner."""
    res = parmap.starmap_async(
        fetch_array,
        zip([img_name] * len(channels), roi2channel.loc[list(channels)]),
    )

    # In serial:
    # return np.asarray(
    #     [fetch_array(img_name, roi2channel.loc[ch]) for ch in channels]
    # )
    return np.asarray(res.get())


@cache.memoize(timeout=300)
def fetch_array(sample_name: str, channel: str) -> Optional[Array]:
    """Get a single array image for a channel in a cached manner."""
    name = f"{sample_name}.{channel}"
    if name in roi2url:
        req_url = roi2url[name]["shared_download_url"]
        print(f"Downloading: {sample_name}, {channel} at {now(False)}")

        res = requests.get(req_url)
        print(f"Download Completed: {sample_name}, {channel} at {now(False)}")

        res.raise_for_status()

        img = np.load(_BytesIO(res.content))["array"]

        return img
    print(f"Could not find '{name}' in database.")
    return None


# Default figure to display
# Empty Figure Layout
def get_empty_fig():
    return {
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


def get_default_figure() -> Fig:
    if DEFAULT_FIGURE == (None, None):
        def_fig = get_empty_fig()
    else:
        output = get_cxy_array(DEFAULT_FIGURE[0], DEFAULT_FIGURE[1])
        output = output.transpose((-1, 1, 0))
        if output.shape[0] > output.shape[1]:
            output = output.transpose((1, 0, 2))

        def_fig = px.imshow(output)
        def_fig.update_xaxes(showticklabels=False, showgrid=False)
        def_fig.update_yaxes(showticklabels=False, showgrid=False)
        def_fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),
            template="plotly_dark",
        )
    return def_fig


# Hover Information Template
custom_hovertemplate = "<br>".join(
    [
        "" + col.capitalize() + ": %{customdata[" + str(i) + "]}"
        for i, col in enumerate(config["HOVERINFO"])
    ]
)


# Initialize app
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server

# Add Cache to Server
cache.init_app(app.server)
print(f"Caching to '{cache.config['CACHE_DIR']}'.")


#### Load data ####
# load pca plot information
pcs = pd.read_csv(APP_PATH / "data" / "pcadata.csv")
pcs = pcs[config["col_interest"]]

loadings = pd.read_csv(APP_PATH / "data" / "loadings.csv", index_col=0)

for col, dtype in zip(pcs.columns, pcs.dtypes):
    if dtype == "O":
        pcs[col] = pcs[col].fillna("")

pcs = pcs.infer_objects()

cat_order = {
    "disease": ["Healthy", "FLU", "ARDS", "COVID19"],
    "Disease Group": [
        "Healthy",
        "Flu",
        "ARDS",
        "Pneumonia",
        "COVID19_early",
        "COVID19_late",
    ],
}

for cat, order in cat_order.items():
    pcs[cat] = pd.Categorical(pcs[cat], categories=order, ordered=True)

# color mapper
color_map = dict()
color_map["Disease Group"] = [
    px.colors.qualitative.D3[x] for x in [2, 0, 1, 5, 4, 3]
]
color_map["disease"] = [px.colors.qualitative.D3[x] for x in [2, 0, 1, 3]]
color_map["default"] = (
    px.colors.qualitative.D3
    + px.colors.qualitative.G10
    + px.colors.qualitative.T10
    + px.colors.qualitative.Plotly
)

fig = plot_from_data(pcs, color_map=color_map, pca_loadings=loadings)
fig.update_layout(clickmode="event+select")

# load options for dropdown box
color_options = []
for label in config["col_interest"][3:]:
    color_options.append(
        {"label": label.capitalize().replace("_", " "), "value": label}
    )

# load image metadata from JSON
roi2url = json.load(open(config["URL_INFO"], "r"))
roi2channel = pd.Series(json.load(open(config["CHANNEL_INFO"], "r")))

channel_options = []
for value, label in enumerate(roi2channel):
    channel_options.append({"label": label, "value": value})


# Build DOM
sidebar = html.Div(
    [
        html.H2(
            "Data explorer",
            className="display-4",
            style={"margin-left": "5px", "margin-top": "15%"},
        ),
        html.Hr(),
        html.Div(
            [
                html.Div(
                    [
                        html.H5("Label Datapoints by:"),
                        dcc.Dropdown(
                            options=color_options,
                            id="color",
                            value="Disease Group",
                        ),
                    ],
                    style={"padding": "10px"},
                ),
                html.Hr(),
                html.Div(
                    [
                        html.H5("IMC Image Markers:"),
                        html.H5(
                            "Red:", style={"color": "red", "margin-left": "5px"}
                        ),
                        dcc.Dropdown(
                            options=channel_options,
                            id="red",
                            value=config["DEFAULT_CHANNELS"][0],
                        ),
                        html.H5(
                            "Green:",
                            style={"color": "green", "margin-left": "5px"},
                        ),
                        dcc.Dropdown(
                            options=channel_options,
                            id="green",
                            value=config["DEFAULT_CHANNELS"][1],
                        ),
                        html.H5(
                            "Blue:",
                            style={"color": "blue", "margin-left": "5px"},
                        ),
                        dcc.Dropdown(
                            options=channel_options,
                            id="blue",
                            value=config["DEFAULT_CHANNELS"][2],
                        ),
                    ],
                    style={"padding": "10px"},
                ),
            ]
        ),
    ],
    style=config["SIDEBAR_STYLE"],
)

content = html.Div(
    id="root",
    children=[
        html.Div(
            id="header",
            children=[
                html.Img(
                    id="logo",
                    src=app.get_asset_url("LOGO_ENGLANDER_2LINE_PMS.png"),
                ),
                html.H2(
                    children="Imaging mass cytometry data explorer",
                    className="display-4",
                ),
                html.H3(
                    children="Lung tissue under infection by SARS-CoV-2 and other pathogens",
                    className="display-4",
                ),
                html.P(
                    id="description",
                    children=[
                        "Data from publication "
                        '"The spatio-temporal landscape of lung pathology in SARS-CoV-2 infection", ',
                        html.A(
                            "doi:10.1101/2020.10.26.20219584v1",
                            href="https://www.medrxiv.org/content/10.1101/2020.10.26.20219584v1",
                        ),
                        html.Br(),
                        "Produced by: Elemento Lab, Weill Cornell Medicine",
                    ],
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
                                    style={
                                        "width": "inherit",
                                        "height": "60vh",
                                        "align": "center",
                                    },
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
                            id="image-container",
                            children=[
                                html.H5(
                                    "ROI Viewer (click points on the left to load images)"
                                ),
                                html.Hr(),
                                dcc.Graph(
                                    id="image-file-data",
                                    figure=get_default_figure(),
                                    style={
                                        "width": "inherit",
                                        "height": "60vh",
                                        "align": "center",
                                    },
                                ),
                                html.Div(
                                    id="application-state",
                                    style={"display": "none"},
                                ),
                                dcc.Loading(
                                    id="loading-2",
                                    children=[
                                        html.Div(
                                            [html.Div(id="loading-output-2")]
                                        )
                                    ],
                                    type="default",
                                ),
                                html.Div(
                                    [
                                        dcc.Markdown(
                                            """**Hover Data**  Mouse over values in the graph."""
                                        ),
                                        html.Pre(id="hover-data"),
                                    ]
                                ),
                                html.Div(
                                    [
                                        dcc.Markdown(
                                            """**Click Data**  Click on points in the graph."""
                                        ),
                                        html.Pre(id="click-data"),
                                    ]
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
            ],
        ),
    ],
    style=config["CONTENT_STYLE"],
)

# App layout
app.layout = html.Div([sidebar, content])


# Callback Functions for the App
@app.callback(
    Output("hover-data", "children"), [Input("live-update-graph", "hoverData")]
)
def display_hover_data(hoverData) -> Optional[str]:
    if hoverData == None:
        return None
    elif "customdata" in hoverData["points"][0]:
        return "HOVERED: {}".format(hoverData["points"][0]["customdata"][-1])
    return None


@app.callback(
    Output("click-data", "children"), [Input("live-update-graph", "clickData")]
)
def display_click_data(clickData) -> Optional[str]:
    if clickData == None:
        return None
    elif "customdata" in clickData["points"][0]:
        return "LOADED: {}".format(clickData["points"][0]["customdata"][-1])
    return None


# Image Loader Function
@app.callback(
    [
        Output("image-file-data", "figure"),
        Output("loading-output-2", "children"),
    ],
    [
        Input("live-update-graph", "clickData"),
        Input("red", "value"),
        Input("green", "value"),
        Input("blue", "value"),
    ],
)
def display_image_data(clickData, *channels: List[int]) -> Tuple[Fig, str]:
    start = now(False)
    # img_dir = "assets/"
    if clickData == None:
        return (
            get_empty_fig(),
            dcc.Markdown("""**Click Any Points on PCA plot to load images**"""),
        )
    if "customdata" not in clickData["points"][0]:
        return (
            get_empty_fig(),
            dcc.Markdown("""**Click Any Points on PCA plot to load images**"""),
        )

    img_name = clickData["points"][0]["customdata"][-1]
    output = get_cxy_array(img_name, channels)
    if len(output.shape) != 3:
        return (get_empty_fig(), "")

    output = output.transpose((-1, 1, 0))
    if output.shape[0] > output.shape[1]:
        output = output.transpose((1, 0, 2))

    # print(output.shape)
    fig2 = px.imshow(output)
    fig2.update_xaxes(showticklabels=False, showgrid=False)
    fig2.update_yaxes(showticklabels=False, showgrid=False)
    fig2.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_dark",
    )
    end = now(False)
    print(f"Took {end - start} to update.")

    return (fig2, "")


# Multiple components can update everytime interval gets fired.
@app.callback(Output("live-update-graph", "figure"), [Input("color", "value")])
def update_graph_live(value) -> Fig:
    fig = plot_from_data(
        pcs, color_map=color_map, label=value, pca_loadings=loadings
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
