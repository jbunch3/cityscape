import chart_studio.plotly as py
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Model.model_class import Model
from Model.model_class import VariableSet

baseLayout = {
    'margin': {'l': 0, 'r': 0, 't': 50, 'b': 50},
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'showlegend': False,
    'title_font_family': "Courier New",
    'title_font_color': "black",
    'font_family': "Courier New",
    'font_color': 'rgba(0, 0, 0, 0.8)',
    'font_size': 15,
    'autosize': True,
}

bar_layout = {
    'barmode': 'stack',
    'xaxis': {
        'showgrid': False,
        'showline': False,
        'zeroline': False,
    },
    'yaxis': {
        'showticklabels': False,
        'showgrid': False,
        'showline': False,
        'zeroline': False, },
}

scatter_layout = {
    'xaxis_gridcolor': 'rgba(0,0,0,0.1)',
    'yaxis_gridcolor': 'rgba(0,0,0,0.1)',
}

quadscatter_layout = {
    'font_size': 12,
    'hoverlabel': dict(
        bgcolor="rgba(0,0,0,0.1)",
        font_size=18,
        font_family="Courier New"
    ),
}


rawData = pd.read_excel(
    "./DraftCityFile.xlsx").sort_values(by=["destruction"], ascending=False).dropna()

rawData["pct_old"] = rawData["old_residential"]/rawData["total_residential"]
rawData["inv_dest"] = 1-rawData["destruction"]

rawData["old_per_km2"] = rawData["very_old_residential"].div(
    rawData["area_km2"].values)
rawData["old_per_capita"] = rawData["very_old_residential"].div(
    rawData["population_2011"].values)
rawData["overnights_per_capita_2021"] = rawData["overnights_2021"].div(
    rawData["population_2011"].values)
rawData["instagram_post_count_cap"] = rawData["instagram_post_count"].div(
    rawData["population_2011"].values)

BaseModel = Model(baseLayout, rawData)

variables = VariableSet(
    yVar="overnights_per_capita_2019",
    xVar="instagram_post_count_cap",
    zVar="population_2011",
    zNames="city",
    yTitle="Overnights per Capita",
    xTitle="Instagram Posts per Capita",
    gTitle="Instagram vs Overnights"
)

fancyScatter = BaseModel.FancyScatterPlot(scatter_layout, variables)

quadVar = [
    VariableSet(
        yVar="instagram_post_count_cap",
        xVar="old_per_capita",
        zVar="population_2011",
        zNames="city",
        yTitle="Instagram Posts p. C.",
        xTitle="Old Buildings p. C.",
        gTitle="<b>Buildings & Instagram</b>"
    ),
    VariableSet(
        yVar="instagram_post_count_cap",
        xVar="destruction",
        zVar="population_2011",
        zNames="city",
        yTitle="Instagram Posts p. C.",
        xTitle="Destruction (%)",
        gTitle="<b>Destruction & Instagram</b>"
    ),
    VariableSet(
        yVar="overnights_per_capita_2019",
        xVar="old_per_capita",
        zVar="population_2011",
        zNames="city",
        yTitle="Overnights p. C.",
        xTitle="Old Buildings p. C.",
        gTitle="<b>Buildings & Overnights</b>"
    ),
    VariableSet(
        yVar="overnights_per_capita_2019",
        xVar="destruction",
        zVar="population_2011",
        zNames="city",
        yTitle="Overnights p. C.",
        xTitle="Destruction (%)",
        gTitle="<b>Destruction & Overnights</b>"
    )
]

quadScatterMobile = BaseModel.QuadScatterPlot(
    True, quadscatter_layout, quadVar)
quadScatter = BaseModel.QuadScatterPlot(False, quadscatter_layout, quadVar)


fancyScatter.show()
# py.plot(fig, filename = 'scatter2', auto_open=True)
quadScatterMobile.show()
#  py.plot(fig, filename = 'scatter1', auto_open=True)
quadScatter.show()
# py.plot(fig, filename = 'scatterMob1', auto_open=True)
