import geopandas as gpd
import geojson
import json
from urllib.request import urlopen
import matplotlib.pyplot as plt
from scipy import stats
import plotly.figure_factory as ff
import plotly.express as px
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
from Model.olstable_template import LatexOLSTableOut
from Model.load_data import LoadData

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

normDist_layout = {
    'showlegend': True,
    'xaxis': dict(
        showgrid=False,
        showline=False,
        zeroline=False,
    ),
    'yaxis': dict(
        showgrid=False,
        showline=False,
        zeroline=False,
    ),
    'title_font_size': 25,
    'legend': dict(
        traceorder="normal",
        font=dict(
            family="Courier New",
            size=15,
            color="black"
        ),
        orientation="h",
        yanchor="bottom",
        y=.96,
        xanchor="right",
        x=1
    )
}

corr_layout = {
    'title': {
        'text': "<b>Correlations</b>",
        'font': dict(
                size=20
        )
    },
    'title_x': 0.5,
    'title_y': 0.9,
    'yaxis_autorange': 'reversed',
    'xaxis_gridcolor': 'rgba(0,0,0,0.1)',
    'yaxis_gridcolor': 'rgba(0,0,0,0.1)',
}


cityData, _ = LoadData()

cityData['logGpdCapita'] = np.log(cityData['GpdCapita'])
cityData['logPopulation'] = np.log(cityData['Population'])
cityData['logHotelsOnly_per_capita'] = np.log(
    cityData['hotelsOnly_per_capita'])
cityData['log_overnights_per_capita'] = np.log(
    cityData['overnights_per_capita'])
cityData['log_overnights_per_capita_foreign'] = np.log(
    cityData['overnights_per_capita_foreign'])
cityData['log_overnights_per_capita_domestic'] = np.log(
    cityData['overnights_per_capita'])
cityData['log_overnights'] = np.log(cityData['Overnights2019'])
cityData['log_overnights_foreign'] = np.log(
    cityData['OvernightsForeign2019'])
cityData['log_overnights_domestic'] = np.log(
    cityData['OvernightsDomestic2019'])

cityData = cityData.dropna()


BaseModel = Model(baseLayout, cityData)

#############################################################################################################################
# Old Building Example
#############################################################################################################################

variables = VariableSet(
    yVar="overnights_per_capita",
    xVar="instagram_post_count_cap",
    zVar="Population",
    zNames="city",
    yTitle="Overnights per Capita",
    xTitle="Instagram Posts per Capita",
    gTitle="Instagram vs Overnights"
)

fancyScatterOvernights = BaseModel.FancyScatterPlot(scatter_layout, variables)

variables = VariableSet(
    yVar="hotelsOnly_per_capita",
    xVar="instagram_post_count_cap",
    zVar="Population",
    zNames="city",
    yTitle="Overnights per Capita",
    xTitle="Instagram Posts per Capita",
    gTitle="Instagram vs Overnights"
)

fancyScatterHotels = BaseModel.FancyScatterPlot(scatter_layout, variables)

quadVar = [
    VariableSet(
        yVar="instagram_post_count_cap",
        xVar="old_per_capita",
        zVar="Population",
        zNames="city",
        yTitle="Instagram Posts p. C.",
        xTitle="Old Buildings p. C.",
        gTitle="<b>Buildings & Instagram</b>"
    ),
    VariableSet(
        yVar="instagram_post_count_cap",
        xVar="destruction",
        zVar="Population",
        zNames="city",
        yTitle="Instagram Posts p. C.",
        xTitle="Destruction (%)",
        gTitle="<b>Destruction & Instagram</b>"
    ),
    VariableSet(
        yVar="overnights_per_capita",
        xVar="old_per_capita",
        zVar="Population",
        zNames="city",
        yTitle="Overnights p. C.",
        xTitle="Old Buildings p. C.",
        gTitle="<b>Buildings & Overnights</b>"
    ),
    VariableSet(
        yVar="overnights_per_capita",
        xVar="destruction",
        zVar="Population",
        zNames="city",
        yTitle="Overnights p. C.",
        xTitle="Destruction (%)",
        gTitle="<b>Destruction & Overnights</b>"
    )
]

quadScatterMobile = BaseModel.QuadScatterPlot(
    True, quadscatter_layout, quadVar)
quadScatter = BaseModel.QuadScatterPlot(False, quadscatter_layout, quadVar)

normDistVar = [
    VariableSet(
        xVar="old_per_km2",
        xTitle="Old Buildings per Km2",
        gTitle="<b>Buildings & Instagram</b>"
    ),
    VariableSet(
        xVar="destruction",
        xTitle="Level of Destruction (%)",
        gTitle="<b>Destruction & Instagram</b>"
    )
]

normDist = BaseModel.SimpleNormalDistOverlay(normDist_layout, normDistVar)

corrVar = [
    VariableSet(
        xVar="log_overnights_per_capita",
        xTitle="Overnights p.C.",
    ),
    VariableSet(
        xVar="instagram_post_count_cap",
        xTitle="Instagram p.C.",
    ),
    VariableSet(
        xVar="Momentum1Yr",
        xTitle="Momentum 1Yr",
    ),
    VariableSet(
        xVar="Momentum5Yr",
        xTitle="Momentum 5Yr",
    ),
    VariableSet(
        xVar="logGpdCapita",
        xTitle="GDP p.C.",
    ),
    VariableSet(
        xVar="old_per_capita",
        xTitle="Old Buildings p. C.",
    ),
    VariableSet(
        xVar="destruction",
        xTitle="Destruction",
    ),
    VariableSet(
        xVar="capital",
        xTitle="Capital City",
    ),
    VariableSet(
        xVar="UNESCO Sites",
        xTitle="Monument",
    ),
]

corrPlot = BaseModel.CorrelationHeatPlot(False, corr_layout, corrVar)
corrPlotM = BaseModel.CorrelationHeatPlot(True, corr_layout, corrVar)

# Model 1 ############################################################################################################################

linearModelXVars_des = [
    VariableSet(
        xVar="Momentum1Yr",
        xTitle="Momentum 1Yr",
    ),
    VariableSet(
        xVar="Momentum5Yr",
        xTitle="Momentum 5Yr",
    ),
    VariableSet(
        xVar="area_km2",
        xTitle="Area Km2",
    ),
    VariableSet(
        xVar="logGpdCapita",
        xTitle="GDP p.C.",
    ),
    VariableSet(
        xVar="logHotelsOnly_per_capita",
        xTitle="Hotels p.C.",
    ),
    VariableSet(
        xVar="UNESCO Sites",
        xTitle="Monuments",
    ),
    VariableSet(
        xVar="capital",
        xTitle="Capital City",
    ),
    VariableSet(
        xVar="Coast",
        xTitle="Coast",
    ),
    VariableSet(
        xVar="destruction",
        xTitle="Destruction",
    ),
]

pldb_scatterMatrix_des = BaseModel.ScatterMatrix(
    False, [VariableSet(
        xVar="log_overnights_per_capita",
        xTitle="Overnights p.C.",
    )] + linearModelXVars_des, {}, 1200)


linearModelYVar = VariableSet(
    yVar="log_overnights_per_capita",
)

model_overnights_des = BaseModel.LinearModel(
    linearModelYVar, linearModelXVars_des)


linearModelYVar = VariableSet(
    yVar="instagram_post_count_cap",
)

model_instagram_des = BaseModel.LinearModel(
    linearModelYVar, linearModelXVars_des)

# Model 2 ############################################################################################################################

linearModelXVars_old = [
    VariableSet(
        xVar="Momentum1Yr",
        xTitle="Momentum 1Yr",
    ),
    VariableSet(
        xVar="Momentum5Yr",
        xTitle="Momentum 5Yr",
    ),
    VariableSet(
        xVar="area_km2",
        xTitle="Area Km2",
    ),
    VariableSet(
        xVar="logGpdCapita",
        xTitle="GDP p.C.",
    ),
    VariableSet(
        xVar="logHotelsOnly_per_capita",
        xTitle="Hotels p.C.",
    ),
    VariableSet(
        xVar="UNESCO Sites",
        xTitle="Monuments",
    ),
    VariableSet(
        xVar="capital",
        xTitle="Capital City",
    ),
    VariableSet(
        xVar="Coast",
        xTitle="Coast",
    ),
    VariableSet(
        xVar="old_per_capita",
        xTitle="Old Buildings p. C.",
    ),
]

pldb_scatterMatrix_old = BaseModel.ScatterMatrix(
    False, [VariableSet(
        xVar="log_overnights_per_capita",
        xTitle="Overnights p.C.",
    )] + linearModelXVars_old, {}, 1200)


linearModelYVar = VariableSet(
    yVar="overnights_per_capita",
)

model_overnights_old = BaseModel.LinearModel(
    linearModelYVar, linearModelXVars_old)


linearModelYVar = VariableSet(
    yVar="instagram_post_count_cap",
)

model_instagram_old = BaseModel.LinearModel(
    linearModelYVar, linearModelXVars_old)

# Results ############################################################################################################################

fancyScatterHotels.show()
fancyScatterOvernights.show()
# py.plot(fig, filename = 'scatter2', auto_open=True)
quadScatterMobile.show()
#  py.plot(fig, filename = 'scatter1', auto_open=True)
quadScatter.show()
# py.plot(fig, filename = 'scatterMob1', auto_open=True)
normDist.show()
# py.plot(normDist, filename = 'normDist1', auto_open=True)
corrPlot.show()
corrPlotM.show()
# py.plot(corrPlot, filename='corr1', auto_open=False)
# py.plot(corrPlotM, filename='corrMob1', auto_open=False)

pldb_scatterMatrix_old.show()
pldb_scatterMatrix_des.show()

print(model_overnights_des.summary2())
print(LatexOLSTableOut("Overnight Stays & Wartime Destruction",
      linearModelXVars_des, model_overnights_des))

print(model_instagram_des.summary2())
print(LatexOLSTableOut("Instagram Posts & Wartime Destruction",
      linearModelXVars_des, model_instagram_des))

print(model_overnights_old.summary2())
print(LatexOLSTableOut("Overnight Stays & Old Buildings",
      linearModelXVars_old, model_overnights_old))

print(model_instagram_old.summary2())
print(LatexOLSTableOut("Instagram Posts & Old Building",
      linearModelXVars_old, model_instagram_old))

# End Old Building Example

linearModelYVar = VariableSet(
    yVar="log_overnights_per_capita_foreign",
)

model_overnights_old_foreign = BaseModel.LinearModel(
    linearModelYVar, linearModelXVars_old)


linearModelYVar = VariableSet(
    yVar="log_overnights_per_capita_domestic",
)

model_overnights_old_domestic = BaseModel.LinearModel(
    linearModelYVar, linearModelXVars_old)

print(model_overnights_old_foreign.summary2())
print(LatexOLSTableOut("Foreign Stays - Old Buildings",
      linearModelXVars_old, model_overnights_old_foreign))

print(model_overnights_old_domestic.summary2())
print(LatexOLSTableOut("Domestic Stays - Old Buildings",
      linearModelXVars_old, model_overnights_old_domestic))


linearModelYVar = VariableSet(
    yVar="log_overnights_per_capita_foreign",
)

model_overnights_old_foreign = BaseModel.LinearModel(
    linearModelYVar, linearModelXVars_des)


linearModelYVar = VariableSet(
    yVar="log_overnights_per_capita_domestic",
)

model_overnights_old_domestic = BaseModel.LinearModel(
    linearModelYVar, linearModelXVars_des)

print(model_overnights_old_foreign.summary2())
print(LatexOLSTableOut("Foreign Stays - Destruction",
      linearModelXVars_des, model_overnights_old_foreign))

print(model_overnights_old_domestic.summary2())
print(LatexOLSTableOut("Domestic Stays - Destruction",
      linearModelXVars_des, model_overnights_old_domestic))

# Results ############################################################################################################################


linearModelXVars_old = [
    VariableSet(
        xVar="instagram_post_count_cap",
        xTitle="Instagram Posts",
    ),
    VariableSet(
        xVar="Momentum1Yr",
        xTitle="Momentum 1Yr",
    ),
    VariableSet(
        xVar="Momentum5Yr",
        xTitle="Momentum 5Yr",
    ),
    VariableSet(
        xVar="area_km2",
        xTitle="Area Km2",
    ),
    VariableSet(
        xVar="logGpdCapita",
        xTitle="GDP p.C.",
    ),
    VariableSet(
        xVar="logHotelsOnly_per_capita",
        xTitle="Hotels p.C.",
    ),
    VariableSet(
        xVar="UNESCO Sites",
        xTitle="Monuments",
    ),
    VariableSet(
        xVar="capital",
        xTitle="Capital City",
    ),
    VariableSet(
        xVar="Coast",
        xTitle="Coast",
    ),
    VariableSet(
        xVar="old_per_capita",
        xTitle="Old Buildings p. C.",
    ),
]

linearModelYVar = VariableSet(
    yVar="overnights_per_capita",
)

model_overnights_old = BaseModel.LinearModel(
    linearModelYVar, linearModelXVars_old)

print(model_overnights_old.summary2())
print(LatexOLSTableOut("Foreign Stays - Destruction",
      linearModelXVars_des, model_overnights_old_foreign))
