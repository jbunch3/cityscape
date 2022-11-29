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
rawData["hotelsOnly_per_capita_2019"] = rawData["only_hotels_2019"].div(
    rawData["population_2011"].values)

rawData["hotels_2019_per_capita"] = rawData["hotels_2019"].div(
    rawData["population_2011"].values)
rawData["beds_2019_per_capita"] = rawData["beds_2019"].div(
    rawData["population_2011"].values)
rawData["arrivals_2019_per_capita"] = rawData["arrivals_2019"].div(
    rawData["population_2011"].values)
rawData["overnights_per_capita_2019"] = rawData["Overnights_2019"].div(
    rawData["population_2011"].values)

rawData["hotels_2021_per_capita"] = rawData["hotels_2021"].div(
    rawData["population_2011"].values)
rawData["beds_2021_per_capita"] = rawData["beds_2021"].div(
    rawData["population_2011"].values)
rawData["arrivals_2021_per_capita"] = rawData["arrivals_2021"].div(
    rawData["population_2011"].values)

capital = {'Landeshauptstadt': 1, 'Stadtkreis': 0,
           'Stadt': 0, 'Universitätsstadt': 0}
rawData['capital'] = rawData['type'].map(capital)

BaseModel = Model(baseLayout, rawData)

#############################################################################################################################
# Old Building Example
#############################################################################################################################

variables = VariableSet(
    yVar="overnights_per_capita_2019",
    xVar="instagram_post_count_cap",
    zVar="population_2011",
    zNames="city",
    yTitle="Overnights per Capita",
    xTitle="Instagram Posts per Capita",
    gTitle="Instagram vs Overnights"
)

fancyScatterOvernights = BaseModel.FancyScatterPlot(scatter_layout, variables)

variables = VariableSet(
    yVar="hotelsOnly_per_capita_2019",
    xVar="instagram_post_count_cap",
    zVar="population_2011",
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
        xVar="capital",
        xTitle="Capital City",
    ),
    VariableSet(
        xVar="average_stay_length_2019",
        xTitle="Average Stay",
    ),
    VariableSet(
        xVar="overnights_per_capita_2019",
        xTitle="Overnights p.C.",
    ),
    VariableSet(
        xVar="instagram_post_count_cap",
        xTitle="Instagram p.C.",
    ),
    VariableSet(
        xVar="gdp_capita",
        xTitle="GDP p.C.",
    ),
    VariableSet(
        xVar="area_km2",
        xTitle="Area Km2",
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
        xVar="monuments",
        xTitle="Monument",
    ),
]


corrPlot = BaseModel.CorrelationHeatPlot(False, corr_layout, corrVar)
corrPlotM = BaseModel.CorrelationHeatPlot(True, corr_layout, corrVar)

linearModelXVars = [
    VariableSet(
        xVar="gdp_capita",
    ),
    VariableSet(
        xVar="area_km2",
    ),
    VariableSet(
        xVar="hotelsOnly_per_capita_2019",
    ),
    VariableSet(
        xVar="old_per_capita",
    ),
    VariableSet(
        xVar="destruction",
    ),
    VariableSet(
        xVar="capital",
    ),
    VariableSet(
        xVar="monuments",
    ),
]

linearModelYVar = VariableSet(
    yVar="overnights_per_capita_2019",
)

model_overnights = BaseModel.LinearModel(linearModelYVar, linearModelXVars)


linearModelYVar = VariableSet(
    yVar="instagram_post_count_cap",
)

model_instagram = BaseModel.LinearModel(linearModelYVar, linearModelXVars)

linearModelXVars = [
    VariableSet(
        xVar="gdp_capita",
    ),
    VariableSet(
        xVar="area_km2",
    ),
    VariableSet(
        xVar="hotelsOnly_per_capita_2019",
    ),
    VariableSet(
        xVar="instagram_post_count_cap",
    ),
    VariableSet(
        xVar="old_per_capita",
    ),
    VariableSet(
        xVar="destruction",
    ),
    VariableSet(
        xVar="capital",
    ),
    VariableSet(
        xVar="monuments",
    ),
]

linearModelYVar = VariableSet(
    yVar="average_stay_length_2019",
)

model_lengthofstay = BaseModel.LinearModel(linearModelYVar, linearModelXVars)

fancyScatterHotels.show()
fancyScatterOvernights.show()
# py.plot(fig, filename = 'scatter2', auto_open=True)
quadScatterMobile.show()
#  py.plot(fig, filename = 'scatter1', auto_open=True)
quadScatter.show()
# py.plot(fig, filename = 'scatterMob1', auto_open=True)
normDist.show()
# py.plot(fig, filename = 'normDist1', auto_open=True)
corrPlot.show()
corrPlotM.show()
# py.plot(fig, filename='corr1', auto_open=False)
# py.plot(fig, filename='corrMob1', auto_open=False)

print(model_overnights.summary())
# print(res.summary().as_latex())
print(model_instagram.summary())
print(model_lengthofstay.summary())

# End Old Building Example

#############################################################################################################################
# Nature Example
#############################################################################################################################


rawData = pd.read_excel("./KreisDatasSmall.xlsx")

modelData = rawData[['ID', 'No', 'GeoName', 'SimpleName', 'GpdCapita', 'Population',
                     'OldBuildings', 'AreaKm2', 'City', 'AllAccomodation2019', 'Beds2019',
                     'Arrivals2019', 'Overnights2019', 'AverageLengthStay2019',
                     'OvernightsCapita2019', 'ArrivalsDomestic', 'ArrivalsForeign',
                     'OvernightsDomestic', 'OvernightsForeign', 'AreaPark',
                     'Hotels', 'TrainLines', 'UNESCO Sites']]

# modelData = rawData[['ID', 'No', 'GeoName', 'SimpleName', 'GpdCapita', 'Population',
#                      'OldBuildings', 'AreaKm2', 'City', 'AllAccomodation2019', 'Beds2019',
#                      'Arrivals2019', 'Overnights2019', 'AverageLengthStay2019',
#                      'OvernightsCapita2019', 'ArrivalsDomestic', 'ArrivalsForeign',
#                      'OvernightsDomestic', 'OvernightsForeign', 'AreaPark', 'AreaKm2Calc',
#                      'Hotels', 'TrainLines', 'UNESCO Sites']]

modelData["old_per_km2"] = modelData["OldBuildings"].div(
    modelData["AreaKm2"].values)
modelData["old_per_capita"] = modelData["OldBuildings"].div(
    modelData["Population"].values)
modelData["overnights_per_capita"] = modelData["Overnights2019"].div(
    modelData["Population"].values)
modelData["overnights_per_capita_foreign"] = modelData["OvernightsForeign"].div(
    modelData["Population"].values)
modelData["overnights_per_capita_domestic"] = modelData["OvernightsDomestic"].div(
    modelData["Population"].values)
modelData["hotelsOnly_per_capita"] = modelData["Hotels"].div(
    modelData["Population"].values)
modelData["trainkm_per_capita"] = modelData["TrainLines"].div(
    modelData["Population"].values)


BaseModel = Model(baseLayout, modelData.dropna())

corrVar = [
    VariableSet(
        xVar="AreaPark",
        xTitle="Area Park",
    ),
    VariableSet(
        xVar="trainkm_per_capita",
        xTitle="Train Km p.C.",
    ),
    VariableSet(
        xVar="overnights_per_capita",
        xTitle="Overnights p.C.",
    ),
    VariableSet(
        xVar="overnights_per_capita_foreign",
        xTitle="Overnights F p.C.",
    ),
    VariableSet(
        xVar="overnights_per_capita_domestic",
        xTitle="Overnights D p.C.",
    ),
    VariableSet(
        xVar="GpdCapita",
        xTitle="GDP p.C.",
    ),
    VariableSet(
        xVar="AreaKm2",
        xTitle="Area Km2",
    ),
    VariableSet(
        xVar="old_per_capita",
        xTitle="Old Buildings p. C.",
    ),
    VariableSet(
        xVar="hotelsOnly_per_capita",
        xTitle="Hotels p.C.",
    ),
    VariableSet(
        xVar="UNESCO Sites",
        xTitle="UNESCO Sites",
    ),
    VariableSet(
        xVar="City",
        xTitle="City",
    ),
]


corrPlot = BaseModel.CorrelationHeatPlot(False, corr_layout, corrVar)
corrPlot.show()


linearModelXVars = [
    VariableSet(
        xVar="AreaPark",
    ),
    VariableSet(
        xVar="GpdCapita",
    ),
    VariableSet(
        xVar="old_per_capita",
    ),
    VariableSet(
        xVar="hotelsOnly_per_capita",
    ),
    VariableSet(
        xVar="UNESCO Sites",
    ),
    VariableSet(
        xVar="City",
    ),
]

linearModelYVar = VariableSet(
    yVar="overnights_per_capita",
)

model_overnights = BaseModel.LinearModel(linearModelYVar, linearModelXVars)
print(model_overnights.summary())

linearModelYVar = VariableSet(
    yVar="overnights_per_capita_foreign",
)

model_overnights_f = BaseModel.LinearModel(linearModelYVar, linearModelXVars)
print(model_overnights_f.summary())

linearModelYVar = VariableSet(
    yVar="overnights_per_capita_domestic",
)

model_overnights_f = BaseModel.LinearModel(linearModelYVar, linearModelXVars)
print(model_overnights_f.summary())
