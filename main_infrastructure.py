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
from Model.olstable_template import LatexOLSTableOut, LatexOLSTableOutSeven, LatexOLSTableOutEight
from Model.load_data import LoadData

#############################################################################################################################
# Infrastructure Example
#############################################################################################################################

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


_, countyData = LoadData()

countyData = countyData.drop(['AreaPark', 'AreaKm2Calc'], axis=1)

countyData = countyData.dropna()

countyData['logtrainkm_per_km2'] = np.log(
    countyData['trainkm_per_km2'][countyData['trainkm_per_km2'] > 0])
countyData['logGpdCapita'] = np.log(countyData['GpdCapita'])
countyData['logPopulation'] = np.log(countyData['Population'])
countyData['logHotelsOnly_per_capita'] = np.log(
    countyData['hotelsOnly_per_capita'])
countyData['log_overnights_per_capita'] = np.log(
    countyData['overnights_per_capita'])
countyData['log_overnights_per_capita_foreign'] = np.log(
    countyData['overnights_per_capita_foreign'])
countyData['log_overnights_per_capita_domestic'] = np.log(
    countyData['overnights_per_capita'])
countyData['log_overnights'] = np.log(countyData['Overnights2019'])
countyData['log_overnights_foreign'] = np.log(
    countyData['OvernightsForeign2019'])
countyData['log_overnights_domestic'] = np.log(
    countyData['OvernightsDomestic2019'])

countyData['ones'] = 1

countyData = countyData.drop(countyData[countyData["ID"] == "08436"].index)

countyData["Resorts"] = countyData["AllAccomodation2019"].sub(
    countyData["Hotels"].values)

countyData["overnights_per_hotel"] = countyData["Overnights2019"].div(
    countyData["Hotels"].values)

countyData["log_overnights_per_hotel"] = np.log(countyData["Overnights2019"].div(
    countyData["Hotels"].values))

countyData = countyData.dropna()

BaseModel = Model(baseLayout, countyData)

# Standard City Model ############################################################################################################################


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
cityData['log_wikipedia'] = np.log(
    cityData['wiki_len'])

cityData = cityData.dropna()

CityModel = Model(baseLayout, cityData)


linearModelXVars_wiki = [
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
        xVar="log_wikipedia",
        xTitle="Wikipedia",
    ),
]

pldb_scatterMatrix_des = CityModel.ScatterMatrix(
    False, [VariableSet(
        xVar="log_overnights_per_capita",
        xTitle="Overnights p.C.",
    )] + linearModelXVars_wiki, {}, 1200)
pldb_scatterMatrix_des.show()

linearModelYVar = VariableSet(
    yVar="log_overnights_per_capita",
)

model_overnights_wikipedia = CityModel.LinearModel(
    linearModelYVar, linearModelXVars_wiki)
print(model_overnights_wikipedia.summary2())
html = model_overnights_wikipedia.summary()
html.tables[0].as_html()
html.tables[1].as_html()
print(LatexOLSTableOut("Overnight Stays (Per Hotel)",
      linearModelXVars_wiki, model_overnights_wikipedia))

wiki_scatter = [
    VariableSet(
        xVar="log_overnights_per_capita",
        xTitle="Overnights p.C.",
    ),
    VariableSet(
        xVar="log_wikipedia",
        xTitle="Wikipedia",
    ),
]

pldb_scatterMatrix_wiki = CityModel.ScatterMatrix(
    False, wiki_scatter, {}, 1200)
# pldb_scatterMatrix_wiki.show()

py.plot(pldb_scatterMatrix_wiki, filename='scatterMatrix_wiki', auto_open=False)

# Resort City Model ############################################################################################################################

cityData["Resorts"] = cityData["AllAccomodation2019"].sub(
    cityData["Hotels"].values)

cityData["overnights_per_hotel"] = cityData["Overnights2019"].div(
    cityData["Hotels"].values)

cityData["log_overnights_per_hotel"] = np.log(cityData["Overnights2019"].div(
    cityData["Hotels"].values))


linearModelXVars_wiki = [
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
        xVar="log_wikipedia",
        xTitle="Wikipedia",
    ),
]

pldb_scatterMatrix_des = CityModel.ScatterMatrix(
    False, [VariableSet(
        xVar="log_overnights_per_hotel",
        xTitle="Overnights p.H.",
    )] + linearModelXVars_wiki, {}, 1200)
pldb_scatterMatrix_des.show()

linearModelYVar = VariableSet(
    yVar="log_overnights_per_hotel",
)

model_overnights_wikipedia = CityModel.LinearModel(
    linearModelYVar, linearModelXVars_wiki)
print(model_overnights_wikipedia.summary2())
print(LatexOLSTableOutEight("Overnight Stays (Per Hotel)",
      linearModelXVars_wiki, model_overnights_wikipedia))


# Everything Model ############################################################################################################################


linearModelXVars = [
    VariableSet(
        xVar="Momentum1Yr",
        xTitle="Momentum 1Yr",
    ),
    VariableSet(
        xVar="Momentum5Yr",
        xTitle="Momentum 5Yr",
    ),
    VariableSet(
        xVar="logtrainkm_per_km2",
        xTitle="Trains p.Km2",
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
        xVar="old_per_capita",
        xTitle="Old Buildings p.C.",
    ),
    VariableSet(
        xVar="AreaCityPerc",
        xTitle="City Area",
    ),
    VariableSet(
        xVar="Coast",
        xTitle="Coast",
    ),
    VariableSet(
        xVar="AreaVineyards",
        xTitle="Vineyards",
    ),
    VariableSet(
        xVar="Airports",
        xTitle="Airports",
    ),
]

corrScatter = BaseModel.ScatterMatrix(False,
                                      [VariableSet(
                                          xVar="log_overnights_per_hotel",
                                          xTitle="Overnights p.H.",
                                      )] + linearModelXVars, {}, 1500)

# corrScatter.show()


countyData[["log_overnights_per_hotel", "Momentum1Yr", "Momentum5Yr", "logtrainkm_per_km2",
            "logGpdCapita", "logHotelsOnly_per_capita", "UNESCO Sites", "AreaCityPerc", "Coast", "old_per_capita", "AreaParkPerc"]].describe()


linearModelYVar = VariableSet(
    yVar="log_overnights_per_capita",
)

model_overnights = BaseModel.LinearModel(linearModelYVar, linearModelXVars)
print(model_overnights.summary2())
html = model_overnights.summary()
html.tables[0].as_html()
html.tables[1].as_html()
print(LatexOLSTableOut("Overnight Stays (Per Hotel)",
      linearModelXVars, model_overnights))


# Main Model ############################################################################################################################

linearModelXVars = [
    VariableSet(
        xVar="Momentum1Yr",
        xTitle="Momentum 1Yr",
    ),
    VariableSet(
        xVar="Momentum5Yr",
        xTitle="Momentum 5Yr",
    ),
    VariableSet(
        xVar="logtrainkm_per_km2",
        xTitle="Trains p.Km2",
    ),
    VariableSet(
        xVar="logGpdCapita",
        xTitle="GDP p.C.",
    ),
    VariableSet(
        xVar="UNESCO Sites",
        xTitle="Monuments",
    ),
    VariableSet(
        xVar="AreaCityPerc",
        xTitle="City Area",
    ),
    VariableSet(
        xVar="Coast",
        xTitle="Coast",
    ),
    VariableSet(
        xVar="Airports",
        xTitle="Airports",
    ),
]

linearModelYVar = VariableSet(
    yVar="log_overnights_per_hotel",
)


model_overnights = BaseModel.LinearModel(linearModelYVar, linearModelXVars)
print(model_overnights.summary2())
print(LatexOLSTableOutSeven("Overnight Stays (Per Hotel)",
      linearModelXVars, model_overnights))

# Main Model ############################################################################################################################

linearModelXVars = [
    VariableSet(
        xVar="Momentum1Yr",
        xTitle="Momentum 1Yr",
    ),
    VariableSet(
        xVar="Momentum5Yr",
        xTitle="Momentum 5Yr",
    ),
    VariableSet(
        xVar="logHotelsOnly_per_capita",
        xTitle="Hotels p.C.",
    ),
    VariableSet(
        xVar="logtrainkm_per_km2",
        xTitle="Trains p.Km2",
    ),
    VariableSet(
        xVar="logGpdCapita",
        xTitle="GDP p.C.",
    ),
    VariableSet(
        xVar="UNESCO Sites",
        xTitle="Monuments",
    ),
    VariableSet(
        xVar="AreaCityPerc",
        xTitle="City Area",
    ),
    VariableSet(
        xVar="Coast",
        xTitle="Coast",
    ),
    VariableSet(
        xVar="Airports",
        xTitle="Airports",
    ),
]

linearModelYVar = VariableSet(
    yVar="log_overnights_per_capita",
)


model_overnights = BaseModel.LinearModel(linearModelYVar, linearModelXVars)
print(model_overnights.summary2())
print(LatexOLSTableOutSeven("Overnight Stays (Per Hotel)",
      linearModelXVars, model_overnights))

# Resort Model ############################################################################################################################

exploreModelXVars = [
    VariableSet(
        xVar="Momentum1Yr",
        xTitle="Momentum 1Yr",
    ),
    VariableSet(
        xVar="Momentum5Yr",
        xTitle="Momentum 5Yr",
    ),
    VariableSet(
        xVar="logtrainkm_per_km2",
        xTitle="Trains p.Km2",
    ),
    VariableSet(
        xVar="logGpdCapita",
        xTitle="GDP p.C.",
    ),
    VariableSet(
        xVar="UNESCO Sites",
        xTitle="Monuments",
    ),
    VariableSet(
        xVar="City",
        xTitle="City",
    ),
    VariableSet(
        xVar="Coast",
        xTitle="Coast",
    ),
    VariableSet(
        xVar="Resorts",
        xTitle="Resorts",
    ),
    VariableSet(
        xVar="Hotels",
        xTitle="Hotels",
    ),
]


linearModelYVar = VariableSet(
    yVar="log_overnights_per_capita",
)

model_overnights = BaseModel.LinearModel(linearModelYVar, exploreModelXVars)
print(model_overnights.summary2())
print(LatexOLSTableOut("Overnight Stays (Per Capita)",
      exploreModelXVars, model_overnights))


# Graphs ############################################################################################################################


CityModel = Model(baseLayout, cityData)

scatter_layout = {
    'xaxis_gridcolor': 'rgba(0,0,0,0.1)',
    'yaxis_gridcolor': 'rgba(0,0,0,0.1)',
}

variables = VariableSet(
    yVar="overnights_per_hotel",
    xVar="instagram_post_count_cap",
    zVar="Hotels",
    zNames="city",
    yTitle="Overnights per Hotel",
    xTitle="Instagram Posts per Capita",
    gTitle="Instagram vs Overnights per Hotel"
)

fancyScatterOvernights = CityModel.FancyScatterPlot(scatter_layout, variables)
fancyScatterOvernights.show()
# py.plot(fancyScatterOvernights, filename='overnightsperhotelscatter', auto_open=False)

quadscatter_layout = {
    'font_size': 12,
    'hoverlabel': dict(
        bgcolor="rgba(0,0,0,0.1)",
        font_size=18,
        font_family="Courier New"
    ),
}


quadVar = [
    VariableSet(
        yVar="overnights_per_hotel",
        xVar="mom1std",
        zVar="Hotels",
        zNames="SimpleName",
        yTitle="Overnights p.H.",
        xTitle="Momentum 1Y",
        gTitle="<b>Infrastructure & 1Y Momentum</b>"
    ),
    VariableSet(
        yVar="overnights_per_hotel",
        xVar="mom5std",
        zVar="Hotels",
        zNames="SimpleName",
        yTitle="Overnights p.H.",
        xTitle="Momentum 5Y",
        gTitle="<b>Infrastructure & 5Y Momentum</b>"
    ),
    VariableSet(
        yVar="overnights_per_capita",
        xVar="mom1std",
        zVar="Population",
        zNames="SimpleName",
        yTitle="Overnights p. C.",
        xTitle="Momentum 1Y",
        gTitle="<b>Population & 1Y Momentum</b>"
    ),
    VariableSet(
        yVar="overnights_per_capita",
        xVar="mom5std",
        zVar="Population",
        zNames="SimpleName",
        yTitle="Overnights p. C.",
        xTitle="Momentum 5Y",
        gTitle="<b>Population & 5Y Momentum</b>"
    )
]

quadScatterMobile = BaseModel.QuadScatterPlot(
    True, quadscatter_layout, quadVar)
quadScatter = BaseModel.QuadScatterPlot(False, quadscatter_layout, quadVar)
# py.plot(quadScatterMobile, filename='infraquad_m', auto_open=False)
# py.plot(quadScatter, filename='infraquad', auto_open=False)
quadScatter.show()

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

corrVars = [
    VariableSet(
        xVar="log_overnights_per_capita",
        xTitle="Overnights p.C.",
    ),
    VariableSet(
        xVar="log_overnights_per_hotel",
        xTitle="Overnights p.H.",
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
        xVar="logtrainkm_per_km2",
        xTitle="Trains p.Km2",
    ),
    VariableSet(
        xVar="logGpdCapita",
        xTitle="GDP p.C.",
    ),
]

corrPlot = BaseModel.CorrelationHeatPlot(False, corr_layout, corrVars)
corrPlotM = BaseModel.CorrelationHeatPlot(True, corr_layout, corrVars)

py.plot(corrPlot, filename='infracorr', auto_open=False)
py.plot(corrPlotM, filename='infracorr_m', auto_open=False)

corrPlot.show()
