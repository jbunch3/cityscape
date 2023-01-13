
from statsmodels.graphics.gofplots import ProbPlot
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
import statsmodels.api as sm

from Model.model_class import Model
from Model.model_class import VariableSet
from Model.olstable_template import LatexOLSTableOut, LatexOLSTableOutSeven, LatexOLSTableOutEight
from Model.load_data import LoadData
from Model.diagnostics import Linear_Reg_Diagnostic
from statsmodels.tools.tools import maybe_unwrap_results


#############################################################################################################################
# County Data
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

countyData["log_overnights_per_hotel"] = np.log(countyData["Overnights2019"].div(
    countyData["Hotels"].values))

countyData = countyData.drop(countyData[countyData["ID"] == "08436"].index)

countyData = countyData.dropna()

BaseModel = Model(baseLayout, countyData)

#############################################################################################################################
# City Data
#############################################################################################################################

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
        xVar="City",
        xTitle="City",
    ),
    VariableSet(
        xVar="Coast",
        xTitle="Coast",
    ),
]

linearModelYVar = VariableSet(
    yVar="log_overnights_per_hotel",
)

model_overnights = BaseModel.LinearModel(linearModelYVar, linearModelXVars)

variables = VariableSet(
    yVar="overnights_per_capita",
    xVar="instagram_post_count_cap",
    zVar="Population",
    zNames="city",
    yTitle="Overnights per Capita",
    xTitle="Instagram Posts per Capita",
    gTitle="Instagram vs Overnights"
)

cls = Linear_Reg_Diagnostic(model_overnights, countyData)
resplot = cls.residual_plot("SimpleName")
resplot.show()
py.plot(resplot,
        filename='corrPlot_hotel', auto_open=True)


results = maybe_unwrap_results(model_overnights)
y_true = results.model.endog
y_predict = results.fittedvalues


residual = np.array(results.resid)
countyData['residual'] = results.resid
influence = results.get_influence()
residual_norm = influence.resid_studentized_internal
countyData['residual_norm'] = residual_norm

ranking = countyData[['SimpleName', 'Population',
                      'log_overnights_per_hotel', 'residual', 'residual_norm']]

ranking = ranking.sort_values(by=['residual'], ascending=False)

ranking.head(10)
ranking.tail(10)

ranking = ranking.sort_values(by=['Population'], ascending=False)

test = ranking.head(50)
test = test.sort_values(by=['residual'], ascending=False)
test.head(10)
test.tail(10)


fig = cls.qq_plot("SimpleName")
fig.show()


cls.scale_location_plot()
cls.leverage_plot()
# https://www.statsmodels.org/dev/examples/notebooks/generated/linear_regression_diagnostics_plots.html
cls.vif_table()


# Capita Model ############################################################################################################################

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
        xVar="City",
        xTitle="City",
    ),
    VariableSet(
        xVar="Coast",
        xTitle="Coast",
    ),
]

linearModelYVar = VariableSet(
    yVar="log_overnights_per_capita_foreign",
)

model_overnights = BaseModel.LinearModel(linearModelYVar, linearModelXVars)

variables = VariableSet(
    yVar="overnights_per_capita",
    xVar="instagram_post_count_cap",
    zVar="Population",
    zNames="city",
    yTitle="Overnights per Capita",
    xTitle="Instagram Posts per Capita",
    gTitle="Instagram vs Overnights"
)

cls = Linear_Reg_Diagnostic(model_overnights, countyData)
resplot = cls.residual_plot("SimpleName")
resplot.show()
py.plot(resplot,
        filename='corrPlot_capita', auto_open=False)


results = maybe_unwrap_results(model_overnights)
y_true = results.model.endog
y_predict = results.fittedvalues


residual = np.array(results.resid)
countyData['residual'] = results.resid
influence = results.get_influence()
residual_norm = influence.resid_studentized_internal
countyData['residual_norm'] = residual_norm

ranking = countyData[['SimpleName', 'Population',
                      'log_overnights_per_hotel', 'residual', 'residual_norm']]

ranking = ranking.sort_values(by=['residual'], ascending=False)

ranking.head(10)
ranking.tail(10)

ranking = ranking.sort_values(by=['Population'], ascending=False)

test = ranking.head(50)
test = test.sort_values(by=['residual_norm'], ascending=False)
test.head(10)
test.tail(10)


fig = cls.qq_plot("SimpleName")
fig.show()
