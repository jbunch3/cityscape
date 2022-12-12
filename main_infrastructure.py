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

countyData = countyData.dropna()

BaseModel = Model(baseLayout, countyData)


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
