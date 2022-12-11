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
    rawData["population_2011"].values).mul(10000)
rawData["overnights_per_capita_2021"] = rawData["overnights_2021"].div(
    rawData["population_2011"].values)
rawData["overnights_foreign_per_capita"] = rawData["overnights_foreign_2019"].div(
    rawData["population_2011"].values)
rawData["overnights_domestic_per_capita"] = rawData["overnights_domestic_2019"].div(
    rawData["population_2011"].values)
rawData["instagram_post_count_cap"] = rawData["instagram_post_count"].div(
    rawData["population_2011"].values)
rawData["hotelsOnly_per_capita_2019"] = rawData["only_hotels_2019"].div(
    rawData["population_2011"].values).mul(10000)

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

rawData['logGpdCapita'] = np.log(rawData['gdp_capita'])
rawData['logHotelsOnly_per_capita'] = np.log(
    rawData['hotelsOnly_per_capita_2019'])
rawData['log_overnights_per_capita'] = np.log(
    rawData['overnights_per_capita_2019'])


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
        xVar="log_overnights_per_capita",
        xTitle="Overnights p.C.",
    ),
    VariableSet(
        xVar="instagram_post_count_cap",
        xTitle="Instagram p.C.",
    ),
    VariableSet(
        xVar="logGpdCapita",
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
    VariableSet(
        xVar="logHotelsOnly_per_capita",
        xTitle="Hotels p.C.",
    ),
]

corrPlot = BaseModel.CorrelationHeatPlot(False, corr_layout, corrVar)
corrPlotM = BaseModel.CorrelationHeatPlot(True, corr_layout, corrVar)

linearModelXVars_des = [
    VariableSet(
        xVar="logGpdCapita",
        xTitle="GDP p.C.",
    ),
    VariableSet(
        xVar="area_km2",
        xTitle="Area Km2",
    ),
    VariableSet(
        xVar="logHotelsOnly_per_capita",
        xTitle="Hotels p.C.",
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
        xVar="monuments",
        xTitle="Monuments",
    ),
]


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

linearModelXVars_old = [
    VariableSet(
        xVar="logGpdCapita",
        xTitle="GDP p.C.",
    ),
    VariableSet(
        xVar="area_km2",
        xTitle="Area Km2",
    ),
    VariableSet(
        xVar="logHotelsOnly_per_capita",
        xTitle="Hotels p.C.",
    ),
    VariableSet(
        xVar="old_per_capita",
        xTitle="Old Buildings p. C.",
    ),
    VariableSet(
        xVar="capital",
        xTitle="Capital City",
    ),
    VariableSet(
        xVar="monuments",
        xTitle="Monuments",
    ),
]

linearModelYVar = VariableSet(
    yVar="overnights_per_capita_2019",
)

model_overnights_old = BaseModel.LinearModel(
    linearModelYVar, linearModelXVars_old)


linearModelYVar = VariableSet(
    yVar="instagram_post_count_cap",
)

model_instagram_old = BaseModel.LinearModel(
    linearModelYVar, linearModelXVars_old)

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

fig = ff.create_scatterplotmatrix(
    rawData[list(map(lambda var: var.xVar, corrVar))], diag='histogram')
fig = fig.update_layout({'width': 1200, 'height': 1200, 'autosize': True})
fig.show()


print(model_overnights_des.summary2())
# print(model_overnights_des.summary2().as_latex())
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

#############################################################################################################################
# Nature Example
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

rawData = pd.read_excel("./KreisDatasSmall.xlsx")

modelData = rawData[['ID', 'No', 'GeoName', 'SimpleName', 'GpdCapita', 'Population',
                     'OldBuildings', 'AreaKm2', 'City', 'AllAccomodation2019', 'Beds2019',
                     'Arrivals2019', 'Overnights2019', 'Overnights2018', 'AverageLengthStay2019',
                     'OvernightsCapita2019', 'ArrivalsDomestic', 'ArrivalsForeign',
                     'OvernightsDomestic2019', 'OvernightsForeign2019', 'AreaPark', 'AreaKm2Calc',
                     'Hotels', 'TrainLines', 'UNESCO Sites', 'Momentum1Yr', 'Momentum5Yr']]

# modelData = rawData[['ID', 'No', 'GeoName', 'SimpleName', 'GpdCapita', 'Population',
#                      'OldBuildings', 'AreaKm2', 'City', 'AllAccomodation2019', 'Beds2019',
#                      'Arrivals2019', 'Overnights2019', 'AverageLengthStay2019',
#                      'OvernightsCapita2019', 'ArrivalsDomestic', 'ArrivalsForeign',
#                      'OvernightsDomestic', 'OvernightsForeign', 'AreaPark', 'AreaKm2Calc',
#                      'Hotels', 'TrainLines', 'UNESCO Sites']]

modelData["ParkPerct"] = modelData["AreaPark"].mul(100).div(
    modelData["AreaKm2Calc"].values)
modelData["old_per_km2"] = modelData["OldBuildings"].div(
    modelData["AreaKm2"].values)
modelData["old_per_capita"] = modelData["OldBuildings"].div(
    modelData["Population"].values).mul(10000)
modelData["Arrivals_perCapita"] = modelData["Arrivals2019"].div(
    modelData["Population"].values)
modelData["overnights_per_capita"] = modelData["Overnights2019"].div(
    modelData["Population"].values)
modelData["overnights_per_capita_foreign"] = modelData["OvernightsForeign2019"].div(
    modelData["Population"].values)
modelData["overnights_per_capita_domestic"] = modelData["OvernightsDomestic2019"].div(
    modelData["Population"].values)
modelData["hotelsOnly_per_capita"] = modelData["Hotels"].div(
    modelData["Population"].values).mul(10000)
modelData["trainkm_per_capita"] = modelData["TrainLines"].div(
    modelData["Population"].values)
modelData["trainkm_per_km2"] = modelData["TrainLines"].div(
    modelData["AreaKm2"].values)


modelData = modelData.dropna()

modelData['logtrainkm_per_km2'] = np.log(modelData['trainkm_per_km2'])
modelData['logGpdCapita'] = np.log(modelData['GpdCapita'])
modelData['logHotelsOnly_per_capita'] = np.log(
    modelData['hotelsOnly_per_capita'])
modelData['log_overnights_per_capita'] = np.log(
    modelData['overnights_per_capita'])
modelData['log_overnights_per_capita_foreign'] = np.log(
    modelData['overnights_per_capita_foreign'])
modelData['log_overnights_per_capita_domestic'] = np.log(
    modelData['overnights_per_capita'])


BaseModel = Model(baseLayout, modelData.dropna())

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
        xVar="ParkPerct",
        xTitle="Park Area",
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
]

linearModelYVar = VariableSet(
    yVar="log_overnights_per_capita",
)


model_overnights = BaseModel.LinearModel(linearModelYVar, linearModelXVars)
print(model_overnights.summary2())
print(LatexOLSTableOut("Overnight Stays (Total)",
      linearModelXVars, model_overnights))

linearModelYVar = VariableSet(
    yVar="log_overnights_per_capita_foreign",
)

model_overnights_f = BaseModel.LinearModel(linearModelYVar, linearModelXVars)
print(model_overnights_f.summary())
print(LatexOLSTableOut("Overnight Stays (Foreign)",
      linearModelXVars, model_overnights_f))

linearModelYVar = VariableSet(
    yVar="log_overnights_per_capita_domestic",
)

model_overnights_d = BaseModel.LinearModel(linearModelYVar, linearModelXVars)
print(model_overnights_d.summary())
print(LatexOLSTableOut("Overnight Stays (Domestic)",
      linearModelXVars, model_overnights))


#############################################################################################################################
# Map Example
#############################################################################################################################

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                 dtype={"fips": str})
df.head()

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                 dtype={"fips": str})


features = counties['features'][0]

fig = px.choropleth_mapbox(df, geojson=counties, locations='fips', color='unemp',
                           color_continuous_scale="Viridis",
                           range_color=(0, 12),
                           mapbox_style="carto-positron",
                           zoom=3, center={"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5,
                           labels={'unemp': 'unemployment rate'}
                           )

fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()

#############################################################################################################################

rawData = pd.read_excel("./Overnight(2019).xlsx", dtype='object')
rawData['NullId'] = rawData['NullId'].astype("string")
rawData['Overnights'] = pd.to_numeric(
    rawData['Overnights pC'], errors='ignore')
modelData = rawData[['NullId', 'Overnights']].dropna()

deGeoJson = gpd.read_file("GermanyKreisen2.geojson")
deGeoJson.dropna(axis=0, subset="geometry", how="any", inplace=True)
deGeoJson = pd.merge(deGeoJson, modelData,
                     left_on='krs_code', right_on='NullId')

deGeoJson['Overnights'] = np.log(pd.to_numeric(
    deGeoJson['Overnights']))
deGeoJson = deGeoJson.set_index('krs_code')

deGeoJson["geometry"] = (
    deGeoJson.to_crs(deGeoJson.estimate_utm_crs()).simplify(
        1000).to_crs(deGeoJson.crs)
)

deGeoJson = deGeoJson.to_crs(epsg=4326)
geojson = deGeoJson.__geo_interface__

deGeoJson['Overnights']


deGeoJson.dropna(axis=0, subset="Overnights", how="any", inplace=True)


fig = px.choropleth_mapbox(deGeoJson, geojson=deGeoJson.geometry, locations=deGeoJson.index, color='Overnights',
                           color_continuous_scale="Viridis",
                           #    range_color=(0, 12),
                           mapbox_style="carto-positron",
                           zoom=3, center={"lat": 54.78252, "lon": 9.43751},
                           opacity=0.5,
                           labels={'Name': 'krs_name_short',
                                   'Log Overnights': 'Overnights'}
                           )

fig.show()
