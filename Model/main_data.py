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

unchangedTourism = [
    VariableSet(
        xVar="Overnights2019",
        xTitle="Overnights",
    ),
    VariableSet(
        xVar="Momentum1Yr",
        xTitle="Momentum 1Yr",
    ),
    VariableSet(
        xVar="Arrivals2019",
        xTitle="Arrivals",
    ),
    VariableSet(
        xVar="AverageLengthStay2019",
        xTitle="Length of Stay",
    ),
]

unchangedTourism_m = [
    VariableSet(
        xVar="Overnights2019",
        xTitle="Overnights",
    ),
    VariableSet(
        xVar="Momentum1Yr",
        xTitle="Momentum 1Yr",
    ),
]

unchangedTourismScatter = BaseModel.ScatterMatrix(unchangedTourism, {}, 1000)
unchangedTourismScatter_m = BaseModel.ScatterMatrix(
    unchangedTourism_m, {}, 800)

unchangedInfrastructure = [
    VariableSet(
        xVar="Hotels",
        xTitle="Hotels",
    ),
    VariableSet(
        xVar="Beds2019",
        xTitle="Beds",
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
        xVar="TrainLines",
        xTitle="Railroad Km",
    ),
]
unchangedInfrastructure_m = [
    VariableSet(
        xVar="Hotels",
        xTitle="Hotels",
    ),
    VariableSet(
        xVar="GpdCapita",
        xTitle="GDP p.C.",
    ),
]

unchanedInfrastructureScatter = BaseModel.ScatterMatrix(
    unchangedInfrastructure, {}, 1000)
unchanedInfrastructureScatter_m = BaseModel.ScatterMatrix(
    unchangedInfrastructure_m, {}, 1000)

corrVar = [
    VariableSet(
        xVar="log_overnights_per_capita",
        xTitle="Overnights p.C.",
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
        xVar="logHotelsOnly_per_capita",
        xTitle="Hotels p.C.",
    ),
    VariableSet(
        xVar="logGpdCapita",
        xTitle="GDP p.C.",
    ),
    VariableSet(
        xVar="logtrainkm_per_km2",
        xTitle="Trains p.Km2",
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


corrPlot = BaseModel.CorrelationHeatPlot(False, corr_layout, corrVar)
corrPlot_m = BaseModel.CorrelationHeatPlot(True, corr_layout, corrVar)

explorevars = [
    VariableSet(
        xVar="log_overnights_per_capita",
        xTitle="Overnights p.C.",
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
        xVar="logHotelsOnly_per_capita",
        xTitle="Hotels p.C.",
    ),
    VariableSet(
        xVar="logGpdCapita",
        xTitle="GDP p.C.",
    ),
    VariableSet(
        xVar="logtrainkm_per_km2",
        xTitle="Trains p.Km2",
    ),
]

explorevars_m = [
    VariableSet(
        xVar="log_overnights_per_capita",
        xTitle="Overnights p.C.",
    ),
    VariableSet(
        xVar="logGpdCapita",
        xTitle="GDP p.C.",
    ),
]


corrScatter = BaseModel.ScatterMatrix(explorevars, {}, 1100)
corrScatter_m = BaseModel.ScatterMatrix(explorevars_m, {}, 1100)
unchangedTourismScatter_m.show()
unchangedTourismScatter.show()
unchanedInfrastructureScatter_m.show()
unchanedInfrastructureScatter.show()
corrPlot.show()
corrScatter.show()
corrScatter_m.show()

py.plot(unchangedTourismScatter,
        filename='unchangedTourismScatter', auto_open=True)
py.plot(unchangedTourismScatter_m,
        filename='unchangedTourismScatter_m', auto_open=True)
py.plot(unchanedInfrastructureScatter,
        filename='unchanedInfrastructureScatter', auto_open=True)
py.plot(unchanedInfrastructureScatter_m,
        filename='unchanedInfrastructureScatter_m', auto_open=True)
py.plot(corrPlot, filename='corrPlotData', auto_open=False)
py.plot(corrPlot_m, filename='corrPlotData_m', auto_open=False)
py.plot(corrScatter, filename='corrScatterData', auto_open=True)
py.plot(corrScatter_m, filename='corrScatterData_m', auto_open=True)


##########################################

exploreVars = [
    VariableSet(
        xVar="log_overnights_per_capita",
        xTitle="Overnights p.C.",
    ),
    VariableSet(
        xVar="logHotelsOnly_per_capita",
        xTitle="Hotels p.C.",
    ),
    VariableSet(
        xVar="logGpdCapita",
        xTitle="GDP p.C.",
    ),
    VariableSet(
        xVar="logtrainkm_per_km2",
        xTitle="Trains p.Km2",
    ),
    VariableSet(
        xVar="UNESCO Sites",
        xTitle="Monuments",
    ),
    VariableSet(
        xVar="old_per_capita",
        xTitle="Old Buildings p. C.",
    ),
    VariableSet(
        xVar="ParkPerct",
        xTitle="Park Area",
    ),
    VariableSet(
        xVar="City",
        xTitle="City",
    ),

]

fig = ff.create_scatterplotmatrix(
    modelData[list(map(lambda var: var.xVar, exploreVars))], diag='histogram')
fig = fig.update_layout({'width': 1200, 'height': 1200, 'autosize': True})
fig.show()


#####

X_lognorm = np.random.lognormal(mean=0.0, sigma=1.7, size=500)
qq = stats.probplot(X_lognorm, dist='lognorm', sparams=(1))
x = np.array([qq[0][0][0], qq[0][0][-1]])

fig = go.Figure()
fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers')
fig.add_scatter(x=x, y=qq[1][1] + qq[1][0]*x, mode='lines')
fig.layout.update(showlegend=False)
fig.show()

###############################################################################

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
        500).to_crs(deGeoJson.crs)
)

deGeoJson = deGeoJson.to_crs(epsg=4326)
geojson = deGeoJson.__geo_interface__

deGeoJson['Overnights']


deGeoJson.dropna(axis=0, subset="Overnights", how="any", inplace=True)


OvernightsGraph = px.choropleth_mapbox(deGeoJson, geojson=deGeoJson.geometry, locations=deGeoJson.index, color='Overnights',
                                       color_continuous_scale="Viridis",
                                       #    range_color=(0, 12),
                                       mapbox_style="carto-positron",
                                       zoom=5.6, center={"lat": 51.5, "lon": 10.2},
                                       opacity=0.5,
                                       hover_data={'krs_name_short': True,
                                                   'Overnights': True},
                                       labels={'krs_name_short': 'Name',
                                               'Overnights': 'Overnights p.C.'},
                                       height=1000,
                                       width=1000,
                                       )

OvernightsGraph.show()
# py.plot(OvernightsGraph, filename = 'OvernightsGraph', auto_open=True)

###############################################################################

rawData = pd.read_excel("./Overnights2(2019).xlsx", dtype='object')
rawData['NullId'] = rawData['Id'].astype("string")
rawData['Overnights'] = pd.to_numeric(
    rawData['Overnights'], errors='ignore')
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
        500).to_crs(deGeoJson.crs)
)

deGeoJson = deGeoJson.to_crs(epsg=4326)
geojson = deGeoJson.__geo_interface__

deGeoJson['Overnights']


deGeoJson.dropna(axis=0, subset="Overnights", how="any", inplace=True)


OvernightsGraph = px.choropleth_mapbox(deGeoJson, geojson=deGeoJson.geometry, locations=deGeoJson.index, color='Overnights',
                                       color_continuous_scale="Viridis",
                                       #    range_color=(0, 12),
                                       mapbox_style="carto-positron",
                                       zoom=5.6, center={"lat": 51.5, "lon": 10.2},
                                       opacity=0.5,
                                       hover_data={'krs_name_short': True,
                                                   'Overnights': True},
                                       labels={'krs_name_short': 'Name',
                                               'Overnights': 'Overnights'},
                                       height=1000,
                                       width=1000,
                                       )

OvernightsGraph.show()
