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

_, countyData = LoadData()

countyData = countyData.drop(["ParkPerct", 'AreaPark', 'AreaKm2Calc'], axis=1)

# Data Transformations ############################################################################################################################
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

unchangedTourismScatter = BaseModel.ScatterMatrix(
    False, unchangedTourism, {}, 1000)
unchangedTourismScatter_m = BaseModel.ScatterMatrix(True,
                                                    unchangedTourism_m, {}, 800)

unchangedInfrastructure = [
    VariableSet(
        xVar="Hotels",
        xTitle="Hotels",
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
    VariableSet(
        xVar="AreaCityPerc",
        xTitle="City Area (%)",
    ),
    VariableSet(
        xVar="Airports",
        xTitle="Airports",
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

unchanedInfrastructureScatter = BaseModel.ScatterMatrix(False,
                                                        unchangedInfrastructure, {}, 1000)
unchanedInfrastructureScatter_m = BaseModel.ScatterMatrix(True,
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
        xVar="MonumentsKm2",
        xTitle="Monuments p.Km2",
    ),
    VariableSet(
        xVar="AreaCityPerc",
        xTitle="City Area (%)",
    ),
    VariableSet(
        xVar="logPopulation",
        xTitle="Population",
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


corrScatter = BaseModel.ScatterMatrix(False, explorevars, {}, 1100)
corrScatter_m = BaseModel.ScatterMatrix(True, explorevars_m, {}, 1100)
unchangedTourismScatter_m.show()
unchangedTourismScatter.show()
unchanedInfrastructureScatter_m.show()
unchanedInfrastructureScatter.show()
corrPlot.show()
corrScatter.show()
corrScatter_m.show()

py.plot(unchangedTourismScatter,
        filename='unchangedTourismScatter', auto_open=False)
py.plot(unchangedTourismScatter_m,
        filename='unchangedTourismScatter_m', auto_open=False)
py.plot(unchanedInfrastructureScatter,
        filename='unchanedInfrastructureScatter', auto_open=False)
py.plot(unchanedInfrastructureScatter_m,
        filename='unchanedInfrastructureScatter_m', auto_open=False)
py.plot(corrPlot, filename='corrPlotData', auto_open=False)
py.plot(corrPlot_m, filename='corrPlotData_m', auto_open=False)
py.plot(corrScatter, filename='corrScatterData', auto_open=False)
py.plot(corrScatter_m, filename='corrScatterData_m', auto_open=False)


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
        xVar="AreaParkPerc",
        xTitle="Park Area",
    ),
    VariableSet(
        xVar="AreaCityPerc",
        xTitle="City Area",
    ),

]

fig = ff.create_scatterplotmatrix(
    countyData[list(map(lambda var: var.xVar, exploreVars))], diag='histogram')
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
countyData = rawData[['NullId', 'Overnights']].dropna()

deGeoJson = gpd.read_file("GermanyKreisen2.geojson")
deGeoJson.dropna(axis=0, subset="geometry", how="any", inplace=True)
deGeoJson = pd.merge(deGeoJson, countyData,
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
countyData = rawData[['NullId', 'Overnights']].dropna()

deGeoJson = gpd.read_file("GermanyKreisen2.geojson")
deGeoJson.dropna(axis=0, subset="geometry", how="any", inplace=True)
deGeoJson = pd.merge(deGeoJson, countyData,
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


# Overnights OLS ############################################################################################################################

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
        xVar="AreaCityPerc",
        xTitle="City Area (%)",
    ),
    VariableSet(
        xVar="Coast",
        xTitle="Coast",
    ),
]

corrScatter = BaseModel.ScatterMatrix(False,
                                      [VariableSet(
                                          xVar="log_overnights_per_capita",
                                          xTitle="Overnights p.C.",
                                      ),
                                          VariableSet(
                                          xVar="log_overnights",
                                          xTitle="Overnights",
                                      )] + linearModelXVars, {}, 2000)

corrScatter.show()

countyData[["log_overnights_per_capita", "Momentum1Yr", "Momentum5Yr", "logtrainkm_per_km2",
            "logGpdCapita", "logHotelsOnly_per_capita", "UNESCO Sites", "City", "Coast"]].describe()


linearModelYVar = VariableSet(
    yVar="log_overnights_per_capita",
)

model_overnights = BaseModel.LinearModel(linearModelYVar, linearModelXVars)
print(model_overnights.summary2())
html = model_overnights.summary()
html.tables[0].as_html() + html.tables[1].as_html() + html.tables[2].as_html()
html.tables[0].as_text() + html.tables[1].as_text() + html.tables[2].as_text()
print(LatexOLSTableOutEight("Overnight Stays (Per Capita)",
      linearModelXVars, model_overnights))

linearModelYVar = VariableSet(
    yVar="log_overnights",
)

model_overnights = BaseModel.LinearModel(linearModelYVar, linearModelXVars)
print(model_overnights.summary2())
print(LatexOLSTableOut("Overnight Stays (Total)",
      linearModelXVars, model_overnights))
