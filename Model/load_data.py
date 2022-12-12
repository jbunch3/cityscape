import pandas as pd
pd.options.mode.chained_assignment = None


def LoadData():

    # Load County Data ############################################################################################################################

    rawCounty = pd.read_excel("./KreisDatasSmall.xlsx", dtype={'ID': str})

    countyData = rawCounty[['ID', 'No', 'GeoName', 'SimpleName', 'GpdCapita', 'Population',
                           'OldBuildings', 'AreaKm2', 'City', 'AllAccomodation2019', 'Beds2019',
                            'Arrivals2019', 'Overnights2019', 'Overnights2018', 'AverageLengthStay2019',
                            'OvernightsCapita2019', 'ArrivalsDomestic', 'ArrivalsForeign',
                            'OvernightsDomestic2019', 'OvernightsForeign2019', 'AreaPark', 'AreaKm2Calc',
                            'Hotels', 'TrainLines', 'UNESCO Sites', 'Momentum1Yr', 'Momentum5Yr', 'Coast']]

    tmp = countyData["AreaPark"].mul(100).div(
        countyData["AreaKm2Calc"].values)

    countyData["ParkPerct"] = [elem if elem > 0 else 0 for elem in tmp]

    countyData["old_per_km2"] = countyData["OldBuildings"].div(
        countyData["AreaKm2"].values)
    countyData["old_per_capita"] = countyData["OldBuildings"].div(
        countyData["Population"].values).mul(10000)
    countyData["Arrivals_perCapita"] = countyData["Arrivals2019"].div(
        countyData["Population"].values)
    countyData["overnights_per_capita"] = countyData["Overnights2019"].div(
        countyData["Population"].values)
    countyData["overnights_per_capita_foreign"] = countyData["OvernightsForeign2019"].div(
        countyData["Population"].values)
    countyData["overnights_per_capita_domestic"] = countyData["OvernightsDomestic2019"].div(
        countyData["Population"].values)
    countyData["hotelsOnly_per_capita"] = countyData["Hotels"].div(
        countyData["Population"].values).mul(10000)
    countyData["trainkm_per_capita"] = countyData["TrainLines"].div(
        countyData["Population"].values)

    tmp = countyData["TrainLines"].div(
        countyData["AreaKm2"].values)

    countyData["trainkm_per_km2"] = [elem if elem > 0 else 0 for elem in tmp]

    # Loat and Merge City Data ############################################################################################################################

    rawCity = pd.read_excel(
        "./DraftCityFile.xlsx", dtype={'id': str}).sort_values(by=["destruction"], ascending=False).dropna()

    capital = {'Landeshauptstadt': 1, 'Stadtkreis': 0,
               'Stadt': 0, 'Universitätsstadt': 0}
    rawCity['capital'] = rawCity['type'].map(capital)

    cityData = rawCity[['id', 'city', 'type', 'area_km2', 'instagram_post_count', 'destruction', 'inner_city_destroyed',
                        'capital']]

    cityData = pd.merge(cityData, countyData, how='inner',
                        left_on="id", right_on="ID")

    cityData = cityData.drop(
        ["ParkPerct", "TrainLines", "trainkm_per_capita", "trainkm_per_km2", 'AreaPark', 'AreaKm2Calc'], axis=1)

    cityData["instagram_post_count_cap"] = cityData["instagram_post_count"].div(
        cityData["Population"].values)

    return cityData, countyData
