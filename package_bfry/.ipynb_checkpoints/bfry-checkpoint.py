import numpy as np
import pandas as pd
import geopandas as gpd

import requests
import geojson
import json
import urllib

import geoplot
import geoplot.crs as gcrs
import folium
import contextily as cx

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn import cluster
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

# DEFINE FUNCTIONS
def trim_all_columns(df):
    """
    Trim whitespace from ends of each value across all series in dataframe
    """
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x
    return df.map(trim_strings)

def read_clean_data(file_name):
    """
    Imports datasets that have semicolon separators for some reason and also weird characters
    """
    df = pd.read_csv(file_name, sep = ';', on_bad_lines='warn')
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('\ufeff', '')
    df = trim_all_columns(df)
    return df

def read_data_link(url, type):
    """ reads data from a download link copied from Bologna open data website """
    # request the url content
    r = requests.get(url)

    # process the data depending on the file format
    if type == "geojson":
        open('data.geojson', 'wb').write(r.content)
        df = gpd.read_file("data.geojson")
    elif type == "csv":
        open('data.csv', 'wb').write(r.content)
        df = read_clean_data("data.csv")
    # return the output
    return df

def create_map(base_df, analysis_df, base_unit, analysis_unit, geo_frame, quotient = 1):
    """
    Creates a chloropleth map given a base map at neighborhood level (i.e. households or population) 
    and geographic analysis dataset at sub-neighborhood level (points, lines, or polygons)
    """
    # create a base map at the neighborhood level
    base = base_df.reset_index().dissolve('zona_fiu')
    # add the points of analysis to the neighborhood base map
    base_w_analysis = base.join(analysis_df)
    # aggregate the points of analysis per the base unit
    base_w_analysis[str(analysis_unit + "_per_" + str(quotient) + "_" + base_unit)] = base_w_analysis[str(analysis_unit + "_count")]/(base_w_analysis['population'] / quotient)
    # round the result to 4 decimal places
    base_w_analysis = base_w_analysis.round(4)
    # create an interactive chlorpleth of this data
    chloropleth = base_w_analysis.explore(column = str(analysis_unit + "_per_" + str(quotient) + "_" + base_unit), cmap = 'RdBu_r', tooltip = ('zona_fiu', 'population', str(analysis_unit + "_count"), str(analysis_unit + "_per_" + str(quotient) + "_" + base_unit)), 
                                          tiles = 'CartoDB positron', legend=True)
    # call explore function
    return geo_frame.explore(m = chloropleth)

def create_geo_df(df):
    """
    Takes a df of a certain format and transforms it into a geopandas compatible data frame
    """
    df[['y','x']] = df['Geo Point'].str.split(', ', expand = True)
    df_geo = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326")
    return df_geo

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()
    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))
    ax.patch.set_facecolor('lightgray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    for (x, y), w in np.ndenumerate(matrix):
        color = 'red' if w > 0 else 'blue'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
        facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    nticks = matrix.shape[0]
    ax.xaxis.tick_top()
    ax.set_xticks(range(nticks))
    ax.set_xticklabels(list(matrix.columns), rotation=90)
    ax.set_yticks(range(nticks))
    ax.set_yticklabels(matrix.columns)
    ax.grid(False)
    ax.autoscale_view()
    ax.invert_yaxis()

def screeplot(pca, standardised_values):
    """ Creates a plot that shows results of principal components in descending order by variation """
    y = np.std(pca.transform(standardised_values), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()

def pca_summary(pca, standardised_data, out=True):
    """ Outputs summary statistics for principal components """
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(list(zip(a, b, c)), index=names, columns=columns)
    if out:
        print("Importance of components:")
        display(summary)
    return summary

def pca_scatter(pca, standardised_values, classifs):
    """ Creates a scatter plot of clusters of the principal components analysis given data and a number of classification groups """
    foo = pca.transform(standardised_values)
    bar = pd.DataFrame(list(zip(foo[:, 0], foo[:, 1], classifs)), columns=["PC1", "PC2", "Class"])
    sns.lmplot(x = "PC1", y = "PC2", data = bar, hue="Class", fit_reg=False)

# IMPORT / GENERATE DATASETS

# IMPORT / GENERATE DATASETS

    # import geojson of base map of statistical areas
base_map_data = read_data_link("https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/aree-statistiche/exports/geojson?lang=it&timezone=Europe%2FRome","geojson")

# statistical area - zona crosswalk 
fiu_xwalk = pd.read_csv('./Data/Bologna - Mappa Zone Fiu.csv', on_bad_lines='warn')
fiu_xwalk['zona_fiu'] = fiu_xwalk['zona_fiu'].str.strip()

# convert CRS to equal area
base_map_data = base_map_data.to_crs(epsg=6933)
# rename zones for consistency
base_map_data.loc[base_map_data.zona == 'S. Viola', 'zona'] = 'Santa Viola'
base_map_data.loc[base_map_data.zona == 'S. Vitale', 'zona'] = 'San Vitale'
base_map_data.loc[base_map_data.zona == 'S. Ruffillo', 'zona'] = 'San Ruffillo'
# create a map at the zone level

base_map_data = base_map_data.set_index('area_statistica').join(fiu_xwalk.set_index('area_statistica'))
base_map_zone = base_map_data.dissolve('zona_fiu')[['geometry']]

#read_data_link("")

    # import population data
#population = read_clean_data("./Data/popolazione-residente-per-eta-sesso-cittadinanza-quartiere-zona-area-statistica-.csv")
population = read_data_link("https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/popolazione-residente-per-eta-sesso-cittadinanza-quartiere-zona-area-statistica-/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B", 'csv')
# cast the year column as a string for future processing
population['Anno'] = population['Anno'].astype(str)
    # clean and combine the dataset with the base map
# aggregate population at the zone level
pop_agg = population.groupby(['Anno', 'Zona di prossimità'])['Residenti'].sum().to_frame().pivot_table('Residenti', ['Zona di prossimità'], 'Anno')
# get 2019 data
pop_2019 = pop_agg[['2019']].rename(columns={'2019':'population'})
# join the base map data with the population data
base_map_2019 = base_map_zone.join(pop_2019)

    # import household data
# number of resident households / families per neighborhood
#households = read_clean_data("./Data/famiglie-residenti-per-quartiere-zona-area-statistica-e-ampiezza-della-famiglia-.csv")
households = read_data_link("https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/famiglie-residenti-per-quartiere-zona-area-statistica-e-ampiezza-della-famiglia-/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B", 'csv')
# cast the year column as a string for future processing
households['Anno'] = households['Anno'].astype(str)
    # clean and combine the dataset with the base map
# aggregate the household count at the zone level
households_agg = households.groupby(['Anno', 'Zona di prossimità'])['Numero Famiglie'].sum().to_frame().pivot_table('Numero Famiglie', ['Zona di prossimità'], 'Anno')
# get 2019 data
hh_2019 = households_agg[['2019']].rename(columns={'2019':'households'})
# create a base map with number of households to explore in comparison to per capita
base_map_2019 = base_map_2019.join(hh_2019)

    # import income data
#income = read_clean_data("./Data/redditi-per-area-statistica.csv")
income = read_data_link("https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/redditi-per-area-statistica/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B", 'csv')
# format the year field
income['Anno reddito'] = income['Anno reddito'].astype(str)
income = income.rename(columns={'Anno reddito':'anno'})
# pivot the income to have each year be a column
#income = income.pivot_table(values = ['Reddito medio contribuente','N contribuenti'], index = ['Area Statistica'], columns = ['anno'])
# get most recent data
income_2019 = income.loc[income['anno'] == '2019']
# join with zones and aggregate at zone level
# value needs to be capitalized for matching 
income_2019['Area Statistica'] = income_2019['Area Statistica'].str.upper()
# rename and filter to relevant columns
income_2019 = income_2019.rename(columns={'Area Statistica':'area_statistica','N contribuenti':'n_taxpayers','Reddito imponibile ai fini irpef':'total_income'})[['area_statistica','n_taxpayers','total_income']]
# aggregate at the statistical area level
income_2019_agg = income_2019.groupby(['area_statistica'])[['n_taxpayers','total_income']].sum()
# join the base map data and aggregate income data at the zone level 
income_zone = income_2019_agg.join(base_map_data.reset_index().set_index('area_statistica')).groupby(['zona_fiu'])[['n_taxpayers','total_income']].sum()
income_zone['avg_income'] = round(income_zone['total_income'] / income_zone['n_taxpayers'],0)
# append income info to 2019 base map
base_map_2019 = base_map_2019.join(income_zone[['n_taxpayers','avg_income']])
# calculate taxpayers per capita
base_map_2019['taxpayers_per_cap'] = round(base_map_2019['n_taxpayers'] / base_map_2019['population'],4)

    # import employment and student data
#occupation = read_clean_data("./Data/occupati_statistica.csv")
occupation = read_data_link("https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/occupati_statistica/exports/csv?lang=it&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B", 'csv')
# format the year field
occupation['Anno di riferimento'] = occupation['Anno di riferimento'].astype(str)
occupation = occupation.rename(columns={'Anno di riferimento':'anno', 'Nome zona':'zona','Sezione censimento (2011)':'census_tract','Numero unità locali':'n_local_units','Numero addetti (dipendenti e indipendenti)':'n_workers','Numero addetti istituzioni pubbliche':'n_public_workers','Numero studenti':'n_students'})
# get only the latest data
occupation_2019 = occupation.loc[occupation['anno'] == '2019']
occupation_2019 = occupation_2019.set_index('Nome area').join(fiu_xwalk.set_index('area_statistica'))
# aggregate the stats at the zona level
occupation_2019_agg = occupation_2019.groupby(['zona_fiu'])[['n_workers','n_students']].sum()
# append occupation info to the base map
base_map_2019 = base_map_2019.join(occupation_2019_agg)
# calculate workers and students per capita
base_map_2019['workers_per_cap'] = round(base_map_2019['n_workers'] / base_map_2019['population'],4)
base_map_2019['students_per_cap'] = round(base_map_2019['n_students'] / base_map_2019['population'],4)

    # calculate additional socioeconomic stats
# density
base_map_2019['pop_density_km2'] = round(base_map_2019['population'] / (base_map_2019.area/1000000), 0)
# avg people per household
base_map_2019['avg_household_size'] = round(base_map_2019['population'] / base_map_2019['households'], 2)

    # import airbnb data
# number of airbnbs from airbnb survey
airbnb = read_data_link("https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/bologna-rilevazione-airbnb/exports/geojson?lang=it&timezone=Europe%2FBerlin", 'geojson')
airbnb = airbnb.to_crs(epsg=6933)
    # process the data set in prepartion for mapping
# aggregate the count of wifi hotspots at the neighborhood level
airbnb_agg = gpd.overlay(base_map_2019, airbnb, how='intersection', keep_geom_type=False).groupby('zona_di_prossimita').count()
# rename and subset the data just to the count of the hotspots per neighborhood
airbnb_agg = airbnb_agg.rename(columns={'id':'airbnb_count'})['airbnb_count'].to_frame()

    # import wifi data
wifi = read_data_link("https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/bolognawifi-elenco-hot-spot/exports/geojson?lang=it&timezone=Europe%2FRome", 'geojson')
wifi = wifi.to_crs(epsg=6933)
    # process the data for mapping
# aggregate the count of wifi hotspots at the neighborhood level
wifi_agg = gpd.overlay(base_map_2019, wifi, how='intersection', keep_geom_type=False).groupby('zona_pross').count()
# rename and subset the data just to the count of the hotspots per neighborhood
wifi_agg = wifi_agg.rename(columns={'hostname':'hotspot_count'})['hotspot_count']

    # import participatory budget data
budg_geo = read_data_link("https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/bilancio-partecipativo/exports/geojson?lang=it&timezone=Europe%2FRome",'geojson')
budg_geo = budg_geo.to_crs(epsg=6933)

    # import the street furniture dataset
furniture = read_data_link("https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/arredo/exports/geojson?lang=it&timezone=Europe%2FRome",'geojson')
furniture = furniture.to_crs(epsg=6933)

# subset columns to relevant columns
furniture = furniture[['classe_arredo','data_agg','geometry','classe_conservazione']]
furniture.data_agg = pd.to_datetime(furniture.data_agg)
# subset data to only those furniture updated before the end of 2019
furniture_2019 = furniture.loc[furniture['data_agg'] <= '2019-12-31']
furniture_2019_zona = gpd.overlay(base_map_2019.reset_index()[['zona_fiu','geometry']], furniture_2019, how='intersection', keep_geom_type=False)

# calculate the percentage of not "good" condition street furniture by zone
# get total street furniture count per zone
total_furniture = furniture_2019_zona['zona_fiu'].value_counts().to_frame().rename(columns={'count':'total_furn'})
# get the count of street furniture in coniditon less than "buono" by zone
not_good_furniture = (furniture_2019_zona.loc[furniture_2019_zona['classe_conservazione'] != 'BUONO'])['zona_fiu'].value_counts().to_frame().rename(columns={'count':'not_good_furn'})
# join the two figures just calculated together
furniture_state_2019 = total_furniture.join(not_good_furniture)
# divide to get the percentage of furniture not good per zone
furniture_state_2019['p_furn_not_good'] = furniture_state_2019['not_good_furn']/furniture_state_2019['total_furn']

    # import bike lane geojson
bike_lanes = read_data_link("https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/piste-ciclopedonali/exports/geojson?lang=it&timezone=Europe%2FRome", "geojson")
bike_lanes = bike_lanes.to_crs(epsg=6933)

    # import traffic accidents dataset
traffic_accidents = read_data_link("https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/incidenti_new/exports/geojson?lang=it&timezone=Europe%2FRome","geojson")
traffic_accidents = traffic_accidents.to_crs(epsg=6933)
# aggregate the data by zone
incidents_agg = traffic_accidents.groupby(['nomezona','anno'])[['n_incident','totale_fer','totale_mor']].sum()
# subset to relevant columns and the year 2019
incidents_2019 = incidents_agg.query("anno == '2019'").reset_index().set_index('nomezona')[['n_incident','totale_fer','totale_mor']]

    # start creating transport dataframe to hold all of the transport related data elements and standardize them by population
transport_2019 = base_map_2019
transport_2019.index = transport_2019.index.str.upper()
# join the incident data to the base map
transport_2019 = transport_2019.join(incidents_2019)
# calculate traffic-related incidents, injured, deaths per capita
transport_2019['incident_per_1000'] = round(transport_2019['n_incident'] / (transport_2019['population']/1000),4)
transport_2019['injured_per_1000'] = round(transport_2019['totale_fer'] / (transport_2019['population']/1000),4)
transport_2019['injured_per_incident'] = round(transport_2019['totale_fer'] / (transport_2019['n_incident']/1000),4)
transport_2019['mortality_per_1000'] = round(transport_2019['totale_mor'] / (transport_2019['population']/1000),4)

    # vehicle traffic flows
# import data *** commented out because it takes a long time to import
traffic_2019 = read_data_link("https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/rilevazione-autoveicoli-tramite-spire-anno-2019/exports/geojson?lang=it&timezone=Europe%2FRome", "geojson")
# add up all traffic for a given day at one flow point *** commented out to avoid redundancy given no import
traffic_2019['day_total_traffic'] = traffic_2019['00_00_01_00'] + traffic_2019['01_00_02_00'] + traffic_2019['02_00_03_00'] + traffic_2019['03_00_04_00'] + traffic_2019['04_00_05_00'] + traffic_2019['05_00_06_00'] + traffic_2019['06_00_07_00'] + traffic_2019['07_00_08_00'] + traffic_2019['08_00_09_00'] + traffic_2019['09_00_10_00'] + traffic_2019['10_00_11_00'] + traffic_2019['11_00_12_00'] + traffic_2019['12_00_13_00'] + traffic_2019['13_00_14_00'] + traffic_2019['14_00_15_00'] + traffic_2019['15_00_16_00'] + traffic_2019['16_00_17_00'] + traffic_2019['17_00_18_00'] + traffic_2019['18_00_19_00'] + traffic_2019['19_00_20_00'] + traffic_2019['20_00_21_00'] + traffic_2019['21_00_22_00'] + traffic_2019['22_00_23_00'] + traffic_2019['23_00_24_00']
# subset to relevant columns
traffic_2019_sub = traffic_2019[['data','day_total_traffic','geometry']]
# calculate average daily traffic flow at each point for the year 2019
traffic_2019_agg = traffic_2019_sub.groupby(['geometry'])[['day_total_traffic']].mean().reset_index()
# convert this dataset to a geoframe
traffic_2019_geo = gpd.GeoDataFrame(traffic_2019_agg, geometry="geometry").to_crs(epsg=6933)
# perform a spatial join to get the zone for each flow point + calculate the mean traffic flow per zone + rename columns
traffic_zone_2019 = traffic_2019_geo.sjoin(base_map_zone, how = "left", predicate="within").groupby(['index_right'])[['day_total_traffic']].mean().reset_index().rename(columns={'index_right':'zona_fiu','day_total_traffic':'avg_daily_traffic'})
# format zone name in preparation for joining this data to the cumulative dataset
traffic_zone_2019['zona_fiu'] = traffic_zone_2019['zona_fiu'].str.upper()
# join the traffic data to the base map
transport_2019 = transport_2019.join(traffic_zone_2019.set_index('zona_fiu'))
# calculate avg daily traffic flow per capita in the cumulative map
transport_2019['traffic_per_1000'] = round(transport_2019['avg_daily_traffic'] / (transport_2019['population']/1000),4)

    # import bike parking dataset
bike_racks = read_data_link("https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/rastrelliere-per-biciclette/exports/geojson?lang=it&timezone=Europe%2FBerlin", "geojson")
bike_racks = bike_racks.to_crs(epsg=6933)
# aggregate and rename columns
bike_racks_agg = bike_racks.groupby(['nomezona'])[['numposti']].sum().rename(columns={'nomezona':'zona_fiu','numposti':'n_bike_parking'})
# append bike parking figures to the cumulative transport dataset
transport_2019 = transport_2019.join(bike_racks_agg)
# calculate number of bike parking spots per capita
transport_2019['bike_parking_per_1000'] = round(transport_2019['n_bike_parking'] / (transport_2019['population']/1000),4)
transport_2019['bike_parking_per_household'] = round(transport_2019['n_bike_parking'] / (transport_2019['households']/1000),4)

    # bus stops (most meaningful mass transit access proxy)
mobility_offer = read_data_link("https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/workshop-bcn-mappatura-poi-completa_mobility/exports/geojson?lang=it&timezone=Europe%2FRome", 'geojson')
mobility_offer = mobility_offer.to_crs(epsg=6933)
# subset to bus stops - other aspects are not significant enough in number
tper_stops = mobility_offer.loc[mobility_offer.tipologia_punto_di_interesse == 'Fermate Tper']
# join the mobility offerings to the base mpa by zone 
# and count the number of each type of mobility point within each zone
tper_stops_grp = tper_stops.sjoin(base_map_2019, how = "left", predicate="within").groupby(['index_right', 'population', 'tipologia_punto_di_interesse']).count()
# clean up the data by subsetting and renaming columns and setting the zone as the index
tper_stops_zone = tper_stops_grp.reset_index()[['index_right','population','geometry']].rename(columns={'index_right':'zona_fiu','geometry':'n_tper_stops'})
tper_stops_zone = tper_stops_zone.set_index('zona_fiu')
# calculate bus stops per capita and check variance
tper_stops_zone['tper_stops_per_1000'] = round(tper_stops_zone['n_tper_stops']/(tper_stops_zone['population']/1000),4)
#tper_stops_zone.tper_stops_per_1000.plot()
# given variance seems significant, append to the transportation data
transport_2019 = transport_2019.join(tper_stops_zone[['tper_stops_per_1000']])