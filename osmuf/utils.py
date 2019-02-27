################################################################################
# Module: core.py
# Description: urban form analysis from OpenStreetMap
# License: MIT, see full license in LICENSE.txt
# Web: https://github.com/atelierlibre/osmuf
################################################################################

import re
import time
import os
import ast
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import seaborn as sns

import osmuf.smallestenclosingcircle as sec

from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.geometry import LineString
from shapely.ops import polygonize
from shapely import wkt

from descartes import PolygonPatch

from osmnx import settings
from osmnx import save_and_show
from osmnx.utils import log
from osmnx.utils import make_str

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.collections import PatchCollection

# extract the coordinates of shapely polygons and multipolygons
# as a list of tuples
def extract_poly_coords(geom):
    if geom.type == 'Polygon':
        exterior_coords = geom.exterior.coords[:]
        interior_coords = []
        for interior in geom.interiors:
            interior_coords += interior.coords[:]
    elif geom.type == 'MultiPolygon':
        exterior_coords = []
        interior_coords = []
        for part in geom:
            epc = extract_poly_coords(part)  # Recursive call
            exterior_coords += epc['exterior_coords']
            interior_coords += epc['interior_coords']
    else:
        raise ValueError('Unhandled geometry type: ' + repr(geom.type))
#    return {'exterior_coords': exterior_coords,
#            'interior_coords': interior_coords}
    return exterior_coords + interior_coords


# takes a shapely polygon or multipolygon and returns a shapely circle polygon
def circlizer(x):
    (cx, cy, buff) = sec.make_circle(extract_poly_coords(x))
    donut = Point(cx, cy).buffer(buff)

    return donut

def dict_to_gdf(place_dict):

    # 0. INVERT COORDINATES FROM LAT, LONG TO LONG, LAT
    place_dict['coordinates']=(place_dict['coordinates'][1], place_dict['coordinates'][0])

    # 1. convert dict to Pandas dataframe
    place_df = pd.DataFrame([place_dict])

    # 2. create 'geometry' column as tuple of Latitude and Longitude
    place_df = place_df.rename(columns={'coordinates': 'geometry'})

    # 3. transform tuples into Shapely Points
    place_df['geometry'] = place_df['geometry'].apply(Point)

    # places_gdf = geopandas.GeoDataFrame(places_df, geometry='geometry')
    place_gdf = gpd.GeoDataFrame(place_df, geometry='geometry')

    # set the crs
    place_gdf.crs = {'init' :'epsg:4326'}

    return place_gdf
