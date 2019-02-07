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

import smallestenclosingcircle as sec

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

# takes a gdf with polgon geometry and returns a new gdf of smallest enclosing
# circles with their areas (in hectares) centroids and regularity ratio
def gdf_circlizer(gdf):
    # create a new gdf that includes the geometry of the old gdf and its area
    new_gdf = gdf.filter(['area_ha_net', 'geometry'])
    # replace the polygon geometry with the smallest enclosing circle
    new_gdf['geometry']=gdf['geometry'].apply(circlizer)

    # calculate centroids for labelling purposes
    new_gdf['centroid'] = gdf.centroid

    # calculate the area of the smallest enclosing circles
    new_gdf['area_sec_ha'] = new_gdf.area/10000

    # calculate 'regularity' as "the ratio between the area of the block and
    # the area of the circumscribed circle C" Barthelemy M. and Louf R., (2014)
    new_gdf['regularity'] = new_gdf['area_ha_net']/new_gdf['area_sec_ha']

    return new_gdf


# def graph_to_polygons(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True):
def graph_to_polygons(G, node_geometry=True, fill_edge_geometry=True):
    """
    Convert the edges of a graph into a GeoDataFrame of polygons
    Parameters
    ----------
    G : networkx multidigraph
    node_geometry : bool
        if True, create a geometry column from node x and y data
    fill_edge_geometry : bool
        if True, fill in missing edge geometry fields using origin and
        destination nodes
    Returns
    -------
    GeoDataFrame
        gdf_polygons
    """

    start_time = time.time()

    # create a list to hold our edges, then loop through each edge in the
    # graph
    edges = []
    for u, v, key, data in G.edges(keys=True, data=True):

        # for each edge, add key and all attributes in data dict to the
        # edge_details
        edge_details = {'u':u, 'v':v, 'key':key}
        for attr_key in data:
            edge_details[attr_key] = data[attr_key]

        # if edge doesn't already have a geometry attribute, create one now
        # if fill_edge_geometry==True
        if 'geometry' not in data:
            if fill_edge_geometry:
                point_u = Point((G.nodes[u]['x'], G.nodes[u]['y']))
                point_v = Point((G.nodes[v]['x'], G.nodes[v]['y']))
                edge_details['geometry'] = LineString([point_u, point_v])
            else:
                edge_details['geometry'] = np.nan

        edges.append(edge_details)

    # extract the edge geometries from the list of edge dictionaries
    edge_geometry = []

    for edge in edges:
        edge_geometry.append(edge['geometry'])

    # create a list to hold our polygons
    polygons = []

    polygons = list(polygonize(edge_geometry))

    # Create a GeoDataFrame from the list of polygons and set the CRS

    # an option here is to feed it a list of dictionaries with a 'geometry' key
    # this would be one step e.g.
    # gdf_polys = gpd.GeoDataFrame(polygons)

    # Greate an empty GeoDataFrame
    gdf_polygons = gpd.GeoDataFrame()

    # Create a new column called 'geometry' to the GeoDataFrame
    gdf_polygons['geometry'] = None

    # Assign the list of polygons to the geometry column
    # Genereate area and centroid columns
    gdf_polygons.geometry = polygons

    # Set the crs
    gdf_polygons.crs = G.graph['crs']
    gdf_polygons.gdf_name = '{}_polygons'.format(G.graph['name'])

    return gdf_polygons
#    return edges

    # create a GeoDataFrame from the list of edges and set the CRS
#    gdf_edges = gpd.GeoDataFrame(edges)
#    gdf_edges.crs = G.graph['crs']
#

    log('Created GeoDataFrame "{}" from graph in {:,.2f} seconds'.format(gdf_edges.gdf_name, time.time()-start_time))
