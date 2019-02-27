################################################################################
# Module: core.py
# Description: urban form analysis from OpenStreetMap
# License: MIT, see full license in LICENSE.txt
# Web: https://github.com/atelierlibre/osmuf
################################################################################

# import re
# import time
# import os
# import ast
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import seaborn as sns

import osmuf.smallestenclosingcircle as sec

from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import polygonize
from shapely import wkt

# from descartes import PolygonPatch

# from osmnx import settings
# from osmnx import save_and_show
# from osmnx.utils import log
# from osmnx.utils import make_str

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.collections import PatchCollection

from .plot import *
from .utils import *

def study_area_from_point(point, distance):

    # define study area as gdf

    bbox = ox.bbox_from_point(point, distance, project_utm=True, return_crs=True)
    # distance is from centre to edge i.e. final square is double the distance
    # north, south, east, west : tuple, if return_crs=False
    # north, south, east, west, crs_proj : tuple, if return_crs=True

    # split the tuple
    n, s, e, w, crs_code = bbox

    # Create geopandas GeoDataFrame which contains the bounding box as a named polygon
    study_area = gpd.GeoDataFrame()
    # study_area.loc[0, 'Location'] = place_name
    study_area.loc[0, 'geometry'] = Polygon([(w, s), (w, n), (e, n), (e, s)])
    study_area['area_ha'] = study_area.area/10000
    study_area.crs = crs_code

    return study_area

def city_blocks_from_point(point, distance):

    # download 'place' polygons
    city_blocks = ox.footprints_from_point(point, distance, footprint_type="place")
    # filter place polygons to retain only city blocks
    city_blocks = city_blocks.loc[city_blocks['place'] == 'city_block']
    # keep only two columns 'place' and 'geometry'
    city_blocks = city_blocks[['place','geometry']].copy()
    # project city_blocks to UTM
    city_blocks = ox.project_gdf(city_blocks)
    # write perimeter length into column
    city_blocks['perimeter_m'] = city_blocks.length
    # calculate areas in hectares (not meters) and include as a column
    city_blocks['area_net_ha'] = city_blocks.area/10000
    # calculate perimeter (m) per net area (ha)
    city_blocks['perimeter_per_area'] = city_blocks['perimeter_m']/city_blocks['area_net_ha']
    # name the index
    city_blocks.index.name='block_id'

    return city_blocks

def street_graph_from_point(point, distance):
    # download the highway network within this boundary
    street_graph = ox.graph_from_point(point, distance, network_type='all',
                                       simplify=True, retain_all=True,
                                       truncate_by_edge=True, clean_periphery=True)
    # project the network to UTM & convert to undirected graph to
    street_graph = ox.project_graph(street_graph)
    # remove duplicates which make polygonization fail
    street_graph = ox.get_undirected(street_graph)

    return street_graph

def gdf_convex_hull(gdf):

    ### INSERT CHECK FOR CRS HERE?

    # project gdf to geographic coordinates as footprints_from_polygon requires it
    gdf_temp = ox.project_gdf(gdf, to_latlong=True)
    # determine the boundary polygon to fetch buildings within
    # buffer originally 0.000225, buffer actually needs to go whole block away
    # to get complete highways therefor trying 0.001
    boundary=gdf_temp.cascaded_union.convex_hull.buffer(0.001)
    # NOTE - maybe more efficient to generate boundary first then reproject second?

    return boundary

def street_graph_from_gdf(gdf):

    # generate boundary
    boundary = gdf_convex_hull(gdf)

    # download the highway network within this boundary
    street_graph = ox.graph_from_polygon(boundary, network_type='all',
                                       simplify=True, retain_all=True,
                                       truncate_by_edge=True, clean_periphery=True)
    # project the network to UTM & convert to undirected graph to
    street_graph = ox.project_graph(street_graph)
    # remove duplicates which make polygonization fail
    street_graph = ox.get_undirected(street_graph)

    return street_graph

def streets_from_street_graph(street_graph):

    streets = ox.graph_to_gdfs(street_graph, nodes=False)

    # insert filtering/processing here for OSMuf purposes

    return streets

def footprints_from_gdf(gdf):

    # generate boundary
    boundary = gdf_convex_hull(gdf)

    # download buildings within boundary
    footprints = ox.footprints_from_polygon(boundary)

    return footprints

def buildings_from_gdf(gdf):

    # download buildings within boundary
    buildings = footprints_from_gdf(gdf)

    # create filtered copy with only key columns
    buildings = buildings[['building','building:levels','geometry']].copy()
    # project filtered copy to UTM
    buildings = ox.project_gdf(buildings)

    ### NEED TO CHECK HERE THAT 'building:levels' IS ACTUALLY RETURNED - MIGHT NOT BE
    ### THERE ACTUALLY IS AN INTEGER TYPE WITH NAN, MAY BE BETTER

    # convert 'building:levels' to float from object (int doesn't support NaN)
    buildings['building:levels']=buildings['building:levels'].astype(float)
    # convert fill NaN with zeroes to allow conversion to int
    buildings = buildings.fillna({'building:levels': 0})
    # convert to int
    buildings["building:levels"] = pd.to_numeric(buildings['building:levels'], downcast='integer')

    # generate footprint areas
    buildings['footprint_m2']=buildings.area
    # generate total_GEA
    buildings['total_GEA_m2']=buildings.area*buildings['building:levels']

    return buildings

def join_buildings_city_block_id(buildings, city_blocks):
    # where they intersect, add the city_block number onto each building
    # how = 'left', 'right', 'inner' sets how the index of the new gdf is determined, left retains buildings index
    # was 'intersects', 'contains', 'within'
    buildings = gpd.sjoin(buildings, city_blocks[['geometry']], how="left", op='intersects')
    buildings.rename(columns={'index_right' : 'block_id'}, inplace=True)

    # convert any NaN values in block_id to zero, then convert to int
    buildings = buildings.fillna({'block_id': 0})
    buildings["block_id"] = pd.to_numeric(buildings['block_id'], downcast='integer')

    return buildings

def link_buildings_highways(buildings, highways):
    return highways.distance(buildings).idxmin()

def join_buildings_street_id(buildings, streets):

    buildings['street_id'] = buildings.geometry.apply(link_buildings_highways, args=(streets,))

    return buildings

def form_factor(poly_gdf):
    # takes a gdf with polgon geometry and returns a new gdf of smallest enclosing
    # circles with their areas (in hectares) centroids and regularity ratio

    # create a new gdf that includes the geometry of the old gdf and its area
    circles_gdf = poly_gdf.filter(['geometry', 'area_net_ha'])
    # replace the polygon geometry with the smallest enclosing circle
    circles_gdf['geometry'] = circles_gdf['geometry'].apply(circlizer)

    # calculate the area of the smallest enclosing circles
    circles_gdf['area_sec_ha'] = circles_gdf.area/10000

    # calculate 'regularity' as "the ratio between the area of the block and
    # the area of the circumscribed circle C" Barthelemy M. and Louf R., (2014)
    circles_gdf['form_factor'] = circles_gdf['area_net_ha']/circles_gdf['area_sec_ha']

    return circles_gdf

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

def gen_city_blocks_gross(street_graph, city_blocks):

    ### BOTH PROJECTED - NEED TO CHECK CRS HERE?

    # polygonize the highway network & return it as a GeoDataFrame
    # this will include edge polygons that are removed in the next steps
    city_blocks_gross = graph_to_polygons(street_graph, node_geometry=False)

    # transfer attributes from city_blocks_net to city_blocks_gross where they intersect
    city_blocks_gross = gpd.sjoin(city_blocks_gross, city_blocks, how="left", op='intersects')
    # dissolve together city_blocks_gross polygons that intersect with the same city_blocks polygon
    city_blocks_gross.rename(columns={'index_right' : 'block_id'}, inplace=True)
    city_blocks_gross = city_blocks_gross.dissolve(by='block_id')
    # convert the index to int
    city_blocks_gross.index = city_blocks_gross.index.astype(int)
    # remove unecessary columns
    city_blocks_gross = city_blocks_gross[['place', 'geometry']]
    # change text city_block to city_block_gross
    city_blocks_gross["place"] = city_blocks_gross['place'].str.replace('city_block', 'city_block_gross')
    # give the dataframe a name
    city_blocks_gross.name = 'city_blocks_gross'
    # calculate gross area in hectares (not meters) and include as a column in city_blocks_net
    city_blocks['area_gross_ha'] = city_blocks_gross.area/10000
    # calculate the net to gross ratio for the blocks and include as a column in city_blocks_net
    city_blocks['net_to_gross'] = round(city_blocks.area_net_ha/city_blocks.area_gross_ha, 2)

    return (city_blocks, city_blocks_gross)

def join_city_blocks_building_data(city_blocks, buildings):

    building_areas_by_block=buildings[['footprint_m2','total_GEA_m2']].groupby([buildings['block_id']]).sum()
    # next line may not be necessary, don't want to create entry '0' in gdf of city_blocks
    building_areas_by_block = building_areas_by_block.drop([0])

    city_blocks = city_blocks.merge(building_areas_by_block, on = 'block_id')

    city_blocks['GSI_net'] = city_blocks['footprint_m2']/(city_blocks['area_net_ha']*10000)
    city_blocks['GSI_gross'] = city_blocks['footprint_m2']/(city_blocks['area_gross_ha']*10000)
    city_blocks['FSI_net'] = city_blocks['total_GEA_m2']/(city_blocks['area_net_ha']*10000)
    city_blocks['FSI_gross'] = city_blocks['total_GEA_m2']/(city_blocks['area_gross_ha']*10000)

    return city_blocks
