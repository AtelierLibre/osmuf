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

# takes a gdf with polgon geometry and returns a new gdf of smallest enclosing
# circles with their areas (in hectares) centroids and regularity ratio
def gdf_circlizer(gdf):
    # create a new gdf that includes the geometry of the old gdf and its area
    new_gdf = gdf.filter(['area_net_ha', 'geometry'])
    # replace the polygon geometry with the smallest enclosing circle
    new_gdf['geometry']=gdf['geometry'].apply(circlizer)

    # calculate centroids for labelling purposes
    new_gdf['centroid'] = gdf.centroid

    # calculate the area of the smallest enclosing circles
    new_gdf['area_sec_ha'] = new_gdf.area/10000

    # calculate 'regularity' as "the ratio between the area of the block and
    # the area of the circumscribed circle C" Barthelemy M. and Louf R., (2014)
    new_gdf['regularity'] = new_gdf['area_net_ha']/new_gdf['area_sec_ha']

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
    
def city_blocks_from_point(point, distance):
    
    # may be better to take the highway network as an input so that it can also be used elsewhere for
    # e.g. connectivity assessment
    
    
    # DOWNLOAD PLACE POLYGONS & SURROUNDING HIGHWAY NETWORK (GEOGRAPHIC COORDINATES)
    
    # net - a filter needs to be added to ensure it only retains place=city_block
    place_polys = ox.footprints_from_point(point, distance, footprint_type="place")
    
    # use a buffered convex hull of the net city blocks to fetch highways within
    # approx. 25m of the net city blocks. ox.graph_from_polygon takes the polygon
    # in geographic coordiantes
    boundary=place_polys.cascaded_union.convex_hull.buffer(0.000225)
    
    # highway network
    highway_network = ox.graph_from_polygon(boundary, network_type='all',
                                            simplify=True, retain_all=True, truncate_by_edge=True)
    
    
    # PROJECT NET CITY BLOCKS AND HIGHWAY NETWORK TO UTM
    
    # create filtered copy of net_city_blocks
    place_polys = place_polys.loc[place_polys['place'] == 'city_block']
    city_blocks = place_polys[['place','geometry']].copy()

    # project city_blocks to UTM
    city_blocks = ox.project_gdf(city_blocks)
    
    # project the network to UTM & convert to undirected graph to remove
    # duplicates which make polygonization fail
    highway_network = ox.project_graph(highway_network)
    highway_network = ox.get_undirected(highway_network)
    
    
    # PROCESS NET CITY BLOCKS
    
    # calculate areas in hectares (not meters) and include as a column
    city_blocks['area_net_ha'] = city_blocks.area/10000
    city_blocks.index.name='block_id'

    
    # PROCESS GROSS CITY BLOCKS
    
    # polygonize the highway network & return it as a GeoDataFrame
    # this will include edge polygons that are removed in the next steps
    city_blocks_gross_raw = graph_to_polygons(highway_network, node_geometry=False)
    
    # first, transfer attributes from city_blocks_net to city_blocks_gross where they intersect
    city_blocks_gross = gpd.sjoin(city_blocks_gross_raw, city_blocks, how="left", op='intersects')
    
    # dissolve together city_blocks_gross polygons that intersect with the same city_blocks polygon
    city_blocks_gross.rename(columns={'index_right' : 'block_id'}, inplace=True)
    # this can be tidied up, just needs to be the index as an integer, doesn't need a column name
    city_blocks_gross = city_blocks_gross.dissolve(by='block_id')
    city_blocks_gross.index = city_blocks_gross.index.astype(int)
    
    # calculate gross area in hectares (not meters) and include as a column in city_blocks_net
    city_blocks['area_gross_ha'] = city_blocks_gross.area/10000
    # calculate the net to gross ratio for the blocks and include as a column in city_blocks_net
    city_blocks['net_to_gross'] = round(city_blocks.area_net_ha/city_blocks.area_gross_ha, 2)
    
    # FUTURE NOTE - include a tare space dataframe in the return
    
    return (city_blocks, city_blocks_gross, city_blocks_gross_raw)

def buildings_from_city_blocks(city_blocks):
    
    # project city_blocks_net back to geographic coordinates as footprints_from_polygon requires it
    city_blocks_temp = ox.project_gdf(city_blocks, to_latlong=True)
    # determine the boundary polygon to fetch buildings within
    boundary=city_blocks_temp.cascaded_union.convex_hull.buffer(0.000225)
    # NOTE - maybe more efficient to generate boundary first then reproject second?
    
    # download buildings within boundary
    buildings_raw = ox.footprints_from_polygon(boundary)
    
    # create filtered copy with only key columns
    buildings_filtered=buildings_raw[['building','building:levels','geometry']].copy()

    # project filtered copy to UTM
    buildings_filtered = ox.project_gdf(buildings_filtered)

    # convert 'building:levels' to float from object (int doesn't support NaN)
    buildings_filtered['building:levels']=buildings_filtered['building:levels'].astype(float)
    # convert fill NaN with zeroes to allow conversion to int
    buildings_filtered = buildings_filtered.fillna({'building:levels': 0})
    buildings_filtered["building:levels"] = pd.to_numeric(buildings_filtered['building:levels'], downcast='integer')
    
    # generate footprint areas
    buildings_filtered['footprint_m2']=buildings_filtered.area

    # generate total_GEA
    buildings_filtered['total_GEA_m2']=buildings_filtered.area*buildings_filtered['building:levels']

    # where they intersect, add the city_block number onto each building
    # how = 'left', 'right', 'inner' sets how the index of the new gdf is determined, left retains buildings index
    # was 'intersects', 'contains', 'within'
    buildings_with_blocks = gpd.sjoin(buildings_filtered, city_blocks[['geometry']], how="left", op='intersects')
    buildings_with_blocks.rename(columns={'index_right' : 'block_id'}, inplace=True)

    # convert any NaN values in block_id to zero, then convert to int
    buildings_with_blocks = buildings_with_blocks.fillna({'block_id': 0})
    buildings_with_blocks["block_id"] = pd.to_numeric(buildings_with_blocks['block_id'], downcast='integer')

    return buildings_with_blocks

def blocks_with_buildings(city_blocks, buildings):
    
    building_areas_by_block=buildings[['footprint_m2','total_GEA_m2']].groupby([buildings['block_id']]).sum()
    # next line may not be necessary, don't want to create entry '0' in gdf of city_blocks
    building_areas_by_block = building_areas_by_block.drop([0])
    
    city_blocks = city_blocks.merge(building_areas_by_block, on = 'block_id')
    
    city_blocks['GSI_net'] = city_blocks['footprint_m2']/(city_blocks['area_net_ha']*10000)
    city_blocks['GSI_gross'] = city_blocks['footprint_m2']/(city_blocks['area_gross_ha']*10000)
    city_blocks['FSI_net'] = city_blocks['total_GEA_m2']/(city_blocks['area_net_ha']*10000)
    city_blocks['FSI_gross'] = city_blocks['total_GEA_m2']/(city_blocks['area_gross_ha']*10000)
    
    return city_blocks