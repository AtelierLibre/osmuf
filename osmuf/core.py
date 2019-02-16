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
def form_factor_of_blocks(poly_gdf):
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
    
    # download 'place' polygons
    city_blocks = ox.footprints_from_point(point, distance, footprint_type="place")
    # create a convex hull of the place polygons and buffer approx. 25m in geographic coordinates
    boundary = city_blocks.cascaded_union.convex_hull.buffer(0.000225)
    # download the highway network within this boundary
    highway_network = ox.graph_from_polygon(boundary, network_type='all',
                                            simplify=True, retain_all=True, truncate_by_edge=True)
    
    
    # PROJECT NET CITY BLOCKS AND HIGHWAY NETWORK TO UTM
    
    # filter place polygons to retain only city blocks
    city_blocks = city_blocks.loc[city_blocks['place'] == 'city_block']
    city_blocks = city_blocks[['place','geometry']].copy()

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
    # transfer attributes from city_blocks_net to city_blocks_gross where they intersect
    city_blocks_gross = gpd.sjoin(city_blocks_gross_raw, city_blocks, how="left", op='intersects')
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

################
### PLOTTING ###
################

# for nicer looking plots
import seaborn
plt.style.use('seaborn')

#  “helper function” that places a text box inside of a plot and acts as an “in-plot title”
# from https://realpython.com/python-matplotlib-guide/
# position format is x,y e.g. 0.02, 0.94
def add_titlebox(ax, text):
    ax.text(0.04, 0.92, text,
        horizontalalignment='left',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.6),
        fontsize=12.5)
    return ax

# label style
style = dict(size=11, color='black', horizontalalignment='center')

# label individual geometries
def label_geom(ax, gdf, column):
    for idx, row in gdf.iterrows():
        label = round(row[column], 2)
        ax.text(row.geometry.centroid.x, row.geometry.centroid.y, label, **style)

# city_blocks plot
def plot_city_blocks(city_blocks_gross_raw, city_blocks_gross, city_blocks):
    gridsize = (2, 3)
    fig = plt.figure(figsize=(18, 10))
    
    ax1 = plt.subplot2grid(gridsize, (0, 1), colspan=2, rowspan=2, facecolor='white')
    ax2 = plt.subplot2grid(gridsize, (0, 0))
    ax3 = plt.subplot2grid(gridsize, (1, 0))

    ax = (ax1, ax2, ax3)
    
    # map
    city_blocks_gross_raw.plot(ax=ax1, color='whitesmoke', edgecolor='white')

    city_blocks_gross.plot(ax=ax1, color='lightgrey',edgecolor='white', alpha=1)

    city_blocks.plot(ax=ax1, column='area_net_ha', cmap='Blues', alpha=0.5)

    # show area and net-to-gross of city_blocks 
    label_geom(ax1, city_blocks, 'area_net_ha')

    ax1.set_title('City blocks net area (ha) and net-to-gross ratio', fontsize=14)

    # histogram - form factor, range=(0,1), 
    ax2.hist(city_blocks['area_net_ha'], bins='auto', color='Blue', alpha=0.5)
    add_titlebox(ax2, 'Area Distribution')
    ax2.set_xlabel("Net area (ha)")
    ax2.set_ylabel("Count")
    ax2.set_xlim([0, 4])

    # scatterplot - 
    ax3.scatter(x=city_blocks.area_net_ha, y=city_blocks.net_to_gross, color='Blue')
    add_titlebox(ax3, 'Net-to-gross by area')
    ax3.set_xlabel("Net area (ha)")
    ax3.set_ylabel("Net-to-gross ratio")
    ax3.set_xlim([0, 4])
    ax3.set_ylim([0, 1])
    
    return fig, ax


def plot_form_factor(city_blocks_gross_raw, city_blocks_gross, city_blocks, circle_gdf):
    gridsize = (2, 3)
    fig = plt.figure(figsize=(18, 10))
    
    ax1 = plt.subplot2grid(gridsize, (0, 1), colspan=2, rowspan=2, facecolor='white')
    ax2 = plt.subplot2grid(gridsize, (0, 0))
    ax3 = plt.subplot2grid(gridsize, (1, 0))

    ax = (ax1, ax2, ax3)
    
    # map
    city_blocks_gross_raw.plot(ax=ax1, color='whitesmoke', edgecolor='white');

    city_blocks_gross.plot(ax=ax1, color='lightgrey',edgecolor='white', alpha=1);

    city_blocks.plot(ax=ax1, color='darkgrey', alpha=1);

    circle_gdf.plot(ax=ax1,
                    edgecolor='red',
                    column='form_factor',
                    cmap='Reds',
                    vmin=0,
                    vmax=1,
                    alpha=0.6,
                    legend=True)

    # show form factor of city blocks
    label_geom(ax1, circle_gdf, 'form_factor')

    ax1.set_title('Form factor of urban blocks (φ)', fontsize=14)

    # histogram - form factor, range=(0,1)
    ax2.hist(circle_gdf['form_factor'], bins=20, color='Red', alpha=0.5)
    add_titlebox(ax2, 'Form factor (φ)')
    ax2.set_xlabel("Form factor (φ)")
    ax2.set_ylabel("Count")
    ax2.set_xlim([0, 1])

    # scatterplot - 
    ax3.scatter(x=circle_gdf.area_net_ha, y=circle_gdf.form_factor, color='Red')
    add_titlebox(ax3, "Form factor (φ)")
    ax3.set_xlabel("Area (ha)")
    ax3.set_ylabel("Form factor (φ)")
    ax3.set_xlim([0, 4])
    ax3.set_ylim([0, 1])
    
    return fig, ax

def plot_buildings(city_blocks_gross_raw, city_blocks_gross, city_blocks, buildings):
    gridsize = (2, 3)
    fig = plt.figure(figsize=(18, 10))
    
    ax1 = plt.subplot2grid(gridsize, (0, 1), colspan=2, rowspan=2, facecolor='white')
    ax2 = plt.subplot2grid(gridsize, (0, 0))
    ax3 = plt.subplot2grid(gridsize, (1, 0))

    ax = (ax1, ax2, ax3)
    
    # MAP BASELAYERS - DON'T CHANGE
    
    # all polygons from highway network
    city_blocks_gross_raw.plot(ax=ax1, color='whitesmoke', edgecolor='white')
    # highway network polygons processed to correspond to net city blocks
    city_blocks_gross.plot(ax=ax1, color='lightgrey',edgecolor='white')
    # net city blocks
    city_blocks.plot(ax=ax1, color='darkgrey',edgecolor='white')
    # buildings with unknown storeys as hatched
    buildings[buildings['building:levels']==0].plot(ax=ax1, color='dimgrey', edgecolor='white', hatch='///')

    # buildings with known storeys - 'categorical=True'
    buildings[buildings['building:levels']>0].plot(ax=ax1,
                                                   column='building:levels',
                                                   cmap='plasma',
                                                   vmin=0,
                                                   vmax=25,
                                                   legend=True,
                                                   )
 
    ax1.set_title('Building Heights (storeys)', fontsize=14)
    
    # histogram
    ax3.hist(buildings['building:levels'], bins=26, color='Purple')
    ax3.set_xlabel('Building storeys')
    ax3.set_ylabel('Count')
    # add_titlebox(ax2, 'Histogram: form factor (φ)')
    ax3.set_xlim([0,25])
    
    return fig, ax

# metric options are GSI_net and FSI_net
def plot_urban_form(city_blocks_gross_raw, city_blocks_gross, city_blocks, buildings, metric='GSI_net'):
    gridsize = (2, 3)
    fig = plt.figure(figsize=(18, 10))
    
    ax1 = plt.subplot2grid(gridsize, (0, 1), colspan=2, rowspan=2, facecolor='white')
    ax2 = plt.subplot2grid(gridsize, (0, 0))
    ax3 = plt.subplot2grid(gridsize, (1, 0))

    ax = (ax1, ax2, ax3)
    
    # MAP BASELAYERS - DON'T CHANGE
    
    # all polygons from highway network
    city_blocks_gross_raw.plot(ax=ax1, color='whitesmoke', edgecolor='white');
    # highway network polygons processed to correspond to net city blocks
    city_blocks_gross.plot(ax=ax1, color='lightgrey',edgecolor='white');
    # buildings with known storeys
    buildings[buildings['building:levels']>0].plot(ax=ax1, color='darkgrey', edgecolor='white')
    # buildings with unknown storeys as hatched
    buildings[buildings['building:levels']==0].plot(ax=ax1, color='lightgrey', edgecolor='white', hatch='///')

    if metric=='GSI_net':
        column='GSI_net'
        cmap='Reds'
        color='Red'
        vmin=0
        vmax=1
        title='Site Coverage (GSI)'
        ylim=[0,1]
        xlim=[0,1]
    elif metric=='FSI_net':
        column='FSI_net'
        cmap='Purples'
        color='Purple'
        vmin=0
        vmax=6
        title='Floor Area Ratio (FSI)'
        ylim=[0,6]
        xlim=[0,6]
    
    # show the city_blocks coloured by metric chosen
    city_blocks.plot(ax=ax1, column=column, cmap=cmap,
                     vmin=vmin, vmax=vmax, alpha=0.6, legend=True)
    
    # label with metric chosen
    label_geom(ax1, city_blocks, column)

    ax1.set_title(title, fontsize=14)


    # scatterplot - 
    ax2.scatter(x=city_blocks.area_net_ha, y=city_blocks[column], color=color)
    add_titlebox(ax2, title + ' by area (ha)')
    ax2.set_xlabel("Area, net (ha)")
    ax2.set_ylabel(title)
    ax2.set_ylim(ylim)
    ax2.set_xlim([0, 4])
    
    # histogram - chosen metric, range=(0,1)
    ax3.hist(city_blocks[column], bins=10, color=color, alpha=0.5)
    ax3.set_xlabel(title)
    # add_titlebox(ax2, 'Histogram: form factor (φ)')
    ax3.set_xlim(xlim)
      
    return fig, ax