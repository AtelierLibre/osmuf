################################################################################
# Module: core.py
# Description: urban form analysis from OpenStreetMap
# License: MIT, see full license in LICENSE.txt
# Web: https://github.com/atelierlibre/osmuf
################################################################################

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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.collections import PatchCollection

from .plot import *
from .utils import *

def study_area_from_point(point, distance):
    """
    Define study area from center point and distance to edge of bounding box.

    Return this as a projected GeoDataFrame for plotting
    and for summary data.

    Parameters
    ----------
    point : tuple
        the (lat, lon) point to create the bounding box around
    distance : int
        how many meters the north, south, east, and west sides of the box should
        each be from the point

    Returns
    -------
    GeoDataFrame
    """
    # use osmnx to define the bounding box, always return the crs
    bbox = ox.bbox_from_point(point, distance, project_utm=True, return_crs=True)

    # split the tuple
    n, s, e, w, crs_code = bbox

    # Create GeoDataFrame
    study_area = gpd.GeoDataFrame()
    # create geometry column with bounding box polygon
    study_area.loc[0, 'geometry'] = Polygon([(w, s), (w, n), (e, n), (e, s)])
    # create column with area in hectares
    study_area['area_ha'] = study_area.area/10000
    # set the crs of the gdf
    study_area.crs = crs_code

    return study_area

def city_blocks_from_point(point, distance):
    """
    Download and create GeoDataFrame of 'place=city_block' polygons.

    use osmnx to download "place" polygons from a center point and distance to
    edge of bounding box. Filter to retain minimal columns. Create new columns
    recording perimeter in metres, area in hectares, and the perimeter per unit
    area.

    Parameters
    ----------
    point : tuple
        the (lat, lon) point to create the bounding box around
    distance : int
        how many meters the north, south, east, and west sides of the box should
        each be from the point

    Returns
    -------
    GeoDataFrame
    """

    # use osmnx to download 'place' polygons
    city_blocks = ox.footprints_from_point(point, distance, footprint_type="place")
    # filter place polygons to retain only city blocks
    city_blocks = city_blocks.loc[city_blocks['place'] == 'city_block']
    # keep only two columns 'place' and 'geometry'
    city_blocks = city_blocks[['place','geometry']].copy()
    # use osmnx to project city_blocks to UTM
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
    """
    Use osmnx to retrieve a networkx graph from OpenStreetMap.

    This function uses osmnx to obtain a graph with some set parameters. It
    projects the graph and converts it to undirected. This is important - a
    directed graph creates overlapping edges which shapely fails to
    polygonize.

    Parameters
    ----------
    point : tuple
        the (lat, lon) point to create the bounding box around
    distance : int
        how many meters the north, south, east, and west sides of the box should
        each be from the point

    Returns
    -------
    networkx multidigraph
    """

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
    """
    Creates a convex hull around the total extent of a GeoDataFrame.

    Used to define a polygon for retrieving geometries within. When calculating
    densities for urban blocks we need to retrieve the full extent of e.g.
    buildings within the blocks, not crop them to an arbitrary bounding box.

    Parameters
    ----------
    gdf : geodataframe
        currently accepts a projected gdf

    Returns
    -------
    shapely polygon
    """
    ### INSERT CHECK FOR CRS HERE?

    # project gdf back to geographic coordinates as footprints_from_polygon
    # requires it
    gdf_temp = ox.project_gdf(gdf, to_latlong=True)
    # determine the boundary polygon to fetch buildings within
    # buffer originally 0.000225, buffer actually needs to go whole block away
    # to get complete highways therefor trying 0.001
    boundary=gdf_temp.cascaded_union.convex_hull.buffer(0.001)
    # NOTE - maybe more efficient to generate boundary first then reproject second?

    return boundary

def street_graph_from_gdf(gdf):
    """
    Download streets within a convex hull around a GeoDataFrame.

    Used to ensure that all streets around city blocks are downloaded, not just
    those inside an arbitrary bounding box.

    Parameters
    ----------
    gdf : geodataframe
        currently accepts a projected gdf

    Returns
    -------
    networkx multidigraph
    """
    # generate convex hull around the gdf
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
    """
    Use osmnx to convert networkx multidigraph to a GeoDataFrame.

    Primarily here to allow future filtering of streets data for osmuf purposes

    Parameters
    ----------
    street_graph : networkx multidigraph

    Returns
    -------
    GeoDataFrame
    """

    streets = ox.graph_to_gdfs(street_graph, nodes=False)

    # insert filtering/processing here for OSMuf purposes

    return streets

def footprints_from_gdf(gdf):
    """
    Download footprints within a convex hull around a GeoDataFrame.

    Used to ensure that all footprints within city blocks are downloaded, not
    just those inside an arbitrary bounding box. Currently defaults to
    buildings. Amend in future to work with other footprint types.

    Parameters
    ----------
    gdf : geodataframe
        currently accepts a projected gdf

    Returns
    -------
    GeoDataFrame
    """
    # generate boundary
    boundary = gdf_convex_hull(gdf)

    # download buildings within boundary
    footprints = ox.footprints_from_polygon(boundary)

    return footprints

def buildings_from_gdf(gdf):
    """
    Download buildings within convex hull around a GeoDataFrame.

    Download buildings within the convex hull of a gdf. Keep only building
    height and area information. Generate measures of footprint size and total
    Gross External Area (footprint x number of storeys).

    Parameters
    ----------
    gdf : geodataframe
        currently accepts a projected gdf

    Returns
    -------
    GeoDataFrame
    """
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
    """
    Add index number of enclosing city_block to buildings geodataframe.

    Where a building is not inside a city_block give it a block_id of zero.

    Convert all block_id numbers to integers.

    Parameters
    ----------
    buildings : geodataframe
        polygons (geometry?) receiving id of enclosing polygons.

    city_blocks : geodataframe
        enclosing polygons to transfering id numbers onto buildings.

    Returns
    -------
    GeoDataFrame
    """
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
    """
    Return id of highway that the building is closest to.

    Parameters
    ----------
    buildings : geodataframe
        polygons (geometry?) receiving id of nearest highway.

    highways : geodataframe
        streets to transfering id numbers onto buildings.

    Check this does actually return an integer.

    Returns
    -------
    integer
    """
    return highways.distance(buildings).idxmin()

def join_buildings_street_id(buildings, streets):
    """
    Add index number of nearest street to buildings geodataframe.

    Parameters
    ----------
    buildings : geodataframe
        polygons (geometry?) receiving id of enclosing polygons.

    streets : geodataframe
        streets to transfer id numbers onto buildings.

    Returns
    -------
    GeoDataFrame
    """

    buildings['street_id'] = buildings.geometry.apply(link_buildings_highways, args=(streets,))

    return buildings

def form_factor(poly_gdf):
    """
    Returns a geodataframe of  smallest enclosing 'circles' (shapely polygons)
    generated from the input geodataframe of polygons.

    It includes columns that contain the area of the original polygon and the
    circles and the 'form factor' ratio of the area of the polygon to the area
    of the enclosing circle.

    Parameters
    ----------
    poly_gdf : geodataframe
        a geodataframe containing polygons.

    Returns
    -------
    GeoDataFrame
    """
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
    Convert the edges of a graph into a GeoDataFrame of polygons.

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
    gdf_polygons.geometry = polygons

    # Set the crs
    gdf_polygons.crs = G.graph['crs']
    gdf_polygons.gdf_name = '{}_polygons'.format(G.graph['name'])

    return gdf_polygons

def gen_city_blocks_gross(street_graph, city_blocks):
    """
    Generate approximate gross urban blocks by polygonizing the highway network.

    It uses a geodataframe of net urban blocks (downloaded as
    'place = city_block') to unify fragments to give better results. It also
    transfers measures of the gross urban blocks onto the net urban blocks for
    calculations.

    Parameters
    ----------
    street_graph : networkx multidigraph
        street centrelines

    city_blocks : GeoDataFrame
        net urban blocks used to unify fragments

    Returns
    -------
    GeoDataFrame
    """
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
    """
    Add summary building data onto city blocks.

    Requires columns to be present in the gdfs generated by other functions in
    osmuf.

    Parameters
    ----------
    city_blocks : geodataframe

    buildings : geodataframe

    Returns
    -------
    GeoDataFrame
    """
    building_areas_by_block=buildings[['footprint_m2','total_GEA_m2']].groupby([buildings['block_id']]).sum()
    # next line may not be necessary, don't want to create entry '0' in gdf of city_blocks
    building_areas_by_block = building_areas_by_block.drop([0])

    city_blocks = city_blocks.merge(building_areas_by_block, on = 'block_id')

    city_blocks['GSI_net'] = city_blocks['footprint_m2']/(city_blocks['area_net_ha']*10000)
    city_blocks['GSI_gross'] = city_blocks['footprint_m2']/(city_blocks['area_gross_ha']*10000)
    city_blocks['FSI_net'] = city_blocks['total_GEA_m2']/(city_blocks['area_net_ha']*10000)
    city_blocks['FSI_gross'] = city_blocks['total_GEA_m2']/(city_blocks['area_gross_ha']*10000)

    return city_blocks
