################################################################################
# Module: core.py
# Description: urban form analysis from OpenStreetMap
# License: MIT, see full license in LICENSE.txt
# Web: https://github.com/atelierlibre/osmuf
################################################################################

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox

from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import polygonize
from shapely import wkt

import matplotlib.pyplot as plt

from .utils import graph_to_polygons, extract_poly_coords, circlizer
from .utils import dict_to_gdf, gdf_convex_hull, footprints_from_gdf

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
    study_area['area_m2'] = study_area.area
    # set the crs of the gdf
    study_area.crs = crs_code
    # set the name of the gdf
    study_area.gdf_name = 'study_area'

    return study_area

def places_from_point(point, distance, place_type='city_block'):
    """
    Download and create GeoDataFrame of 'place' polygons of the type 'place_type'.

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
    places = ox.footprints_from_point(point, distance, footprint_type="place")
    # filter place polygons to retain only city blocks
    places = places.loc[places['place'] == place_type]
    # keep only two columns 'place' and 'geometry'
    places = places[['place','geometry']].copy()
    # project and measure places
    # places = project_and_measure_gdf(places)
    # name the index
    places.index.name='place_id'

    return places

def street_graph_from_point(point, distance, network_type='all'):
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
    network_type : string
        network_type as defined in osmnx

    Returns
    -------
    networkx multidigraph
    """

    # download the highway network within this boundary
    street_graph = ox.graph_from_point(point, distance, network_type,
                                       simplify=True, retain_all=True,
                                       truncate_by_edge=True, clean_periphery=True)
    # project the network to UTM & convert to undirected graph to
    street_graph = ox.project_graph(street_graph)
    # remove duplicates which make polygonization fail
    street_graph = ox.get_undirected(street_graph)

    return street_graph

def street_graph_from_gdf(gdf, network_type='all'):
    """
    Download streets within a convex hull around a GeoDataFrame.

    Used to ensure that all streets around city blocks are downloaded, not just
    those inside an arbitrary bounding box.

    Parameters
    ----------
    gdf : geodataframe
        currently accepts a projected gdf

    network_type : string
        network_type as defined in osmnx

    Returns
    -------
    networkx multidigraph
    """
    # generate convex hull around the gdf
    boundary = gdf_convex_hull(gdf)

    # download the highway network within this boundary
    street_graph = ox.graph_from_polygon(boundary, network_type,
                                       simplify=True, retain_all=True,
                                       truncate_by_edge=True, clean_periphery=False)
    # remove duplicates which make polygonization fail
    street_graph = ox.get_undirected(street_graph)

    return street_graph

def net_city_blocks_from_places(place_gdf):
    """
    Filter place gdf to retain net city_blocks and minimal columns.

    Parameters
    ----------
    gdf: GeoDataFrame

    Returns
    -------
    GeoDataFrame
    """

    # filter place polygons to retain only city blocks
    net_city_blocks = place_gdf.loc[place_gdf['place'] == 'city_block']
    # keep only two columns 'place' and 'geometry'
    net_city_blocks = net_city_blocks[['place','geometry']].copy()

    # change text city_block to net_city_block
    net_city_blocks["place"] = net_city_blocks['place'].str.replace('city_block', 'net_city_block')

    # name the dataframe - osmnx uses 'gdf_name' not sure if this is standard
    net_city_blocks.gdf_name = 'net_city_blocks'
    # name the index
    net_city_blocks.index.name='net_city_blocks_id'

    return net_city_blocks

def gen_gross_city_blocks(street_graph, net_city_blocks):
    """
    Generate approximate gross urban blocks by polygonizing the highway network.

    Use a geodataframe of net urban blocks to unify fragments to give better results.

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

    # polygonize the highway network & return it as a GeoDataFrame
    # this will include edge polygons that are removed in the next steps
    gross_city_blocks = graph_to_polygons(street_graph, node_geometry=False)

    # transfer attributes from city_blocks_net to city_blocks_gross where they intersect
    gross_city_blocks = gpd.sjoin(gross_city_blocks, net_city_blocks, how="left", op='intersects')
    # dissolve together gross_city_blocks polygons that intersect with the same net_city_blocks polygon
    gross_city_blocks.rename(columns={'index_right' : 'net_city_block_id'}, inplace=True)
    gross_city_blocks = gross_city_blocks.dissolve(by='net_city_block_id')
    # convert the index to int
    gross_city_blocks.index = gross_city_blocks.index.astype(int)
    # remove unecessary columns
    gross_city_blocks = gross_city_blocks[['place', 'geometry']]
    # change text city_block to city_block_gross
    gross_city_blocks["place"] = gross_city_blocks['place'].str.replace('net_city_block', 'gross_city_block')
 
    # name the dataframe
    gross_city_blocks.gdf_name = 'gross_city_blocks'
    # name the index
    gross_city_blocks.index.name='gross_city_blocks_id'


    return gross_city_blocks

def buildings_from_gdf(gdf):
    """
    Download buildings within convex hull around a GeoDataFrame.

    Parameters
    ----------
    gdf : geodataframe

    Returns
    -------
    GeoDataFrame
    """
    # download buildings within boundary
    buildings = footprints_from_gdf(gdf)
    # name the dataframe
    buildings.gdf_name = 'buildings'

    return buildings

def streets_proj_from_street_graph(street_graph):
    """
    Project and convert networkx multidigraph to a GeoDataFrame.

    Primarily here to allow future filtering of streets data for osmuf purposes

    Parameters
    ----------
    street_graph : networkx multidigraph

    Returns
    -------
    GeoDataFrame
    """
    # project the network to UTM
    street_graph = ox.project_graph(street_graph)
    # convert to gdf
    streets_proj = ox.graph_to_gdfs(street_graph, nodes=False)

    # insert filtering/processing here for OSMuf purposes

    return streets_proj

def merge_gdfs(gdf_list):
    """
    Merge GeoDataFrames.

    Parameters
    ----------
    gdf_list : list of geodataframes

    Returns
    -------
    GeoDataFrame
    """

    # dataframesList = [landuse, leisure]
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True, sort=True), crs=gdf_list[0].crs)

    # name the dataframe
    merged_gdf.gdf_name = gdf_list[0].gdf_name

    return merged_gdf

def project_measure_gdf(gdf):
    """
    Project GeoDataFrame to UTM and measure perimeter and area.

    Use osmnx to project a GeoDataFrame to UTM. Create new columns
    recording perimeter in metres, area in hectares, and the perimeter per unit
    area.

    Parameters
    ----------
    gdf : GeoDataFrame

    Returns
    -------
    GeoDataFrame
    """
    # use osmnx to project gdf to UTM
    gdf_proj = ox.project_gdf(gdf)
    # write perimeter length into column
    gdf_proj['perimeter_m'] = gdf_proj.length
    # write area into column
    gdf_proj['area_m2'] = gdf_proj.area

    gdf_proj[['perimeter_m', 'area_m2']] = gdf_proj[['perimeter_m', 'area_m2']].round(2)

    return gdf_proj

def project_measure_buildings(buildings):
    """
    Project, filter and measure GeoDataFrame of buildings.

    Keep only building height and area information. Generate measures of footprint size and total
    Gross External Area (footprint x number of storeys).

    Parameters
    ----------
    gdf : geodataframe
        gdf of buildings

    Returns
    -------
    GeoDataFrame
    """
    # project to UTM
    buildings_proj = ox.project_gdf(buildings)
    # create filtered copy with only key columns, creating 'building:levels' if not present
    buildings_proj = buildings_proj.reindex(columns=['building','building:levels','geometry'])

    ### THERE ACTUALLY IS AN INTEGER TYPE WITH NAN, MAY BE BETTER

    # convert 'building:levels' to float from object (int doesn't support NaN)
    buildings_proj['building:levels']=buildings_proj['building:levels'].astype(float)
    # convert fill NaN with zeroes to allow conversion to int
    buildings_proj = buildings_proj.fillna({'building:levels': 0})
    # convert to int
    buildings_proj["building:levels"] = pd.to_numeric(buildings_proj['building:levels'], downcast='integer')

    # generate footprint areas
    buildings_proj['footprint_m2']=buildings_proj.area
    # generate total_GEA
    buildings_proj['total_GEA_m2']=buildings_proj.area*buildings_proj['building:levels']

    return buildings_proj

def join_buildings_place_id(buildings, places):
    """
    Add index number of enclosing place polygon to buildings geodataframe.

    Where a building is not inside a place polygon give it a place_id of zero.

    Convert all place_id numbers to integers.

    Parameters
    ----------
    buildings : geodataframe
        polygons (geometry?) receiving id of enclosing polygons.

    places : geodataframe
        enclosing polygons to transfering id numbers onto buildings.

    Returns
    -------
    GeoDataFrame
    """
    # where they intersect, add the city_block number onto each building
    # how = 'left', 'right', 'inner' sets how the index of the new gdf is determined, left retains buildings index
    # was 'intersects', 'contains', 'within'
    buildings = gpd.sjoin(buildings, places[['geometry']], how="left", op='intersects')

    place_column_name = places.gdf_name + '_id'
    buildings.rename(columns={'index_right' : place_column_name}, inplace=True)

    # convert any NaN values in block_id to zero, then convert to int
    buildings = buildings.fillna({place_column_name: 0})
    buildings[place_column_name] = pd.to_numeric(buildings[place_column_name], downcast='integer')

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

def dist_buildings_highways(buildings, highways):
    """
    Return distance to highway that the building is closest to.

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
    return highways.distance(buildings).min()

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

def join_buildings_street_dist(buildings_proj, streets_proj):
    """
    Add distance to nearest street centreline to buildings geodataframe.

    Parameters
    ----------
    buildings_proj : geodataframe
        projected polygons (geometry?) receiving id of enclosing polygons.

    streets_proj : geodataframe
        projected streets to transfer id numbers onto buildings.

    Returns
    -------
    GeoDataFrame
    """

    buildings_proj['street_dist'] = buildings_proj.geometry.apply(dist_buildings_highways, args=(streets_proj,))
    buildings_proj['street_dist'] = buildings_proj['street_dist'].round(2)

    return buildings_proj

def gen_net_to_gross(net_gdf, gross_gdf):
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
    net_gdf['net_to_gross'] = round(net_gdf.area/gross_gdf.area, 2)
    
    return net_gdf

def gen_regularity(gdf):
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
    gdf_regularity = gdf['geometry'].copy()
    gdf_regularity['original_area'] = gdf_regularity.area

    # replace the polygon geometry with the smallest enclosing circle
    gdf_regularity['geometry'] = gdf_regularity['geometry'].apply(circlizer)

    # calculate the area of the smallest enclosing circles
    gdf_regularity['circle_area'] = gdf_regularity.area

    # calculate 'regularity' as "the ratio between the area of the block and
    # the area of the circumscribed circle C" Barthelemy M. and Louf R., (2014)
    gdf_regularity['regularity'] = gdf_regularity['original_area']/gdf_regularity['circle_area']

    return gdf_regularity

def join_places_building_data(places_proj, buildings_proj):
    """
    Add summary building data onto city blocks.

    Requires columns to be present in the gdfs generated by other functions in
    osmuf.

    Parameters
    ----------
    places_proj : geodataframe

    buildings_proj : geodataframe

    Returns
    -------
    GeoDataFrame
    """
    column_id = places_proj.gdf_name + '_id'
    building_areas_by_place=buildings_proj[['footprint_m2','total_GEA_m2']].groupby([buildings_proj[column_id]]).sum()
    # next line may not be necessary, don't want to create entry '0' in gdf of city_blocks
    building_areas_by_place = building_areas_by_place.drop([0])

    places_proj = places_proj.merge(building_areas_by_place, on = column_id)

    places_proj['GSI'] = places_proj['footprint_m2']/(places_proj['area_m2'])
    places_proj['FSI'] = places_proj['total_GEA_m2']/(places_proj['area_m2'])

    return places_proj
