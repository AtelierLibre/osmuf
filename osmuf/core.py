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
from osmnx.utils_geo import bbox_from_point

from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import polygonize
from shapely import wkt

import matplotlib.pyplot as plt

from .utils import graph_to_polygons, extract_poly_coords, circlizer
from .utils import dict_to_gdf, gdf_convex_hull, footprints_from_gdf
from .utils import extend_line_by_factor

def study_area_from_point(point, distance):
    """
    Define a study area from a center point and distance to edge of bounding box.

    Returns a GeoDataFrame for plotting and for summary data.

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
    # use osmnx to define the bounding box
    bbox = bbox_from_point(point, distance)

    # split the tuple
    n, s, e, w = bbox

    # Create GeoDataFrame
    study_area = gpd.GeoDataFrame()
    # create geometry column with bounding box polygon
    study_area.loc[0, 'geometry'] = Polygon([(w, s), (w, n), (e, n), (e, s)])
    # set the name of the gdf
    study_area.gdf_name = 'study_area'

    study_area.crs = {'init' :'epsg:4326'}

    return study_area

def projected_study_area_from_point(point, distance):
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
    bbox = bbox_from_point(point, distance, project_utm=True, return_crs=True)

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
    # filter place polygons to retain only those of 'place_type'
    places = places.loc[places['place'] == place_type]
    # write the index into a column
    places[place_type + '_id'] = places.index
    # reindex the columns
    places = places.reindex(columns=[place_type + '_id', 'place', 'geometry'])
    # name the dataframe
    places.gdf_name=place_type

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

def streets_from_street_graph(street_graph):
    """
    Convert a networkx multidigraph to a GeoDataFrame.

    Primarily here to allow future filtering of streets data for osmuf purposes

    Parameters
    ----------
    street_graph : networkx multidigraph

    Returns
    -------
    GeoDataFrame
    """

    # convert to gdf
    streets = ox.graph_to_gdfs(street_graph, nodes=False)
    # write index into a column
    streets['street_id'] = streets.index

    # insert filtering/processing here for OSMuf purposes

    return streets

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
    # write the index into a column
    net_city_blocks['city_block_id'] = net_city_blocks.index
    # name the index
    net_city_blocks.index.name='net_city_blocks_id'

    return net_city_blocks

def gen_gross_city_blocks(street_graph, net_city_blocks=None):
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

    if net_city_blocks is not None:
        # transfer attributes from city_blocks_net to city_blocks_gross where they intersect
        gross_city_blocks = gpd.sjoin(gross_city_blocks, net_city_blocks, how="left", op='intersects')
        # dissolve together gross_city_blocks polygons that intersect with the same net_city_blocks polygon
        gross_city_blocks.rename(columns={'index_right' : 'net_city_block_id'}, inplace=True)
        gross_city_blocks = gross_city_blocks.dissolve(by='net_city_block_id')
        # convert the index to int
        gross_city_blocks.index = gross_city_blocks.index.astype(int)
        # change text city_block to city_block_gross
        gross_city_blocks["place"] = gross_city_blocks['place'].str.replace('net_city_block', 'gross_city_block')
 
    # name the dataframe
    gross_city_blocks.gdf_name = 'gross_city_blocks'
    # write the index into a column
    gross_city_blocks['city_block_id'] = gross_city_blocks.index
    # reindex the columns
    gross_city_blocks = gross_city_blocks.reindex(columns=['city_block_id', 'place', 'geometry'])
    # delete the index name
    # del gross_city_blocks.index.name

    return gross_city_blocks

def buildings_from_gdf(gdf):
    """
    Download buildings within a convex hull around a GeoDataFrame.

    Parameters
    ----------
    gdf : geodataframe

    Returns
    -------
    GeoDataFrame
    """
    # download buildings within boundary
    buildings = footprints_from_gdf(gdf)
    # write the index into a column
    buildings['building_id'] = buildings.index
    # reindex the columns
    buildings = buildings.reindex(columns=['building_id', 'building', 'building:levels', 'geometry'])
    # name the dataframe
    buildings.gdf_name = 'buildings'

    return buildings

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

def measure_buildings(buildings_proj):
    """
    Measure GeoDataFrame of buildings.

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

    # fill NaN with zeroes
    buildings_proj = buildings_proj.fillna({'building:levels': 0})

    # convert 'building:levels' to float then Int64 (with Nan) one step is not possible
    buildings_proj['building:levels']=buildings_proj['building:levels'].astype('float').astype('int')
 
    # generate footprint areas
    buildings_proj['footprint_m2']=buildings_proj.area.round(decimals=1)
    # generate total_GEA
    buildings_proj['total_GEA_m2']=(buildings_proj.area*buildings_proj['building:levels']).round(decimals=1)

    return buildings_proj

def measure_city_blocks(net_city_blocks_gdf, gross_city_blocks_gdf):
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
    # net urban block measures
    net_city_blocks_gdf['net_area_m2'] = net_city_blocks_gdf.area.round(decimals=2)
    net_city_blocks_gdf['frontage_m'] = net_city_blocks_gdf.length.round(decimals=2)
    net_city_blocks_gdf['PAR'] = (net_city_blocks_gdf.length/net_city_blocks_gdf.area).round(decimals=4)
    net_city_blocks_gdf['net_frontage_density_m_m2'] = (net_city_blocks_gdf['frontage_m']/net_city_blocks_gdf['net_area_m2']).round(decimals=4)

    # gross urban block measures
    net_city_blocks_gdf['gross_area_m2'] = gross_city_blocks_gdf.area.round(decimals=2)
    net_city_blocks_gdf['inner_streets_m'] = gross_city_blocks_gdf['inner_streets_m'].round(decimals=2)
    net_city_blocks_gdf['outer_streets_m'] = gross_city_blocks_gdf['outer_streets_m'].round(decimals=2)
    net_city_blocks_gdf['network_length_m'] = ((net_city_blocks_gdf['outer_streets_m']/2) + net_city_blocks_gdf['inner_streets_m']).round(decimals=2)
    net_city_blocks_gdf['network_density_m_ha'] = gross_city_blocks_gdf['network_density_m_ha'].round(decimals=4)
    net_city_blocks_gdf['gross_frontage_density_m_m2'] = (net_city_blocks_gdf['frontage_m']/net_city_blocks_gdf['gross_area_m2']).round(decimals=4)

    # net to gross area ratio
    net_city_blocks_gdf['net:gross'] = (net_city_blocks_gdf.area/gross_city_blocks_gdf.area).round(decimals=2)

    # Change the order (the index) of the columns
    columnsTitles = ['city_block_id', 'place', 'net_area_m2', 'frontage_m', 'PAR', 'net_frontage_density_m_m2', 'gross_frontage_density_m_m2',
                     'gross_area_m2', 'inner_streets_m', 'outer_streets_m', 'network_length_m', 'network_density_m_ha',
                     'net:gross', 'geometry']
    net_city_blocks_gdf = net_city_blocks_gdf.reindex(columns=columnsTitles)

    return net_city_blocks_gdf

def measure_network_density(streets_for_networkd_prj, gross_city_blocks_prj):
    """
    Adds network density (m/ha.) onto a gdf of gross urban blocks

    Requires a gdf of streets to overlay with the gross city blocks. Streets
    that are within a gross urban blocks (i.e. do not coincide with its perimeter)
    have the block id added to them. The length of these streets are then aggregated
    by block id and their complete length added to the gross city blocks gdf. Half
    the lenght of the perimeter (i.e. the bounding roads) are then added to the gdf
    as well and the network density calculated as the sum of these two numbers divided
    by the gross area of the block.

    Parameters
    ----------
    streets_for_networ_prj : geodataframe
        a projected gdf of streets
    gross_city_blocks_prj: geodataframe
        a projected gdf of gross city blocks

    Returns
    -------
    gross_city_blocks_prj
        GeoDataFrame
    """
    # OSMnx returns some highway values as lists, this converts them to strings
    streets_for_networkd_prj['highway'] = streets_for_networkd_prj['highway'].apply(lambda x: ', '.join(x) if type(x) is list else x)

    # make a new gdf which only contains street fragments completely within a gross city block
    streets_in_gross_blocks = gpd.sjoin(streets_for_networkd_prj, gross_city_blocks_prj, how="inner", op="within")

    # Write the length of these inner streets into a new column 'inner_streets_m'
    streets_in_gross_blocks['inner_streets_m'] = streets_in_gross_blocks.length.round(decimals=1)

    # aggregate the total length of inner streets for each block
    inner_streets_agg_by_block = streets_in_gross_blocks.groupby(['city_block_id']).sum().round(decimals=2)

    # reindex to keep onlt the columns necessary
    keep_columns = ['inner_streets_m']
    inner_streets_agg_by_block = inner_streets_agg_by_block.reindex(columns=keep_columns)

    # merge the total inner street length onto the gross blocks
    gross_city_blocks_prj = gross_city_blocks_prj.merge(inner_streets_agg_by_block, how='outer', left_index=True, right_index=True)

    # Fill NaN with zeroes - Why is this? For those without inner streets?
    print(gross_city_blocks_prj.columns)
    gross_city_blocks_prj.fillna({'inner_streets_m':0, 'outer_streets_m':0, 'gross_area_ha':0}, inplace=True)

    gross_city_blocks_prj['outer_streets_m'] = gross_city_blocks_prj.length.round(decimals=2)
    gross_city_blocks_prj['gross_area_ha'] = (gross_city_blocks_prj.area/10000).round(decimals=4)
    gross_city_blocks_prj['network_density_m_ha'] = (((gross_city_blocks_prj['outer_streets_m']/2)
                                                      +(gross_city_blocks_prj['inner_streets_m']))
                                                     /((gross_city_blocks_prj.area/10000))).round(decimals=2)

    return gross_city_blocks_prj

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

    place_column_name = 'city_block_id'
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

    Returns
    -------
    float
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
    gdf_regularity = gdf[['city_block_id', 'geometry']].copy()

    # write the area of each polygon into a column
    gdf_regularity['poly_area_m2'] = gdf.area.round(decimals=1)

    # replace the polygon geometry with the smallest enclosing circle
    gdf_regularity['geometry'] = gdf_regularity['geometry'].apply(circlizer)

    # calculate the area of the smallest enclosing circles
    gdf_regularity['SEC_area_m2'] = gdf_regularity.area.round(decimals=1)

    # calculate 'regularity' as "the ratio between the area of the polygon and
    # the area of the circumscribed circle C" Barthelemy M. and Louf R., (2014)
    gdf_regularity['regularity'] = gdf_regularity['poly_area_m2']/gdf_regularity['SEC_area_m2']

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
    building_areas_by_place=buildings_proj[['footprint_m2','total_GEA_m2']].groupby(buildings_proj['city_block_id']).sum()
    # if there are buildings not associated with a city_block they aggregate under 0
    # if this happens remove them from the dataframe
    if 0 in building_areas_by_place.index:#building_areas_by_place.index.contains(0):
        building_areas_by_place = building_areas_by_place.drop([0])

    places_proj = places_proj.merge(building_areas_by_place, on = 'city_block_id')

    places_proj['net_GSI'] = (places_proj['footprint_m2']/places_proj.area).round(decimals=3)
    places_proj['net_FSI'] = (places_proj['total_GEA_m2']/places_proj.area).round(decimals=3)
    places_proj['gross_GSI'] = (places_proj['footprint_m2']/places_proj['gross_area_m2']).round(decimals=3)
    places_proj['gross_FSI'] = (places_proj['total_GEA_m2']/places_proj['gross_area_m2']).round(decimals=3)
    places_proj['avg_building:levels'] = (places_proj['total_GEA_m2']/places_proj['footprint_m2']).round(decimals=1)

    return places_proj

def gen_building_depth(row):
    # get the id of the street nearest the building
    street_id = row.street_id
    # extract street geometry as shapely linestring
    street_linestring = streets_proj.loc[street_id, 'geometry']
    # extract building centroid as shapely point
    building_centroid = row.geometry.centroid
    # distance along line to nearest point
    dist_along_line = street_linestring.project(building_centroid)
    # coordinates of the nearest point as shapely point
    point_on_line = street_linestring.interpolate(dist_along_line)
    # line projected back through centroid extended by factor
    projected_line = extend_line_by_factor(point_on_line, building_centroid, 10)
    # extract buildings outline as shapely linestring
    building_outline = row.geometry.boundary
    # calculate all points of intersection between projected line and building outline
    intersection_points = projected_line.intersection(building_outline)
    # calculate the building depth as shapely linestring from first intersection point to last
    if intersection_points.geom_type == 'MultiPoint':
        building_depth=LineString([intersection_points[0], intersection_points[-1]])
        return building_depth
    elif intersection_points.geom_type == 'Point':
        pass
    else:
        pass