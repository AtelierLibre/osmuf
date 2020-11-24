################################################################################
# Module: core.py
# Description: urban form analysis from OpenStreetMap
# License: MIT, see full license in LICENSE.txt
# Web: https://github.com/atelierlibre/osmuf
################################################################################

import math
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from osmnx.utils_geo import bbox_from_point

from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import polygonize
from shapely import wkt

import matplotlib.pyplot as plt

from .utils import graph_to_polygons, extract_poly_coords, circlizer
from .utils import dict_to_gdf, gdf_convex_hull, footprints_from_gdf
from .utils import extend_line_by_factor


def streets_blocks_buildings_from_graph(G, landuse_tags=None):
    '''

    '''
    if landuse_tags is None:
        landuse_tags = {'landuse':True,
                        'amenity':True,
                        'leisure':True,
                        }

    streets = _streets_from_graph(G)

    street_polygons = _street_polygons_from_streets(streets)

    street_polygons_union = street_polygons.unary_union

    landuse = _landuse_from_street_polygons(street_polygons_union, landuse_tags=landuse_tags)

    buildings = _buildings_from_street_polygons(street_polygons_union)

    gross_blocks, net_blocks = _blocks_from_streets_polygons_landuse(streets, street_polygons, landuse)

    # join the block_id of the gross_block to the net_block, landuse and buildings
    net_blocks = gpd.sjoin(net_blocks, gross_blocks, how="left", op='intersects').drop(columns=['index_right'])
    landuse = gpd.sjoin(landuse, gross_blocks, how="left", op='intersects').drop(columns=['index_right'])
    buildings = gpd.sjoin(buildings, gross_blocks, how="left", op='intersects').drop(columns=['index_right'])

    return streets, gross_blocks, net_blocks, landuse, buildings


def _streets_from_graph(G):
    # Make Graph undirected to remove duplicate edges
    G_undirected = G.to_undirected()
    # extract the edges as streets
    streets = ox.graph_to_gdfs(G_undirected, nodes=False)

    return streets


def _street_polygons_from_streets(streets):
    # polygonize the edges & create a new geodataframe
    polygons = list(polygonize(streets['geometry']))
    street_polygons = gpd.GeoDataFrame(geometry=polygons, crs=streets.crs)

    return street_polygons


def _landuse_from_street_polygons(street_polygons_union, landuse_tags):

    landuse = ox.geometries_from_polygon(street_polygons_union, tags=landuse_tags)
    # filter to only retain polygonal geometry
    landuse = landuse[landuse.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()

    return landuse


def _buildings_from_street_polygons(street_polygons_union):
    buildings = ox.geometries_from_polygon(street_polygons_union, tags={'building':True})
    # filter to retain only polygonal geometry
    buildings = buildings[buildings.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()

    return buildings


def _blocks_from_streets_polygons_landuse(streets, street_polygons, landuse):
    
    landuse_polygons = gpd.GeoDataFrame(geometry=list(landuse.unary_union), crs=streets.crs)

    # increase the landuse_polygons index numbers
    landuse_polygons.index = landuse_polygons.index + street_polygons.index.max()

    gdf = gpd.sjoin(street_polygons,
                    landuse_polygons,
                    how="inner",
                    op='intersects')

    tuple_list = list(zip(gdf.index.to_list(), gdf['index_right'].to_list()))

    graph = nx.Graph(tuple_list)
    result = list(nx.connected_components(graph))

    # dictionary of index:group
    d_ = dict()

    for i, idxs in enumerate(result):
        for idx in idxs:
            d_.update({idx:i})
            
    street_polygons['group'] = street_polygons.index.map(d_)
    landuse_polygons['group'] = landuse_polygons.index.map(d_)

    gross_blocks = street_polygons.dissolve(by='group')
    net_blocks = landuse_polygons.dissolve(by='group')

    # write the index number of each gross block into a column 'block_id'
    gross_blocks['block_id'] = gross_blocks.index

    return gross_blocks, net_blocks


def dissolve_gdf1_by_gdf2(gdf1, gdf2):
    '''
    dissolve together geometries in gdf1 that intersect a common shape in gdf2
    '''
    # adds the index of intersecting geometries in gdf2 to gdf1
    gdf1 = gpd.sjoin(gdf1, gdf2, how="left", op='intersects')

    # extract rows that received an index number and dissolve them together if they share the index
    gdf1_dissolved = gdf1[gdf1['index_right'].notna()].dissolve(by='index_right')
    
    # drop any duplicates
    gdf1_dissolved = gdf1_dissolved[~gdf1_dissolved.index.duplicated(keep='first')].copy()

    # extract rows that didn't receive an index number & drop that column
    #gdf1_not_dissolved = gdf1[gdf1['index_right'].isna()].copy()
    #gdf1_not_dissolved.drop(columns='index_right', inplace=True)

    # append the dissolved and non-dissolved
    gdf = gdf1_dissolved#.append(gdf1_not_dissolved)

    gdf.reset_index(drop=True, inplace=True)

    return gdf


def project_and_measure_streets_blocks_buildings(streets, gross_blocks, net_blocks, landuse, buildings, crs=None):
    '''
    '''
    utm_crs = determine_utm_crs(streets)

    # project the GeoDataFrame to the UTM CRS
    streets_prj = streets.to_crs(utm_crs)
    gross_blocks_prj = gross_blocks.to_crs(utm_crs)
    net_blocks_prj = net_blocks.to_crs(utm_crs)
    landuse_prj = landuse.to_crs(utm_crs)
    buildings_prj = buildings.to_crs(utm_crs)

    # measure area, perimeter and PAR of gross and net blocks
    gross_blocks_prj = measure_blocks(gross_blocks_prj)
    net_blocks_prj = measure_blocks(net_blocks_prj)

    # measure streets per gross block
    gross_blocks_prj = measure_streets_per_gross_block(gross_blocks_prj, streets_prj)

    # measure land use
    landuse_prj = measure_landuse(landuse_prj)

    # measure buildings
    buildings_prj = measure_buildings(buildings_prj)

    # add building GSI and FSI to both net and gross blocks
    gross_blocks_prj = join_building_data_to_blocks(gross_blocks_prj, buildings_prj)
    net_blocks_prj = join_building_data_to_blocks(net_blocks_prj, buildings_prj)

    # set the index to the block_id column and remove the column
    gross_blocks_prj.set_index('block_id', drop=True, inplace=True, verify_integrity=True)
    net_blocks_prj.set_index('block_id', drop=True, inplace=True, verify_integrity=True)

    # calculate the block net:gross ratio
    gross_blocks_prj['block_net:gross'] = net_blocks_prj['area_m2']/gross_blocks_prj['area_m2']

    print(f"Projected GeoDataFrame to {utm_crs}") #utils.log

    return streets_prj, gross_blocks_prj, net_blocks_prj, landuse_prj, buildings_prj

def determine_utm_crs(gdf):
    # calculate longitude of centroid of union of all geometries in streets
    minx, miny, maxx, maxy = gdf.geometry.total_bounds
    avg_lng = minx+(maxx-minx/2)
    # calculate UTM zone from avg longitude to define CRS to project to
    utm_zone = int(math.floor((avg_lng + 180) / 6.0) + 1)
    utm_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    return utm_crs


def measure_blocks(gdf):
    """
    Write the area, perimeter and perimeter:area ratio into columns

    Parameters
    ----------
    gdf : geodataframe
        a geodataframe containing polygons.

    Returns
    -------
    GeoDataFrame
    """
    # check if projected

    # basic block measurements
    gdf['area_m2'] = gdf.area.round(decimals=2)
    gdf['perimeter_m'] = gdf.length.round(decimals=2)
    gdf['PAR'] = (gdf.length/gdf.area).round(decimals=4)

    return gdf


def measure_streets_per_gross_block(gross_blocks_prj, streets_prj):
    """
    Adds network density (m/ha.) onto a gdf of gross urban blocks

    Requires a gdf of streets to overlay with the gross city blocks. Streets
    that are within a gross urban blocks (i.e. do not coincide with its perimeter)
    have the block id added to them. The length of these streets are then aggregated
    by block id and their complete length added to the gross city blocks gdf. Half
    the length of the perimeter (i.e. the bounding roads) are then added to the gdf
    as well and the network density calculated as the sum of these two numbers divided
    by the gross area of the block.

    Parameters
    ----------
    streets_prj : geodataframe
        a projected gdf of streets
    gross_blocks_prj: geodataframe
        a projected gdf of gross blocks

    Returns
    -------
    gross_blocks_prj
        GeoDataFrame
    """
    # OSMnx returns some highway values as lists, this converts them to strings
    streets_prj['highway'] = streets_prj['highway'].apply(lambda x: ', '.join(x) if type(x) is list else x)

    # make a new gdf which only contains street fragments completely within a gross city block
    streets_in_gross_blocks = gpd.sjoin(streets_prj, gross_blocks_prj, how="inner", op="within")

    # Write the length of these inner streets into a new column 'inner_streets_m'
    streets_in_gross_blocks['inner_streets_m'] = streets_in_gross_blocks.length.round(decimals=1)

    # aggregate the total length of inner streets for each block
    inner_streets_agg_by_block = streets_in_gross_blocks.groupby(['block_id']).sum().round(decimals=2)

    # reindex to keep only the columns necessary
    keep_columns = ['inner_streets_m']
    inner_streets_agg_by_block = inner_streets_agg_by_block.reindex(columns=keep_columns)

    # merge the total inner street length onto the gross blocks
    gross_blocks_prj = gross_blocks_prj.merge(inner_streets_agg_by_block, how='outer', left_index=True, right_index=True)

    # Fill NaN with zeroes
    gross_blocks_prj.fillna({'inner_streets_m':0, 'outer_streets_m':0, 'gross_area_ha':0}, inplace=True)

    gross_blocks_prj['outer_streets_m'] = gross_blocks_prj.length.round(decimals=2)
    gross_blocks_prj['area_ha'] = (gross_blocks_prj.area/10000).round(decimals=4)
    gross_blocks_prj['network_density_m_ha'] = (((gross_blocks_prj['outer_streets_m']/2)
                                                      +(gross_blocks_prj['inner_streets_m']))
                                                     /((gross_blocks_prj.area/10000))).round(decimals=2)

    return gross_blocks_prj

def measure_landuse(landuse_prj):
    """
    Measure GeoDataFrame of landuse.

    Keep only building height and area information. Generate measures of footprint size and total
    Gross External Area (footprint x number of storeys).

    Parameters
    ----------
    buildings_prj : geodataframe
        gdf of buildings

    Returns
    -------
    GeoDataFrame
    """
    # reduce columns of data
    landuse_prj = landuse_prj[['geometry', 'unique_id', 'landuse', 'amenity', 'leisure', 'block_id']].copy()

    landuse_prj['Combined land use'] = ''

    for col in ['landuse', 'amenity', 'leisure']:
        landuse_prj['Combined land use'] += landuse_prj[col].fillna('')
 
    # generate footprint areas
    landuse_prj['area_m2'] = landuse_prj.area.round(decimals=1)

    return landuse_prj

def measure_buildings(buildings_prj):
    """
    Measure GeoDataFrame of buildings.

    Keep only building height and area information. Generate measures of footprint size and total
    Gross External Area (footprint x number of storeys).

    Parameters
    ----------
    buildings_prj : geodataframe
        gdf of buildings

    Returns
    -------
    GeoDataFrame
    """
    # reduce columns of data
    buildings_prj = buildings_prj[['geometry', 'unique_id', 'building', 'building:levels', 'block_id']]

    # fill NaN with zeroes
    buildings_prj = buildings_prj.fillna({'building:levels': 0})

    # convert 'building:levels' to float then Int64 (with Nan) one step is not possible
    buildings_prj['building:levels']=buildings_prj['building:levels'].astype('float').astype('int')
 
    # generate footprint areas
    buildings_prj['footprint_m2']=buildings_prj.area.round(decimals=1)
    # generate total_GEA
    buildings_prj['total_GEA_m2']=(buildings_prj.area*buildings_prj['building:levels']).round(decimals=1)

    return buildings_prj

def join_building_data_to_blocks(blocks_prj, buildings_prj):
    """
    Add summary building data onto blocks.

    Requires columns to be present in the gdfs generated by other functions in
    osmuf.

    Parameters
    ----------
    places_prj : geodataframe

    buildings_prj : geodataframe

    Returns
    -------
    GeoDataFrame
    """
    building_areas_by_block = buildings_prj[['footprint_m2','total_GEA_m2']].groupby(buildings_prj['block_id']).sum()

    # if there are buildings not associated with a block they aggregate under 0
    # if this happens remove them from the dataframe
    #if 0 in building_areas_by_block.index:
    #    building_areas_by_block = building_areas_by_block.drop([0])

    blocks_prj = blocks_prj.merge(building_areas_by_block, how='left', on='block_id').copy()

    blocks_prj['GSI'] = (blocks_prj['footprint_m2']/blocks_prj.area).round(decimals=3)
    blocks_prj['FSI'] = (blocks_prj['total_GEA_m2']/blocks_prj.area).round(decimals=3)
    blocks_prj['avg_building:levels'] = (blocks_prj['total_GEA_m2']/blocks_prj['footprint_m2']).round(decimals=1)

    return blocks_prj

def calculate_landuse_frontage(net_blocks_prj, landuse_prj):
    """
    Note: This currently doesn't handle areas of overlapping land use
    """

    # Create a GeoDataFrame of net block boundaries
    net_block_boundaries = net_blocks_prj.copy()
    net_block_boundaries['geometry'] = net_block_boundaries.boundary

    # Create a GeoDataFrame of land use boundaries
    landuse_boundaries_prj = landuse_prj.copy()
    landuse_boundaries_prj['geometry'] = landuse_boundaries_prj['geometry'].boundary

    # Create a GeoDataFrame of land use frontage lengths
    landuse_frontage_prj = gpd.overlay(landuse_boundaries_prj, net_block_boundaries[['geometry']], how='intersection')

    # Calculate the length of the frontages
    landuse_frontage_prj['length_m'] = landuse_frontage_prj.length.round(1)

    return landuse_frontage_prj

###################################################################################################3


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
    try:
        del gross_city_blocks.index.name
    except AttributeError:
        pass

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
    gdf_proj = ox.projection.project_gdf(gdf)
    # write perimeter length into column
    gdf_proj['perimeter_m'] = gdf_proj.length
    # write area into column
    gdf_proj['area_m2'] = gdf_proj.area

    gdf_proj[['perimeter_m', 'area_m2']] = gdf_proj[['perimeter_m', 'area_m2']].round(2)

    return gdf_proj


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


def add_block_ids_to_geometries(gdf, blocks):
    """
    Add the index number of enclosing blocks to geometries in a geodataframe.

    Convert all block_id numbers to integers.

    Parameters
    ----------
    gdf : geodataframe
        geometries receiving id of enclosing polygons.

    blocks : geodataframe
        enclosing polygons to transfer id numbers onto geometries.

    Returns
    -------
    GeoDataFrame
    """
    # where they intersect, add the block_id onto each geometry
    gdf = gpd.sjoin(gdf, blocks[['geometry']], how="left", op='intersects')

    gdf.rename(columns={'index_right' : 'block_id'}, inplace=True)

    # convert any NaN values in block_id to zero, then convert to int
    gdf = gdf.fillna({'block_id': 0})
    gdf['block_id'] = pd.to_numeric(gdf['block_id'], downcast='integer')

    return gdf


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




def gen_building_depth(row):
    # get the id of the street nearest the building
    street_id = row.street_id
    # extract street geometry as shapely linestring
    street_linestring = streets_prj.loc[street_id, 'geometry']
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