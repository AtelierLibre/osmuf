################################################################################
# Module: utils.py
# Description: urban form analysis from OpenStreetMap
# License: MIT, see full license in LICENSE.txt
# Web: https://github.com/atelierlibre/osmuf
################################################################################

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox

import osmuf.smallestenclosingcircle as sec

from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import Polygon, MultiPolygon

from shapely.ops import polygonize

def extract_poly_coords(geom):
    # extract the coordinates of shapely polygons and multipolygons
    # as a list of tuples
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

    return exterior_coords + interior_coords

def circlizer(x):
    # takes a shapely polygon or multipolygon and returns a shapely circle polygon
    (cx, cy, buff) = sec.make_circle(extract_poly_coords(x))
    donut = Point(cx, cy).buffer(buff)

    return donut

def dict_to_gdf(place_dict):

    # 0. INVERT COORDINATES FROM LAT, LONG TO LONG, LAT # changed 'coordinates' to 'geometry'
    place_dict['geometry']=(place_dict['coordinates'][1], place_dict['coordinates'][0])

    # 1. convert dict to Pandas dataframe
    place_df = pd.DataFrame([place_dict])

    # 2. create 'geometry' column as tuple of Latitude and Longitude # unecessary
    # place_df = place_df.rename(columns={'coordinates': 'geometry'})

    # 3. transform tuples into Shapely Points
    place_df['geometry'] = place_df['geometry'].apply(Point)

    # places_gdf = geopandas.GeoDataFrame(places_df, geometry='geometry')
    place_gdf = gpd.GeoDataFrame(place_df, geometry='geometry')

    # set the crs
    place_gdf.crs = {'init' :'epsg:4326'}

    return place_gdf

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
    gdf_temp = ox.projection.project_gdf(gdf, to_latlong=True)
    # determine the boundary polygon to fetch buildings within
    # buffer originally 0.000225, buffer actually needs to go whole block away
    # to get complete highways therefor trying 0.001
    boundary=gdf_temp.cascaded_union.convex_hull.buffer(0.001)
    # NOTE - maybe more efficient to generate boundary first then reproject second?

    return boundary

def footprints_from_gdf(gdf, footprint_type='building'):
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
    footprints = ox.footprints_from_polygon(boundary, footprint_type)

    # name the dataframe
    footprints.gdf_name = footprint_type

    return footprints

def extend_line_by_factor(p1,p2, extension_factor):
    'Creates a line from p1 through p2 extended by factor'
    p3 = (p1.x+extension_factor*(p2.x-p1.x), p1.y+extension_factor*(p2.y-p1.y))
    return LineString([p1,p3])

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