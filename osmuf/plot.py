################################################################################
# Module: plot.py
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

from .utils import dict_to_gdf

# for nicer looking plots
import seaborn
plt.style.use('seaborn')

#  “helper function” that places a text box inside of a plot and acts as an “in-plot title”
# from https://realpython.com/python-matplotlib-guide/
# position format is x,y e.g. 0.04, 0.92, text then aligns around this point ('left', 'right', 'center')
def add_titlebox(ax, text):
    ax.text(0.5, 0.93, text,
        horizontalalignment='center',
        transform=ax.transAxes,
        # bbox=dict(facecolor='white', alpha=0.6),
        fontsize=12.5,
        fontweight='bold')
    return ax

# label style
style = dict(size=11, color='black', horizontalalignment='center')

# label individual geometries
def label_geom(ax, gdf, column):
    for idx, row in gdf.iterrows():
        label = round(row[column], 2)
        ax.text(row.geometry.centroid.x, row.geometry.centroid.y, label, **style, clip_on=True)

#####################
# NEW PLOT ELEMENTS #
#####################

# Layout - 3 plots - map, top and bottom
def layout_3_plots():

    gridsize = (2, 3)
    # was (18,10)
    fig = plt.figure(figsize=(18, 10), facecolor='gainsboro', dpi=200)

    ax_top = plt.subplot2grid(gridsize, (0, 0))
    ax_bottom = plt.subplot2grid(gridsize, (1, 0))
    ax_map = plt.subplot2grid(gridsize, (0, 1), colspan=2, rowspan=2)

    ax = (ax_top, ax_bottom, ax_map)

    return fig, ax

########
# MAPS #
########

def ax_map_settings(ax_, study_area):

    # draw the boundary and centre of the study area
    study_area.plot(ax=ax_, color='none', edgecolor='black', linestyle=':', linewidth=1)
    study_area.centroid.plot(ax=ax_, color='black', marker='+')

    # temporarily hard code in clip distance, potentially allow size of this
    # to vary with the study area in future
    distance = 500

    # clip
    offset = distance*0.05
    left, bottom, right, top = study_area.total_bounds
    ax_.set_xlim((left - offset, right + offset))
    ax_.set_ylim((bottom - offset, top + offset))

    # turn off the axis display set the margins to zero and point the ticks in
    # so there's no space around the plot
    ax_.axis('on')
    ax_.grid(True)
    ax_.margins(0)
    ax_.grid(color='darkgrey', linestyle=':')
    ax_.tick_params(which='both', direction='in')

    # make everything square
    ax_.set_aspect('equal')

    # set the background to transparent
    ax_.set_facecolor('None')

def ax_background_map(ax_, city_blocks_gross, city_blocks):

    # disabling the edge color here as should show if measurement is net or gross
    city_blocks_gross.plot(ax=ax_, color='silver', edgecolor='none') #edgecolor='white', linewidth=1.5)
    city_blocks.plot(ax=ax_, color='darkgrey')
    # Remove streets and buildings for now - add to individual plots as req'd
    # streets.plot(ax=ax_, edgecolor='white', linestyle=':', linewidth=1)
    # buildings.plot(ax=ax_, color='grey')

def ax_map_block_size(ax_, study_area, city_blocks_gross, city_blocks):

    # draw the background map
    ax_background_map(ax_, city_blocks_gross, city_blocks)

    # show city blocks coloured by their net_to_gross, labeled with sizes and ratio
    city_blocks.plot(ax=ax_, column='net:gross', cmap='Blues', alpha=0.5, vmin=0, vmax=1, legend=True)

    for idx, row in city_blocks.iterrows():
        s = str(round((row['net_area_m2']/10_000),2)) + '\n (' + str(round((row['gross_area_m2']/10_000),2)) + ') \n' + str(round((row['net:gross']),2))
        ax_.text(row.geometry.centroid.x, row.geometry.centroid.y, s=s, ha='center', va='center',fontsize=9, clip_on=True)

    ax_map_settings(ax_, study_area)

def ax_map_form_factor(ax_, study_area, city_blocks_gross, city_blocks, city_blocks_form_factor):

    # draw the background map
    ax_background_map(ax_, city_blocks_gross, city_blocks)

    city_blocks.plot(ax=ax_,
                     column='PAR',
                     cmap='Reds',
                     linewidth=1.0,
                     alpha=0.5,
                     legend=True)

    city_blocks_form_factor.plot(ax=ax_,
                            edgecolor='red',
                            linewidth=1.0,
                            color='None',
                            linestyle='-'
                            )

    # show form factor of city blocks
    label_geom(ax_, city_blocks_form_factor, 'regularity')

    ax_.set_title('Form factor of urban blocks (φ)', fontsize=14)

    ax_map_settings(ax_, study_area)

def ax_map_GSI(ax_, study_area, city_blocks_gross, city_blocks, buildings):

    # draw the background map
    ax_background_map(ax_, city_blocks_gross, city_blocks)

    # show buildings
    buildings.plot(ax=ax_, color='grey')

    # show city blocks coloured by their net_to_gross, labeled with sizes and ratio
    city_blocks.plot(ax=ax_, column='net_GSI', cmap='Oranges', alpha=0.5, vmin=0, vmax=1, legend=True)

    for idx, row in city_blocks.iterrows():
        s = str(round((row.net_GSI),2))
        ax_.text(row.geometry.centroid.x, row.geometry.centroid.y, s=s, ha='center', va='center', clip_on=True)

    ax_map_settings(ax_, study_area)

def ax_map_building_heights(ax_, study_area, city_blocks_gross, city_blocks, buildings):

    # draw the background map
    ax_background_map(ax_, city_blocks_gross, city_blocks)

    # buildings with unknown storeys as hatched
    no_levels_filter = buildings['building:levels'].isna()
    buildings[no_levels_filter].plot(ax=ax_, color='grey', edgecolor='white', hatch='///')

    # buildings with known storeys - 'categorical=True'
    buildings[buildings['building:levels']>0].plot(ax=ax_,
                                                   column='building:levels',
                                                   cmap='viridis_r',
                                                   vmin=0,
                                                   vmax=25,
                                                   legend=True,
                                                   )

    ax_map_settings(ax_, study_area)

def ax_map_FSI(ax_, study_area, city_blocks_gross, city_blocks, buildings):

    # draw the background map
    ax_background_map(ax_, city_blocks_gross, city_blocks)

    # buildings with unknown storeys as hatched
    no_levels_filter = buildings['building:levels'].isna()
    buildings[no_levels_filter].plot(ax=ax_, color='grey', edgecolor='white', hatch='///')

    # buildings with known storeys solid
    with_levels_filter = buildings['building:levels'].notna()
    buildings[with_levels_filter].plot(ax=ax_, color='grey', edgecolor='None')


    # show city blocks coloured by their net_to_gross, labeled with sizes and ratio
    city_blocks.plot(ax=ax_, column='net_FSI', cmap='Purples', alpha=0.5, vmin=0, vmax=6, legend=True)
    city_blocks.plot(ax=ax_, facecolor='none', edgecolor='white', linewidth=1)

    for idx, row in city_blocks.iterrows():
        s = str(round((row['net_FSI']),1))
        ax_.text(row.geometry.centroid.x, row.geometry.centroid.y, s=s, ha='center', va='center', clip_on=True, fontweight='bold')

    ax_map_settings(ax_, study_area)

def ax_map_FSI_with_building_heights(ax_, study_area, city_blocks_gross, city_blocks, buildings):

    # add_titlebox(ax_, 'Building Heights and FSI per Urban Block')

    # draw the background map
    ax_background_map(ax_, city_blocks_gross, city_blocks)

    # show city blocks coloured by their net_to_gross, labeled with sizes and ratio
    city_blocks.plot(ax=ax_, column='net_FSI', cmap='Purples', alpha=0.6, vmin=0, vmax=6, legend=True)
    city_blocks.plot(ax=ax_, facecolor='none', edgecolor='white', linewidth=1)

    # buildings with unknown storeys as hatched
    no_levels_filter = buildings['building:levels'].isna()
    buildings[no_levels_filter].plot(ax=ax_, color='grey', edgecolor='white', hatch='///')

    # buildings with known storeys - 'categorical=True'
    buildings[buildings['building:levels']>0].plot(ax=ax_,
                                                   column='building:levels',
                                                   cmap='viridis_r',
                                                   vmin=0,
                                                   vmax=25,
                                                   alpha=1,
                                                   legend=True,
                                                   )

    for idx, row in city_blocks.iterrows():
        s = str(round((row['net_FSI']),1))
        ax_.text(row.geometry.centroid.x, row.geometry.centroid.y, s=s, ha='center', va='center', clip_on=True, fontweight='bold')

    ax_map_settings(ax_, study_area)

def ax_map_network_density(ax_, study_area, city_blocks_gross, city_blocks, buildings):

    add_titlebox(ax_, 'Network Density (m/ha)')

    # draw the background map
    ax_background_map(ax_, city_blocks_gross, city_blocks)

    # show city blocks coloured by their net_to_gross, labeled with sizes and ratio
    city_blocks_gross.plot(ax=ax_, column='network_density_m_ha', cmap='Greens', alpha=0.5, vmin=0, vmax=800, legend=True)

    # buildings with unknown storeys as hatched
    no_levels_filter = buildings['building:levels'].isna()
    buildings[no_levels_filter].plot(ax=ax_, color='grey', edgecolor='white', hatch='///')

    # buildings with known storeys solid
    with_levels_filter = buildings['building:levels'].notna()
    buildings[with_levels_filter].plot(ax=ax_, color='grey', edgecolor='None')

    for idx, row in city_blocks_gross.iterrows():
        s = str(round((row['network_density_m_ha'])))
        ax_.text(row.geometry.centroid.x, row.geometry.centroid.y, s=s, ha='center', va='center', clip_on=True)

    ax_map_settings(ax_, study_area)

def ax_map_spacemate(ax_, study_area, city_blocks_gross, city_blocks, buildings):

    # draw the background map
    ax_background_map(ax_, city_blocks_gross, city_blocks)

    # show city blocks coloured by their net_to_gross, labeled with sizes and ratio
    city_blocks.plot(ax=ax_, column='gross_FSI', cmap='viridis', alpha=0.5, vmin=0, vmax=6, legend=True)

    # buildings with unknown storeys as hatched
    buildings[buildings['building:levels']==0].plot(ax=ax_, color='grey', edgecolor='white', hatch='///')

    for idx, row in city_blocks.iterrows():
        s = str(round((row['gross_FSI']),2))
        ax_.text(row.geometry.centroid.x, row.geometry.centroid.y, s=s, ha='center', va='center', clip_on=True)

    ax_map_settings(ax_, study_area)

##########
# GRAPHS #
##########

def ax_graph_settings(ax_):
    ax_.set_facecolor('None')
    ax_.grid(color='darkgrey', linestyle=':')

def ax_empty(ax_):

    ax_graph_settings(ax_)

def ax_block_ntg_to_size(ax_, city_blocks):
    # ax block net_to_gross against size
    ax_.scatter(x=city_blocks.area/10_000, y=city_blocks['net:gross'], color='steelblue')
    add_titlebox(ax_, 'Net-to-gross by net area')
    ax_.set_xlabel("Net area (ha)")
    ax_.set_ylabel("Net-to-gross ratio")
    ax_.set_xlim([0, 8])
    ax_.set_ylim([0, 1])

    ax_graph_settings(ax_)

def ax_block_area_distribution(ax_, city_blocks):
    # histogram - form factor, range=(0,1),
    ax_.hist((city_blocks['net_area_m2']/10_000), bins='auto', color='steelblue', density=True)
    add_titlebox(ax_, 'Area - Normalised Distribution')
    ax_.set_xlabel("Net area (ha)")
    # ax_.set_ylabel("Density")
    ax_.set_xlim([0, 8])
    # ax_.set_ylim([0, 1])

    ax_graph_settings(ax_)

def ax_form_factor_to_area(ax_, city_blocks_form_factor):
    # form factor scatterplot
    ax_.scatter(x=(city_blocks_form_factor['net_area_m2']/10_000), y=city_blocks_form_factor['form_factor'], color='Red')
    add_titlebox(ax_, "Form factor (φ)")
    ax_.set_xlabel("Area (ha)")
    ax_.set_ylabel("Form factor (φ)")
    ax_.set_xlim([0, 8])
    ax_.set_ylim([0, 1])

    ax_graph_settings(ax_)

def ax_block_perimeter_to_area(ax_, city_blocks):
    # ax block area:perimeter ratio
    ax_.scatter(x=(city_blocks['net_area_m2']/10_000), y=city_blocks['frontage_m'], color='Red')
    add_titlebox(ax_, 'Perimeter to area')
    ax_.set_xlabel("Net area (ha)")
    ax_.set_ylabel("Perimeter (m)")
    ax_.set_xlim([0, 8])
    ax_.set_ylim([0, 2500])

    ax_graph_settings(ax_)

    # plot line of ratio for square blocks
    x = np.linspace(0, 8, 80)
    ax_.plot(x, 4*100*np.sqrt(x), color='lightcoral')

def ax_net_GSI_to_net_area(ax_, city_blocks):
    # scatterplot -
    ax_.scatter(x=(city_blocks['net_area_m2']/10_000), y=city_blocks['net_GSI'], color='Orange')
    add_titlebox(ax_, 'net GSI' + ' by area (ha)')
    ax_.set_xlabel("Area, net (ha)")
    ax_.set_ylabel("GSI")
    ax_.set_ylim([0,1])
    ax_.set_xlim([0, 4])
    ax_graph_settings(ax_)

def ax_net_GSI_to_frontage_density(ax_, city_blocks):
    # scatterplot -
    ax_.scatter(x=(city_blocks['frontage_density_m_ha']), y=city_blocks['net_GSI'], color='Orange')
    add_titlebox(ax_, 'net GSI' + ' by frontage density(m/ha)')
    ax_.set_xlabel("Frontage Density (m/ha)")
    ax_.set_ylabel("net GSI")
    ax_.set_ylim([0,1])
    ax_.set_xlim([0,0.15])
    ax_graph_settings(ax_)

def ax_building_GEA_to_frontage(ax_, city_blocks):
    # scatterplot -
    ax_.scatter(x=(city_blocks['frontage_m']), y=city_blocks['total_GEA_m2'], color='Orange')
    add_titlebox(ax_, 'Total Building GEA, Frontage')
    ax_.set_xlabel("Frontage (m)")
    ax_.set_ylabel("Total Building GEA (m2)")
    ax_.set_ylim([0,25_000])
    ax_.set_xlim([0, 600])

    ax_graph_settings(ax_)

def ax_building_GEA_to_net_area(ax_, city_blocks):
    # scatterplot -
    ax_.scatter(x=(city_blocks['net_area_m2']/10_000), y=city_blocks['total_GEA_m2'], color='Orange')
    add_titlebox(ax_, 'Total Building GEA, Net Block Area')
    ax_.set_xlabel("Net Block Area (ha)")
    ax_.set_ylabel("Total Building GEA (m2)")
    ax_.set_ylim([0,25_000])
    ax_.set_xlim([0, 1.5])

    ax_graph_settings(ax_)

def ax_GSI_distribution(ax_, city_blocks):
    # histogram - chosen metric, range=(0,1)
    ax_.hist(city_blocks['net_GSI'], bins='auto', color='Orange', alpha=0.5, density=True)
    ax_.set_xlabel('net GSI')
    # add_titlebox(ax2, 'Histogram: form factor (φ)')
    ax_.set_xlim([0,1])

    ax_graph_settings(ax_)

def ax_building_height_distribution(ax_, buildings):
    # histogram
    ax_.hist(buildings['building:levels'], bins=26, cmap='Purples')
    ax_.set_xlabel('Building storeys')
    ax_.set_ylabel('Count')
    # add_titlebox(ax2, 'Histogram: form factor (φ)')
    ax_.set_xlim([0,25])

    ax_graph_settings(ax_)

def ax_net_FSI_to_net_area(ax_, city_blocks):
    # scatterplot -
    ax_.scatter(x=(city_blocks['net_area_m2']/10_000), y=city_blocks['net_FSI'], color='Purple')
    add_titlebox(ax_, 'FSI, Net Urban Block Area')
    ax_.set_xlabel("Urban Block, Net Area (ha)")
    ax_.set_ylabel("Floor Area Ratio (FSI)")
    ax_.set_ylim([0,6])
    ax_.set_xlim([0,4])

    ax_graph_settings(ax_)

def ax_net_FSI_to_frontage(ax_, city_blocks):
    # scatterplot -
    ax_.scatter(x=(city_blocks['frontage_m']), y=city_blocks['net_FSI'], color='Purple')
    add_titlebox(ax_, 'FSI' + ' by frontage (m)')
    ax_.set_xlabel("Frontage (m)")
    ax_.set_ylabel("FSI")
    ax_.set_ylim([0,6])
    ax_.set_xlim([0,800])

    ax_graph_settings(ax_)

def ax_net_FSI_to_frontage_density(ax_, city_blocks):
    # scatterplot -
    ax_.scatter(x=(city_blocks['frontage_density_m_ha']), y=city_blocks['net_FSI'], color='Purple')
    add_titlebox(ax_, 'FSI, Frontage Density(m/ha)')
    ax_.set_xlabel("Urban Block, Frontage Density (m/ha)")
    ax_.set_ylabel("Floor Area Ratio (FSI)")
    ax_.set_ylim([0,6])
    ax_.set_xlim([0,0.25])
    ax_graph_settings(ax_)

def ax_gross_FSI_to_gross_area(ax_, city_blocks):
    '''
    Plot Gross FSI agains Gross Urban Block Area

    Parameters
    ----------
    ax_: ax to plot to
    city_blocks: gdf containing data
    '''
    # scatterplot -
    ax_.scatter(x=(city_blocks['gross_area_m2']/10_000), y=city_blocks['gross_FSI'], color='Purple')
    add_titlebox(ax_, 'Gross FSI' + ', Gross Area')
    ax_.set_xlabel("Gross Area (ha)")
    ax_.set_ylabel("Gross FSI")
    ax_.set_ylim([0,6])
    ax_.set_xlim([0,8])

    ax_graph_settings(ax_)

def ax_FSI_to_network_density(ax_, city_blocks):
    # scatterplot -
    ax_.scatter(x=(city_blocks['network_density_m_ha']), y=city_blocks['gross_FSI'], color='Purple')
    add_titlebox(ax_, 'Gross FSI' + '/ Network Density')
    ax_.set_xlabel("Network Density (m/ha)")
    ax_.set_ylabel("Gross FSI")
    ax_.set_ylim([0,6])
    ax_.set_xlim([0,800])

    ax_graph_settings(ax_)

def ax_network_density_to_gross_area(ax_, city_blocks_gross):
    # scatterplot -
    ax_.scatter(x=(city_blocks_gross['gross_area_ha']), y=city_blocks_gross['network_density_m_ha'], color='Green')
    add_titlebox(ax_, 'Network Density, Gross Area')
    ax_.set_xlabel("Gross Area (ha)")
    ax_.set_ylabel("Network Density (m/ha)")
    ax_.set_ylim([0,800])
    ax_.set_xlim([0,8])

    ax_graph_settings(ax_)

def ax_FSI_distribution(ax_, city_blocks):
    # histogram - chosen metric, range=(0,1)
    ax_.hist(city_blocks['net_FSI'], bins=10, color='Purple', alpha=0.5)
    ax_.set_xlabel('net FSI')
    # add_titlebox(ax2, 'Histogram: form factor (φ)')
    ax_.set_xlim([0,6])

    ax_graph_settings(ax_)

def ax_spacemate(ax_, city_blocks):
    # ax Spacemate
    # draw lines of building levels
    x = np.linspace(0, 0.6)
    for l in np.arange(1,14):
        ax_.plot(x, l*x, color='darkgrey', label=l, linewidth=0.5)

    city_blocks[['gross_GSI', 'net_area_m2', 'gross_FSI']].plot(ax=ax_,
                                                                kind='scatter',
                                                                x='gross_GSI',
                                                                y='gross_FSI')
    ax_.set_xlim(0, 0.6)
    ax_.set_ylim(0, 3)

    ax_graph_settings(ax_)

def ax_FSI_perimeter_area(ax_, city_blocks):
    # ax FSI against perimeter and area of urban block
    city_blocks[['frontage_m', 'net_area_m2', 'net_FSI']].plot(ax=ax_, kind='scatter', x='net_perimeter_m',
                                                                y='net_area_m2', c=city_blocks['net_FSI'].values,
                                                                cmap='inferno',
                                                                s=20*city_blocks['net_FSI'].values,
                                                                alpha=0.5)
    ax_.set_xlim(0, None)
    ax_.set_ylim(0, None)

    ax_graph_settings(ax_)
