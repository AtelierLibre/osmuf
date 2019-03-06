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
# position format is x,y e.g. 0.02, 0.94
def add_titlebox(ax, text):
    ax.text(0.04, 0.92, text,
        horizontalalignment='left',
        transform=ax.transAxes,
        # bbox=dict(facecolor='white', alpha=0.6),
        fontsize=12.5)
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

    city_blocks_gross.plot(ax=ax_, color='silver', edgecolor='white', linewidth=1.5)
    city_blocks.plot(ax=ax_, color='darkgrey')
    # Remove streets and buildings for now - add to individual plots as req'd
    # streets.plot(ax=ax_, edgecolor='white', linestyle=':', linewidth=1)
    # buildings.plot(ax=ax_, color='grey')

def ax_map_block_size(ax_, study_area, city_blocks_gross, city_blocks):

    # draw the background map
    ax_background_map(ax_, city_blocks_gross, city_blocks)

    # show city blocks coloured by their net_to_gross, labeled with sizes and ratio
    city_blocks.plot(ax=ax_, column='net_to_gross', cmap='Blues', alpha=0.5, vmin=0, vmax=1, legend=True)

    for idx, row in city_blocks.iterrows():
        s = str(round((row.area_net_ha),2)) + '\n (' + str(round((row.area_gross_ha),2)) + ') \n' + str(round((row.net_to_gross),2))
        ax_.text(row.geometry.centroid.x, row.geometry.centroid.y, s=s, ha='center', va='center',fontsize=9, clip_on=True)

    ax_map_settings(ax_, study_area)

def ax_map_form_factor(ax_, study_area, city_blocks_gross, city_blocks, city_blocks_form_factor):

    # draw the background map
    ax_background_map(ax_, city_blocks_gross, city_blocks)

    city_blocks.plot(ax=ax_,
                     column='perimeter_per_area',
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
    label_geom(ax_, city_blocks_form_factor, 'form_factor')

    ax_.set_title('Form factor of urban blocks (φ)', fontsize=14)

    ax_map_settings(ax_, study_area)

def ax_map_GSI(ax_, study_area, city_blocks_gross, city_blocks, buildings):

    # draw the background map
    ax_background_map(ax_, city_blocks_gross, city_blocks)

    # show buildings
    buildings.plot(ax=ax_, color='grey')

    # show city blocks coloured by their net_to_gross, labeled with sizes and ratio
    city_blocks.plot(ax=ax_, column='GSI_net', cmap='Oranges', alpha=0.5, vmin=0, vmax=1, legend=True)

    for idx, row in city_blocks.iterrows():
        s = str(round((row.GSI_net),2))
        ax_.text(row.geometry.centroid.x, row.geometry.centroid.y, s=s, ha='center', va='center', clip_on=True)

    ax_map_settings(ax_, study_area)

def ax_map_building_heights(ax_, study_area, city_blocks_gross, city_blocks, buildings):

    # draw the background map
    ax_background_map(ax_, city_blocks_gross, city_blocks)

    # buildings with unknown storeys as hatched
    buildings[buildings['building:levels']==0].plot(ax=ax_, color='grey', edgecolor='white', hatch='///')

    # buildings with known storeys - 'categorical=True'
    buildings[buildings['building:levels']>0].plot(ax=ax_,
                                                   column='building:levels',
                                                   cmap='plasma',
                                                   vmin=0,
                                                   vmax=25,
                                                   legend=True,
                                                   )

    ax_map_settings(ax_, study_area)

def ax_map_FSI(ax_, study_area, city_blocks_gross, city_blocks, buildings):

    # draw the background map
    ax_background_map(ax_, city_blocks_gross, city_blocks)

    # show city blocks coloured by their net_to_gross, labeled with sizes and ratio
    city_blocks.plot(ax=ax_, column='FSI_net', cmap='Purples', alpha=0.5, vmin=0, vmax=6, legend=True)

    # buildings with unknown storeys as hatched
    buildings[buildings['building:levels']==0].plot(ax=ax_, color='grey', edgecolor='white', hatch='///')

    for idx, row in city_blocks.iterrows():
        s = str(round((row.FSI_net),2))
        ax_.text(row.geometry.centroid.x, row.geometry.centroid.y, s=s, ha='center', va='center', clip_on=True)

    ax_map_settings(ax_, study_area)

def ax_map_spacemate(ax_, study_area, city_blocks_gross, city_blocks, buildings):

    # draw the background map
    ax_background_map(ax_, city_blocks_gross, city_blocks)

    # show city blocks coloured by their net_to_gross, labeled with sizes and ratio
    city_blocks.plot(ax=ax_, column='FSI_gross', cmap='viridis', alpha=0.5, vmin=0, vmax=6, legend=True)

    # buildings with unknown storeys as hatched
    buildings[buildings['building:levels']==0].plot(ax=ax_, color='grey', edgecolor='white', hatch='///')

    for idx, row in city_blocks.iterrows():
        s = str(round((row.FSI_gross),2))
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
    ax_.scatter(x=city_blocks.area, y=city_blocks.net_to_gross, color='steelblue')
    add_titlebox(ax_, 'Net-to-gross by area')
    ax_.set_xlabel("Net area (ha)")
    ax_.set_ylabel("Net-to-gross ratio")
    ax_.set_xlim([0, 8])
    ax_.set_ylim([0, 1])

    ax_graph_settings(ax_)

def ax_block_area_distribution(ax_, city_blocks):
    # histogram - form factor, range=(0,1),
    ax_.hist(city_blocks['area_net_ha'], bins='auto', color='steelblue', density=True)
    add_titlebox(ax_, 'Area - Normalised Distribution')
    ax_.set_xlabel("Net area (ha)")
    # ax_.set_ylabel("Density")
    ax_.set_xlim([0, 8])
    # ax_.set_ylim([0, 1])

    ax_graph_settings(ax_)

def ax_form_factor_to_area(ax_, city_blocks_form_factor):
    # form factor scatterplot
    ax_.scatter(x=city_blocks_form_factor.area_net_ha, y=city_blocks_form_factor.form_factor, color='Red')
    add_titlebox(ax_, "Form factor (φ)")
    ax_.set_xlabel("Area (ha)")
    ax_.set_ylabel("Form factor (φ)")
    ax_.set_xlim([0, 8])
    ax_.set_ylim([0, 1])

    ax_graph_settings(ax_)

def ax_block_perimeter_to_area(ax_, city_blocks):
    # ax block area:perimeter ratio
    ax_.scatter(x=city_blocks.area_net_ha, y=city_blocks.perimeter_m, color='Red')
    add_titlebox(ax_, 'Perimeter to area')
    ax_.set_xlabel("Net area (ha)")
    ax_.set_ylabel("Perimeter (m)")
    ax_.set_xlim([0, 8])
    ax_.set_ylim([0, 2500])

    ax_graph_settings(ax_)

    # plot line of ratio for square blocks
    x = np.linspace(0, 8, 80)
    ax_.plot(x, 4*100*np.sqrt(x), color='lightcoral')

def ax_GSI_to_net_area(ax_, city_blocks):
    # scatterplot -
    ax_.scatter(x=city_blocks.area_net_ha, y=city_blocks['GSI_net'], color='Orange')
    add_titlebox(ax_, 'GSI' + ' by area (ha)')
    ax_.set_xlabel("Area, net (ha)")
    ax_.set_ylabel("GSI")
    ax_.set_ylim([0,1])
    ax_.set_xlim([0, 8])

    ax_graph_settings(ax_)

def ax_GSI_distribution(ax_, city_blocks):
    # histogram - chosen metric, range=(0,1)
    ax_.hist(city_blocks['GSI_net'], bins='auto', color='Orange', alpha=0.5, density=True)
    ax_.set_xlabel('GSI')
    # add_titlebox(ax2, 'Histogram: form factor (φ)')
    ax_.set_xlim([0,1])

    ax_graph_settings(ax_)

def ax_building_height_distribution(ax_, buildings):
    # histogram
    ax_.hist(buildings['building:levels'], bins=26, color='Purple')
    ax_.set_xlabel('Building storeys')
    ax_.set_ylabel('Count')
    # add_titlebox(ax2, 'Histogram: form factor (φ)')
    ax_.set_xlim([0,25])

    ax_graph_settings(ax_)

def ax_FSI_to_net_area(ax_, city_blocks):
    # scatterplot -
    ax_.scatter(x=city_blocks.area_net_ha, y=city_blocks['FSI_net'], color='Purple')
    add_titlebox(ax_, 'FSI' + ' by area (ha)')
    ax_.set_xlabel("Area, net (ha)")
    ax_.set_ylabel("FSI")
    ax_.set_ylim([0,6])
    ax_.set_xlim([0,8])

    ax_graph_settings(ax_)

def ax_FSI_distribution(ax_, city_blocks):
    # histogram - chosen metric, range=(0,1)
    ax_.hist(city_blocks['FSI_net'], bins=10, color='Purple', alpha=0.5)
    ax_.set_xlabel('FSI')
    # add_titlebox(ax2, 'Histogram: form factor (φ)')
    ax_.set_xlim([0,6])

    ax_graph_settings(ax_)

def ax_spacemate(ax_, city_blocks):
    # ax Spacemate
    # draw lines of building levels
    x = np.linspace(0, 0.6)
    for l in np.arange(1,14):
        ax_.plot(x, l*x, color='darkgrey', label=l, linewidth=0.5)

    city_blocks[['GSI_gross', 'area_net_ha', 'FSI_gross']].plot(ax=ax_,
                                                                kind='scatter',
                                                                x='GSI_gross',
                                                                y='FSI_gross')
    ax_.set_xlim(0, 0.6)
    ax_.set_ylim(0, 3)

    ax_graph_settings(ax_)

def ax_FSI_perimeter_area(ax_, city_blocks):
    # ax FSI against perimeter and area of urban block
    city_blocks[['perimeter_m', 'area_net_ha', 'FSI_net']].plot(ax=ax_, kind='scatter', x='perimeter_m',
                                                                y='area_net_ha', c=city_blocks['FSI_net'].values,
                                                                cmap='inferno',
                                                                s=20*city_blocks['FSI_net'].values,
                                                                alpha=0.5)
    ax_.set_xlim(0, None)
    ax_.set_ylim(0, None)

    ax_graph_settings(ax_)
