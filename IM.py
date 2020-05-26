# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:49:24 2020

@author: annes
"""

#Import
import fiona
import shapefile
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import itertools
from itertools import product
from scipy.sparse import dok_matrix
import pandas as pd
import numpy as np
import scipy.sparse
import scipy.io
import numpy.ma as ma
from matplotlib.colors import ListedColormap
import matplotlib.transforms as mtransforms
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.geometry import Polygon, Point
import math
