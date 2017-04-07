# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:21:33 2017

@author: Yam
"""

import numpy as np
from fdata import fgrid

inner_X = 5E3 #inner span
inner_Y = 7E3
outer_X = 13E3 #outer allowance
outer_Y = 15E3 

div_X_inner = 14
div_Y_inner = 15
div_X_outer = 7
div_Y_outer = 7
outer_dist_min = 400.

center = [525479.453,774586.588]
degree_rotation = 60. #counter-clockwise

#Z layers
Z_top = 2000.
Z_bottom = -500
Z_base = -2000.
Za = np.arange(Z_top,Z_bottom,-250)
Zb = np.arange(Z_bottom,Z_base-500,-500)

Z = np.concatenate((Za,Zb))

#x nodes
X_inner = np.linspace(center[0] - inner_X/2, center[0] + inner_X/2, 
                       div_X_inner, endpoint=True)

#log spacing for outer
X_outer_left = (center[0] - inner_X/2 - 
               np.logspace(np.log10(outer_dist_min), np.log10(outer_X), 
                           div_X_outer, endpoint=True)[::-1])
X_outer_right = (center[0] + inner_X/2 + 
               np.logspace(np.log10(outer_dist_min), np.log10(outer_X), 
                           div_X_outer, endpoint=True))
X = np.concatenate((X_outer_left,X_inner,X_outer_right))

#y nodes
Y_inner = np.linspace(center[1] - inner_Y/2, center[1] + inner_Y/2, 
                       div_Y_inner, endpoint=True)

#log spacing for outer
Y_outer_bot = (center[1] - inner_Y/2 - 
               np.logspace(np.log10(outer_dist_min), np.log10(outer_Y), 
                           div_Y_outer, endpoint=True)[::-1])
Y_outer_top = (center[1] + inner_Y/2 + 
               np.logspace(np.log10(outer_dist_min), np.log10(outer_Y), 
                           div_Y_outer, endpoint=True))
Y = np.concatenate((Y_outer_bot,Y_inner,Y_outer_top))



#make the 2D grid
grid = fgrid()
grid.make('MGPF_Grid_2D.inp', x = X, y = Y, z = [])
grid.rotate(degree_rotation,center)
grid.write('MGPF_Grid_2D.inp')

#make the 3D grid
grid = fgrid()
grid.make('MGPF_Grid.inp', x = X, y = Y, z = Z)
grid.rotate(degree_rotation,center)
grid.write('MGPF_Grid.inp')
