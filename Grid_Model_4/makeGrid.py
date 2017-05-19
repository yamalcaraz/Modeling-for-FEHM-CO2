# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:21:33 2017

@author: Yam
"""

import numpy as np
from fdata import fgrid
np.set_printoptions(threshold=np.nan)

inner_X = 5E3 #inner span
inner_Y = 7E3
outer_X = 13E3 #outer allowance
outer_Y = 15E3 

#div_X_inner = 14
#div_Y_inner = 15
#div_X_outer = 7
#div_Y_outer = 7

#2x finer
#div_X_inner = 28
#div_Y_inner = 30
#div_X_outer = 8
#div_Y_outer = 7
#mult_x = 2
#mult_y = 2

#4x finer
div_X_inner = 14*4
div_Y_inner = 15*4
div_X_outer = 10
div_Y_outer = 9
mult_x = 2
mult_y = 2

center = [525479.453,774586.588]
degree_rotation = 60. #counter-clockwise

#Z layers
#Z_top = 2000.
#Z_bottom = -500
#Z_base = -2000.
##dz_top = 250
##dz_bot = 500
#dz_top = 250
#dz_bot = 500
#Za = np.arange(Z_top,Z_bottom,-dz_top)
#Zb = np.arange(Z_bottom,Z_base-dz_bot,-dz_bot)
#
#Z = np.concatenate((Za,Zb))

Z = [2000,
1750,
1500,
1250,
1000,
775,
750,
500,
250,
25,
0,
-250,
-500,
-1000,
-1500,
-2000
]

Z = np.array(Z)

def linspacing(start, end, multiplier=1):
    results = [start]
    next_r = start*2
    i=1
    while next_r < end:
        results+=[results[-1]+start*i*multiplier]
        next_r=results[-1]
        i+=1

        
    return np.array(results)

#x nodes
dx_inner = np.linspace(0, inner_X/2, 
                       div_X_inner/2, endpoint=True)

dx_outer = linspacing(dx_inner[1],outer_X,mult_x)


X_half = np.concatenate((dx_inner,inner_X/2 + dx_outer))
X = center[0] + np.concatenate((-X_half[-1:0:-1],X_half))

dy_inner = np.linspace(0, inner_Y/2, 
                       div_Y_inner/2, endpoint=True)

dy_outer = linspacing(dy_inner[1],outer_Y,mult_y)

Y_half = np.concatenate((dy_inner,inner_Y/2 + dy_outer))
Y = center[1] + np.concatenate((-Y_half[-1:0:-1],Y_half))


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
