# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:17:23 2017

@author: Yam
"""

#make a grid shapefile

from fgrid import fgrid
import shapefile as shp

grid = fgrid()
grid.read('MGPF_Grid_2D.inp')

shp_writer=shp.Writer()
shp_writer.field('NODE','C','40')

#for node in grid.nodelist:
#    shp_writer.point(node.position[0].astype(float),node.position[1].astype(float))
#    shp_writer.record(node)

for c in grid.connlist:
    
    points = [[c.nodes[0].position[0],c.nodes[0].position[1]],
              [c.nodes[1].position[0],c.nodes[1].position[1]]]
    shp_writer.poly(shapeType=shp.POLYLINE, parts=[points])
    shp_writer.record(c)
    

shp_writer.save('shapefiles/Grid')
