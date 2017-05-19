# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:25:27 2017

@author: Yam
"""

#make shapefile from deviation survey
data_dir = "D:/Users/Yam/Desktop/Thesis/Modeling Directory/Data/"
wells_file = data_dir + "MGPF Deviation Survey and Well Data1.xlsx"
well_list = data_dir + "ModelWellList.xlsx"

import pandas as pd
import shapefile as shp

wells = pd.read_excel(well_list)
wells = wells.values.flatten().astype(str)
wells = wells[wells!='nan']

wells_DS=pd.read_excel(wells_file,sheetname=None)


shp_writer=shp.Writer()
shp_writer.field('WELL_NAME','C','40')

for w in wells:
    wells_DS[w]=wells_DS[w].loc[:,['Easting','Northing']]
    shp_writer.poly(shapeType=shp.POLYLINE, parts=[wells_DS[w].values])
    shp_writer.record(w)

shp_writer.save('shapefiles/MGPF_wells')