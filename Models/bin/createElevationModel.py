# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:12:26 2017

@author: Yam
"""

import pandas as pd
import os
from scipy.interpolate import RectBivariateSpline as rbs
import numpy as np
from fdata import fdata,fzone

work_dir=os.getcwd()

dirs=pd.read_excel(work_dir+"\\ModelData.xlsx",sheetname='Dir')
dirs=dirs.set_index('Variable').T

grid_dir = grid_3D_path = dirs['grid_3D_path'].values[0]
data_dir = data_dir = dirs['data_dir'].values[0]
elevation_data= data_dir+'\\Mt. Apo Elevations\\Elevations.xyz'

data = pd.read_csv(elevation_data,names=['x','y','z'],header=None,sep=' ')

x=data['x'].unique()
y=data['y'].unique()
z=data['z'].values.reshape(len(x),len(y),order='F')

rbsf=rbs(x,y,z)

dat=fdata()
dat.grid.read(grid_dir)

node_array=[(n.index, n.position[0],n.position[1],n.position[2]) for n in dat.grid.nodelist]
node_df=pd.DataFrame(node_array,columns=['ni','x','y','z'])

#find inactive blocks
node_df['active']=node_df.apply(lambda i: i['z']<rbsf.ev(i['x'],i['y']),axis=1)

x_columns = node_df['x'].unique()
node_df['is_top']=False

for x in x_columns:
    column=node_df[node_df['x']==x]
    indices=column.index.tolist()
    for i, row in node_df[node_df['x']==x].iterrows():
        if row['active']==False:
            node_df.set_value(indices[indices.index(i)-1],'is_top',True)
            break
        elif i==indices[-1]:
            node_df.set_value(i,'is_top',True)


node_df.to_csv('elevationModel.csv',index=False)
#dat.paraview(exe=r"D:\ParaView 5.0.0\bin\paraview.exe")
