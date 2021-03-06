# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 01:41:11 2017

@author: Yam
"""

#create contours

#process NS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.interpolate import Rbf

#files
data_dir = "D:/Thesis/Modeling Directory/Data/"
wells_file = data_dir + "MGPF Deviation Survey and Well Data1.xlsx"
ns_temp_file = data_dir + "MAGBU New Temperature Interpretation.xlsx"
well_list = data_dir + "ModelWellList.xlsx"

#get the well list
wells = pd.read_excel(well_list)
wells = wells.values.flatten().astype(str)
wells = wells[wells!='nan']

#get DS
wells_DS=pd.read_excel(wells_file,sheetname=None)

#get NS Temp
ns_temp_df=pd.read_excel(ns_temp_file,sheetname=None,parse_cols='A:B',names=['T','MRSL'])




def plot_contour(z, method = 'multiquadric'):
    xyt=[]
    wells_with_data=[]
    
    for w in wells:
        w_DS = wells_DS[w].loc[:,['Easting','Northing','MRSL']]
        w_DS=w_DS.sort_values('MRSL')
        #get only DS on cont.z values
        x=np.interp(z,w_DS['MRSL'],w_DS['Easting'])
        y=np.interp(z,w_DS['MRSL'],w_DS['Northing'])
        w_df = ns_temp_df[w]
        w_df=w_df.sort_values('MRSL')
        if not(any(w_df.MRSL>=z) and any(w_df.MRSL<=z)):
            print w_df.MRSL
            print z
            print '{} at {} not within well'.format(w,z)
            continue
        #get only DS on cont.z values
        xyt+=[[x,y,np.interp(z,w_df['MRSL'],w_df['T'])]]
        wells_with_data+=[w]
        
    T=np.array(xyt)
    
    f=Rbf(T[:,0],T[:,1],T[:,2], function = method)
    
    print 'epsilon is {}'.format(f.epsilon)
    
    # interpolate onto a 100x100 regular grid
    xmin,ymax=521536.243,777812.558
    xmax,ymin=529641.372,770788.113
    
    X, Y = np.meshgrid(np.linspace(xmin,xmax,100,),np.linspace(ymin,ymax,100,))
    Z = f(X.ravel(), Y.ravel()).reshape(X.shape)
    
    # plotting
    fig, ax = plt.subplots(1, 1,figsize=(10,10))
    ax.contourf(X, Y, Z, levels=np.arange(180,360,20),cmap=plt.get_cmap('RdYlBu_r'),vmin=50.,vmax=350.)
    contour_lines=ax.contour(X, Y, Z, levels=np.arange(180,360,20),colors='k')
    contour_lines.clabel(inline=1)
    
    ax.scatter(T[:,0], T[:,1], c='k', s=40)
    #label wells
    for label, x, y in zip(wells_with_data, T[:,0], T[:,1]):
        ax.annotate(label, xy=(x, y),color='b',size='12')
    
    ax.grid(True)
    ax.set_xlabel('Easting, mE')
    ax.set_ylabel('Northing, mN')
    
    fig.tight_layout()
    fig.savefig('contourT_'+str(z)+'_'+method+'.png',dpi=300)
    plt.close(fig)


zlist = [0,-250,-500]
for z in zlist:
    plot_contour(z,method = 'multiquadric')

zlist=[-250]

for z in zlist:
    plot_contour(z,method = 'multiquadric')
    plot_contour(z,method = 'gaussian')
    plot_contour(z,method = 'linear')
    plot_contour(z,method = 'cubic')