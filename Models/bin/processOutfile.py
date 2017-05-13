# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 02:52:52 2017

@author: Yam
"""

#parse massflow
import pandas as pd
import numpy as np

def process_outfile(run_dir, outfile):
    
    times=[]
    pos=[]
    co2_sim=False
    
    with open(run_dir+outfile,'r') as fo:
        line=fo.readline()
        while line!='':
            line=fo.readline()
            if '           Years              Days         Step Size (Days)' in line:
                year,day,dt=fo.readline().strip().split()
                times+=[day]
                pos+=[fo.tell()]
            if 'water-CO2 problem' in line:
                co2_sim=True
        
        df = pd.DataFrame(columns=['time','Node','mf_water','mf_co2'])
        for i,p in enumerate(pos):
            water_inf=[]
            co2_inf=[]
            fo.seek(p)
            line=' '
            while 'Node P(MPa)' not in line and 'Node   P (MPa)' not in line:
                line=fo.readline()
            while line!='\n':
                line = fo.readline()
                water_inf+=[line.strip().split()]
                if line=='\n': water_inf.pop(-1)
            if co2_sim:
                while 'Node Pco2(MPa)' not in line:
                    line=fo.readline()
                while line!='\n':
                    line = fo.readline()
                    co2_inf+=[line.strip().split()]
                    if line=='\n': co2_inf.pop(-1)
            water_df=pd.DataFrame(water_inf)
            co2_df=pd.DataFrame(co2_inf)
            if co2_sim:
                water_df=water_df.loc[:,[0,6]]
                co2_df=co2_df.loc[:,[0,5]]
                co2_df.columns=['Node','mf_co2']
            else:
                water_df=water_df.loc[:,[0,5]]
            water_df.columns=['Node','mf_water']
            temp_df=pd.DataFrame()
            temp_df['Node']=water_df['Node'].astype(int)
            temp_df['time']=[float(times[i])]*temp_df.shape[0]
            temp_df['mf_water']=water_df['mf_water'].astype(float)
            if co2_sim:
                temp_df['mf_co2']=co2_df['mf_co2'].astype(float)
            else:temp_df['mf_co2']=np.nan
            df=df.append(temp_df)
            
    df=df.reset_index(drop=True)
    
    print 'output written to ' + run_dir + 'massflow_his.csv'
    df.to_csv(run_dir+'massflow_his.csv',index=False)
    
    return df
