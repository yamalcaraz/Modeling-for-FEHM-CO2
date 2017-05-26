# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 01:43:50 2017

@author: Yam
"""
import sys,os
bin_path = os.path.abspath(os.path.join('..', 'bin'))
sys.path.append(bin_path)

from wellboreModel import bottom_up
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from CoolProp.CoolProp import PropsSI, PhaseSI
import argparse
from matplotlib.ticker import AutoMinorLocator
from fdata import fcontour


def get_well_top(well):
    dirs=pd.read_excel(work_dir+"\\ModelData.xlsx",sheetname='Dir')
    dirs=dirs.set_index('Variable').T
    data_dir = dirs['data_dir'].values[0]
    wells_file = os.path.abspath(work_dir + '\\' + data_dir + "MGPF Deviation Survey and Well Data1.xlsx")
    w_DS=pd.read_excel(wells_file,sheetname=well)
    w_DS = w_DS.loc[:,['Easting','Northing','MRSL']]
    
    w_top = w_DS['MRSL'].max()
    
    return w_top

def get_well_names(work_dir):
    model_data = os.path.abspath(work_dir+"\\ModelData.xlsx")
    hist=pd.read_excel(model_data,sheetname='hist')
    col_names = hist['Name'].values.tolist()
    
    return col_names

def get_data(fluid, work_dir):
    #files
    model_name=work_dir.split('\\')[-1]
    
   
    
    param='presWAT'
    
    s1=pd.read_csv(work_dir + '\\{}_Stage1\\{}_{}_his.csv'.format(fluid,model_name,param))
    s2=pd.read_csv(work_dir + '\\{}_Stage2\\{}_{}_his.csv'.format(fluid,model_name,param))
    pres_df = s1.append(s2)
    
    #remove last row if error
    if pres_df.iloc[-1][0] == 1.0:
        pres_df = pres_df.iloc[:-1]
    
    pres_df['Time (days)']=pres_df['Time (days)'].round(5)
    pres_df.sort_values('Time (days)',inplace=True)
    pres_df.drop_duplicates(inplace=True)
    pres_df.set_index('Time (days)',inplace=True)
    pres_df.columns=[int(i.split()[-1]) for i in pres_df.columns.tolist()]

    
    param='temp'
    
    s1=pd.read_csv(work_dir + '\\{}_Stage1\\{}_{}_his.csv'.format(fluid,model_name,param))
    s2=pd.read_csv(work_dir + '\\{}_Stage2\\{}_{}_his.csv'.format(fluid,model_name,param))
    temp_df = s1.append(s2)
    
    if temp_df.iloc[-1][0] == 1.0:
        temp_df = temp_df.iloc[:-1]
    
    temp_df['Time (days)']=temp_df['Time (days)'].round(5)
    temp_df.sort_values('Time (days)',inplace=True)
    temp_df.drop_duplicates(inplace=True)
    temp_df.set_index('Time (days)',inplace=True)
    temp_df.columns=[int(i.split()[-1]) for i in temp_df.columns.tolist()]
    
    s1=pd.read_csv(work_dir + '\\{}_Stage1\\massflow_his.csv'.format(fluid,model_name))
    s2=pd.read_csv(work_dir + '\\{}_Stage2\\massflow_his.csv'.format(fluid,model_name))
    mass_df=s1.append(s2)
    mass_df=mass_df.astype({'Node':int})
    mass_df['time']=mass_df['time'].round(5)
    
    return pres_df, temp_df, mass_df

def get_heat_extraction_rate(work_dir, fluid,inj_temp=30):
    pres_df, temp_df, mass_df = get_data(fluid, work_dir)
    
    times=mass_df.time.unique().tolist()
    columns=temp_df.columns.tolist()
    wnames=get_well_names(work_dir)
    wname_map={k:wnames[i] for i,k in enumerate(columns)}
    
    h_df=pd.DataFrame(columns=columns,index=times)
    state_df=pd.DataFrame(columns=columns,index=times)
    
    for t in times:
        print "DEBUG: time is " + str(t)
        for col in columns:
            
            if fluid=='Water':
                mf = mass_df[(mass_df['Node']==col) & (mass_df['time']==t)]['mf_water'].values[0]
            elif fluid=='CO2':
                mf = mass_df[(mass_df['Node']==col) & (mass_df['time']==t)]['mf_co2'].values[0]
            
            if mf>0.0: 
                T=temp_df.loc[t,col] + 273.15
            elif mf<0.0:                
                T=inj_temp + 273.15
                inj_well = col
                
            P=pres_df.loc[t,col]*1E6
            h_df.loc[t,col]=PropsSI('H','T',T,'P',P,fluid)/1E3
            state_df.loc[t,col]=PhaseSI('T',T,'P',P,fluid)
        
    prod_wells = [col for col in columns if col!=inj_well]
    q_df=pd.DataFrame(columns=prod_wells,index=times)
    
    for t in times:
        print "DEBUG: time is " + str(t)
        for col in prod_wells:
            if fluid=='Water':
                mf = mass_df[(mass_df['Node']==col) & (mass_df['time']==t)]['mf_water'].values[0]
            elif fluid=='CO2':
                mf = mass_df[(mass_df['Node']==col) & (mass_df['time']==t)]['mf_co2'].values[0]
                
            q_df.loc[t,col] = mf * (h_df.loc[t,col] - h_df.loc[t,inj_well])
    
    q_df['Total']=q_df.sum(axis=1)
    q_df.rename(columns=wname_map,inplace=True)
    h_df.rename(columns=wname_map,inplace=True)
    state_df.rename(columns=wname_map,inplace=True)
    
    return q_df, h_df, state_df
            

def get_viscosity(work_dir, fluid,inj_temp=30):
    pres_df, temp_df, mass_df = get_data(fluid, work_dir)
    
    times=mass_df.time.unique().tolist()
    columns=temp_df.columns.tolist()
    wnames=get_well_names(work_dir)
    wname_map={k:wnames[i] for i,k in enumerate(columns)}
    
    v_df=pd.DataFrame(columns=columns,index=times)
    
    for t in times:
        print "DEBUG: time is " + str(t)
        for col in columns:
            
            if fluid=='Water':
                mf = mass_df[(mass_df['Node']==col) & (mass_df['time']==t)]['mf_water'].values[0]
            elif fluid=='CO2':
                mf = mass_df[(mass_df['Node']==col) & (mass_df['time']==t)]['mf_co2'].values[0]
            
            
            if mf>0.0: 
                T=temp_df.loc[t,col] + 273.15
            elif mf<0.0:                
                T=inj_temp + 273.15
            elif mf==0:
                continue
                
            P=pres_df.loc[t,col]*1E6
            v_df.loc[t,col]=PropsSI('V','T',T,'P',P,fluid)
    
    
    v_df.rename(columns=wname_map,inplace=True)
    
    return v_df

def get_whp(work_dir, fluid,z_bot,z_top,dz,inj_temp=30,plot_times=[],well_diameter=0.22,roughness=55E-6):
    
    pres_df, temp_df, mass_df = get_data('CO2', work_dir) #force to get data from CO2
    
    times=mass_df.time.unique().tolist()
    columns=temp_df.columns.tolist()
    whp_df=pd.DataFrame(columns=columns,index=times)
    wnames=get_well_names(work_dir)
    wname_map={k:wnames[i] for i,k in enumerate(columns)}
    
    results=[]
    if len(plot_times)>0: times = plot_times
    
    for t in times:
        print "DEBUG: time is " + str(t)
        result_dict={}
        for col in columns:
            if fluid=='Water':
                mf = mass_df[(mass_df['Node']==col) & (mass_df['time']==t)]['mf_co2'].values[0] #still get values from co2
            elif fluid=='CO2':
                mf = mass_df[(mass_df['Node']==col) & (mass_df['time']==t)]['mf_co2'].values[0]
            
            if mf>0.0: 
                pres = pres_df.loc[t,col]*1E6
                temp = temp_df.loc[t,col]
                result=bottom_up(fluid,z_bot,z_top,dz,mf,pres,temp,well_diameter,roughness,9.81)
            elif mf<0.0:
                temp = inj_temp
                pres = pres_df.loc[t,col]*1E6
                result=bottom_up(fluid,z_bot,z_top,dz,mf,pres,temp,well_diameter,roughness,9.81)
            else:
                whp_df.loc[t,col]=np.nan
                continue
            
            if result.iloc[-1]['z']<z_top:
                whp_df.loc[t,col]=0.0
            else: whp_df.loc[t,col]=result.iloc[-1]['P'] #get the last point, i.e. wellhead, pressure
            
            if t in plot_times:
                result_dict[col]=result
        if t in plot_times:
            results+=[result_dict]
    
    
    whp_df.rename(columns=wname_map,inplace=True)
    for r in results:
        for k in r.keys():
            new_k = wname_map[k]
            r[new_k]=r.pop(k)
    
    return whp_df,results
            

def plot_results(time,fluid,ax):
    #for marbel, 1500 to 750, dz=35
    #for matingao, 1250 to 0, dz= 30
    params=get_well_params(work_dir)
    _,results=get_whp(work_dir,fluid,params['min_elev'],params['max_elev'],
                      params['dz'],plot_times=[time],
                      inj_temp=params['inj_temp'],
                       well_diameter=params['well_diameter'],
                       roughness=params['roughness'])
    #_,co2_results=get_whp('CO2',-1000,0,10,plot_times=[733.0])
    
    
    result_dict = results[0]
    
    k=result_dict.keys()
    k.sort()
    for i,node in enumerate(k):
        if fluid=='CO2':
            ax.plot(result_dict[node]['P']/1E6,result_dict[node]['z'], 
                    label = fluid + ' ' + str(node), color=colors[i], 
                    ls=linestyles[0])
        if fluid=='Water':
            ax.plot(result_dict[node]['P']/1E6,result_dict[node]['z'], 
                    label = fluid + ' ' + str(node), color=colors[i], 
                    ls=linestyles[1])
    #    ax.plot(water_result_dict[node][:,1]/1E6,water_result_dict[node][:,0], label = 'Water ' + str(node))


def plot_wellsim(time):
    fig, ax = plt.subplots()  
    plot_results(time,'CO2',ax)
    plot_results(time,'Water',ax)
    ax.legend()
    ax.set_xlabel('Pressure (MPa)')
    ax.set_xlim(0,10)
    ax.set_ylabel('Elevation (mRSL)')
    ax.grid(True)
    
    fig=ax.get_figure()
    fig.set_size_inches((7,5))
    fig.savefig('wellsim_at_t{}.png'.format(time))

def plot_whp(fluid):
    #for marbel, 1500 to 750, dz=35
    #for matingao, 1250 to 0, dz= 30
    fig, ax = plt.subplots()
    
    if len(args.multi)>0:
        model_dirs = [work_dir + '\\' + m for m in args.multi]
    else:
        model_dirs = [work_dir]
        
    for i,model_dir in enumerate(model_dirs):
        model_name = model_dir.split('\\')[-1]
        params=get_well_params(model_dir)
    
        
        
        if args.plotwater:
            fluid2 = ['CO2','Water']
            for i2, f2 in enumerate(fluid2):
                whp_df,_ = get_whp(model_dir, f2,params['min_elev'],params['max_elev'],
                               params['dz'],inj_temp=params['inj_temp'],
                               well_diameter=params['well_diameter'],
                               roughness=params['roughness'])
                whp_df = whp_df/1E6 #in MPa
                whp_df.columns = [model_name+'_'+c+'_'+f2 for c in whp_df.columns]
                whp_df.plot(ax=ax,ls=linestyles[i2],color=colors)       
        else:
            whp_df,_ = get_whp(model_dir, fluid,params['min_elev'],params['max_elev'],
                           params['dz'],inj_temp=params['inj_temp'],
                           well_diameter=params['well_diameter'],
                           roughness=params['roughness'])
            whp_df = whp_df/1E6 #in MPa
            whp_df.columns = [model_name+'_'+c for c in whp_df.columns]
            whp_df.plot(ax=ax,ls=linestyles[i],color=colors)
        
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Pressure (MPa)')
    ax.set_xlim(0,5000)
    ax.grid(True)
    minorLocator=AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.legend(prop={'size':8})
    
    fig=ax.get_figure()
    fig.set_size_inches((7,5))
    fig.savefig(args.figname+fluid+'_'+'whp_history.png')

def plot_q(fluid):
    fig, ax = plt.subplots()
    
    if len(args.multi)>0:
        model_dirs = [work_dir + '\\' + m for m in args.multi]
    else:
        model_dirs = [work_dir]
    
    for i,model_dir in enumerate(model_dirs):
        model_name = model_dir.split('\\')[-1]
    
        params=get_well_params(model_dir)
        
        
        if args.plotwater:
            fluid2 = ['CO2','Water']
            for i2, f2 in enumerate(fluid2):
                q_df, _, _ = get_heat_extraction_rate(model_dir,f2,params['inj_temp'])
                q_df.columns = [model_name+'_'+c+'_'+f2 for c in q_df.columns]
                q_df.plot(ax=ax,ls=linestyles[i2],color=colors)
        else:
            q_df, _, _ = get_heat_extraction_rate(model_dir,fluid,params['inj_temp'])
            q_df.columns = [model_name+'_'+c for c in q_df.columns]
            q_df.plot(ax=ax,ls=linestyles[i],color=colors)
            
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Heat Extraction Rate (kW)')
    ax.set_xlim(0,5000)
    ax.legend(loc='best')
    ax.grid(True)
    minorLocator=AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.legend(prop={'size':8})
    
    fig=ax.get_figure()
    fig.set_size_inches((7,5))
    fig.savefig(args.figname+fluid+'_'+'q_history.png')
    
    
def plot_visc(fluid):
    fig, ax = plt.subplots()
    
    if len(args.multi)>0:
        model_dirs = [work_dir + '\\' + m for m in args.multi]
    else:
        model_dirs = [work_dir]
    
    for i,model_dir in enumerate(model_dirs):
        model_name = model_dir.split('\\')[-1]
    
        params=get_well_params(model_dir)
        v_df = get_viscosity(model_dir,fluid,params['inj_temp'])
        v_df.columns = [model_name+'_'+c for c in v_df.columns]
        v_df.plot(ax=ax,ls=linestyles[i],color=colors)
            
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Viscosity (Pa-s)')
    ax.set_xlim(0,5000)
    ax.legend(loc='best')
    ax.grid(True)
    minorLocator=AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.legend(prop={'size':8})
    
    fig=ax.get_figure()
    fig.set_size_inches((7,5))
    fig.savefig(args.figname+fluid+'_'+'visc_history.png')    
    
def cont_hist(work_dir,fluid,stage):
    model_name=work_dir.split('\\')[-1]
    cont=fcontour(work_dir+'\\{}_Stage{}\\{}.*_days_sca_node.csv'.format(fluid,stage,model_name))
    
    return cont
    
    
def extract_node_hist(node, cont):
    n_dict = cont.node(node-1)
    n_df=pd.DataFrame(n_dict)
    n_df['time'] = cont.times  
    
    
    return n_df

def get_well_params(work_dir):
    model_data = os.path.abspath(work_dir+"\\ModelData.xlsx")
    params=pd.read_excel(model_data,sheetname='Well_Params')
    params=params.set_index('Variable').T
    
    params_dict={}
    for col in params.columns.tolist():
        params_dict[col]=params[col].values[0]
    return params_dict

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process WHP and Qrate')
    parser.add_argument('-q', dest= 'qplot', action='store_true',
                       help='plot q hist')
    parser.add_argument('-whp', dest= 'whpplot', action='store_true',
                       help='plot whp hist')
    parser.add_argument('-v', dest= 'viscplot', action='store_true',
                       help='plot viscosity hist')
    parser.add_argument('-ws', dest= 'wsplot', action='store_true',
                       help='plot well simulation')
    parser.add_argument('-m', dest= 'multi', type=str, nargs='+',default=[],
                       help='create multi q and whp plots')
    parser.add_argument('-fn', dest= 'figname', type=str, default='',
                       help='name of plot' )
    parser.add_argument('-tws', dest= 'timews', type=float, default=3652.5,
                       help='time for ws plot' )
    parser.add_argument('-pw', dest= 'plotwater', action='store_true',
                       help='plot water in whp simulation')
    
    args = parser.parse_args()
    
    work_dir=os.getcwd()
    linestyles = ['-',(0,[5,1]),(0,[1,1])]
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    
    if args.qplot:
        plot_q('CO2')
    if args.whpplot:
        plot_whp('CO2')
    if args.viscplot:
        plot_visc('CO2')
    if args.wsplot:
        plot_wellsim(args.timews)
    