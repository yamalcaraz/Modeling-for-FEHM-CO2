# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:28:00 2017

@author: Yam
"""

#process NS
from fpost import fcontour
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, glob
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from matplotlib.ticker import AutoMinorLocator

def read_contours(model_names=[],multi=False):
    cont_list=[]
    if multi:
        for name in model_names:
            cont_list += [fcontour(models_dir+'\\'+name+'\\NS\\'+name+'.*_days_sca_node.csv',latest=True)]
    else:
        cont_list= [fcontour(ns_dir+'\\'+model_names[0]+'.*_days_sca_node.csv',latest=True)]
#        cont_list= [fcontour(ns_dir+'\\'+model_names[0]+'.0.*_days_sca_node.csv')]
        
    return cont_list

def plot_simulated(ax, zs, fs, color, model_name):
    ax.plot(fs,zs,color=color,marker='o',linestyle='-',label=model_name)
    
def plot_measured(ax, zd, fd):
    ax.plot(fd,zd,color='red',marker='^',linestyle='',label='Measured')
    
def setup_plot(ax, well):
    ax.legend()
    ax.set_title(well)
    ax.set_xlabel('Temperature (degC)')
    ax.set_ylabel('Elevation (mRSL)')
    ax.set_xlim(0,350)

def create_NS_plots(cont_list,model_names):

    #get NS Temp
    ns_temp_df=pd.read_excel(ns_temp_file,sheetname=None,parse_cols='A:B',names=['T','MRSL'])
    
    pdf= PdfPages(work_dir+'\\NSresults.pdf')
    
    well_chunks = [wells[i:i+4] for i in range(0, len(wells), 4)]
    
    for chunk in well_chunks:
        fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(10,7))
        ax=ax.flatten()
        for i,w in enumerate(chunk):
            plot_measured(ax[i],ns_temp_df[w]['MRSL'],ns_temp_df[w]['T'])
            
            w_DS = wells_DS[w].loc[:,['Easting','Northing','MRSL']]
            w_DS=w_DS.sort_values('MRSL')
            #get only DS on cont.z values
            for ci,cont in enumerate(cont_list):
                z=cont.z
                x=np.interp(z,w_DS['MRSL'],w_DS['Easting'])
                y=np.interp(z,w_DS['MRSL'],w_DS['Northing'])
                coords = np.array((x,y,z)).T
                prof=cont.profile('T',coords)
                
                plot_simulated(ax[i],prof[:,2],prof[:,3],colors[ci],model_names[ci])
            setup_plot(ax[i],w)
            
        fig.tight_layout()
        pdf.savefig(fig) 
        plt.close(fig)
    pdf.close()
    
def create_pcp_plot(cont_list,model_names):
    #get pcp data
    pcp_df=pd.read_excel(pcp_data,parse_cols='A:C')
    
    pcp_df['Well'] = pcp_df.Well.apply(lambda x:x.replace('-',''))
    
    fig, ax = plt.subplots(figsize=(10,7)) 
    
    for w in wells:
        
        if w not in pcp_df['Well'].tolist(): continue
        
        #plot measured pcp first
        pcp_well = pcp_df[pcp_df['Well']==w]
        pcp_depth = pcp_well['Depth, mRSL'].values[0]
        pcp_meas = pcp_well['Pressure (stable), MPag'].values[0]
        meas_plot=ax.plot(pcp_meas,pcp_depth,'r^',label='Measured')
        ax.annotate(w, xy=(pcp_meas,pcp_depth),color='r',size='8')
        sim_plots=[]
        for ci, cont in enumerate(cont_list):
            w_DS = wells_DS[w].loc[:,['Easting','Northing','MRSL']]
            w_DS=w_DS.sort_values('MRSL')
            #get only DS on cont.z values
            z=cont.z
            x=np.interp(z,w_DS['MRSL'],w_DS['Easting'])
            y=np.interp(z,w_DS['MRSL'],w_DS['Northing'])
            coords = np.array((x,y,z)).T
            prof=cont.profile('P',coords)
            
            pcp_sim = np.interp(pcp_well['Depth, mRSL'],prof[:,2],prof[:,3])[0]
        
            sim_plots+=ax.plot(pcp_sim,pcp_depth,'bo',color=colors[ci],label=model_names[ci])
            if len(cont_list)==1:
                ax.annotate(w, xy=(pcp_sim,pcp_depth),color=colors[ci],size='8')
    
    ax.legend(meas_plot + sim_plots,['Measured']+model_names, loc = 'best')
    ax.set_title('PCP Plots')
    ax.set_xlabel('Pressure (MPag)')
    ax.set_ylabel('Elevation (mRSL)')
    
    fig.savefig('PCP_Plot.pdf')
    
def launch_paraview(run_dir):
    #post-processing
    delete_temp_files()
    cont=fcontour(run_dir+'\\'+model_name+'.*_days_sca_node.csv')
    
    #launch paraview
    cont.paraview(grid_3D_path, exe=r"D:\ParaView 5.0.0\bin\paraview.exe",time_derivatives=False)
    
    
def delete_temp_files():
    for filename in glob.glob(work_dir+"/temp*"):
        print 'deleting ', filename
        os.remove(filename)
        
def plot_co2():
    co2s1=pd.read_csv(work_dir + '\\CO2_Stage1\\{}_co2mt_his.csv'.format(model_name))
    co2s2=pd.read_csv(work_dir + '\\CO2_Stage2\\{}_co2mt_his.csv'.format(model_name))
    co2_df = co2s1.append(co2s2)
    
    #remove last row if error
    if co2_df.iloc[-1][0] == 1.0:
        co2_df = co2_df.iloc[:-1]
    
    co2_df.sort_values('Time (days)',inplace=True)
    co2_df.set_index('Time (days)',inplace=True)
    ax = co2_df.plot(marker='o', ls='')
    ax.set_ylabel('CO2 mass (kg)')
    ax.set_xlim([0,365.25*10])
    ax.grid(True)
    minorLocator=AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minorLocator)
    
    fig = ax.get_figure()
    fig.set_size_inches((7,5))
    fig.savefig('co2 mass history.png')
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process NS model results')
    parser.add_argument('-t', dest= 'tplots', action='store_true',
                       help='create temp NS plots')
    parser.add_argument('-m', dest= 'multi', type=str, nargs='+',default=[],
                       help='create multi temp NS and pcp plots')
    parser.add_argument('-p', dest= 'pplots', action='store_true',
                       help='create pcp NS plots')
    parser.add_argument('-pv', dest= 'paraview', action='store_true',
                       help='run paraview ns')
    parser.add_argument('-pvp', dest= 'paraview_preNS', action='store_true',
                       help='run paraview prens')
    parser.add_argument('-pvs1', dest= 'paraview_stage1', action='store_true',
                       help='run paraview stage1')
    parser.add_argument('-co2', dest= 'co2', action='store_true',
                       help='process co2 run results')
    
    args = parser.parse_args()
    
    work_dir=os.getcwd()
    
    if not args.multi:

        dirs=pd.read_excel(work_dir+"\\ModelData.xlsx",sheetname='Dir')
        dirs=dirs.set_index('Variable').T
        
        exe = dirs['exe'].values[0]
        
        #files
        model_name=work_dir.split('\\')[-1]
        model_data = os.path.abspath(work_dir+"\\ModelData.xlsx")
        grid_3D_path = dirs['grid_3D_path'].values[0]
        
        #files
        ns_dir = work_dir+'\\NS'
        preNS_dir = work_dir+'\\Pre_NS'
        s1_dir = work_dir+'\\CO2_Stage1'
        data_dir = dirs['data_dir'].values[0]
     
    else:
        data_dir = os.path.abspath(work_dir+'\\..\\Data') + '\\'
        models_dir = work_dir
        
    wells_file = data_dir + "MGPF Deviation Survey and Well Data1.xlsx"
    ns_temp_file = data_dir + "MAGBU New Temperature Interpretation.xlsx"
    well_list = data_dir + "ModelWellList.xlsx"
    pcp_data = data_dir + 'PCP Plot.xls'
    
    #get the well list
    wells = pd.read_excel(well_list)
    wells = wells.values.flatten().astype(str)
    wells = wells[wells!='nan']
    
    #get DS
    wells_DS=pd.read_excel(wells_file,sheetname=None)
    
    
    colors = ['blue','green','magenta','cyan']
    
    if len(args.multi)>0:
        cont_list = read_contours(model_names=args.multi,multi=True)
        create_NS_plots(cont_list,model_names=args.multi)
        create_pcp_plot(cont_list,model_names=args.multi)
    else:
        cont_list = read_contours([model_name],multi=False)
        
    if args.tplots:
        create_NS_plots(cont_list,[model_name])
    if args.pplots:
        create_pcp_plot(cont_list,[model_name])
    if args.co2:
        plot_co2()
    if args.paraview:
        launch_paraview(ns_dir)
    if args.paraview_preNS:
        launch_paraview(preNS_dir)
    if args.paraview_stage1:
        launch_paraview(s1_dir)
