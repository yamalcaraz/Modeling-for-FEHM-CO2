# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:28:00 2017

@author: Yam
"""

#process NS
from fpost import fcontour
from fgrid import fgrid
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
#            print '**********' + models_dir
            cont_list += [fcontour(models_dir+'\\'+name+'\\NS\\'+name+'.*_days_sca_node.csv',latest=True)]
    else:
        cont_list= [fcontour(ns_dir+'\\'+model_names[0]+'.*_days_sca_node.csv',latest=True)]
#        cont_list= [fcontour(ns_dir+'\\'+model_names[0]+'.0.*_days_sca_node.csv')]
        
    return cont_list

def plot_simulated(ax, zs, fs, color, model_name, ls='-', lw=2):
    ax.plot(fs,zs,color=color,marker='o',linestyle=ls,lw=lw, ms=4,label=model_name)
    
def plot_measured(ax, zd, fd):
    ax.plot(fd,zd,color='red',marker='^',linestyle='',label='Measured')
    
def setup_plot(ax, well, xmin=0, xmax=350):
    ax.legend(prop={'size':8},loc='best')
    ax.set_title(well)
    ax.set_xlabel('Temperature (degC)')
    ax.set_ylabel('Elevation (mRSL)')
    ax.set_xlim(xmin,xmax)

def setup_kplot(ax, well, xmin=np.log10(1E-18), xmax=np.log10(1E-11)):
    ax.legend(prop={'size':8},loc='best')
    ax.set_title(well)
    ax.set_xlabel('Permeability ($log_{10}m^2$)')
    ax.set_ylabel('Elevation (mRSL)')
    ax.set_xlim(xmin,xmax)

def create_NS_plots(cont_list,model_names,savetopdf=True, basemodel='', threshold=30):

    #get NS Temp
    ns_temp_df=pd.read_excel(ns_temp_file,sheetname=None,parse_cols='A:B',names=['T','MRSL'])
    
    if savetopdf:
        pdf= PdfPages(work_dir+'\\'+args.figname+'NSresults.pdf')
        if args.plot_k:
            kpdf= PdfPages(work_dir+'\\'+args.figname+'Kprofile.pdf')
    
    well_chunks = [wells[i:i+4] for i in range(0, len(wells), 4)]
    
    profiles = {}
    kprofiles = {}
    
    for i_chunk, chunk in enumerate(well_chunks):
        fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(10,7))
        ax=ax.flatten()
        if args.plot_k:
            k_fig,k_ax=plt.subplots(nrows=2,ncols=2,figsize=(10,7))
            k_ax=k_ax.flatten()
        
        
        for i,w in enumerate(chunk):
            plot_measured(ax[i],ns_temp_df[w]['MRSL'],ns_temp_df[w]['T'])
            
            w_DS = wells_DS[w].loc[:,['Easting','Northing','MRSL']]
            w_DS=w_DS.sort_values('MRSL')
            #get only DS on cont.z values
            profiles[w]={}
            for ci,cont in enumerate(cont_list):
                z=cont.z
                x=np.interp(z,w_DS['MRSL'],w_DS['Easting'])
                y=np.interp(z,w_DS['MRSL'],w_DS['Northing'])
                coords = np.array((x,y,z)).T
                prof=cont.profile('T',coords)
                profiles[w][model_names[ci]]=prof
                plot_simulated(ax[i],prof[:,2],prof[:,3],colors[ci],model_names[ci])
                if args.plot_k:
                    kx_prof = cont.profile('perm_x',coords)
                    kz_prof = cont.profile('perm_z',coords)
                    plot_simulated(k_ax[i],kx_prof[:,2],kx_prof[:,3],colors[ci],model_names[ci] + '_kx', ls = '-',lw=1)
                    plot_simulated(k_ax[i],kz_prof[:,2],kz_prof[:,3],colors[ci],model_names[ci]+ '_kz', ls = '--',lw=1)
                if model_names[ci] == basemodel:
                    base_elev = prof[:,2]
                    base_temp = prof[:,3]
            
            if len(basemodel)>0 and threshold>0.:
                ax[i].plot(base_temp+threshold,base_elev,'k--', lw=1)
                ax[i].plot(base_temp-threshold,base_elev,'k--', lw=1, label = '{}$^\circ$C threshold'.format(threshold))
                
            setup_plot(ax[i],w)
            if args.plot_k:
                setup_kplot(k_ax[i],w)
        fig.tight_layout()
        if args.plot_k:
            k_fig.tight_layout()
        if savetopdf:
            pdf.savefig(fig)
            if args.plot_k:
                kpdf.savefig(k_fig)
        else:
            fig.savefig(args.figname + 'NSresults_{}.png'.format(i_chunk))
            if args.plot_k:
                k_fig.savefig(args.figname + 'Kprofile_{}.png'.format(i_chunk))
        plt.close(fig)
    if savetopdf:
        pdf.close()
        if args.plot_k:
            kpdf.close()
    
    if args.sens:
        plot_sensitivity(profiles,args.base_model,threshold,savetopdf)
        
    if args.rsquared:
        compute_rsquared(profiles,args.base_model)
        
def plot_sensitivity(profiles,base_model,threshold,savetopdf,temp_bounds=170):
    
    well_chunks = [wells[i:i+4] for i in range(0, len(wells), 4)]
    
    if savetopdf:
        pdf= PdfPages(work_dir+'\\'+args.figname+'NSsens.pdf')
    
    for i_chunk, chunk in enumerate(well_chunks):
        fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(10,7))
        ax=ax.flatten()
        for i,w in enumerate(chunk):
            prof_dict = profiles[w]
            models = [j for j in prof_dict.keys() if j!=base_model]
            prof_df = pd.DataFrame(index = prof_dict[base_model][:,2],data= {base_model:prof_dict[base_model][:,3]})
            sens_df = pd.DataFrame(index = prof_dict[base_model][:,2])
            for mi, m in enumerate(models):
                df_m=pd.DataFrame(index = prof_dict[m][:,2],data= {m:prof_dict[m][:,3]})
                prof_df = prof_df.join(df_m)
                sens_df[m] = prof_df[base_model]-prof_df[m]
                plot_simulated(ax[i],sens_df.index.values,sens_df[m].values,colors[mi],m)
            setup_plot(ax[i],w,-temp_bounds,temp_bounds)
        
            ax[i].plot([0,0],[sens_df.index.values[0],sens_df.index.values[-1]],'k--')
            if threshold>0.:
                ax[i].plot([threshold,threshold],[sens_df.index.values[0],sens_df.index.values[-1]],'r--')
                ax[i].plot([-threshold,-threshold],[sens_df.index.values[0],sens_df.index.values[-1]],'r--')
        
        fig.tight_layout()
        if savetopdf:
            pdf.savefig(fig)
        else:
            fig.savefig(args.figname + 'NSsens_{}.png'.format(i_chunk))
        plt.close(fig)
        
    if savetopdf:
        pdf.close()

def compute_rsquared(profiles,base_model):
    
    rsquared_dict={}
    num_points={}
    
    for i,w in enumerate(wells):
        prof_dict = profiles[w]
        models = [j for j in prof_dict.keys() if j!=base_model]
        prof_df = pd.DataFrame(index = prof_dict[base_model][:,2],data= {base_model:prof_dict[base_model][:,3]})
        sens_df = pd.DataFrame(index = prof_dict[base_model][:,2])
        for mi, m in enumerate(models):
            df_m=pd.DataFrame(index = prof_dict[m][:,2],data= {m:prof_dict[m][:,3]})
            prof_df = prof_df.join(df_m)
            sens_df[m] = prof_df[base_model]-prof_df[m]
        rsquared_dict[w] = (sens_df**2).sum()
        num_points[w] = sens_df.shape[0]
        
    n=sum([num_points[k] for k in num_points.keys()])
    rsquared=pd.DataFrame(rsquared_dict).T.sum()
    std = np.sqrt(rsquared/n)    
    pd.DataFrame({'rsquared':rsquared,'std':std}).to_excel(args.figname+'rsquared.xlsx')
    
def compute_rsquared_pcp(profiles,base_model):
    
    profiles = {k:profiles[k] for k in profiles.keys() if profiles[k]}
    models = [j for j in profiles[wells[0]].keys()  if j!=base_model]
    rsquared_dict={m:[] for m in models}

    
    for i,w in enumerate(profiles.keys()):
        print w
        prof_dict = profiles[w]
        base_p = prof_dict[base_model][0]
        for mi, m in enumerate(models):
            other_p=prof_dict[m][0]
            rsquared_dict[m] += [(base_p-other_p)**2]
    
    rsquared=pd.DataFrame(rsquared_dict).sum()
    n=rsquared.shape[0]
    std = np.sqrt(rsquared/n)    
    pd.DataFrame({'rsquared':rsquared,'std':std}).to_excel(args.figname+'rsquaredPCP.xlsx')
    
    
def create_pcp_plot(cont_list,model_names, savetopdf=True):
    #get pcp data
    pcp_df=pd.read_excel(pcp_data,parse_cols='A:C')
    
    pcp_df['Well'] = pcp_df.Well.apply(lambda x:x.replace('-',''))
    
    fig, ax = plt.subplots(figsize=(10,7)) 
    
    profiles={}
    
    for w in wells:
        profiles[w]= {}
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
            profiles[w][model_names[ci]]=(pcp_sim,pcp_depth)
            sim_plots+=ax.plot(pcp_sim,pcp_depth,'bo',color=colors[ci],label=model_names[ci])
#            if len(cont_list)==1:
#                ax.annotate(w, xy=(pcp_sim,pcp_depth),color=colors[ci],size='8')
    
    ax.legend(meas_plot + sim_plots,['Measured']+model_names, loc = 'best',prop={'size':8})
    ax.set_title('PCP Plots')
    ax.set_xlabel('Pressure (MPag)')
    ax.set_ylabel('Elevation (mRSL)')
    
    if savetopdf:
        fig.savefig(args.figname + 'PCP_Plot.pdf')
    else:
        fig.savefig(args.figname + 'PCP_Plot.png')
        
    if args.sens:
        create_pcp_plot_sa(profiles,model_names,args.base_model)
    
    if args.rsquared:
        compute_rsquared_pcp(profiles,args.base_model)
        
def create_pcp_plot_sa(profiles,model_names,base_model):
    
    print 'plotting Pressure SA'
    fig, ax = plt.subplots(figsize=(7,7))
#    markers = ['o','+','^','*']
    
    model_names = [j for j in model_names if j!=base_model]
    for iw,w in enumerate(wells):
#        print w
        if base_model in profiles[w].keys():
            base_p = profiles[w][base_model][0]
        else:
            continue
        sim_plots = []
        for im,m in enumerate(model_names):
            sim_p = profiles[w][m][0]
            sim_plots += ax.plot(base_p,sim_p,c=colors[im],marker='o',ms=4, label=m)
    
    baseplot=ax.plot([0,30],[0,30],'k--',lw=2)
    thres=0.5
    thresplot=ax.plot([0+thres,30+thres],[0,30],'r--',lw=1)
    ax.plot([0-thres,30-thres],[0,30],'r--',lw=1)
    
    ax.legend(sim_plots+baseplot+thresplot,
              model_names+[base_model]+['{} MPa deviation'.format(thres)],
              loc = 'best',prop={'size':8})
    ax.set_title('PCP Plots')
    ax.set_xlabel(base_model + ' Pressure (MPag)')
    ax.set_ylabel('Simulated Pressure (MPag)')
    ax.set_xlim(2,14)
    ax.set_ylim(2,14)
    
    fig.savefig(args.figname + 'PCP_Plot_sa.png')
    
def launch_paraview(run_dir,td):
    #post-processing
    delete_temp_files()
    cont=fcontour(run_dir+'\\'+model_name+'.*_days_sca_node.csv')
    
    #launch paraview
    cont.paraview(grid_3D_path, exe=r"D:\ParaView 5.0.0\bin\paraview.exe",
                  time_derivatives=td,filename=args.figname)
    
    
def delete_temp_files():
    for filename in glob.glob(work_dir+"/temp*"):
        print 'deleting ', filename
        os.remove(filename)
        
def plot_co2(model_names=[]):
    plot_hist('co2','co2mt','CO$_2$ Mass (kg)',model_names)
    plot_hist('co2','temp','Temperature ($^\degree$C)',model_names)
    plot_hist('co2','presCO2','CO$_2$ pressure (MPa)',model_names)
    plot_hist('co2','presWAT','Water pressure (MPa)',model_names)
    plot_hist('co2','co2sg','CO$_2$ gas saturation',model_names)
    plot_hist('co2','co2sl','CO$_2$ liquid saturation',model_names)
    plot_hist('co2','denCO2l','CO$_2$ Density (kg/m$^3$)',model_names)
    plot_hist('co2','denCO2g','CO$_2$ Density (kg/m$^3$)',model_names)
    
def plot_water(model_names=[]):
    plot_hist('water','temp','Temperature ($^\degree$C)',model_names)
    plot_hist('water','presWAT','Water pressure (MPa)',model_names)
    plot_hist('water','denWAT','Water Density (kg/m$^3$)',model_names)

def plot_hist(fluid,param,param_label,model_names):
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    linestyles = ['-',(0,[5,1]),(0,[1,1])]
    if fluid in ['CO2','co2']:
        fluid='CO2'
    elif fluid in ['Water','water']:
        fluid='Water'
    else:
        print "Cannot find fluid: ", fluid 
        return
    
    fig, ax = plt.subplots() 
    
    if args.multi:
        if args.multirow: fig, ax = plt.subplots(len(model_names))

        for imodel,model_name,ls in zip(range(len(model_names)),model_names,linestyles[:len(model_names)]):
            hist=pd.read_excel('{}\\{}\\ModelData.xlsx'.format(work_dir,model_name),sheetname='hist')
            
            s1=pd.read_csv(work_dir + '\\{1}\\{0}_Stage1\\{1}_{2}_his.csv'.format(fluid,model_name,param))
            s2=pd.read_csv(work_dir + '\\{1}\\{0}_Stage2\\{1}_{2}_his.csv'.format(fluid,model_name,param))
            co2_df = s1.append(s2)    
            labels = ['{}_{}'.format(model_name,p) for p in hist['Name'].values.tolist()]
            co2_df.columns=['Time (days)'] + labels
        #remove last row if error
            if co2_df.iloc[-1][0] == 1.0:
                co2_df = co2_df.iloc[:-1]
            
            co2_df.sort_values('Time (days)',inplace=True)
            co2_df.set_index('Time (days)',inplace=True)
            for column,color in zip(co2_df.columns,colors[:len(co2_df.columns)]):
                if args.multirow:
                    co2_df[column].plot(ax = ax[imodel], marker='', c=color, ms=2)
                else:
                    co2_df[column].plot(ax = ax, marker='', c=color, ls=ls, ms=2)
    else:
        model_name=model_names[0]
        hist=pd.read_excel(model_data,sheetname='hist')
        
        s1=pd.read_csv(work_dir + '\\{}_Stage1\\{}_{}_his.csv'.format(fluid,model_name,param))
        s2=pd.read_csv(work_dir + '\\{}_Stage2\\{}_{}_his.csv'.format(fluid,model_name,param))
        co2_df = s1.append(s2)    
        
        co2_df.columns=['Time (days)'] + hist['Name'].values.tolist()
        #remove last row if error
        if co2_df.iloc[-1][0] == 1.0:
            co2_df = co2_df.iloc[:-1]
        
        co2_df.sort_values('Time (days)',inplace=True)
        co2_df.set_index('Time (days)',inplace=True)
        co2_df.plot(ax = ax, marker='o', ls='', ms=2)
    
    if not isinstance(ax,np.ndarray):
        ax = [ax]
    
    for _ax in ax:    
        _ax.set_ylabel(param_label)
        _ax.set_xlim([0,365.25*10])
        _ax.grid(True)
        minorLocator=AutoMinorLocator(2)
        _ax.xaxis.set_minor_locator(minorLocator)
        
        if fluid=='CO2' and 'pres' in param:
            pcrit_CO2 = 7.39 #MPa
            _ax.plot([0,365.25*10],[pcrit_CO2,pcrit_CO2],'k--',label = 'CO$_2$ Critical Pressure')
        
    #    lgd=ax.legend(prop={'size':8})
        if 'mt' in param:
            lgd=_ax.legend(bbox_to_anchor=(1.1,1.1),prop={'size':8})
        else:
            lgd=_ax.legend(loc='best',prop={'size':8})
        
    fig.set_size_inches((7,len(model_names)*3))
    fig.tight_layout()
    fig.savefig(args.figname+fluid+'_'+param+'_history.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    return co2_df

def plot_perm_slices(cont_list):
    cont = cont_list[0]
    x1,y1 = [517147.435,767258.665]
    x2,y2 = [534070.715,781211.863]
    
#    fig,ax= plt.subplots()
    gridlines = []
    grid = fgrid()
    grid.read(dirs['grid_2D_path'].values[0])
        
    for c in grid.connlist:
        
        points = np.array([(c.nodes[0].position[0],c.nodes[0].position[1]),
                           (c.nodes[1].position[0],c.nodes[1].position[1])])
        ind = np.lexsort((points[:,0],points[:,1]))
        gridlines += [points[ind]]

    for z in cont.z:
        ax=cont.slice_plot('perm_z',slice=['z',z],divisions=[1000,1000],
                    levels=15,cbar=True, xlims=[x1,x2], ylims=[y1,y2])
    
        for points in gridlines:
            ax.plot(points[:,0],points[:,1],'k-',lw=0.5)
        
        fig = ax.get_figure()
        fig.set_size_inches(10,10)
        fig.savefig('permzplot_{}.png'.format(z))
        
    for z in cont.z:
        ax=cont.slice_plot('perm_x',slice=['z',z],divisions=[1000,1000],
                    levels=15,cbar=True, xlims=[x1,x2], ylims=[y1,y2])
    
        for points in gridlines:
            ax.plot(points[:,0],points[:,1],'k-',lw=0.5)
        
        fig = ax.get_figure()
        fig.set_size_inches(10,10)
        fig.savefig('permxplot_{}.png'.format(z))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process NS model results')
    parser.add_argument('-t', dest= 'tplots', action='store_true',
                       help='create temp NS plots')
    parser.add_argument('-m', dest= 'multi', type=str, nargs='+',default=[],
                       help='create multi temp NS and pcp plots')
    parser.add_argument('-mr', dest= 'multirow', action='store_true',
                       help='set multi plot with multiple rows in figure')
    parser.add_argument('-p', dest= 'pplots', action='store_true',
                       help='create pcp NS plots')
    parser.add_argument('-pv', dest= 'paraview', action='store_true',
                       help='run paraview ns')
    parser.add_argument('-pvp', dest= 'paraview_preNS', action='store_true',
                       help='run paraview prens')
    parser.add_argument('-pvco2s1', dest= 'paraview_co2_stage1', action='store_true',
                       help='run paraview co2 stage1')
    parser.add_argument('-pvco2s2', dest= 'paraview_co2_stage2', action='store_true',
                       help='run paraview co2 stage2')
    parser.add_argument('-pvwats1', dest= 'paraview_wat_stage1', action='store_true',
                       help='run paraview water stage1')
    parser.add_argument('-co2', dest= 'co2', action='store_true',
                       help='process co2 run results')
    parser.add_argument('-wat', dest= 'water', action='store_true',
                       help='process water run results')
    parser.add_argument('-td', dest= 'td', action='store_true', default=False,
                       help='add time delta in paraview')
    parser.add_argument('-png', dest= 'png', action='store_true',default=False,
                       help='save plots as png' )
    parser.add_argument('-fn', dest= 'figname', type=str, default='',
                       help='name of plot,file name of vtk' )
    parser.add_argument('-sa', dest= 'sens', action='store_true',default=False,
                       help='plot sensitivity' )
    parser.add_argument('-bm', dest= 'base_model', type=str, default='',
                       help='base model for sensitivity' )
    parser.add_argument('-tt', dest= 't_thres', type=float, default=0.0,
                       help='temperature threshold' )
    parser.add_argument('-rs', dest= 'rsquared', action='store_true',default=False,
                       help='compute r-squared error and deviation from base model' )
    parser.add_argument('-pk', dest= 'plot_k', action='store_true',default=False,
                       help='plot permeability profiles' )
    
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
        co2_s1_dir = work_dir+'\\CO2_Stage1'
        co2_s2_dir = work_dir+'\\CO2_Stage2'
        wat_s1_dir = work_dir+'\\Water_Stage1'
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
    wells = wells.values.T.flatten().astype(str)
    wells = wells[wells!='nan']
    
    #get DS
    wells_DS=pd.read_excel(wells_file,sheetname=None)
    
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    
    if len(args.multi)>0:
        if args.tplots or args.pplots:
            cont_list = read_contours(model_names=args.multi,multi=True)
            if args.tplots:
                create_NS_plots(cont_list,model_names=args.multi,savetopdf=not args.png,basemodel=args.base_model,threshold=args.t_thres)
            if args.pplots:
                create_pcp_plot(cont_list,model_names=args.multi,savetopdf=not args.png)
        if args.co2:
            plot_co2(model_names=args.multi)
        if args.water:
            plot_water(model_names=args.multi)
    else:
        cont_list = read_contours([model_name],multi=False)
        
        if args.tplots and args.png:
            create_NS_plots(cont_list,[model_name],savetopdf=False)
        elif args.tplots:
            create_NS_plots(cont_list,[model_name],savetopdf=True)
        if args.pplots and args.png:
            create_pcp_plot(cont_list,[model_name],savetopdf=False)
        elif args.pplots:
            create_pcp_plot(cont_list,[model_name],savetopdf=True)
        if args.co2:
            plot_co2(model_names=[model_name])
        if args.water:
            plot_water(model_names=[model_name])
        if args.paraview:
            launch_paraview(ns_dir,args.td)
        if args.paraview_preNS:
            launch_paraview(preNS_dir,args.td)
        if args.paraview_co2_stage1:
            launch_paraview(co2_s1_dir,args.td)
        if args.paraview_co2_stage2:
            launch_paraview(co2_s2_dir,args.td)
        if args.paraview_wat_stage1:
            launch_paraview(wat_s1_dir,args.td)
