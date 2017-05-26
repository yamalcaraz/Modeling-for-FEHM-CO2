# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 20:37:30 2017

@author: Yam
"""
import sys,os
bin_path = os.path.abspath(os.path.join('..', 'bin'))
sys.path.append(bin_path)

import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import numpy as np
import shapefile as shp
from fgrid import fgrid
from fdata import fdata, fzone, fmacro, fmodel
import os, glob, shutil
import pandas as pd
from fpost import fhistory
import argparse
from processOutfile import process_outfile

work_dir=os.getcwd()

dirs=pd.read_excel(work_dir+"\\ModelData.xlsx",sheetname='Dir')
dirs=dirs.set_index('Variable').T

exe = dirs['exe'].values[0]

#files
model_name=work_dir.split('\\')[-1]
grid_2D_path = dirs['grid_2D_path'].values[0]
grid_3D_path = dirs['grid_3D_path'].values[0]
data_dir = dirs['data_dir'].values[0]
model_data = os.path.abspath(work_dir+"\\ModelData.xlsx")

#read the 2D grid
grid_2D = fgrid()
grid_2D.read(grid_2D_path)

def point_inside_poly(point, poly):
    
    bbPath = mplPath.Path(poly)
    return bbPath.contains_point(point)
    
def collect_nodes(shapefile, grid_2D, grid_3D, Z):   
    sfo = shp.Reader(shapefile)
    shape_rec = sfo.shapeRecords()
    gridpoints = [n.position[:2] for n in grid_2D.nodelist]
    
    zones = []
    
    for s in shape_rec:
        zone = {'order':s.record[0],'rock':s.record[1],'nodes':[]}
        for p in gridpoints:
            if point_inside_poly(p,s.shape.points):
                p_3D = np.append(p,Z)
#                zone['points'] += [p_3D]
                zone['nodes'] += [grid_3D.node_nearest_point(p_3D)]
            zone['nodes'] = sorted(zone['nodes']) 
        zones += [zone]
    zones = sorted(zones,key=lambda d:d['order'])
    
    for z1 in zones:
        for z2 in zones:
            if z1['order']<z2['order']:
                clipped_nodes=list(set(z1['nodes']).difference(set(z2['nodes'])))
                if clipped_nodes!=None:
#                    print 'clipped', str(len(z1['nodes'])-len(clipped_nodes)), 'from', z1['rock']
#                    print 'clipped_nodes', len(clipped_nodes), len(sorted(clipped_nodes))
                    z1['nodes']=sorted(clipped_nodes)
#                    print z1['nodes'][:10]
    
    return zones
    
def assign_zones(zones,layer_id, dat, rocks):
    global debugstring, debugstring2
    """Add zone objects to fdata object"""
    for z in zones:
        debugstring = z
        idx=rocks.loc[:,'ROCK']==z['rock']
        z['zone_index']=layer_id*1000+z['order']
        zone_obj = fzone(index=int(z['zone_index']),type='nnum',
                         nodelist=z['nodes'],name=z['rock']+'_layer'+str(layer_id))
        dat.add(zone_obj)
        assign_rock_props(zone_obj,rocks[idx])

def assign_rock_props(zone,rock):
    global debugstring
    debugstring = zone
    
    zone.permeability = [rock['k1'].values[0],rock['k2'].values[0],rock['k3'].values[0]]
    
    zone.density=rock['density'].values[0]
    zone.specific_heat=rock['specific_heat'].values[0]
    zone.porosity=rock['porosity'].values[0]

    zone.conductivity=rock['therm_con'].values[0]
        
def assign_layers(dat, grid_2D):
    print 'Assigning materials...',
    layers=pd.read_excel(model_data,sheetname='Layers')
    rocks=pd.read_excel(model_data,sheetname='Rocks')

    layers['Abspath']=work_dir+layers['Path']
    for i,row in layers.iterrows():
        zones=collect_nodes(row['Abspath'],grid_2D,dat.grid,row['Layer']) 
        assign_zones(zones, row['Layer_id'], dat,rocks)
    
    print '\rAssigning materials... Done'
        


#set BC
def apply_ns_bc(dat,preNS_run=False):
    bc=pd.read_excel(model_data,sheetname='BC')
    bc['Abspath']=work_dir+bc['Path']
    for i, row in bc.iterrows():
        if preNS_run and not bool(row['PreNS_BC']):
            continue
        if pd.isnull(row['zone']):
            zone_index = i+100
            zones=collect_nodes(row['Abspath'],grid_2D,dat.grid,row['Z']) 
            zone = (z for z in zones if z['rock']==row['BC']).next()
            zone_obj = fzone(index=zone_index, type='nnum', nodelist=zone['nodes'], name=zone['rock']+'_bc')
            dat.add(zone_obj)
        else:
            zone_index=dat.zone[row['zone']].index
            
        if row['Type']=='fix_t':
            print 'Adding fixed_T BC'
            dat.zone[zone_index].fix_temperature(T=row['Temperature'],multiplier=row['impedance'])
        if row['Type']=='fix_p':
            print 'Adding fixed_T BC'
            dat.zone[zone_index].fix_pressure(P=row['Pressure'],T=row['Temperature'],impedance=row['impedance'])
        if row['Type']=='flow':
            print 'Adding flow BC'
            flow_macro=fmacro('flow')
            flow_macro.zone=dat.zone[zone_index]
            flow_macro.param['rate']=row['rate']
            flow_macro.param['energy']=row['energy']
            flow_macro.param['impedance']=row['impedance']
            dat.add(flow_macro)

def apply_bc(dat,sheetname,rates=[]):
    bc=pd.read_excel(model_data,sheetname=sheetname)
    
    wells_file = data_dir + "MGPF Deviation Survey and Well Data1.xlsx"
    
    i_opt = 0
    
    for i, row in bc.iterrows():
        
        if row['zone_name'] in dat.zone:
            row['zone_index'] = dat.zone[row['zone_name']].index
        else:
            if pd.isnull(row['well']):
                node = dat.grid.node_nearest_point([row['x'],row['y'],row['z']])
            else:
                w_DS=pd.read_excel(wells_file,sheetname=row['well'])
                node = find_well_node(dat,w_DS,row['z'])
                
            #create the zone using found node
            dat.new_zone(int(row['zone_index']), name=row['zone_name'], nodelist=node, overwrite=True)
        
            
        if row['Type']=='co2flow':
            print 'Adding co2flow BC'
            flow_macro=fmacro('co2flow')
            flow_macro.zone=dat.zone[row['zone_index']]
            if row['optimize']==1:
                print i_opt, rates
                flow_macro.param['rate']=rates[i_opt]
                i_opt+=1
            else:
                flow_macro.param['rate']=row['rate']
            flow_macro.param['energy']=row['energy']
            flow_macro.param['impedance']=row['impedance']
            flow_macro.param['bc_flag']=row['bc_flag']
            dat.add(flow_macro)
        if row['Type']=='flow':
            print 'Adding flow BC'
            flow_macro=fmacro('flow')
            flow_macro.zone=dat.zone[row['zone_index']]
            flow_macro.param['rate']=row['rate']
            flow_macro.param['energy']=row['energy']
            flow_macro.param['impedance']=row['impedance']
            dat.add(flow_macro)
        if row['Type']=='monitor':
            print 'node {} added to hist'.format(node.index)
            dat.hist.nodelist.append(node)
            

def find_well_node(dat,w_DS,z):
        w_DS = w_DS.loc[:,['Easting','Northing','MRSL']]
        w_DS=w_DS.sort_values('MRSL')
        #get only DS on cont.z values
        z=float(z)
        x=np.interp(z,w_DS['MRSL'],w_DS['Easting'])
        y=np.interp(z,w_DS['MRSL'],w_DS['Northing'])
        
        if not(any(w_DS.MRSL>=z) and any(w_DS.MRSL<=z)):
            print '{} not within well'.format(z)
            return
        
        return dat.grid.node_nearest_point([x,y,z])
      
#Pre-NS
def run_PreNS():
    """this is just to work out the initial state pressures"""
    dat = fdata(work_dir=work_dir+r'\Pre_NS')
    dat.grid.read(grid_3D_path)
    
    #set initial permeability of everything
    dat.zone[0].permeability=1E-14
    
    read_elevation_model(dat,'elevationModel.csv')
    
    #fix top temp and pressure
    dat.zone[0].Pi=10. #note: Pi should be called first before Ti
    dat.zone[0].Ti=20.
    
    apply_ns_bc(dat,preNS_run=True)
    
    dat.tf = 1.e16
    dat.dtmax = dat.tf/10.
    dat.dtn = 5000
    dat.files.rsto = 'restart_file.ini'
    
    dat.cont.variables.append(['pressure', 'temperature', 'perm_x','perm_y','perm_z','porosity','saturation' ])
    dat.cont.format='surf'
    dat.cont.timestep_interval=50
    
    dat.write(os.path.join(work_dir+'/'+model_name+'.dat'))
    delete_model_files(work_dir+'\\Pre_NS',model_name)
    if args.paraview:
        launch_paraview(dat)
    
    dat.run(model_name+'.dat',exe=exe)
    
    #launch_paraview(dat)
    return dat

#NS
def run_NS(restart_from_end=False):
    run_params=pd.read_excel(work_dir+"\\ModelData.xlsx",sheetname='NS_Params')
    run_params=run_params.set_index('Variable').T
    
    dat = fdata(work_dir=work_dir+r"\NS")
    dat.grid.read(grid_3D_path)
    
    assign_layers(dat,grid_2D)
    read_elevation_model(dat,'elevationModel.csv')
    apply_ns_bc(dat)
    
    if restart_from_end:
        copy_incon(work_dir+'\\NS\\natural_state.ini',work_dir+'\\NS\\natural_state_restart.ini')
        dat.incon.read(work_dir+r'\\NS\\natural_state_restart.ini')
    else:
        dat.incon.read(work_dir+r'\\Pre_NS\\restart_file.ini')
     
    dat.files.rsto = 'natural_state.ini'
    dat.ti = run_params['ti'].values[0]
    dat.tf = run_params['tf'].values[0]
    dat.dtmax = run_params['dtmax'].values[0]
    dat.dtn = run_params['dtn'].values[0]
    
    dat.cont.variables.append(['xyz','pressure', 'temperature', 'perm_x','perm_y','perm_z','porosity','saturation'])
    dat.cont.format='surf'
    dat.cont.timestep_interval=int(run_params['cont.timestep_interval'].values[0])
    dat.cont.time_interval=run_params['cont.time_interval'].values[0]
    
    dat.hist.variables.append(['temperature','pressure'])
    dat.hist.zonelist.append('upf_bc')
    dat.hist.timestep_interval=1
    dat.hist.format='csv'
    
    dat.ctrl['max_newton_iterations_MAXIT'] = 10
    dat.ctrl['newton_cycle_tolerance_EPM'] = 1E-2
    dat.iter['machine_tolerance_TMCH'] = -1E-2
    
    delete_model_files(work_dir+'\\NS',model_name)
    dat.run(model_name+'.dat',exe=exe)
    plot_timesteps(work_dir+'\\NS\\'+model_name+'_temp_his.csv')
    
    
    return dat
#

def run_first_stage(fluid):
    run_params=pd.read_excel(work_dir+"\\ModelData.xlsx",sheetname='S1_Params')
    run_params=run_params.set_index('Variable').T
    
    if fluid in ['CO2','co2']:
        co2_sim=True
        fluid='CO2'
        dat = fdata(work_dir=work_dir+r"\CO2_Stage1")
    elif fluid in ['Water','water']:
        co2_sim=False
        fluid='Water'
        dat = fdata(work_dir=work_dir+r"\Water_Stage1")
    else:
        print "Cannot find fluid: ", fluid 
        return
    dat.grid.read(grid_3D_path)
    
#    assign_layers(dat,grid_2D)
#    read_elevation_model(dat,'elevationModel.csv')
#    apply_ns_bc(dat)
    
    dat.read(r'{0}\\NS\\{1}.dat'.format(work_dir,model_name))
    
    dat.incon.read(work_dir+r'\\NS\\natural_state.ini')
    dat.files.rsto = '{}_first_stage.ini'.format(fluid)
    
    if co2_sim:
        #turn on CO2 computation
        dat.carb.on(iprtype=int(run_params['iprtype'].values[0]))
        dat.files.co2in = data_dir+"co2_interp_table.txt"
    
        co2frac=fmacro('co2frac', zone=0, 
                       param=(('water_rich_sat',1.0), 
                              ('co2_rich_sat',0.0),('co2_mass_frac',1.0), 
                              ('init_salt_conc',0.), ('override_flag',0)))
        dat.add(co2frac)

    apply_bc(dat,'{}_S1_BC'.format(fluid))
    
    dat.hist.nodelist=[]
    dat.hist.zonelist=[]
    add_monitor_nodes(dat)

    if co2_sim:
    ##    
    #    add relative perm
    #     linear relperm model, nice to sub in when things are misbehaving
        rlp = fmodel('rlp',index=17,param=[.05,1,1,0,1,1,0,0,1,1,1,0,1,0])
        dat.add(rlp)
        #rlpm=frlpm(group=1,zone=dat.zone[0])
        #rlpm.add_relperm ('water','exponential',[0.2,1.,3.1,1.])
        #rlpm.add_relperm ('co2_liquid','exponential',[0.2,1.,3.1,0.8])
        #dat.add (rlpm)
    
    dat.cont.variables=[]
    dat.cont.variables.append(['xyz','pressure', 'temperature', 'perm_x','perm_y','perm_z','porosity','saturation'])
    if co2_sim:
        dat.cont.variables.append(['co2'])
    dat.cont.format='surf'
    dat.cont.timestep_interval=int(run_params['cont.timestep_interval'].values[0])
    dat.cont.time_interval=run_params['cont.time_interval'].values[0]
    
    #flow history
    dat.hist.variables=[]
    dat.hist.variables.append(['temperature', 'pressure', 'flow','density','zfl'])
    if co2_sim:
        dat.hist.variables.append(['co2m','co2s'])
    dat.hist.format='surf'
    
    #timesteps
    dat.ti = run_params['ti'].values[0]
    dat.tf = run_params['tf'].values[0]
    dat.dtmax = run_params['dtmax'].values[0]
    dat.dtmin = run_params['dtmin'].values[0]
    dat.dti = 1.
    dat.dtn = run_params['dtn'].values[0]
    
    delete_model_files('{}\\{}_Stage1'.format(work_dir,fluid),model_name)
    dat.run(model_name+'.dat',exe=exe)
    
    print 'exporting mass flow'
    process_outfile('{}\\{}_Stage1\\'.format(work_dir,fluid),model_name+'.outp')
    
    return dat

def run_second_stage(fluid,rates=[]):
    run_params=pd.read_excel(work_dir+"\\ModelData.xlsx",sheetname='S2_Params')
    run_params=run_params.set_index('Variable').T
    
    if fluid in ['CO2','co2']:
        co2_sim=True
        fluid='CO2'
        dat = fdata(work_dir=work_dir+r"\CO2_Stage2")
    elif fluid in ['Water','water']:
        co2_sim=False
        fluid='Water'
        dat = fdata(work_dir=work_dir+r"\Water_Stage2")
    else:
        print "Cannot find fluid: ", fluid 
        return
    
    dat.grid.read(grid_3D_path)
    
#    assign_layers(dat,grid_2D)
#    read_elevation_model(dat,'elevationModel.csv')
#    apply_ns_bc(dat)

    #read dat file from Stage1
    dat.read(r'{0}\\{1}_Stage1\\{2}.dat'.format(work_dir,fluid,model_name))
    
    dat.incon.read(r'{0}\\{1}_Stage1\\{1}_first_stage.ini'.format(work_dir,fluid))
    dat.files.rsto = '{}_second_stage.ini'.format(fluid)
    
    if co2_sim:
    #turn on CO2 computation
        dat.carb.on(iprtype=int(run_params['iprtype'].values[0]))
        dat.files.co2in = data_dir+"co2_interp_table.txt"
    
    #remove bc from previous run
    while len(dat.co2flowlist) > 0:
        for m in dat.co2flowlist:
            dat.delete(m)
        
    apply_bc(dat,'{}_S2_BC'.format(fluid),rates)
    dat.hist.nodelist=[]
    dat.hist.zonelist=[]
    add_monitor_nodes(dat)

    if co2_sim:
    ##    
    #    add relative perm
    #     linear relperm model, nice to sub in when things are misbehaving
        rlp = fmodel('rlp',index=17,param=[.05,1,1,0,1,1,0,0,1,1,1,0,1,0])
        dat.add(rlp)
        #rlpm=frlpm(group=1,zone=dat.zone[0])
        #rlpm.add_relperm ('water','exponential',[0.2,1.,3.1,1.])
        #rlpm.add_relperm ('co2_liquid','exponential',[0.2,1.,3.1,0.8])
        #dat.add (rlpm)
    
    #contour values every 6 months
    dat.cont.variables = []
    dat.cont.variables.append(['xyz','pressure', 'temperature', 'perm_x','perm_y','perm_z','porosity','saturation'])
    if co2_sim:
        dat.cont.variables.append(['co2'])

    dat.cont.format='surf'
    dat.cont.timestep_interval=int(run_params['cont.timestep_interval'].values[0])
    dat.cont.time_interval=run_params['cont.time_interval'].values[0]
    
    #flow history
    dat.hist.variables = []
    dat.hist.variables.append(['temperature', 'pressure', 'flow','density','zfl'])
    if co2_sim:
        dat.hist.variables.append(['co2m','co2s'])
    dat.hist.format='surf'
    
    #timesteps
    dat.ti = run_params['ti'].values[0]
    dat.tf = run_params['tf'].values[0]
    dat.dtmax = run_params['dtmax'].values[0]
    dat.dtmin = run_params['dtmin'].values[0]
    dat.dti = 1.
    dat.dtn = run_params['dtn'].values[0]
    
    delete_model_files('{}\\{}_Stage2'.format(work_dir,fluid),model_name)
    
    print 'writing input file'
    dat.run(model_name+'.dat',exe=exe)
    
    print 'exporting mass flow'
    process_outfile('{}\\{}_Stage2\\'.format(work_dir,fluid),model_name+'.outp')
    
    return dat

def add_monitor_nodes(dat):
    hist=pd.read_excel(model_data,sheetname='hist')
    
    wells_file = data_dir + "MGPF Deviation Survey and Well Data1.xlsx"
    
    for i, row in hist.iterrows():
        
        if pd.isnull(row['well']):
            node = dat.grid.node_nearest_point([row['x'],row['y'],row['z']])
        else:
            w_DS=pd.read_excel(wells_file,sheetname=row['well'])
            node = find_well_node(dat,w_DS,row['z'])
        
        if row['Type']=='monitor':
            print 'node {} added to hist'.format(node.index)
            dat.hist.nodelist.append(node)
            

def plot_timesteps(histfile):
    hist=fhistory(histfile)
    fig,ax = plt.subplots(1,1)
    timesteps = np.diff(hist.times)
    ax.plot(range(len(timesteps)),timesteps)
    fig.savefig(work_dir+'\\timesteps.png')

def delete_model_files(work_dir,model_name):
    for filename in glob.glob(work_dir+"/"+model_name+"*"):
        print 'deleting ', filename
        os.remove(filename)
        
def copy_incon(incon1, incon2):
    #overwrites incon2
    shutil.copy(incon1, incon2)

def read_elevation_model(dat, elev_model_file):
    #create inactive zone and assign porosity to zero
    print 'Reading elevation model...',
    node_df=pd.read_csv(elev_model_file)
    node_index_to_remove = node_df[~node_df['active']]['ni']
    nodes_to_remove = [dat.grid.node[ni] for ni in node_index_to_remove]
    
    #initialize then edit later
    zone_obj = fzone(index=100000,type='nnum', nodelist=nodes_to_remove, name='atmos')
    dat.add(zone_obj)
    
    top_node_index=node_df[node_df['is_top']]['ni']
    top_nodes=[dat.grid.node[ni] for ni in top_node_index]
    
    #convert top_nodes with 4 or more connections to the atmos zone
    for nd in top_nodes:
        atmos_cons=[]
        for con in nd.connected_nodes:
            if 100000 in con.zone:
                atmos_cons+=[con]
        if len(atmos_cons)>=(len(nd.connected_nodes)-2): 
            nodes_to_remove += [nd]
            top_nodes.remove(nd)
#            print nd, str(len(atmos_cons))
#            print nd in nodes_to_remove
    nodes_to_remove.sort()
    zone_obj = fzone(index=100000,type='nnum', nodelist=nodes_to_remove, name='atmos')
    dat.delete(dat.zone[100000])
    dat.add(zone_obj)
    
    dat.zone['atmos'].porosity=0.001
    
#    dat.zone['atmos'].Pi=0.1
#    dat.zone['atmos'].Ti=20.

    #add 'sides' to top_zone
    for nd in dat.grid.nodelist:
        if nd not in nodes_to_remove:
            for con in nd.connected_nodes:
                if 100000 in con.zone:
#                    print nd
                    if nd not in top_nodes:
                        top_nodes+=[nd]
                    break
    
    print str(len(top_nodes)) + ' nodes found in top of model'
    #Warning: add this for the sake of h_sens only
    if model_name=='Model_5_h_sens':
        top_nodes+=[n for n in dat.grid.nodelist if n.index==63209]
    top_nodes.sort()
    zone_obj = fzone(index=11,type='nnum', nodelist=top_nodes, name='top_zone')
    dat.add(zone_obj)
    

def launch_paraview(dat):
    #launch paraview
    dat.paraview(exe=r"D:\ParaView 5.0.0\bin\paraview.exe")

#            
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Create and run NS model')
    parser.add_argument('-p', dest= 'preNS', action='store_true',
                       help='run preNS before running NS')
    parser.add_argument('-r', dest= 'restart', action='store_true',
                       help='run NS using ending restart file')
    parser.add_argument('-pv', dest= 'paraview', action='store_true',
                       help='launch paraview before running preNS')
    parser.add_argument('-cs1', dest= 'run_co2_stage1', action='store_true',
                       help='run CO2 stage 1')
    parser.add_argument('-cs2', dest= 'run_co2_stage2', action='store_true',
                       help='run CO2 stage 2')
    parser.add_argument('-ws1', dest= 'run_water_stage1', action='store_true',
                       help='run Water stage 1')
    parser.add_argument('-ws2', dest= 'run_water_stage2', action='store_true',
                       help='run Water stage 2')
    parser.add_argument('-ns', dest= 'ns', action='store_true',
                       help='run NS')
    
    args = parser.parse_args()
    
    
    if args.preNS:
        dat = run_PreNS()
    if args.ns:
        if args.restart:
            run_NS(restart_from_end=True)
        else:
            run_NS()
    if args.run_co2_stage1:
        dat=run_first_stage('co2')
    if args.run_co2_stage2:
        dat=run_second_stage('co2')
    if args.run_water_stage1:
        dat=run_first_stage('water')
    if args.run_water_stage2:
        dat=run_second_stage('water')
        
    
    plot_timesteps(work_dir+'\\NS\\'+model_name+'_temp_his.csv')
