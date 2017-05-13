# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 04:30:39 2017

@author: Yam
"""

import numpy as np
from scipy.optimize import fixed_point
from CoolProp.CoolProp import PropsSI, PhaseSI
import matplotlib.pyplot as plt 
import pandas as pd

#use fixed point iteration for Colebrook equation
def friction_factor(V,d,rho,ni,k):
    #velocity
    #diameter
    #density
    #viscosity
    #roughness
    
    Re = V*d*rho/ni
    
    
    def colebrook(x):
        LHS = -2*np.log10((2.51/(Re*np.sqrt(x))) + (k/(3.71*d)))
        return (1.0/LHS)**2
    
    
    return fixed_point(colebrook, 0.2)

def wellbore_step(fluid,z,dz,h,V,P,rho,d,k,g):
    #change in Kinetic Energy is ignored @Adams, 2014
    Vsquared=V**2
    z2 = z + dz

    ni = PropsSI('V','D',rho,'P',P,fluid) #V here is vicosity
    f = friction_factor(abs(V),d,rho,ni,k)
       
    Ploss = f*dz*rho*Vsquared/d/2
    if V<0: #if V is opposite dz
        Ploss = -Ploss

    #first law
    h2 = h + g*z - g*z2 #no change in velocity, velocity is updated with  density
    #bernoulli
    P2 = P + rho*g*z - rho*g*z2 - Ploss
    return z2, h2, P2
    

def bottom_up(fluid, z_bot, z_top, dz, mflow, P, T_celsius, d, k, g=9.81):
#    
#   bottom-up simulation
    z=z_bot
    h = PropsSI('H','T',T_celsius+273.15,'P',P,fluid)
    rho = PropsSI('D','T',T_celsius+273.15,'P',P,fluid)
    
    A=np.pi * d**2 / 4.
    V=mflow/rho/A #negative massflow is injection
    phase=PhaseSI('T',T_celsius+273.15,'P',P,fluid)
    
    results_vals=[[z,P,h,rho,V,T_celsius,phase]]
    for i in np.arange(z_bot,z_top,dz):
        z2,h2,P2=wellbore_step(fluid,z,dz,h,V,P,rho,d,k,g)
        if P2<0.0: break
    
        h=h2
        P=P2
        z=z2
        #update rho
        rho = PropsSI('D','H',h,'P',P,fluid)
        phase = PhaseSI('H',h,'P',P,fluid)
        #update V
        V=mflow/rho/A #velocity updated with density
        #check Temp
        T_celsius=PropsSI('T','H',h,'P',P,fluid)-273.15
        results_vals+=[[z,P,h,rho,V,T_celsius,phase]]
        
#        print 'Debug: ', z, P, h, rho, V
    results_df=pd.DataFrame(results_vals,columns=['z','P','h','rho','V','T','phase']) 
    
    return results_df