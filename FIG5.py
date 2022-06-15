#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 17:27:50 2021

@author: maria
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date

prepath1 = '/Users/maria/cluster_CSG/'
prepath2 = '/home/masoliverm/'
path = prepath1+ 'Documents/FilesFigures/force_method_2/testing/Resultats_switching_chimera/as_perturbation/c_random'
os.chdir(path)


cvec = np.loadtxt('cvec_6_N_2000_Q_0.8_c_imin_random.txt')
output = np.loadtxt('output_llavor_6_N_2000_Q_0.8_c_imin_random.txt')
nodes = 3
pca = np.loadtxt('PCA-firing-rates_switching_chimera_crandom.txt')

fr1 = np.loadtxt('firing_rates_llavor_6_N_2000_Q_0.8_c_imin_random.txt',skiprows=145000,usecols=(1,5,23,45,54,65),max_rows =150000 )

nodes = 3

sintheta = output[:,0:nodes] 
costheta = output[:,nodes:2*nodes] 
sinphi = output[:,2*nodes:3*nodes] 
cosphi = output[:,3*nodes:4*nodes] 

theta = np.arctan2(sintheta,costheta)
phi = np.arctan2(sinphi,cosphi)


"""Plotting"""
params = {
                'axes.labelsize': 22,
                'xtick.labelsize': 22,
                'ytick.labelsize': 22,
                    'legend.fontsize': 22,
                        
                        'text.usetex': False,
                            #'font': 'Helvetica',
                            'mathtext.bf': 'helvetica:bold',
                        }

plt.rcParams.update(params)

#%%
#Time series theta and phi
import matplotlib.cm as cm

size = 20
params = {
    'axes.labelsize': size,
        'xtick.labelsize': size,
            'ytick.labelsize': size,
                'legend.fontsize': size,
                     'axes.titlesize':size,
                    'text.usetex': False,
                        #'font': 'Helvetica',
                        'mathtext.bf': 'helvetica:bold',

                    }

plt.rcParams.update(params)

t1before = 31500
t2after = 32500

# t1before = 29000
# t2after = 30000
t1fr = t1before - 29000
t2fr = t2after - 29000
dt = 0.2

nt1before = int(t1before/dt)
nt2after = int(t2after/dt)

nt1fr = int(t1fr/dt)
nt2fr = int(t2fr/dt)

dtc = 0.1
nt1cB = int(t1before/dtc)
nt2cA = int(t2after/dtc)

time_pca = np.arange(0,len(pca[:,0])*dt,dt)
time_c = np.arange(0,len(cvec[:])*dtc,dtc)
orange_out ='#e6550d'
purple_out = '#54278f'


fig = plt.figure(figsize = (15,7.5))

ax0 = fig.add_subplot(2,4,1)
ax0.text(0,1.05,'A',color='k',weight='bold',fontsize = size)
ax0.set_yticks([])
ax0.set_xticks([])

ax0b = fig.add_subplot(2,4,2)
ax0b.text(0,1.05,'B',color='k',weight='bold',fontsize = size)
ax0b.set_yticks([])
ax0b.set_xticks([])

# #################
# #FIRE RATES
# ################

nodes_cmap = cm.get_cmap('bone')

step = (0.9-0.1)/12
color_nodes_and_fr = []
for i in range(10):
    ind = 0.1+i*step
    color_nodes_and_fr.append(nodes_cmap(ind))

ax1 = fig.add_subplot(2,4,(3,4))
ax1.set_title(r'Firing rates mean $FR_{mean}$')
ax1.text(4950,13,'C',color='k',weight='bold',fontsize = size)
for i in range (1,6):
    ax1.plot(time_pca[nt1before:nt2after], fr1[nt1fr:nt2fr,i]+2.2*i, color=color_nodes_and_fr[i], linewidth=2,alpha =0.8) 
    ax1.set_xticks([])



ax = fig.add_subplot(2,4,(5,6))
ax.set_title(r'Embedded switching chimera $\hat\theta(t)$ and $\hat\phi(t)$')
ax.text(4950,50,'D',color='k',weight='bold',fontsize = size)
for i in range (3):
    ax.plot(time_pca[nt1before:nt2after], theta[nt1before:nt2after,i]+8*i, color=orange_out, linewidth=3,alpha =0.8) 
    ax.plot(time_pca[nt1before:nt2after], phi[nt1before:nt2after,i]+8*(i+3), color=purple_out, linewidth=3,alpha =0.8)
    ax.plot(time_c[nt1cB:nt2cA], (cvec[nt1cB:nt2cA]/10)*np.pi - 8,color='grey', linewidth=2)
    #ax.set_xticks([])
ax.set_xlabel(r'time $t$',color='k')

color_pca = ['#8c3839', '#2b4743', '#1f7f95', '#da7073', '#dda83f', '#737373', '#3c6ca8', '#5d6c89']

ax3 = fig.add_subplot(2,4,(7,8))
# ax3.text(4950,125,'E',color='k',weight='bold',fontsize = size)
ax3.set_title('Principal component analysis \n of the firing rates $PC_{1,3,5}$')

for i in range (3):
    ax3.plot(time_pca[nt1before:nt2after], pca[nt1before:nt2after,i*2]+45*i, color=color_pca[i*2], linewidth=3,alpha =0.8) 
    ax3.plot(time_c[nt1cB:nt2cA], (cvec[nt1cB:nt2cA]) - 40,color='grey', linewidth=2)
#ax3.set_xlim(3500,4500)
ax3.set_xlabel(r'time $t$',color='k')

prepath1 = '/home/maria/cluster_CSG' 
prepath2 = '/home/masoliverm/' 

path = prepath2 + '/Documents/Files&Figures/figures-for-articles/2021'
os.chdir(path)
today = date.today()
figure_nom = 'FIG6_prova_'+str(today)+'.svg'
# , bbox_inches='tight'
plt.tight_layout()
plt.savefig(figure_nom, format='svg',dpi=300)
plt.show()