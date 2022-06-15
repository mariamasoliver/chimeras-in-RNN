#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 09:49:10 2021

@author: maria
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date

prepath1 = '/home/maria/cluster_CSG/'
prepath2 = '/home/masoliverm/'
prepath3 = '/Users/maria/cluster_CSG/'
path = prepath3+'Documents/FilesFigures/force_method_2/testing/Resultats_switching_chimera/as_perturbation/c_random'
os.chdir(path)



# it is size: (500001,1500)
pca = np.loadtxt('PCA-firing-rates_switching_chimera_crandom.txt')
cvec = np.loadtxt('cvec_6_N_2000_Q_0.8_c_imin_random.txt')
nodes = 3



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

size = 22
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

ca = '#d6604d'
cb = '#878787'
plt.rcParams.update(params)

#before a positive pulse
t1before = 31000
t2before = 32000
#after a positive pulse
t1after = 32400
t2after = 33400

#before a positive pulse
t1before = 31000
t2before = 31500
#after a positive pulse
t1after = 32400
t2after = 33000

dt = 0.2

nt1before = int(t1before/dt)
nt2before = int(t2before/dt)
nt1after = int(t1after/dt)
nt2after = int(t2after/dt)

dtc = 0.1
nt1cB = int(t1before/dtc)
nt2cB = int(t2before/dtc)
nt1cA = int(t1after/dtc)
nt2cA = int(t2after/dtc)

zbefore = pca[nt1before:nt2before, 2]
xbefore = pca[nt1before:nt2before, 0]
ybefore = pca[nt1before:nt2before, 1]

zafter = pca[nt1after:nt2after, 2]
xafter = pca[nt1after:nt2after, 0]
yafter = pca[nt1after:nt2after, 1]

fig = plt.figure(figsize = (15,7.5))

ax0 = fig.add_subplot(241, projection='3d')
ax0.plot3D(xbefore,ybefore, zbefore, cb, label = 'Before')
ax0.text(-30,-10,18, 'A',color='k',weight='bold',fontsize = size)
ax0.set_ylabel(r'$PC_2$',color='k')
ax0.set_xlabel(r'$PC_1$',color='k')
ax0.set_zlabel(r'$PC_3$',color='k')
ax0.set_zlim(-12,12)

ax = fig.add_subplot(242)
ax.text(-30,35, 'B',color='k',weight='bold',fontsize = size)
ax.plot(xbefore, ybefore ,cb, label = 'before')
ax.set_ylabel(r'$PC_2$',color='k')
# ax.set_xticklabels([])

ax2 = fig.add_subplot(243)
ax2.plot(xbefore, zbefore, cb, label = 'before')
ax2.text(-30,17, 'C',color='k',weight='bold',fontsize = size)
ax2.set_ylabel(r'$PC_3$',color='k')
# ax2.set_xticklabels([])
# ax2.set_ylim(-14,14)


ax = fig.add_subplot(244)
ax.plot(ybefore, zbefore ,cb, label = 'before')
ax.text(-30,17, 'D',color='k',weight='bold',fontsize = size)
ax.set_ylabel(r'$PC_3$',color='k')
# ax.set_xticklabels([])
# ax.set_ylim(-14,14)

ax0b = fig.add_subplot(245, projection='3d')
ax0b.text(-30,-2,4.5, 'E',color='k',weight='bold',fontsize = size)
ax0b.plot3D(xafter, yafter, zafter, ca, label = 'After')
ax0b.set_ylabel(r'$PC_2$',color='k')
ax0b.set_xlabel(r'$PC_1$',color='k')
ax0b.set_zlabel(r'$PC_3$',color='k')
ax0b.set_zlim(-3.5,3.5)


ax = fig.add_subplot(246)
ax.text(-30,33, 'F',color='k',weight='bold',fontsize = size)
ax.plot(xafter, yafter ,ca, label = 'after')
ax.set_xlabel(r'$PC_1$',color='k')
# ax.set_ylabel(r'$PC_3$',color='k')


ax = fig.add_subplot(247)
ax.plot(xafter, zafter, ca, label = 'after')
ax.text(-30,2.2, 'G',color='k',weight='bold',fontsize = size)
ax.set_ylabel(r'$PC_3$',color='k')
ax.set_xlabel(r'$PC_1$',color='k')
# ax.set_ylim(-3.5,3.5)

ax = fig.add_subplot(248)
ax.text(-30,2.2, 'H',color='k',weight='bold',fontsize = size)
ax.plot(yafter, zafter, ca, label = 'after')
ax.set_xlabel(r'$PC_2$',color='k')
ax.set_ylabel(r'$PC_3$',color='k')
# ax.set_ylim(-3.5,3.5)


path = prepath3 + 'Documents/FilesFigures/figures-for-articles/2021'
os.chdir(path)
today = date.today()
figure_nom = 'Suplementary_sc_3D_before_after_positive_pulse_t1before_+'+str(t1before)+str(today)+'.svg'
# , bbox_inches='tight'
plt.tight_layout()
plt.savefig(figure_nom, format='svg',dpi=300)
plt.show()

#%%
#Time series theta and phi

size = 22
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

ca = '#d6604d'
cb = '#878787'
plt.rcParams.update(params)

#before a positive pulse
t1before = 39600
t2before = 40200
#after a positive pulse
t1after = 40400
t2after = 40900

dt = 0.2

nt1before = int(t1before/dt)
nt2before = int(t2before/dt)
nt1after = int(t1after/dt)
nt2after = int(t2after/dt)

dtc = 0.1
nt1cB = int(t1before/dtc)
nt2cB = int(t2before/dtc)
nt1cA = int(t1after/dtc)
nt2cA = int(t2after/dtc)

zafter= pca[nt1before:nt2before, 2]
xafter = pca[nt1before:nt2before, 0]
yafter= pca[nt1before:nt2before, 1]

zbefore = pca[nt1after:nt2after, 2]
xbefore = pca[nt1after:nt2after, 0]
ybefore = pca[nt1after:nt2after, 1]

fig = plt.figure(figsize = (15,7.5))

ax0 = fig.add_subplot(241, projection='3d')
ax0.plot3D(xbefore,ybefore, zbefore, cb, label = 'Before')
ax0.text(-30,-10,18, 'A',color='k',weight='bold',fontsize = size)
ax0.set_ylabel(r'$PC_2$',color='k')
ax0.set_xlabel(r'$PC_1$',color='k')
ax0.set_zlabel(r'$PC_3$',color='k')
ax0.set_zlim(-12,12)

ax = fig.add_subplot(242)
ax.text(-30,35, 'B',color='k',weight='bold',fontsize = size)
ax.plot(xbefore, ybefore ,cb, label = 'before')
ax.set_ylabel(r'$PC_2$',color='k')
# ax.set_xticklabels([])

ax2 = fig.add_subplot(243)
ax2.plot(xbefore, zbefore, cb, label = 'before')
ax2.text(-30,17, 'C',color='k',weight='bold',fontsize = size)
ax2.set_ylabel(r'$PC_3$',color='k')
# ax2.set_xticklabels([])
# ax2.set_ylim(-14,14)


ax = fig.add_subplot(244)
ax.plot(ybefore, zbefore ,cb, label = 'before')
ax.text(-30,17, 'D',color='k',weight='bold',fontsize = size)
ax.set_ylabel(r'$PC_3$',color='k')
# ax.set_xticklabels([])
# ax.set_ylim(-14,14)

ax0b = fig.add_subplot(245, projection='3d')
ax0b.text(-30,-2,4.5, 'E',color='k',weight='bold',fontsize = size)
ax0b.plot3D(xafter, yafter, zafter, ca, label = 'After')
ax0b.set_ylabel(r'$PC_2$',color='k')
ax0b.set_xlabel(r'$PC_1$',color='k')
ax0b.set_zlabel(r'$PC_3$',color='k')
ax0b.set_zlim(-3.5,3.5)


ax = fig.add_subplot(246)
ax.text(-30,33, 'F',color='k',weight='bold',fontsize = size)
ax.plot(xafter, yafter ,ca, label = 'after')
ax.set_xlabel(r'$PC_1$',color='k')
# ax.set_ylabel(r'$PC_3$',color='k')


ax = fig.add_subplot(247)
ax.plot(xafter, zafter, ca, label = 'after')
ax.text(-30,2.2, 'G',color='k',weight='bold',fontsize = size)
ax.set_ylabel(r'$PC_3$',color='k')
ax.set_xlabel(r'$PC_1$',color='k')
# ax.set_ylim(-3.5,3.5)

ax = fig.add_subplot(248)
ax.text(-30,2.2, 'H',color='k',weight='bold',fontsize = size)
ax.plot(yafter, zafter, ca, label = 'after')
ax.set_xlabel(r'$PC_2$',color='k')
ax.set_ylabel(r'$PC_3$',color='k')
# ax.set_ylim(-3.5,3.5)


path = prepath3 + 'Documents/FilesFigures/figures-for-articles/2021'
os.chdir(path)
today = date.today()
figure_nom = 'Suplementary_sc_3D_before_after_positive_pulse_t1before_+'+str(t1before)+str(today)+'.svg'
# , bbox_inches='tight'
plt.tight_layout()
plt.savefig(figure_nom, format='svg',dpi=300)
plt.show()