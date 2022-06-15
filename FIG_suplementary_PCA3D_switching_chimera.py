#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 14:57:05 2021

@author: masoliverm
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 17:13:09 2021

@author: masoliverm
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.graphics.tsaplots
from sklearn.decomposition import PCA

prepath1 = '/home/maria/cluster_CSG/'
prepath2 = '/home/masoliverm/'
prepath3 = '/Users/maria/cluster_CSG/'
path = prepath3+'Documents/FilesFigures/force_method_2/testing/Resultats_switching_chimera/as_perturbation/c_random'
os.chdir(path)



# it is size: (500001,1500)
pca = np.loadtxt('PCA-firing-rates_switching_chimera_crandom.txt')
cvec = np.loadtxt('cvec_6_N_2000_Q_0.8_c_imin_random.txt')
nodes = 3




#%%
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

plt.rcParams.update(params)
dtc = 0.1
dt = 0.2
time_pca = np.arange(0,len(pca[:,0])*dt,dt)
time_c = np.arange(0,len(cvec[:])*dtc,dtc)

#%%

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

zbefore = pca[nt1before:nt2before, 4]
xbefore = pca[nt1before:nt2before, 0]
ybefore = pca[nt1before:nt2before, 2]

zafter = pca[nt1after:nt2after, 4]
xafter = pca[nt1after:nt2after, 0]
yafter = pca[nt1after:nt2after, 2]

fig = plt.figure(figsize = (10,5))

ax0 = fig.add_subplot(121, projection='3d')
ax0.plot3D(xbefore,ybefore, zbefore, cb,linewidth=2)
ax0.set_ylabel(r'$PC_2$',color='k')
ax0.set_xlabel(r'$PC_1$',color='k')
ax0.set_zlabel(r'$PC_3$',color='k')
# ax0.set_zlim(-5,5)


ax0b = fig.add_subplot(122, projection='3d')
ax0b.plot3D(xafter, yafter, zafter, ca, linewidth=2)
ax0b.set_ylabel(r'$PC_2$',color='k')
ax0b.set_xlabel(r'$PC_1$',color='k')
ax0b.set_zlabel(r'$PC_3$',color='k')
# ax0b.set_zlim(-14,14)


# ax0.legend(fontsize=size)
# ax0b.legend(fontsize=size)

figure_nom = 'Fig_PCA-switching_chimera-fr_before_after_positive_pulse_t1_'+str(t1before)+'.svg'

plt.tight_layout()
plt.savefig(figure_nom, format='svg', bbox_inches='tight',dpi=250)
plt.show()

#%%

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

zbefore = pca[nt1before:nt2before, 4]
xbefore = pca[nt1before:nt2before, 0]
ybefore = pca[nt1before:nt2before, 2]

zafter = pca[nt1after:nt2after, 4]
xafter = pca[nt1after:nt2after, 0]
yafter = pca[nt1after:nt2after, 2]

fig = plt.figure(figsize = (10,5))

ax0 = fig.add_subplot(121, projection='3d')
ax0.plot3D(xbefore,ybefore, zbefore, cb,linewidth=2)
ax0.set_ylabel(r'$PC_2$',color='k')
ax0.set_xlabel(r'$PC_1$',color='k')
ax0.set_zlabel(r'$PC_3$',color='k')
# ax0.set_zlim(-5,5)


ax0b = fig.add_subplot(122, projection='3d')
ax0b.plot3D(xafter, yafter, zafter, ca, linewidth=2)
ax0b.set_ylabel(r'$PC_2$',color='k')
ax0b.set_xlabel(r'$PC_1$',color='k')
ax0b.set_zlabel(r'$PC_3$',color='k')
# ax0b.set_zlim(-14,14)


# ax0.legend(fontsize=size)
# ax0b.legend(fontsize=size)

figure_nom = 'Fig_PCA-switching_chimera-fr_before_after_positive_pulse_t1_'+str(t1before)+'.svg'

plt.tight_layout()
plt.savefig(figure_nom, format='svg', bbox_inches='tight',dpi=250)
plt.show()


#%%
fig = plt.figure(figsize = (10,5))
ax = fig.add_subplot(111)

t = 3000
dt = 0.2
nt = int(t/dt)
dtc = 0.1
ntc = int(t/dtc)

color_pca = ['#8c3839', '#2b4743', '#1f7f95', '#da7073', '#dda83f', '#737373', '#3c6ca8', '#5d6c89']


for i in range (3):
    ax.plot(time_pca[:nt], pca[:nt,i*2]+45*i, color=color_pca[i*2], linewidth=1,alpha =0.8) 
    ax.plot(time_c[:ntc], (cvec[:ntc]) - 40,color='grey', linewidth=1)
# ax.set_xlim(800,2500)
ax.set_xlabel(r'time $t$',color='k')


# ax.plot(time_c[:ntc],cvec[:ntc])
# ax.plot(time_pca[:nt],pca[:nt,:3])

