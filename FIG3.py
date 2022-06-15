#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 17:46:00 2021

@author: maria
FIG4: DALE'S LAW, W SPARSE, N 25'
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 
import os
from datetime import date
import seaborn as sns
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

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

"""
1A. Retrieve data: DALES LAW WEIGHTS
"""
NDL = 2500
mu = 0.5 #ratio of E to total 
NE = int(mu*NDL)
NI = NDL - NE
prepath1 = '/Users/maria/cluster_CSG/'
prepath2 = '/home/masoliverm/'
path = prepath2+'Documents/Files&Figures/force_method_1/N_3/RNN_small_pop/dales_law/Results'
os.chdir(path)
nodes = 10
omega_total = np.loadtxt('total_weights_plus_neg_seed_3_N_2500.txt')
omega_perturbation = np.loadtxt('perturbation_weights_plus_neg_seed_3_N_2500.txt')
cmap = sns.color_palette("Spectral", as_cmap=True)

"""
1B. Retrieve data: ETA & PHI SPARSE
"""

path = prepath2+'Documents/Files&Figures/force_method_2/training/twopop/sparsity_eta_and_phi'
os.chdir(path)
omega_learned_sparse = np.loadtxt('omega_learned_3_N_5000500_Q_1.txt')

N = 5000
p = 0.1
g = 1.5
np.random.seed(3)
m = 12
A = np.random.normal(0., 1.0, (N,N)) #NxN Matrix with a normal distribution
B = np.random.rand(N,N)<p #NxN Matrix with a uniform distribution. 
                          #It will return true and false, depending on p. 
omega_initial = g*np.multiply(A,B)/np.sqrt(N*p) #Initial weight matrix
omega_total_sparse = omega_learned_sparse+omega_initial


"""
1C. CHIMERA NETWORK drawing N = 25
"""


G = nx.complete_graph(60)
pos = nx.circular_layout(G)  # positions for all nodes

    
nodelist_1 = np.arange(18,43)
nodelist_2 = np.arange(0,13)
nodelist_2=np.append(nodelist_2,np.arange(48,60))
  
intra_edges=[]
for i in range(len(nodelist_1)):
    for j in range(len(nodelist_2)):
        intra_edges.append((nodelist_1[i], nodelist_2[j]))

inter_edges_1=[]
for i in range(len(nodelist_1)):
    for j in range(len(nodelist_1)):
        if(j!=i):
            inter_edges_1.append((nodelist_1[i], nodelist_1[j]))
        
inter_edges_2=[]
for i in range(len(nodelist_2)):
    for j in range(len(nodelist_2)):
        if(j!=i):
            inter_edges_2.append((nodelist_2[i], nodelist_2[j]))
            
            
            
#%%   

"""
R
"""
def R_v1(spacetime):
    """Returns the Kuramoto order parameter for an TxN numpy array.
    Parameters:
    -----------
    spacetime: TxN numpy array (N: number of nodes, T: time).
               Each row represents a timeseries of the corresponding node.
    Returns:
    --------
    r: Global order parameter (Scalar number at each time)
    """
    scos = np.cos(spacetime).sum(axis=1)
    ssin = np.sin(spacetime).sum(axis=1)
    r = np.sqrt((scos*scos + ssin*ssin)) / (1.0*spacetime.shape[1])
    return r   


"""
1D. SUPERVISOR DATA
(i) N = 3
"""

path = prepath2+'Documents/Files&Figures/force_method_1/N_3/RNN_small_pop/Results/Files'
os.chdir(path)
dt = 0.1
phi_s = np.loadtxt('phi_A_0.1_N_3_beta_0.025_ic_7_TC_1.txt',skiprows=20000)
theta_s = np.loadtxt('theta_A_0.1_N_3_beta_0.025_ic_7_TC_1.txt',skiprows=20000)

# """
# MPV 
# """
from scipy.signal import find_peaks

dtplot = 0.1
deltat = 1000
ntf_output = len(phi_s[:,0])
nodes = len(phi_s[0,1:])*2
ndeltat = int(deltat/dtplot)
ntotal_deltat = int(ntf_output/ndeltat)
ndeltat_vec = np.arange(ntotal_deltat)*ndeltat

phase_sup = np.append(theta_s[:,1:],phi_s[:,1:],axis=1)

j = 0
rotations_phase = np.zeros((ntotal_deltat,nodes))
for l in range (nodes):
    peaks_t = find_peaks(phase_sup[:,l], height=2.5)
    peaks_1 = peaks_t[0]
    for i in range (len(ndeltat_vec)):
        rotations_phase[i, l] = len(np.nonzero((peaks_1<ndeltat_vec[i])*1)[0]) - sum(rotations_phase[:i, l])

mpv_s_t = np.zeros((ntotal_deltat-1,nodes))
for i in range(nodes):
    mpv_s_t[:,i] = (2*np.pi*(rotations_phase[1:,i])/deltat)
    
mean_mpv_s_t = np.mean(mpv_s_t, axis=0)


#########
#R sup
########

rvtheta_sup_t = R_v1(theta_s[0:499999,1:])
rvphi_sup_t = R_v1(phi_s[0:499999,1:])

rvphiround_s_t = np.round(rvphi_sup_t,2)
rvthetaround_s_t = np.round(rvtheta_sup_t,2)


"""
1D. OUTPUT DATA
(ii) N = 3 DALES LAW
"""
path = prepath2+'Documents/Files&Figures/force_method_1/N_3/RNN_small_pop/dales_law/Results'
os.chdir(path)

output = np.loadtxt('output_seed_3.txt')

nodes_t = 3
totaltime = output[:,0]
intime = 0
costheta = output[intime:-1,0:nodes_t] 
cosphi = output[intime:-1,nodes_t:2*nodes_t] 
sintheta = output[intime:-1,2*nodes_t:3*nodes_t] 
sinphi = output[intime:-1,3*nodes_t:4*nodes_t] 

theta_o_t_dl = np.arctan2(sintheta,costheta)
phi_o_t_dl = np.arctan2(sinphi,cosphi)

# """
# MPV OUTPUT
# """

from scipy.signal import find_peaks

dtplot = 0.2
deltat = 1000
ntf_output = len(phi_o_t_dl[:,0])
nodes_f = len(phi_o_t_dl[0,:])*2
ndeltat = int(deltat/dtplot)
ntotal_deltat = int(ntf_output/ndeltat)
ndeltat_vec = np.arange(ntotal_deltat)*ndeltat

phase_t_dl = np.append(theta_o_t_dl,phi_o_t_dl,axis=1)

j = 0
rotations_phase_t = np.zeros((ntotal_deltat,nodes_f))
for l in range (nodes_f):
    peaks_t = find_peaks(phase_t_dl[:,l],  height=2.5)
    peaks_1 = peaks_t[0]
    for i in range (len(ndeltat_vec)):
        rotations_phase_t[i, l] = len(np.nonzero((peaks_1<ndeltat_vec[i])*1)[0]) - sum(rotations_phase_t[:i, l])


mpv_t_dl = np.zeros((ntotal_deltat-1,nodes_f))
for i in range(nodes_f):
    mpv_t_dl[:,i] = (2*np.pi*(rotations_phase_t[1:,i])/deltat)
    
mean_mpv_t_dl = np.mean(mpv_t_dl, axis=0)

#########
#R OUT
########

rvtheta_o_t_dl = R_v1(theta_o_t_dl[0:499999,:])
rvphi_o_t_dl = R_v1(phi_o_t_dl[0:499999,:])

rvphiround_dl = np.round(rvphi_o_t_dl,2)
rvthetaround_dl = np.round(rvtheta_o_t_dl,2)


"""
1D. OUTPUT DATA
(iii) PHI&ETA SPARSE
"""

path = prepath2+'Documents/Files&Figures/force_method_2/training/twopop/sparsity_eta_and_phi'
os.chdir(path)

output = np.loadtxt('output_seed_3_N_5000_Q_1.txt')

nodes_t = 3
totaltime = output[:,0]
intime = 0
costheta = output[intime:-1,0:nodes_t] 
cosphi = output[intime:-1,nodes_t:2*nodes_t] 
sintheta = output[intime:-1,2*nodes_t:3*nodes_t] 
sinphi = output[intime:-1,3*nodes_t:4*nodes_t] 

theta_o_t_sparse = np.arctan2(sintheta,costheta)
phi_o_t_sparse = np.arctan2(sinphi,cosphi)

# """
# MPV OUTPUT
# """

from scipy.signal import find_peaks

dtplot = 0.2
deltat = 1000
ntf_output = len(phi_o_t_sparse[:,0])
nodes_f = len(phi_o_t_sparse[0,:])*2
ndeltat = int(deltat/dtplot)
ntotal_deltat = int(ntf_output/ndeltat)
ndeltat_vec = np.arange(ntotal_deltat)*ndeltat

phase_t_sparse = np.append(theta_o_t_sparse,phi_o_t_sparse,axis=1)

j = 0
rotations_phase_t = np.zeros((ntotal_deltat,nodes_f))
for l in range (nodes_f):
    peaks_t = find_peaks(phase_t_sparse[:,l],  height=2.5)
    peaks_1 = peaks_t[0]
    for i in range (len(ndeltat_vec)):
        rotations_phase_t[i, l] = len(np.nonzero((peaks_1<ndeltat_vec[i])*1)[0]) - sum(rotations_phase_t[:i, l])


mpv_t_sparse = np.zeros((ntotal_deltat-1,nodes_f))
for i in range(nodes_f):
    mpv_t_sparse[:,i] = (2*np.pi*(rotations_phase_t[1:,i])/deltat)
    
mean_mpv_t_sparse = np.mean(mpv_t_sparse, axis=0)

#########
#R
########

rvtheta_o_t_sp = R_v1(theta_o_t_sparse[0:499999,:])
rvphi_o_t_sp = R_v1(phi_o_t_sparse[0:499999,:])

rvphiround_sp = np.round(rvphi_o_t_sp,2)
rvthetaround_sp = np.round(rvtheta_o_t_sp,2)


"""
1D. SUPERVISOR AND OUTPUT DATA
(iv) N = 25
"""


path = prepath2+'Documents/Files&Figures/force_method_1/N_25/RNN_small_pop/Results'
os.chdir(path)
output = np.loadtxt('output_llavor_3.txt')

nodes_tf = 25
totaltime = output[:,0]
intime = 0
costheta = output[intime:-1,0:nodes_tf] 
cosphi = output[intime:-1,nodes_tf:2*nodes_tf] 
sintheta = output[intime:-1,2*nodes_tf:3*nodes_tf] 
sinphi = output[intime:-1,3*nodes_tf:4*nodes_tf] 

theta_o_tf = np.arctan2(sintheta,costheta)
phi_o_tf = np.arctan2(sinphi,cosphi)

# """
# MPV OUTPUT
# """

from scipy.signal import find_peaks

dtplot = 0.2
deltat = 1000
ntf_output = len(phi_o_tf[:,0])
nodes_f = len(phi_o_tf[0,:])*2
ndeltat = int(deltat/dtplot)
ntotal_deltat = int(ntf_output/ndeltat)
ndeltat_vec = np.arange(ntotal_deltat)*ndeltat

phase_tf = np.append(theta_o_tf,phi_o_tf,axis=1)

j = 0
rotations_phase_tf = np.zeros((ntotal_deltat,nodes_f))
for l in range (nodes_f):
    peaks_t = find_peaks(phase_tf[:,l],  height=2.5)
    peaks_1 = peaks_t[0]
    for i in range (len(ndeltat_vec)):
        rotations_phase_tf[i, l] = len(np.nonzero((peaks_1<ndeltat_vec[i])*1)[0]) - sum(rotations_phase_tf[:i, l])


mpv_tf = np.zeros((ntotal_deltat-1,nodes_f))
for i in range(nodes_f):
    mpv_tf[:,i] = (2*np.pi*(rotations_phase_tf[1:,i])/deltat)
    
mean_mpv_tf = np.mean(mpv_tf, axis=0)
#########
#R output
########

rvtheta_o_tf = R_v1(theta_o_tf[0:499999,:])
rvphi_o_tf = R_v1(phi_o_tf[0:499999,:])

rvphiround_otf = np.round(rvphi_o_tf,2)
rvthetaround_otf = np.round(rvtheta_o_tf,2)


# """
# MPV SUP
# """

path = prepath2+'Documents/Files&Figures/force_method_1/N_25/RNN_small_pop/'
os.chdir(path)
dt = 0.1
phi_s_tf = np.loadtxt('phi_A_0.1_N_25_beta_0.025_ic_2_TC_1.txt',skiprows=180000)
theta_s_tf = np.loadtxt('theta_A_0.1_N_25_beta_0.025_ic_2_TC_1.txt',skiprows=180000)

cosphi_s = np.cos(phi_s_tf)
sinphi_s = np.sin(theta_s_tf)

from scipy.signal import find_peaks


deltat = 1000
ntf_sup = len(phi_s_tf[:,0])
nodes_f = len(phi_s_tf[0,1:])*2
ndeltat = int(deltat/dt)
ntotal_deltat = int(ntf_sup/ndeltat)
ndeltat_vec = np.arange(ntotal_deltat)*ndeltat

phase_tf_s = np.append(theta_s_tf[:,1:],phi_s_tf[:,1:],axis=1)

j = 0
rotations_phase_tf_s = np.zeros((ntotal_deltat,nodes_f))
for l in range (nodes_f):
    peaks_t = find_peaks(phase_tf_s[:,l], distance=int(8/dtplot), height=2.5)
    peaks_1 = peaks_t[0]
    for i in range (len(ndeltat_vec)):
        rotations_phase_tf_s[i, l] = len(np.nonzero((peaks_1<ndeltat_vec[i])*1)[0]) - sum(rotations_phase_tf_s[:i, l])


mpv_tf_s = np.zeros((ntotal_deltat-1,nodes_f))
for i in range(nodes_f):
    mpv_tf_s[:,i] = (2*np.pi*(rotations_phase_tf_s[1:,i])/deltat)
    
mean_mpv_tf_s = np.mean(mpv_tf_s, axis=0)

#########
#R SUP
########

rvtheta_s_tf = R_v1(theta_s_tf[0:499999,1:])
rvphi_s_tf = R_v1(phi_s_tf[0:499999,1:])

rvphiround_stf = np.round(rvphi_s_tf,2)
rvthetaround_stf = np.round(rvtheta_s_tf,2)

#%%
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

"""
2. PLOTS
"""
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
path = prepath2+'Documents/Files&Figures/figures-for-articles'
os.chdir(path)
plt.rcParams.update(params)

orange_sup = '#fdae6b'
purple_sup = '#9e9ac8'

orange_out_softer = '#e6550d'
purple_out_softer = '#756bb1','#54278f'

orange_out ='#e6550d'
purple_out = '#54278f'

fig = plt.figure(figsize = (20,15))

#####################
# Dales law drawing
####################
ax = fig.add_subplot(3,4,1)
ax.text(-2,-1,'B',color='k',weight='bold',fontsize = 22)
cmap = sns.color_palette("Spectral", as_cmap=True)
sns.heatmap(omega_total[NE-nodes:NE+nodes,NE-nodes:NE+nodes], square=True,cmap=cmap, vmin = omega_total[NE-nodes:NE+nodes,NE-nodes:NE+nodes].min(), vmax = omega_total[NE-nodes:NE+nodes,NE-nodes:NE+nodes].max(), cbar_kws={'label': 'Weigths w', "shrink": .52})

ax.set_yticks([])
ax.set_xticks([])

###############
# SPARSE W
##############
#We just want to plot the histogram of the values different to zero
#I delete zero values

om_init_resize = np.resize(omega_initial,N*N)
om_total_sparse_resize = np.resize(omega_total_sparse,N*N)

om_in_zeros = np.where(om_init_resize==0)
om_total_sp_zeros = np.where(om_total_sparse_resize==0)

om_init_nozeros = np.delete(om_init_resize, om_in_zeros)
om_total_sp_nozeros = np.delete(om_total_sparse_resize,om_total_sp_zeros)



color_v = ['#128f8b', '#fa6559']
ax2 = fig.add_subplot(3,4,5)
ax2.hist(om_init_nozeros,100, color=color_v[0], density=True)
ax2.hist(om_total_sp_nozeros+1,100,  color=color_v[1], density=True)
ax2.set_title('Non-zero weight distribution',fontsize = 20)
ax2.set_ylabel('Density')
# ax2.set_xlabel(r'$w_0$ weights')
ax2.text(-0.7,7,'G',color='k',weight='bold',fontsize = 22)
ax2.text(-0.45,-1.7,'Pre-learning \n weights',color=color_v[0],fontsize = 20)
ax2.text(0.7,-1.7,'Post-learning \n weights',color=color_v[1],fontsize = 20)
ax2.set_xlim(-0.5,1.5)
ax2.set_xticks([-0.3,0,0.3,0.7,1,1.3])
ax2.set_xticklabels([-0.3,0,0.3,-0.3,0,0.3])
colors = [color_v[0], color_v[0], color_v[0], color_v[1], color_v[1], color_v[1]]
for xtick, color in zip(ax2.get_xticklabels(), colors):
    xtick.set_color(color)


#############################
# Network drawing N = 25
############################

# fig = plt.figure(figsize = (15,15))
orange = '#fdae6b'
purple = '#9e9ac8'
ax3 = fig.add_subplot(3,4,9)
ax3.text(0,1,'L',color='k',fontsize = 22,weight='bold',)
ax3.axis("off")

#############################
# SUPERVISOR OUTPUT N = 3 
# DALES LAW
############################

ti = 55100
tf = ti+250
ntis = int(round((ti)/dt,0))
ntfs = int(round((tf)/dt,0))

nti = int(round(ti/dtplot,0))
ntf = int(round(tf/dtplot,0))
time2 = np.arange(ti,tf,dtplot)
time = np.arange(ti,tf,dt)

delay_pu = 0
ndelay_s_pu = int(delay_pu/dt)
ndelay_o_pu = int(delay_pu/dtplot)

delay_or = -5
ndelay_s_or = int(delay_or/dt)
ndelay_o_or = int(delay_or/dtplot)


ax4 = fig.add_subplot(3,4,2)
ax4.text(55050,47,'C',color='k',fontsize = 22,weight='bold',)

ntis = int(round((ti)/dt,0))
ntfs = int(round((tf)/dt,0))

nti = int(round(ti/dtplot,0))
ntf = int(round(tf/dtplot,0))
time2 = np.arange(ti,tf,dtplot)
time = np.arange(ti,tf,dt)

for j in range (3):
    if (j==0):
        ax4.plot(time2, theta_o_t_dl[nti+delay_or:ntf+delay_or,j]+8*(j), color=orange_out ,linewidth=2, label = ' ') 
        # ax4.plot(time,theta_s[ntis:ntfs,j+1]+8*(j), color=orange_sup, linewidth=5,alpha =0.6, label = ' ')
    else:
        ax4.plot(time2, theta_o_t_dl[nti+delay_or:ntf+delay_or,j]+8*(j), color=orange_out ,linewidth=2) 
        # ax4.plot(time,theta_s[ntis:ntfs,j+1]+8*(j), color=orange_sup, linewidth=5,alpha =0.6) 
    
    if (j==0):
        ax4.plot(time2, phi_o_t_dl[nti+delay_pu:ntf+delay_pu,j]+8*(j+3), color=purple_out , linewidth=2,  label = 'Output') 
        # ax4.plot(time,phi_s[ntis:ntfs,j+1]+8*(j+3), color=purple_sup, linewidth=5,alpha =0.9, label = 'Supervisor') 
    else:
        ax4.plot(time2, phi_o_t_dl[nti+delay_pu:ntf+delay_pu,j]+8*(j+3), color=purple_out , linewidth=2) 
        # ax4.plot(time,phi_s[ntis:ntfs,j+1]+8*(j+3), color=purple_sup, linewidth=5,alpha =0.6) 

# ax4.legend(bbox_to_anchor=(0.09, 0.95),  frameon=False, ncol=2)
ax4.axis("off")
####################
#MPV N = 3 DALES LAW
###################

ax5 = fig.add_subplot(3,4,3)
ax5.text(-0.5,0.42,'D',color='k',fontsize = 22,weight='bold',)

ax5.scatter(np.arange(1,4),np.round(mean_mpv_t_dl[:3],1), s =8**2 , color = orange_out, label = ' ')
ax5.scatter(np.arange(4,7),np.round(mean_mpv_t_dl[3:],1), s =8**2 , color = purple_out, label = 'Output')
ax5.scatter(np.arange(1,4),np.round(mean_mpv_s_t[:3],1), s =5**2 , color = orange)
ax5.scatter(np.arange(4,7),np.round(mean_mpv_s_t[3:],1), s =5**2 , color = purple)
ax5.legend(bbox_to_anchor=(0.55, 1.35),  frameon=False, ncol=1)
plt.xlabel(r'Node $i$', size = '18')

x = [1,2,3,4,5,6]
ax5.set_xticks(x)
xlabels = [1,2,3,1,2,3]
ax5.set_xticklabels(xlabels)
# ax5.set_xticklabels([])


colors = [orange_out, orange_out, orange_out, purple_out, purple_out, purple_out]
for xtick, color in zip(ax5.get_xticklabels(), colors):
    xtick.set_color(color)
    xtick.set_weight('bold')
    
ybox3 = TextArea(r'$\Omega_{\theta_i}, $', textprops=dict(color=orange_out, size=22,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$\Omega_{\phi_i}$', textprops=dict(color=purple_out, size=22,rotation=90,ha='left',va='bottom'))
ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.2 , 0.3), 
                                  bbox_transform=ax5.transAxes, borderpad=0.)

ax5.add_artist(anchored_ybox)  

#############################
# SUPERVISOR OUTPUT N = 3 
# eta & phi sparse
############################

ti = 55100
tf = ti+250
ntis = int(round((ti)/dt,0))
ntfs = int(round((tf)/dt,0))

nti = int(round(ti/dtplot,0))
ntf = int(round(tf/dtplot,0))
time2 = np.arange(ti,tf,dtplot)
time = np.arange(ti,tf,dt)

delay_pu = 0
ndelay_s_pu = int(delay_pu/dt)
ndelay_o_pu = int(delay_pu/dtplot)

delay_or = 0
ndelay_s_or = int(delay_or/dt)
ndelay_o_or = int(delay_or/dtplot)


ax6 = fig.add_subplot(3,4,6)
ax6.text(55050,47,'H',color='k',fontsize = 22,weight='bold',)

ntis = int(round((ti)/dt,0))
ntfs = int(round((tf)/dt,0))

nti = int(round(ti/dtplot,0))
ntf = int(round(tf/dtplot,0))
time2 = np.arange(ti,tf,dtplot)
time = np.arange(ti,tf,dt)

for j in range (3):
    ax6.plot(time2, theta_o_t_sparse [nti+delay_or:ntf+delay_or,j]+8*(j), color=orange_out ,linewidth=2) 
    # ax6.plot(time,theta_s[ntis:ntfs,j+1]+8*(j), color=orange_sup, linewidth=5,alpha =0.6)

    ax6.plot(time2, phi_o_t_sparse[nti+delay_pu:ntf+delay_pu,j]+8*(j+3), color=purple_out , linewidth=2) 
    # ax6.plot(time,phi_s[ntis:ntfs,j+1]+8*(j+3), color=purple_sup, linewidth=5,alpha =0.6) 
# ax6.axis("off")
####################
#MPV N = 3 SPARSE
###################

ax7 = fig.add_subplot(3,4,7)
ax7.text(-0.5,0.42,'I',color='k',fontsize = 22,weight='bold',)

ax7.scatter(np.arange(1,4),np.round(mean_mpv_t_sparse[:3],1), s =8**2 , color = orange_out)
ax7.scatter(np.arange(4,7),np.round(mean_mpv_t_sparse[3:],1), s =8**2 , color = purple_out)
ax7.scatter(np.arange(1,4),np.round(mean_mpv_s_t[:3],1), s =5**2 , color = orange, label = ' ')
ax7.scatter(np.arange(4,7),np.round(mean_mpv_s_t[3:],1), s =5**2 , color = purple, label = 'Supervisor')
ax7.legend(bbox_to_anchor=(0.7, 1.35),  frameon=False, ncol=1)

plt.xlabel(r'Node $i$', size = '18')
# ax9.set_yticks([0.15,0.2,0.25,0.3,0.35])

x = [1,2,3,4,5,6]
ax7.set_xticks(x)
xlabels = [1,2,3,1,2,3]
ax7.set_xticklabels(xlabels)


colors = [orange_out, orange_out, orange_out, purple_out, purple_out, purple_out]
for xtick, color in zip(ax7.get_xticklabels(), colors):
    xtick.set_color(color)
    xtick.set_weight('bold')
    
ybox3 = TextArea(r'$\Omega_{\theta_i}, $', textprops=dict(color=orange_out, size=22,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$\Omega_{\phi_i}$', textprops=dict(color=purple_out, size=22,rotation=90,ha='left',va='bottom'))
ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.2 , 0.3), 
                                  bbox_transform=ax7.transAxes, borderpad=0.)

ax7.add_artist(anchored_ybox)  


#############################
# SUPERVISOR OUTPUT N = 25
############################

ti = 55100
tf = ti+250
ntis = int(round((ti)/dt,0))
ntfs = int(round((tf)/dt,0))

nti = int(round(ti/dtplot,0))
ntf = int(round(tf/dtplot,0))
time2 = np.arange(ti,tf,dtplot)
time = np.arange(ti,tf,dt)

delay_pu = -50
ndelay_s_pu = int(delay_pu/dt)
ndelay_o_pu = int(delay_pu/dtplot)

delay_or = -70
ndelay_s_or = int(delay_or/dt)
ndelay_o_or = int(delay_or/dtplot)


ax8 = fig.add_subplot(3,4,10)
ax8.text(55050,200,'M',color='k',fontsize = 22,weight='bold',)

ntis = int(round((ti)/dt,0))
ntfs = int(round((tf)/dt,0))

nti = int(round(ti/dtplot,0))
ntf = int(round(tf/dtplot,0))
time2 = np.arange(ti,tf,dtplot)
time = np.arange(ti,tf,dt)

for j in range (4):
    ax8.plot(time2, phi_o_tf[nti+delay_or:ntf+delay_or,j]+10*(j), color=orange_out ,linewidth=2) 
    ax8.plot(time2, theta_o_tf[nti+delay_pu:ntf+delay_pu,j]+10*(j+10), color=purple_out , linewidth=2) 

for j in range (6,10):
    ax8.plot(time2, phi_o_tf[nti+delay_or:ntf+delay_or,j]+10*(j), color=orange_out ,linewidth=2) 

    ax8.plot(time2, theta_o_tf[nti+delay_pu:ntf+delay_pu,j]+10*(j+10), color=purple_out , linewidth=2) 
# ax8.legend(bbox_to_anchor=(0.093, 0.95),  frameon=False, ncol=2)
ax8.axis("off")
ax8.text(55225,49,'.',color='k',fontsize = 25)
ax8.text(55225,44,'.',color='k',fontsize = 25)
ax8.text(55225,39,'.',color='k',fontsize = 25)

ax8.text(55225,149,'.',color='k',fontsize = 25)
ax8.text(55225,144,'.',color='k',fontsize = 25)
ax8.text(55225,139,'.',color='k',fontsize = 25)
######
#MPV N = 25
#####

ax9 = fig.add_subplot(3,4,11)
ax9.text(-4,0.52,'N',color='k',fontsize = 22,weight='bold',)

ax9.scatter(np.arange(1,26),np.round(mean_mpv_tf[25:],1), s =8**2 , color = orange_out)
ax9.scatter(np.arange(26,51),np.round(mean_mpv_tf[:25],1), s =8**2 , color = purple_out)
ax9.scatter(np.arange(1,26),np.round(mean_mpv_tf_s[25:],1), s =5**2 , color = orange)
ax9.scatter(np.arange(26,51),np.round(mean_mpv_tf_s[:25],1), s =5**2 , color = purple)

plt.xlabel(r'Node $i$', size = '18')
# ax9.set_yticks([0.15,0.2,0.25,0.3,0.35])

x = [1,10,20,26,35,45]
ax9.set_xticks(x)
xlabels = [1,10,20,1,10,20]
ax9.set_xticklabels(xlabels)


colors = [orange_out, orange_out, orange_out, purple_out, purple_out, purple_out]
for xtick, color in zip(ax9.get_xticklabels(), colors):
    xtick.set_color(color)
    xtick.set_weight('bold')
    
ybox3 = TextArea(r'$\Omega_{\theta_i}, $', textprops=dict(color=orange_out, size=22,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$\Omega_{\phi_i}$', textprops=dict(color=purple_out, size=22,rotation=90,ha='left',va='bottom'))
ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.15 , 0.3), 
                                  bbox_transform=ax9.transAxes, borderpad=0.)

ax9.add_artist(anchored_ybox)  

##################
# Order parameter
#################
ti = 0
tf = 1e5
ntis = int(round((ti)/dt,0))
ntfs = int(round((tf)/dt,0))

nti = int(round(ti/dtplot,0))
ntf = int(round(tf/dtplot,0))
time2 = np.arange(ti,tf,dtplot)
time = np.arange(ti,tf,dt)


axd = fig.add_subplot(3,4,4)
axd.text(1200,1.1,'E',color='k',fontsize = 22,weight='bold',)


axd.plot(time2[:len(rvphi_o_t_dl)], rvphi_o_t_dl,color=purple_out, linewidth=4,alpha =0.8)
axd.plot(time2[:len(rvphi_o_t_dl)], rvtheta_o_t_dl,color=orange_out, linewidth=4,alpha =0.8)
axd.plot(time[:len(rvphi_sup_t)], rvphi_sup_t,color=purple, linewidth=2,alpha =0.8)
axd.plot(time[:len(rvphi_sup_t)], rvtheta_sup_t,color=orange, linewidth=2,alpha =0.8)
plt.xlabel(r'time $t$')

axd.set_ylim([-0.05, 1.05])
axd.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ybox3 = TextArea(r'$R_{\theta}$, ', textprops=dict(color=orange_out, size=22,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$R_{\phi}$', textprops=dict(color=purple_out, size=22,rotation=90,ha='left',va='bottom'))

ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.18, 0.3), 
                                  bbox_transform=axd.transAxes, borderpad=0.)

axd.add_artist(anchored_ybox)  
axd.set_xlim(1200,1300)
axd.margins(0.8)
axe = fig.add_subplot(3,4,8)
axe.text(1200,1.1,'J',color='k',fontsize = 22,weight='bold',)

axe.plot(time2[:len(rvphi_o_t_sp)], rvphi_o_t_sp,color=purple_out, linewidth=4,alpha =0.8)
axe.plot(time2[:len(rvphi_o_t_sp)], rvtheta_o_t_sp,color=orange_out, linewidth=4,alpha =0.8)
axe.plot(time[:len(rvphi_sup_t)], rvphi_sup_t,color=purple, linewidth=2,alpha =0.8)
axe.plot(time[:len(rvphi_sup_t)], rvtheta_sup_t,color=orange, linewidth=2,alpha =0.8)
plt.xlabel(r'time $t$')

axe.set_ylim([-0.05, 1.05])
axe.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ybox3 = TextArea(r'$R_{\theta}$, ', textprops=dict(color=orange_out, size=22,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$R_{\phi}$', textprops=dict(color=purple_out, size=22,rotation=90,ha='left',va='bottom'))

ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.18, 0.3), 
                                  bbox_transform=axd.transAxes, borderpad=0.)

axe.add_artist(anchored_ybox)  

axe.set_xlim(1200,1300)
axe.margins(0.20)

axf = fig.add_subplot(3,4,12)
axf.margins(0.20)
axf.text(1200,1.1,'0',color='k',fontsize = 22,weight='bold',)

axf.plot(time2[:len(rvphi_o_tf)], rvphi_o_tf,color=orange_out, linewidth=4,alpha =0.8)
axf.plot(time2[:len(rvphi_o_tf)], rvtheta_o_tf,color=purple_out, linewidth=4,alpha =0.8)
axf.plot(time[:len(rvphi_s_tf[135:])], rvphiround_stf[135:],color=orange, linewidth=2,alpha =0.8)
axf.plot(time[:len(rvphi_s_tf[135:])], rvthetaround_stf[135:],color=purple, linewidth=2,alpha =0.8)
plt.xlabel(r'time $t$')

axf.set_ylim([-0.05, 1.05])
axf.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ybox3 = TextArea(r'$R_{\theta}$, ', textprops=dict(color=orange_out, size=22,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$R_{\phi}$', textprops=dict(color=purple_out, size=22,rotation=90,ha='left',va='bottom'))

ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.18, 0.3), 
                                  bbox_transform=axd.transAxes, borderpad=0.)

axf.add_artist(anchored_ybox)  

axf.set_xlim(1200,1300)





path = prepath2+'Documents/Files&Figures/figures-for-articles/2021'
os.chdir(path)

today = date.today()
figure_nom = 'FIG4_'+str(today)+'.svg'

plt.tight_layout()
# , bbox_inches='tight'
plt.savefig(figure_nom,format='svg',bbox_inches='tight',dpi=300)
plt.show()


#%%
###############
# SPARSE W
##############
#We just want to plot the histogram of the values different to zero
#I delete zero values

# from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

# """
# 2. PLOTS
# """
# params = {
#     'axes.labelsize': 22,
#         'xtick.labelsize': 22,
#             'ytick.labelsize': 22,
#                 'legend.fontsize': 22,
                    
#                     'text.usetex': False,
#                         #'font': 'Helvetica',
#                         'mathtext.bf': 'helvetica:bold',

#                     }

# plt.rcParams.update(params)
# path = '/home/maria/cluster_CSG/Documents/Files&Figures/figures-for-articles'
# os.chdir(path)
# plt.rcParams.update(params)

# orange_sup = '#fdae6b'
# purple_sup = '#9e9ac8'

# orange_out_softer = '#e6550d'
# purple_out_softer = '#756bb1','#54278f'

# orange_out ='#e6550d'
# purple_out = '#54278f'

# fig = plt.figure(figsize = (5,5))

# om_init_resize = np.resize(omega_initial,N*N)
# om_total_sparse_resize = np.resize(omega_total_sparse,N*N)

# om_in_zeros = np.where(om_init_resize==0)
# om_total_sp_zeros = np.where(om_total_sparse_resize==0)

# om_init_nozeros = np.delete(om_init_resize, om_in_zeros)
# om_total_sp_nozeros = np.delete(om_total_sparse_resize,om_total_sp_zeros)



# color_v = ['#128f8b', '#fa6559']
# ax2 = fig.add_subplot(1,1,1)
# ax2.hist(om_init_nozeros,100, color=color_v[0], density=True)
# ax2.hist(om_total_sp_nozeros+1,100,  color=color_v[1], density=True)
# ax2.set_title('Non-zero weight distribution')
# ax2.set_ylabel('Density')
# # ax2.set_xlabel(r'$w_0$ weights')
# #ax2.text(-0.7,135,'G',color='k',weight='bold',fontsize = 22)
# ax2.text(-0.45,-50000,r'Pre-learning weights',color=color_v[0],fontsize = 20)
# ax2.text(0.7,-50000,r'Post-learning weights',color=color_v[1],fontsize = 20)
# ax2.set_xlim(-0.5,1.5)
# ax2.set_xticks([-0.3,0,0.3,0.7,1,1.3])
# ax2.set_xticklabels([-0.3,0,0.3,-0.3,0,0.3])
# colors = [color_v[0], color_v[0], color_v[0], color_v[1], color_v[1], color_v[1]]
# for xtick, color in zip(ax2.get_xticklabels(), colors):
#     xtick.set_color(color)

# path = '/home/maria/cluster_CSG/Documents/Files&Figures/figures-for-articles/2021'
# os.chdir(path)

# today = date.today()
# figure_nom = 'subfirgure_FIG4_'+str(today)+'.svg'

# # plt.tight_layout()
# # , bbox_inches='tight'
# plt.savefig(figure_nom, format='svg',dpi=300)
# plt.show()


# axa = fig.add_subplot(3,5,1)
# axa.text(0,1,'A',color='k',fontsize = 22,weight='bold',)
# axa.axis("off")
# axb = fig.add_subplot(3,5,6)
# axb.text(0,1,'F',color='k',fontsize = 22,weight='bold',)
# axb.axis("off")
# axc = fig.add_subplot(3,5,11)
# axc.text(0,1,'K',color='k',fontsize = 22,weight='bold',)
# axc.axis("off")

