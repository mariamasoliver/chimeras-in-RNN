#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 15:51:57 2021
@author: maria

FIG 3
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 
import os
from datetime import date


"""
1. Retrieve data: output and supervisors
"""

# prepath1 = '/home/maria/cluster_CSG'
prepath1 = '/Users/maria/cluster_CSG'
prepath2 = '/home/masoliverm/' 
path = prepath1 +'/Documents/Files&Figures/force_method_1/N_3/retrain/Results'
os.chdir(path)

dtplot = 0.2

fr = np.loadtxt('firing_rates_plot_3.txt')

output = np.loadtxt('output_plot_3.txt')

nodes = 3
totaltime = output[:,0]
intime = 0
costheta = output[intime:-1,0:nodes] 
cosphi = output[intime:-1,nodes:2*nodes] 
sintheta = output[intime:-1,2*nodes:3*nodes] 
sinphi = output[intime:-1,3*nodes:4*nodes] 

theta_o = np.arctan2(sintheta,costheta)
phi_o = np.arctan2(sinphi,cosphi)


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


path = prepath1+'/Documents/Files&Figures/force_method_1/N_3/RNN_small_pop/Results/Files'
os.chdir(path)
dt = 0.1
phi_s = np.loadtxt('phi_A_0.1_N_3_beta_0.025_ic_7_TC_1.txt',skiprows=20000)
theta_s = np.loadtxt('theta_A_0.1_N_3_beta_0.025_ic_7_TC_1.txt',skiprows=20000)

cosphi_s = np.cos(phi_s)
sinphi_s = np.sin(theta_s)

"""
MPV & ORDER PARAMETER R
"""

"""
Rotations
"""
from scipy.signal import find_peaks

dtplot = 0.2
deltat = 1000
ntf_output = len(phi_o[:,0])
nodes = len(phi_o[0,:])*2
ndeltat = int(deltat/dtplot)
ntotal_deltat = int(ntf_output/ndeltat)
ndeltat_vec = np.arange(ntotal_deltat)*ndeltat

phase = np.append(theta_o,phi_o,axis=1)

j = 0
rotations_phase = np.zeros((ntotal_deltat,nodes))
for l in range (nodes):
    peaks_t = find_peaks(phase[:,l],  height=2.5)
    peaks_1 = peaks_t[0]
    for i in range (len(ndeltat_vec)):
        rotations_phase[i, l] = len(np.nonzero((peaks_1<ndeltat_vec[i])*1)[0]) - sum(rotations_phase[:i, l])

      
"""
MPV
"""

mpv = np.zeros((ntotal_deltat-1,nodes))
for i in range(nodes):
    mpv[:,i] = (2*np.pi*(rotations_phase[1:,i])/deltat)
    
mean_mpv = np.mean(mpv, axis=0)

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

t = 0.1
ntf_output = len(phi_o[:,0])

rvtheta = R_v1(theta_o[249999:499999,:])
rvphi = R_v1(phi_o[249999:499999,:])


rvphiround = np.round(rvphi,2)
rvthetaround = np.round(rvtheta,2)

"""
NETWORK construction (as a drawing)
"""

p = 0.05
g = 1.5

np.random.seed(3)
N = 20
A = np.random.normal(0., 1.0, (N,N)) #NxN Matrix with a normal distribution
B = np.random.rand(N,N)<p #NxN Matrix with a uniform distribution. 
                          #It will return true and false, depending on p. 
omega = g*np.multiply(A,B)/np.sqrt(N*p) #Initial weight matrix
positiu = np.where(omega>0)
negatiu = np.where(omega<0)

#OMEGA_0 (95 % sparse: p = 0.05)
edges_fixed=[]
for i in range(len(positiu[0])):
    edges_fixed.append((positiu[0][i], positiu[1][i]))
for i in range(len(negatiu[0])):
    edges_fixed.append((negatiu[0][i], negatiu[1][i]))

G_initial = nx.Graph()
G_initial.add_edges_from(edges_fixed)


#OMEGA_LEARNED: 85% sparse (only 15% of the total connections: 80 connections)
edges_learned = []
for i in range(N+10):
    irand = np.random.randint(0,20,1)
    jrand = np.random.randint(0,20,1)
    if (jrand == irand):
        jrand = np.random.randint(0,20,1)
    edges_learned.append((irand[0],jrand[0]))
        
G_l = nx.Graph()
G_l.add_edges_from(edges_learned)
edges_learned = list(G_l.edges)

G_initial.add_nodes_from(G_l)
nodes_num = G_initial.number_of_nodes()



#%%

params = {
    'axes.labelsize': 22,
        'xtick.labelsize': 22,
            'ytick.labelsize': 22,
                'legend.fontsize': 22,
                'axes.titlesize':22,
                    'text.usetex': False,
                        #'font': 'Helvetica',
                        'mathtext.bf': 'helvetica:bold',

                    }

plt.rcParams.update(params)


from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

# VERSION 1: OUTPUT I SUPERVISOR ARE COSPHI COSTHETA 
path = prepath1+'/Documents/Files&Figures/figures-for-articles/2021'
os.chdir(path)
fig = plt.figure(figsize = (20,10))

edges_l_cmap = cm.get_cmap('twilight_shifted')
edges_f_cmap = cm.get_cmap('twilight')
nodes_cmap = cm.get_cmap('bone')
redblue_cmap = cm.get_cmap('RdBu')
red = '#c84c81ff'
blue = '#9cb9faff'

red='#cb181d'
blue = '#2171b5'
color_edges_l = []
for i in range(len(edges_learned)):
    c = int(np.random.randint(1,3))
    if (c==1):
        color_edges_l.append(red)
    else:
        color_edges_l.append(blue)

color_edges_f = []
for i in range(len(edges_fixed)):
    c = int(np.random.randint(1,3))
    if (c==1):
        color_edges_f.append(red)
    else:
        color_edges_f.append(blue)

step = (0.9-0.1)/N
color_nodes_and_fr = []
for i in range(N):
    ind = 0.1+i*step
    color_nodes_and_fr.append(nodes_cmap(ind))
    
step = 0.8/(len(edges_learned)+len(edges_fixed))
redblue_color = []
for i in range(0,int((len(edges_learned)+len(edges_fixed)))//2):
    ind = i*step
    redblue_color.append(redblue_cmap(ind))

for i in range(int((len(edges_learned)+len(edges_fixed)))//2):
    ind =1- i*step
    redblue_color.append(redblue_cmap(ind))

import random
random.shuffle(redblue_color)


trans_f = np.random.default_rng().uniform(0.2,1,len(edges_fixed))
trans_l = np.random.default_rng().uniform(0.2,1,len(edges_learned))


#################
#Network drawing    
################
ax = fig.add_subplot(2,4,1)

ax.text(-1.2,1.17,'A',color='k',fontsize = 22,weight='bold')
ax.set_title(r'$w_0$ ')
pos = nx.circular_layout(G_initial)
nx.draw_networkx_nodes(G_initial, pos, node_size=350,alpha=0.98,node_color=color_nodes_and_fr)
for i in range(len(edges_fixed)):
    nx.draw_networkx_edges(G_initial, pos, edgelist=[edges_fixed[i]], width=4,alpha=trans_f[i] , edge_color=color_edges_f[i])
plt.axis("off")

axb = fig.add_subplot(2,4,2)

axb.set_title(r'$w_0$ + $\eta d^T$ ')
axb.text(-1.4,1.17,'C',color='k',fontsize = 22,weight='bold',)

pos2 = nx.circular_layout(G_initial)
nx.draw_networkx_nodes(G_initial, pos2, node_size=350,alpha=0.98,node_color=color_nodes_and_fr)
for i in range(len(edges_fixed)):
    nx.draw_networkx_edges(G_initial, pos, edgelist=[edges_fixed[i]], width=4,alpha=trans_f[i] , edge_color=color_edges_f[i])
for i in range(len(edges_learned)):
    nx.draw_networkx_edges(G_initial, pos2, edgelist=[edges_learned[i]], width=2, alpha=trans_l[i], edge_color=color_edges_l[i])
    
plt.axis("off")



orange_cmap = cm.get_cmap('Oranges')
purpule_cmap = cm.get_cmap('Purples')

#######################
#Output & supervisor plot: theta plots#  
#######################

orange_sup = '#fdae6b'
purple_sup = '#9e9ac8'

orange_out_softer = '#e6550d'
purple_out_s = '#756bb1'

orange_out ='#e6550d'
purple_out = '#54278f'


ax1 = fig.add_subplot(2,4,3)
ax1.text(600,18,'E',color='k',fontsize = 22,weight='bold',)

# pos1 = ax1.get_position() # get the original position
# pos2 = [pos1.x0, pos1.y0-0.2,  pos1.width, pos1.height]
# ax1.set_position(pos2) # set a new position

# ax1.text(660,34,'A',color='k',fontsize = 22,weight='bold',)

ti = 600
tf = 1000
ntis = int(round((ti)/dt,0))
ntfs = int(round((tf)/dt,0))

nti = int(round(ti/dtplot,0))
ntf = int(round(tf/dtplot,0))
time2 = np.arange(ti,tf,dtplot)
time = np.arange(ti,tf,dt)
delay = -0
ndelay_s = int(delay/dt)
ndelay_o = int(delay/dtplot)

for j in range (3):  
    ax1.plot(time2, cosphi[nti-ndelay_o:ntf-ndelay_o,j]+3*(j+3), color=purple_out_s ,linewidth=3) 
    ax1.plot(time,np.cos(phi_s[ntis:ntfs,j+1])+3*(j+3), color='k', linestyle = 'dashed', linewidth=1)
    
    # ax1.plot(time2, sinphi[nti-ndelay_o:ntf-ndelay_o,j]+3*(j+6), color=purple_out ,linewidth=2) 
    # ax1.plot(time,np.sin(phi_s[ntis:ntfs,j+1])+3*(j+6), color=purple_sup, linewidth=5,alpha =0.6) 

    if (j==0):
        ax1.plot(time2, costheta[nti-ndelay_o:ntf-ndelay_o,j]+3*(j), color=orange_out ,linewidth=3, label = r' ') 
        ax1.plot(time,np.cos(theta_s[ntis:ntfs,j+1])+3*(j), color='k', linestyle = 'dashed',linewidth=1, label = r'$\cos{{\phi}_i(t)}$, $\cos{\theta}_i(t)$,') 
    else:
        ax1.plot(time2, costheta[nti-ndelay_o:ntf-ndelay_o,j]+3*(j), color=orange_out ,linewidth=3) 
        ax1.plot(time,np.cos(theta_s[ntis:ntfs,j+1])+3*(j), color='k', linestyle = 'dashed', linewidth=1)
    
    # ax1.plot(time2, sintheta[nti-ndelay_o:ntf-ndelay_o,j]+3*(j), color=orange_out ,linewidth=2) 
    # ax1.plot(time,np.sin(theta_s[ntis:ntfs,j+1])+3*(j), color=orange_sup, linewidth=5,alpha =0.6) 
# ax1.axis("off")
ax1.set_xlabel(r' time $t$')
ybox3 = TextArea(r'$\cos\hat{\theta}_i(t)$, ', textprops=dict(color=orange_out, size=22,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$\cos{\hat{\phi}_i(t)}$', textprops=dict(color=purple_out_s, size=22,rotation=90,ha='left',va='bottom'))

ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.18, 0.1), 
                                  bbox_transform=ax1.transAxes, borderpad=0.)

ax1.add_artist(anchored_ybox)  

ax1.legend(bbox_to_anchor=(0.1, 0.95),  frameon=False, ncol=1)

ax2 = fig.add_subplot(2,4,7)
ax2.text(600,48,'F',color='k',fontsize = 22,weight='bold',)

ntis = int(round((ti)/dt,0))
ntfs = int(round((tf)/dt,0))

nti = int(round(ti/dtplot,0))
ntf = int(round(tf/dtplot,0))
time2 = np.arange(ti,tf,dtplot)
time = np.arange(ti,tf,dt)

for j in range (3):
 
    ax2.plot(time2, theta_o[nti:ntf,j]+8*(j), color=orange_out ,linewidth=3) 
    ax2.plot(time,theta_s[ntis:ntfs,j+1]+8*(j), color='k',linestyle = 'dashed', linewidth=1) 
    
    if (j==0):
        ax2.plot(time2, phi_o[nti:ntf,j]+8*(j+3), color=purple_out_s , linewidth=3,  label = r' ') 
        ax2.plot(time,phi_s[ntis:ntfs,j+1]+8*(j+3), color='k',linestyle = 'dashed', linewidth=1, label = r'${\phi}_i(t)$, ${\theta}_i(t)$') 
    else:
        ax2.plot(time2, phi_o[nti:ntf,j]+8*(j+3), color=purple_out_s , linewidth=3) 
        ax2.plot(time,phi_s[ntis:ntfs,j+1]+8*(j+3), color='k', linestyle = 'dashed',linewidth=1) 

plt.xlabel(r' time $t$')

ax2.legend(bbox_to_anchor=(0.1, 0.95),  frameon=False, ncol=1)
ybox3 = TextArea(r'$\hat{\theta}_i(t)$, ', textprops=dict(color=orange_out, size=22,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$\hat{\phi}_i(t)$', textprops=dict(color=purple_out_s, size=22,rotation=90,ha='left',va='bottom'))

ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.18, 0.3), 
                                  bbox_transform=ax2.transAxes, borderpad=0.)

ax2.add_artist(anchored_ybox)  


######
#MPV
#####

ax6 = fig.add_subplot(2,4,4)
ax6.text(0.8,0.42,'G',color='k',fontsize = 22,weight='bold',)

ax6.scatter(np.arange(1,4),np.round(mean_mpv[:3],2), s =9**2 , color = orange_out)
ax6.scatter(np.arange(4,7),np.round(mean_mpv[3:],2), s =9**2 , color = purple_out_s)

plt.xlabel(r'Node $i$')

ax6.set_ylim(0.1,0.4)
# ax6.set_yticks([0.15,0.2,0.25,0.3,0.35])
ax6.set_xticks([1,2,3,4,5,6])
ax6.set_xticklabels([1,2,3,1,2,3])


colors = [orange_out, orange_out, orange_out, purple_out_s, purple_out_s, purple_out_s]
for xtick, color in zip(ax6.get_xticklabels(), colors):
    xtick.set_color(color)
    xtick.set_weight('bold')
    
ybox3 = TextArea(r'$\Omega_{\hat{\theta}_i}, $', textprops=dict(color=orange_out, size=22,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$\Omega_{\hat{\phi}_i}$', textprops=dict(color=purple_out_s, size=22,rotation=90,ha='left',va='bottom'))
ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.25 , 0.3), 
                                  bbox_transform=ax6.transAxes, borderpad=0.)

ax6.add_artist(anchored_ybox)  

######
#R
#####
ax7 = fig.add_subplot(2,4,8)
ax7.text(700,1.1,'H',color='k',weight='bold',fontsize = 22)

ax7.plot(time2, rvphi[nti:ntf],color=purple_out, linewidth=4,alpha =0.8)
ax7.plot(time2, rvtheta[nti:ntf],color=orange_out, linewidth=4,alpha =0.8)
plt.xlabel(r'time $t$')

ax7.set_ylim([-0.05, 1.05])
ax7.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ybox3 = TextArea(r'$R_{\hat{\theta}}$, ', textprops=dict(color=orange_out, size=22,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$R_{\hat{\phi}}$', textprops=dict(color=purple_out_s, size=22,rotation=90,ha='left',va='bottom'))

ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.18, 0.3), 
                                  bbox_transform=ax7.transAxes, borderpad=0.)

ax7.add_artist(anchored_ybox)  

ax7.set_xlim(700,800)

#############
#FIRING RATES (caotic regime)
#############
axc = fig.add_subplot(2,4,5)
axc.text(0,24,'B',color='k',fontsize = 22,weight='bold',)

ti = 0
tf = 400

nti = int(round(ti/dtplot,0))
ntf = int(round(tf/dtplot,0))
time2 = np.arange(0,len(fr[nti:ntf,0])*dtplot,dtplot)
for j in range(5):
    axc.plot(time2, (fr[nti:ntf,j]+1)+2.2*(j),color=color_nodes_and_fr[-1-j], linewidth=2) 
for j in range(5,10):
    axc.plot(time2, (fr[nti:ntf,j]+1)+2.2*(j),color=color_nodes_and_fr[10-(j+1)], linewidth=2) 
    

plt.ylabel(r' Firing rates $r_j(t)$')
plt.xlabel(r' time $t$')

#############
#FIRING RATES (regular regime)
#############
axd = fig.add_subplot(2,4,6)
axd.set_title(r'$d_{i1}\cdot r_1(t)+\cdots+d_i\cdot r_N=\cos\theta_i$ ')
# axd.text(0,24,'D',color='k',fontsize = 22,weight='bold')

ti = 90000
tf = 90400

nti = int(round(ti/dtplot,0))
ntf = int(round(tf/dtplot,0))
time2 = np.arange(0,len(fr[nti:ntf,0])*dtplot,dtplot)
for j in range(5):
    axd.plot(time2, (fr[nti:ntf,j]+1)+2.2*(j),color=color_nodes_and_fr[-1-j], linewidth=2) 
for j in range(5,10):
    axd.plot(time2, (fr[nti:ntf,j]+1)+2.2*(j),color=color_nodes_and_fr[10-(j+1)], linewidth=2) 

plt.ylabel(r' Firing rates $r_j(t)$')
plt.xlabel(r' time $t$')

path = prepath1+'/Documents/Files&Figures/figures-for-articles/2021'
os.chdir(path)

today = date.today()
figure_nom = 'FIG3_'+str(today)+'_transparency_v2.svg'

plt.tight_layout()
plt.savefig(figure_nom, format='svg', bbox_inches='tight',dpi=300)
plt.show()
# plt.xlim(0,500)

