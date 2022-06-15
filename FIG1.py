#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 15:05:30 2021

@author: maria
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 
import os
from datetime import date

"""
1. CHIMERA NETWORK drawing
"""
G = nx.complete_graph(6)
pos = nx.circular_layout(G)  # positions for all nodes

# nodes
nodelist_1 = [2,3,4]
nodelist_2 = [1,0,5]

intra_edges=[]
for i in range(len(nodelist_1)):
    for j in range(len(nodelist_2)):
        intra_edges.append((nodelist_1[i], nodelist_2[j]))

"""
2A. Retrieve data: chimera 
"""

path = '/home/masoliverm/Documents/Files&Figures/force_method_1/N_3/RNN_small_pop/Results/Files'
os.chdir(path)
dt = 0.1
phi_s = np.loadtxt('phi_A_0.1_N_3_beta_0.025_ic_7_TC_1.txt',skiprows=20000)
theta_s = np.loadtxt('theta_A_0.1_N_3_beta_0.025_ic_7_TC_1.txt',skiprows=20000)

"""
2B. MPV & ORDER PARAMETER R
"""

"""
Rotations
"""
from scipy.signal import find_peaks

dtplot = 0.1
deltat = 1000
ntf_output = len(phi_s[:,0])
nodes = len(phi_s[0,1:])*2
ndeltat = int(deltat/dtplot)
ntotal_deltat = int(ntf_output/ndeltat)
ndeltat_vec = np.arange(ntotal_deltat)*ndeltat

phase = np.append(theta_s[:,1:],phi_s[:,1:],axis=1)

j = 0
rotations_phase = np.zeros((ntotal_deltat,nodes))
for l in range (nodes):
    peaks_t = find_peaks(phase[:,l], height=2.5)
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
ntf_output = len(phi_s[:,0])
theta = theta_s[:,1:]
phi = phi_s[:,1:]


rvtheta = R_v1(theta)
rvphi = R_v1(phi)


rvphiround = np.round(rvphi,2)
rvthetaround = np.round(rvtheta,2)
#%%
# import matplotlib.gridspec as gridspec
# gs1 = gridspec.GridSpec(32, 5)
# gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

"""
3. PLOT VERSION A (scheme, snapshot, polar-plot, time-series)
"""
path = '/home/masoliverm/Documents/Files&Figures/figures-for-articles/2021'
os.chdir(path)


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


dtplot = 0.1


orange = '#fdae6b'
purple = '#9e9ac8'

orange_cmap = cm.get_cmap('Oranges')
purpule_cmap = cm.get_cmap('Purples')

step = (0.8-0.2)/len(nodelist_1)

color_nodes_o = []
color_nodes_p = []
for i in range(3):
    ind = 0.2+i*step
    color_nodes_o.append(orange_cmap(ind))
    color_nodes_p.append(purpule_cmap(ind))
    
fig = plt.figure(figsize = (30,5))

################
#NETWORK DRAWING
################
ax1 = fig.add_subplot(1, 6, 1)
ax1.text(-0.6,1.05,'A',color='k',fontsize = 22,weight='bold',)
options = {"node_size": 600}
nx.draw_networkx_edges(G, pos, edgelist=intra_edges, width=1, style="dashed", edge_color='k')
nx.draw_networkx_nodes(G, pos, nodelist=nodelist_1, node_color=orange, **options)
nx.draw_networkx_nodes(G, pos, nodelist=nodelist_2, node_color=purple,**options)

# edges
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[(2,3), (3,4), (4,2)],
    width=3,
    edge_color='k',
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[(1,0),(0,5),(5,1)],
    width=3,
    edge_color='k',
)

ax1.axis("off")

###################
#SNAPSHOT
##################
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker


ti = 4900
tf = ti+300
nti = int(round(ti/dtplot,0))
ntf = int(round(tf/dtplot,0))
time2 = np.arange(ti,tf,dtplot)

y = np.zeros(6)
for j in range (3):
    y[j] = theta_s[nti,j+1]
    y[j+3] = phi_s[nti,j+1]
    
ax2 = fig.add_subplot(1, 6, 2)
pos1 = ax2.get_position() # get the original position
pos2 = [pos1.x0 + 0.05, pos1.y0,  pos1.width-0.025, pos1.height]
ax2.set_position(pos2) # set a new position

for j in range (3):
    ind = j*step +0.2
    ax2.scatter(np.arange(1,4), y[:3], s =9**2 , color = orange)     
    ax2.scatter(np.arange(4,7), y[3:],s = 9**2, color = purple) 


ax2.text(0.9,3.7,'B',color='k',fontsize = 22,weight='bold',)
ax2.set_ylim(-3.5,3.5)
ax2.set_xlabel('Nodes $n$') 
     
ybox3 = TextArea(r'$\theta_i(t = 4500), $', textprops=dict(color=orange, size=18,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$\phi_i(t = 4500)$', textprops=dict(color=purple, size=18,rotation=90,ha='left',va='bottom'))

ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.14, 0.05), 
                                  bbox_transform=ax2.transAxes, borderpad=0.)

ax2.add_artist(anchored_ybox)     

# ax3.axis("off")

orange = '#fdae6b'
purple = '#9e9ac8'

#############
#POLAR PLOT
#############
ax3 = fig.add_subplot(1, 6, 3,projection='polar')
ax3.text(-180,1.25,'C',color='k',fontsize = 22,weight='bold',)
ax3.set_title(r'$\theta_i(t = $'+str(int(ti))+'$)$',fontsize = 22)
for j in range (3):
    ind = j*step +0.2
    ax3.plot(theta_s[nti,j+1],1,'o',color = orange,markersize = 9) 
ax3.set_yticklabels([])

ax3.set_xticklabels([r'0, 2$\pi$','', r'$\frac{\pi}{2}$','', r'$\pi$', '', r'$\frac{3\pi}{2}$'])
ax3.grid(True)


# Create second axes, the top-left plot with orange plot
ax4 = fig.add_subplot(1,6,4, projection='polar') # two rows, two columns, second cell
ax4.set_title(r'$\phi_i(t = $'+str(int(ti))+'$)$',fontsize = 22)
for j in range(3):
    ind = j*step +0.2
    ax4.plot(phi_s[nti,j+1],1,'o',color=purple,markersize = 9) 
ax4.set_yticklabels([])
# ax2.set_xticks([0, 270, 180, 90])
ax4.set_xticklabels([r'0, 2$\pi$','', r'$\frac{\pi}{2}$','', r'$\pi$', '', r'$\frac{3\pi}{2}$'])
ax4.grid(True)
    # Create third axes, a combination of third and fourth cell


#############
#Time-series
#############
ax5 = fig.add_subplot(1,6,(5,6))
ax5.text(4490,47,'D',color='k',weight='bold',fontsize = 22)
for j in range (3):
    ind = j*step +0.2  
    ax5.plot(time2, phi_s[nti:ntf,j+1]+8*(j+3), color=purple, linewidth=4)
    ax5.plot(time2, theta_s[nti:ntf,j+1]+8*(j), color=orange, linewidth=4)
ax5.set_xlabel('time $t$') 
ax5.set_yticklabels([])
ax5.set_yticks([])    

ybox3 = TextArea(r'$\theta_i(t), $', textprops=dict(color=orange, size=18,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$\phi_i(t)$', textprops=dict(color=purple, size=18,rotation=90,ha='left',va='bottom'))

ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.025, 0.3), 
                                  bbox_transform=ax5.transAxes, borderpad=0.)

ax5.add_artist(anchored_ybox)  

path = '/home/masoliverm/Documents/Files&Figures/figures-for-articles/2021'
os.chdir(path)

today = date.today()
figure_nom = 'FIG1_'+str(today)+'.png'
plt.tight_layout()
plt.savefig(figure_nom, bbox_inches='tight',format='png', dpi=300)
plt.show()

#%%
"""
4. PLOT VERSION B (scheme, snapshot, polar-plot, time-series, MPV, R)
"""
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


dtplot = 0.1


orange = '#fdae6b'
purple = '#9e9ac8'

orange_cmap = cm.get_cmap('Oranges')
purpule_cmap = cm.get_cmap('Purples')

step = (0.8-0.2)/len(nodelist_1)

color_nodes_o = []
color_nodes_p = []
for i in range(3):
    ind = 0.2+i*step
    color_nodes_o.append(orange_cmap(ind))
    color_nodes_p.append(purpule_cmap(ind))
    
fig = plt.figure(figsize = (25,10))

################
#NETWORK DRAWING
################
ax1 = fig.add_subplot(2, 5, 3)
pos1 = ax1.get_position() # get the original position
pos2 = [pos1.x0 , pos1.y0,  pos1.width, pos1.height]
ax1.set_position(pos2)
ax1.text(-1.3,1.175,'B',color='k',fontsize = 22,weight='bold',)
options = {"node_size": 600}
ax1.set_title('Kuramoto oscillators \n network')
nx.draw_networkx_edges(G, pos, edgelist=intra_edges, width=1, style="dashed", edge_color='k')
nx.draw_networkx_nodes(G, pos, nodelist=nodelist_1, node_color=orange, **options)
nx.draw_networkx_nodes(G, pos, nodelist=nodelist_2, node_color=purple,**options)

# edges
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[(2,3), (3,4), (4,2)],
    width=3,
    edge_color='k',
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[(1,0),(0,5),(5,1)],
    width=3,
    edge_color='k',
)

ax1.axis("off")

###################
#SNAPSHOT
##################
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker


ti = 4900
tf = ti+300
nti = int(round(ti/dtplot,0))
ntf = int(round(tf/dtplot,0))
time2 = np.arange(ti,tf,dtplot)

y = np.zeros(6)
for j in range (3):
    y[j] = theta_s[nti,j+1]
    y[j+3] = phi_s[nti,j+1]
    
ax2 = fig.add_subplot(2, 5, 6)
ax2.set_title('Ocillators dynamics: \n Snapshot')
pos1 = ax2.get_position() # get the original position
pos2 = [pos1.x0 + 0.05, pos1.y0,  pos1.width-0.025, pos1.height]
ax2.set_position(pos2) # set a new position


ax2.scatter(np.arange(1,4), y[:3], s =9**2 , color = orange)     
ax2.scatter(np.arange(4,7), y[3:],s = 9**2, color = purple) 

ax2.set_xticks([1,2,3,4,5,6])
ax2.set_xticklabels([1,2,3,1,2,3])

colors = [orange, orange, orange, purple, purple, purple]
for xtick, color in zip(ax2.get_xticklabels(), colors):
    xtick.set_color(color)
    xtick.set_weight('bold')

ax2.text(0.3,4.32,'C',color='k',fontsize = 22,weight='bold',)
ax2.set_ylim(-3.5,3.5)
ax2.set_xlabel('Nodes $i$') 
     
ybox3 = TextArea(r'$\theta_i(t = 4500), $', textprops=dict(color=orange, size=18,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$\phi_i(t = 4500)$', textprops=dict(color=purple, size=18,rotation=90,ha='left',va='bottom'))

ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.14, 0.05), 
                                  bbox_transform=ax2.transAxes, borderpad=0.)

ax2.add_artist(anchored_ybox)     

# ax3.axis("off")


  
#############
#Time-series
#############
ax5 = fig.add_subplot(2,5,(4,5))
ax5.set_title('Ocillators dynamics: Time-series \n ')

#sax5.text(4490,52,'D',color='k',weight='bold',fontsize = 22)
for j in range (3):
    ind = j*step +0.2  
    ax5.plot(time2, phi_s[nti:ntf,j+1]+8*(j+3), color=purple, linewidth=4)
    ax5.plot(time2, theta_s[nti:ntf,j+1]+8*(j), color=orange, linewidth=4)
ax5.set_xlabel('time $t$') 
ax5.set_yticklabels([])
ax5.set_yticks([])    

ybox3 = TextArea(r'$\theta_i(t), $', textprops=dict(color=orange, size=20,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$\phi_i(t)$', textprops=dict(color=purple, size=20,rotation=90,ha='left',va='bottom'))

ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.025, 0.3), 
                                  bbox_transform=ax5.transAxes, borderpad=0.)

ax5.add_artist(anchored_ybox)  

#############
#POLAR PLOT
#############
ax3 = fig.add_subplot(2, 5, 7,projection='polar')

ax3.text(-180,1.7,'E',color='k',fontsize = 22,weight='bold',)
ax3.set_title(r'                Polar plot for $\theta_i(t = $'+str(int(ti))+'$)$',fontsize = 22)
for j in range (3):
    ind = j*step +0.2
    ax3.plot(theta_s[nti,j+1],1,'o',color = orange,markersize = 9) 
ax3.set_yticklabels([])

ax3.set_xticklabels([r'0, 2$\pi$','', r'$\frac{\pi}{2}$','', r'$\pi$', '', r'$\frac{3\pi}{2}$'])
ax3.grid(True)


# Create second axes, the top-left plot with orange plot
ax4 = fig.add_subplot(2,5,8, projection='polar')

 # two rows, two columns, second cell
ax4.set_title(r'and $\phi_i(t = $'+str(int(ti))+'$)$',fontsize = 22)
for j in range(3):
    ind = j*step +0.2
    ax4.plot(phi_s[nti,j+1],1,'o',color=purple,markersize = 9) 
ax4.set_yticklabels([])
# ax2.set_xticks([0, 270, 180, 90])
ax4.set_xticklabels([r'0, 2$\pi$','', r'$\frac{\pi}{2}$','', r'$\pi$', '', r'$\frac{3\pi}{2}$'])
ax4.grid(True)

######
#MPV
#####
ax6 = fig.add_subplot(2,5,9)
ax6.text(-0.15,0.435,'F',color='k',fontsize = 22,weight='bold',)
ax6.set_title('Mean phase velocity \n    ')

ax6.scatter(np.arange(1,4),np.round(mean_mpv[:3],2), s =9**2 , color = orange)
ax6.scatter(np.arange(4,7),np.round(mean_mpv[3:],2), s =9**2 , color = purple)

plt.xlabel(r'Node $i$')
ax6.set_ylim(0.1,0.4)
ax6.set_xticks([1,2,3,4,5,6])
ax6.set_xticklabels([1,2,3,1,2,3])


colors = [orange, orange, orange, purple, purple, purple]
for xtick, color in zip(ax6.get_xticklabels(), colors):
    xtick.set_color(color)
    xtick.set_weight('bold')
    
ybox3 = TextArea(r'$\Omega_{\theta_i}, $', textprops=dict(color=orange, size=22,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$\Omega_{\phi_i}$', textprops=dict(color=purple, size=22,rotation=90,ha='left',va='bottom'))
ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.25 , 0.3), 
                                  bbox_transform=ax6.transAxes, borderpad=0.)

ax6.add_artist(anchored_ybox)  

######
#R
#####
ax7 = fig.add_subplot(2,5,10)
#ax7.text(4495,1.18,'G',color='k',weight='bold',fontsize = 22)
ax7.set_title('Order Parameter \n     ')

ax7.plot(time2, rvphi[nti:ntf],color=purple, linewidth=4)
ax7.plot(time2, rvtheta[nti:ntf],color=orange, linewidth=4)
plt.xlabel(r'$time$ $t$')

ax7.set_ylim([-0.05, 1.05])
ybox3 = TextArea(r'$R_{\theta}$, ', textprops=dict(color=orange, size=22,rotation=90,ha='left',va='bottom'))
ybox1 = TextArea(r'$R_{\phi}$', textprops=dict(color=purple, size=22,rotation=90,ha='left',va='bottom'))

ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.25, 0.3), 
                                  bbox_transform=ax7.transAxes, borderpad=0.)

ax7.add_artist(anchored_ybox)  

ax7.set_xlim(4900,5000)
path = '/home/masoliverm/Documents/Files&Figures/figures-for-articles/2021'
os.chdir(path)

today = date.today()
figure_nom = 'FIG1_'+str(today)+'_3.svg'
plt.tight_layout()
plt.savefig(figure_nom, bbox_inches='tight',format='svg', dpi=300)
plt.show()









