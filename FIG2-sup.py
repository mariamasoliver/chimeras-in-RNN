#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:39:04 2021

@author: maria
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 
import os
from datetime import date

"""
1. NETWORK construction (as a drawing)
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



"""
2. Retrieve data: firing rates, decoders, output and supervisors
"""

path = '/Users/maria/Documents/files&figures/basics_force_method'
os.chdir(path)

dtplot = 0.1

fr = np.loadtxt('store_r_sine.txt')
output = np.loadtxt('store_s_sine.txt')
phi = np.loadtxt('store_phi_sine.txt')
sup = np.loadtxt('store_sup_sine.txt')

nodes = 3
totaltime = fr[:,0]

initial_t = 400 #RLS turned on
time_fig = 3800
learning_t = int(time_fig/2)

n_initial_t = int(initial_t/dtplot)
n_time_fig = int(time_fig/dtplot)
n_learning_t = int(learning_t/dtplot)

time = np.arange(0,n_time_fig)*dtplot

#%%
"""
3. PLOT
"""
params = {
    'axes.labelsize': 22,
        'xtick.labelsize': 22,
            'ytick.labelsize': 22,
                'legend.fontsize': 22,
                    'axes.titlesize':22

                    }

plt.rcParams.update(params)


fig = plt.figure(figsize = (20,5.5))


edges_l_cmap = cm.get_cmap('coolwarm')
edges_f_cmap = cm.get_cmap('twilight')
nodes_cmap = cm.get_cmap('bone')

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


trans_f = np.random.default_rng().uniform(0.2,1,len(edges_fixed))
trans_l = np.random.default_rng().uniform(0.2,1,len(edges_learned))



#################
#Network drawing    
################
ax1 = plt.subplot2grid((2,4), (0,0), rowspan=2)
ax1.text(-1.2,1.17,'A',color='k',fontsize = 22,weight='bold')
ax1.set_title(r'Initial weight matrix $w_0$ ')
pos = nx.circular_layout(G_initial)
nx.draw_networkx_nodes(G_initial, pos, node_size=500,alpha=0.98,node_color=color_nodes_and_fr)
for i in range(len(edges_fixed)):
    nx.draw_networkx_edges(G_initial, pos, edgelist=[edges_fixed[i]], width=4,alpha=trans_f[i] , edge_color=color_edges_f[i])
plt.axis("off")

 
    
ax2 = plt.subplot2grid((2,4), (0,1), rowspan=2)
ax2.set_title(r'Learned weight matrix $\eta d^T$ ')
ax2.text(-1.4,1.17,'B',color='k',fontsize = 22,weight='bold',)
ax2.text(-1.25,-0.05,'+',color='k',fontsize = 25,weight='bold',)

pos2 = nx.circular_layout(G_initial)
nx.draw_networkx_nodes(G_initial, pos2, node_size=500,alpha=0.98,node_color=color_nodes_and_fr)
# nx.draw_networkx_edges(G_initial, pos6, edgelist=edges_fixed, width=3, alpha=0.97,edge_color='k')
for i in range(len(edges_learned)):
    nx.draw_networkx_edges(G_initial, pos2, edgelist=[edges_learned[i]], width=2, alpha=trans_l[i], edge_color=color_edges_l[i])
plt.axis("off")

  
    
  

#################
#Firing rates
################
    
ti = 1000
tf = ti+500
nti = int(round(ti/dtplot,0))
ntf = int(round(tf/dtplot,0))

ax3 = plt.subplot2grid((2,4), (0, 2), rowspan=2)
ax3.text(1000-40,23.5,'C',color='k',weight='bold',fontsize = 22)
# ax3.text(1500+40,0.4,'N',color='k',fontsize = 22)
ax3.set_title('Firing rates')
for j in range(5):
    ax3.plot(time[nti:ntf], (fr[nti:ntf,j]+1)+2.2*(j),color=color_nodes_and_fr[-1-j], linewidth=2) 
for j in range(5,10):
    ax3.plot(time[nti:ntf], (fr[nti:ntf,j]+1)+2.2*(j),color=color_nodes_and_fr[10-(j+1)], linewidth=2) 

# plt.axis("off")

#######
#output 
#######
ax4 = plt.subplot2grid((2,4), (0, 3))
ax4.set_title('Network output: \n learned dynamics')
ax4.text(1000-20,10.2,'D',color='k',fontsize = 22,weight='bold',)
ax4.text(1500+35,8.7,'e(t)',color='k',fontsize = 22)

ax4.plot(time[nti:ntf], (output[nti:ntf]+1)/2+j,color='#2b8cbe', linewidth=4,alpha = 0.8) 
plt.axis("off")

###########
#supervisor
###########


ax5 = plt.subplot2grid((2,4), (1, 3))
ax5.set_title('Supervisor')
ax5.plot(time[nti:ntf], (sup[nti:ntf]+1)/2+j,color='k', linewidth=1,alpha = 0.8) 
plt.axis("off")

path = '/Users/maria/cluster_CSG/Documents/Files&Figures/figures-for-articles/2021'
os.chdir(path)

today = date.today()
figure_nom = 'FIG2_'+str(today)+'_preliminary.svg'
plt.tight_layout()
plt.savefig(figure_nom, bbox_inches='tight',format='svg', dpi=300)
plt.show()

#%%
# ########
# #decoder
# ########
# N2 = len(fr[0,:])
# step = (0.8-0.2)/N2
# color_nodes_and_fr = []
# for i in range(N2):
#     ind = 0.1+i*step
#     color_nodes_and_fr.append(nodes_cmap(ind))

# ti = 400
# tf = ti+200
# nti = int(round(ti/dtplot,0))
# ntf = int(round(tf/dtplot,0))

# axextra = plt.subplot2grid((2,5), (0,2), rowspan=2)
# axextra.text(400,0.0295,'C',color='k',fontsize = 22, weight='bold',)
# axextra.text(615,-0.005,'N',color='k',fontsize = 22)

# for j in range(4):
#     axextra.plot(time[nti:ntf]+j*0.0025, phi[nti:ntf,j], color=color_nodes_and_fr[j+2], linewidth=2)

# for j in range(7,10):
#     axextra.plot(time[nti:ntf], phi[nti:ntf,j]+j*0.0025, color=color_nodes_and_fr[j], linewidth=2)
# plt.axis("off")
