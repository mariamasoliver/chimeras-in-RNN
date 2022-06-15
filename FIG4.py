#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIG1 of the manuscript
@author: maria
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm



prepath1 = '/home/maria/cluster_CSG/'
prepath2 = '/Users/maria/cluster_CSG/'
prepath3 = '/home/masoliverm/'

path = prepath3+'/Documents/Files&Figures/force_method_1/N_3/retrain/Results'
os.chdir(path)


dtplot = 0.2
fr = np.loadtxt('firing_rates_plot_3.txt')
ti = 1000
tf = ti+300
nti = int(round(ti/dtplot,0))
ntf = int(round(tf/dtplot,0))
time = np.arange(0,len(fr)*dtplot, dtplot)


"""
1. Retrieve data: firing rates, pca, fft
"""
path = prepath3 +'/Documents/Files&Figures/force_method_1/N_3/RNN_small_pop/Results/Files/'
os.chdir(path)

pca = np.loadtxt('PCA-firing-rates_time.txt')
time_pca = np.arange(0,len(pca[:-1,0])*dtplot,dtplot)
fr_mean = np.loadtxt('rmean_vs_time_llavor_3.txt')
time_mean = np.arange(0,len(fr_mean)*dtplot,dtplot)
fr1 = np.loadtxt('firing_rates_llavor_3_N_1500.txt',skiprows=20000,usecols=(1,5,23,45),max_rows =80000 )

time_fr = np.arange(0,len(fr1)*dtplot,dtplot)
phi_s = np.loadtxt('phi_A_0.1_N_3_beta_0.025_ic_7_TC_1.txt',skiprows=20000)
theta_s = np.loadtxt('theta_A_0.1_N_3_beta_0.025_ic_7_TC_1.txt',skiprows=20000)

#%%
path = prepath3 +'Documents/Files&Figures/force_method_1/N_3/retrain/Results'
os.chdir(path)

dtplot = 0.2

output = np.loadtxt('output_plot_3.txt')

nodes = 3
totaltime = output[:,0]
intime = 0
cos = output[intime:-1,0:2*nodes] 


#%%
"""
1.A FFT PCA
"""
from scipy.fft import fft, fftfreq
# Number of sample points
ntf_output=len(pca[:,0])
yf_pca_plot = np.zeros((ntf_output//2,6))

for i in range (6):
    yf_pca = fft(pca[:,i])
    yf_pca_plot[:,i] = (1/np.abs(yf_pca[0:ntf_output//2]).max())*np.abs(yf_pca[0:ntf_output//2])
    
xf_pca = fftfreq(int(ntf_output), dtplot)[:ntf_output//2] #sense la part negativa

"""
1.B FFT MEAN
"""
ntf_output=len(fr_mean)
yf_m_plot = np.zeros(ntf_output//2)

yf_m = fft(fr_mean)
yf_m_plot = (1/np.abs(yf_m[0:ntf_output//2]).max())*np.abs(yf_m[0:ntf_output//2])


"""
1.C FFT CHIMERA
"""
# Number of sample points
ntf_output=len(phi_s[:,1:])
yf_T_plot = np.zeros((ntf_output//2,3))
yf_P_plot = np.zeros((ntf_output//2,3))

for i in range (3):
    yf_T = fft(np.cos(theta_s[:,i+1]))
    yf_T_plot[:,i] = (1/np.abs(yf_T[0:ntf_output//2]).max())*np.abs(yf_T[0:ntf_output//2])
    yf_P = fft(np.cos(phi_s[:,i+1]))
    yf_P_plot[:,i] = (1/np.abs(yf_P[0:ntf_output//2]).max())*np.abs(yf_P[0:ntf_output//2])

dt = 0.1
xfc = fftfreq(int(ntf_output), dt)[:ntf_output//2] #sense la part negativa


"""
1.D FFT Network Output
"""
# Number of sample points
ntf_output=len(cos[:,0])
yf_cos_plot = np.zeros((ntf_output//2,6))


for i in range (6):
    yf_cos = fft(cos[:,i])
    yf_cos_plot[:,i] = (1/np.abs(yf_cos[0:ntf_output//2]).max())*np.abs(yf_cos[0:ntf_output//2])

dt = 0.2
xfcos = fftfreq(int(ntf_output), dt)[:ntf_output//2] #sense la part negativa


"""
1.E FFT INDIVIDUAL FR
"""
# Number of sample points
ntf_output=len(fr1)
num = len(fr1[0,:])

yf_fr1_plot = np.zeros((num,ntf_output//2))


for i in range(4):
    yf_fr1= fft(fr1[:,i])
    yf_fr1_plot[i,:] = (1/np.abs(yf_fr1[0:ntf_output//2]).max())*np.abs(yf_fr1[0:ntf_output//2])


xf_fri = fftfreq(int(ntf_output), dtplot)[:ntf_output//2] #sense la part negativa



from datetime import date
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

size = 24
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

"""
2. Plot
"""
path = prepath3 + 'Documents/Files&Figures/figures-for-articles/2021'
os.chdir(path)

fig = plt.figure(figsize = (12,30))

#######################################
#CHIMERA cos theta and cos phi Time-series
#######################################
ax0 = fig.add_subplot(5,2,1)
orange = '#fdae6b'
purple = '#9e9ac8'
dt = 0.1
ti = 4500
tf = ti+300
nti = int(round(ti/dt,0))
ntf = int(round(tf/dt,0))
time2 = np.arange(ti,tf,dt)

ax0.set_title('Network supervisor \n $\cos{\theta_i(t)}, \cos{\phi_i(t)}$')

ax0.text(4450,11.8,'A',color='k',weight='bold',fontsize = 22)
for j in range (3):
    ax0.plot(time2, np.cos(phi_s[nti:ntf,j+1])+2*(j+3), color=purple, linewidth=4)
    ax0.plot(time2, np.cos(theta_s[nti:ntf,j+1])+2*(j), color=orange, linewidth=4)
ax0.set_yticklabels([])
ax0.set_yticks([])    

# ybox3 = TextArea(r'$\cos{\theta_i(t)}, $', textprops=dict(color=orange, size=20,rotation=90,ha='left',va='bottom'))
# ybox1 = TextArea(r'$\cos{\phi_i(t)}$', textprops=dict(color=purple, size=20,rotation=90,ha='left',va='bottom'))
# ybox = VPacker(children=[ybox1, ybox3],align="bottom", pad=0, sep=5)

# anchored_ybox = AnchoredOffsetbox(loc=4, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.025, 0.3), 
#                                   bbox_transform=ax0.transAxes, borderpad=0.)
# ax0.add_artist(anchored_ybox)  


#################
#FFT CHIMERA
################
axb = fig.add_subplot(5,2,2)
axb.text(-0.005,10**27,'B',color='k',weight='bold',fontsize = 22)
tf = xfc
linewidth=np.array([3,2,1])
# +1.25*i
# +4+1.25*i
for i in range (3):
    plt.plot(tf, (10**(5*i))*yf_T_plot[:,i], '-', color=orange,  linewidth=3)
    plt.plot(tf, (10**(5*(i+3)))*yf_P_plot[:,i], '-', color=purple,  linewidth=3)
# plt.xlabel(r'Frequency $f$', size = '18')
plt.title(r'FFT$(\cos{\theta_i},\cos{\phi_i)}$ ')
plt.yscale('log')
axb.set_xlim([0, 0.1])
axb.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
axb.set_xticklabels([])



###############
#NETWORK OUTPUT
###############
ax0b = fig.add_subplot(5,2,3)
ax0b.set_title('Network output \n $\cos{\hat{\theta}_i(t)}, \cos{\hat{\phi}_i(t)}$')
ax0b.text(550,11.8,'C',color='k',weight='bold',fontsize = 22)

orange = '#fdae6b'
purple = '#9e9ac8'

orange_out ='#e6550d'
purple_out = '#54278f'

ti = 600
tf = 900
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
    ax0b.plot(time2, cos[nti-ndelay_o:ntf-ndelay_o,j]+2*(j), color=orange_out ,linewidth=3) 
    ax0b.plot(time2, cos[nti-ndelay_o:ntf-ndelay_o,j+3]+2*(j+3), color=purple_out ,linewidth=3)

#################
#FFT network output
################
axb2 = fig.add_subplot(5,2,4)
axb2.text(-0.005,10**27,'D',color='k',weight='bold',fontsize = 22)
tf = xfcos
linewidth=np.array([3,2,1])

for i in range (3):
    plt.plot(tf, (10**(5*i))*yf_cos_plot[:,i], '-', color=orange_out,  linewidth=3)
    plt.plot(tf, (10**(5*(i+3)))*yf_cos_plot[:,i+3], '-', color=purple_out,  linewidth=3)
plt.yscale('log')
# axb.set_ylim([10**-5,1])
plt.title(r'FFT$(\cos{\hat{\theta_i}},\hat{\cos{\phi_i)}}$ ')
axb2.set_xlim([0, 0.1])
axb2.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
axb2.set_xticklabels([])

# #################
# #FIRE RATES
# ################

nodes_cmap = cm.get_cmap('bone')

step = (0.9-0.1)/12
color_nodes_and_fr = []
for i in range(10):
    ind = 0.1+i*step
    color_nodes_and_fr.append(nodes_cmap(ind))

# for j in range(10):
#     ax1.plot(time[nti:ntf], (fr[nti:ntf,j]+1)+2.2*(j),color=color_nodes_and_fr[j], linewidth=2) 
# ax1.set_title('Firing rates')
# ax1.axis("off")


#################
#PCA
################
color_pca = ['#8c3839', '#2b4743', '#1f7f95', '#da7073', '#dda83f', '#737373', '#3c6ca8', '#5d6c89']

ax2 = fig.add_subplot(5,2,5)
ax2.text(175,98,'E',color='k',weight='bold',fontsize = 22)
for i in range (3):
    ax2.plot(time_pca, pca[:-1, i*2]+40*(i),color=color_pca[i*2], linewidth=3)
ax2.set_xlim(200,500)

# ax2.set_xticklabels([])
# ax2.axis("off")
ax2.set_title('Principal component analysis \n of the firing rates $PC_{1,3,5}$')
# ax2.set_xlabel(r'time $t$', size = '18')

#################
#FR MEAN 
################
ax3 = fig.add_subplot(5,2,9)
ax3.text(200+5,0.11,'I',color='k',weight='bold',fontsize = 22)
ax3.plot(time_mean[:-1], fr_mean,color='k', linewidth=2)
ax3.set_xlim(175,500)
ax3.set_ylim(-0.1,0.1)
# ax3.axis("off")
ax3.set_title('Mean of the \n firing rates $FR_{mean}$')
ax3.set_xlabel(r'time $t$')

#################
#FFT PCA
################
ax4 = fig.add_subplot(5,2,6)
ax4.text(-0.005,10**14,'F',color='k',weight='bold',fontsize = 22)
tf = xf_pca
linewidth=np.array([3,2,1])
ax4.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
ax4.set_xticklabels([])
plt.yscale('log')
for i in range (3):
    plt.plot(tf, (10**(6*i))*yf_pca_plot[:,i*2], '-', color=color_pca[2*i],  linewidth=2)
# plt.xlabel(r'Frequency $f$', size = '18')
ax4.set_ylim([10**(-7),10**(14)])
plt.title(r'FFT$(PC_{1,3,5})$ ')
ax4.set_xticklabels([])
ax4.set_xlim([0, 0.1])



#################
#FFT PCA MEAN
################
ax5 = fig.add_subplot(5,2,10)
ax5.text(0.005,10**1,'J',color='k',weight='bold',fontsize = 22)
plt.plot(tf, yf_m_plot, '-', color='k',  linewidth=2)
ax5.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
ax5.set_xticklabels([0, 0.02, 0.04, 0.06, 0.08, 0.1])
plt.xlabel(r'Frequency $f$')
plt.title(r'FFT$(FR_{mean})$ ')
ax5.set_xlim([0, 0.1])
plt.yscale('log')
ax5.set_ylim([10**-7,10**1])
# #################
# # VOID
# ################
# axD = fig.add_subplot(4,3,10)
# axD.axis("off")

#################
#Firing Rates N = 12
################
axE = fig.add_subplot(5,2,7)
axE.text(175,9,'G',color='k',weight='bold',fontsize = 22)
for i in range(num):
    print(i)
    plt.plot(time_fr[:], (fr1[:,i]+1)+2.2*i, color=color_nodes_and_fr[i+2], linewidth=2)
axE.set_xlim(200,500)
#axE.set_ylim(-2,2)
# ax3.axis("off")
axE.set_title(r'Individual firing rates FR')
axE.set_xlabel(r'time $t$')

#################
#FFT PCA MEAN
################

tf=xf_fri
ax5 = fig.add_subplot(5,2,8)
ax5.text(-0.005,10**19,'H',color='k',weight='bold',fontsize = 22)
for i in range (4):
    plt.plot(tf, yf_fr1_plot[i,:]*(10**(6*i)), '-', color=color_nodes_and_fr[i+2],  linewidth=3)
plt.title(r'FFT of individual FR ')
ax5.set_xlim([0, 0.1])
ax5.set_xticks([0,0.025,0.05,0.075,0.1])
ax5.set_xticklabels([0,0.025,0.05,0.075,0.1])
ax5.set_xlim([0, 0.1])
plt.yscale('log')
# ax5.set_ylim([10**-5,1])

today = date.today()
figure_nom = 'FIG5_'+str(today)+'.svg'

plt.tight_layout()
plt.savefig(figure_nom, format='svg', bbox_inches='tight',dpi=250)
plt.show()








