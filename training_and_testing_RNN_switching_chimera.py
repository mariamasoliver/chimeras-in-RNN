#This is a force code. 
#It reads from a file the supervisors. Which are two small populations
#of N Kuramoto oscillators each (nodes) displaying a chimera state.
#The chimera state switches between populations, to recreate the 
#switching of the uni-hemispheric sleep

import numpy as np
#supervisor (KRM)
nodes = 3
#We have 2 populations with 3 nodes each. 
#Each node will result in 2 supervisors: cosine and sine.
 
m = 2*2*nodes
K = 0.1
beta = 0.025
TC = 1
ic = 7
data = np.loadtxt('theta_A_'+str(K)+'_N_'+str(nodes)+'_beta_'+str(beta)+'_ic_'+str(ic)+'_TC_'+str(TC)+'.txt')
data2 = np.loadtxt('phi_A_'+str(K)+'_N_'+str(nodes)+'_beta_'+str(beta)+'_ic_'+str(ic)+'_TC_'+str(TC)+'.txt')
   

dt = 0.1

#Removing transients
ti_file = 3000
nti_file = int(ti_file/dt)
theta = np.transpose(data[nti_file:,1:])
phi = np.transpose(data2[nti_file:,1:])

ntf_file = len(phi[0,:])

sup_A = np.zeros((m,ntf_file))
sup_A[0:nodes] = np.sin(theta)
sup_A[nodes:2*nodes] = np.cos(theta)
sup_A[2*nodes:3*nodes] = np.sin(phi)
sup_A[3*nodes:4*nodes] = np.cos(phi)

sup_B = np.zeros((m,ntf_file))
sup_B[0:nodes] = np.sin(phi)
sup_B[nodes:2*nodes] = np.cos(phi)
sup_B[2*nodes:3*nodes] = np.sin(theta)
sup_B[3*nodes:4*nodes] = np.cos(theta)    

N = 2500 #Number of neurons
T = 1e4 #Total simulation time
nt = int(T/dt)
p = 0.1 #
g = 1.5 #initial coupling strength on static weight matrix

llavor = 3
np.random.seed(llavor)

A = np.random.normal(0., 1.0, (N,N)) #NxN Matrix with a normal distribution
B = np.random.rand(N,N)<p #NxN Matrix with a uniform distribution. 
                          #It will return true and false, depending on p. 
omega = g*np.multiply(A,B)/np.sqrt(N*p) #Initial weight matrix
Q = 1
eta = Q*(2*np.random.rand(N,m)-1) #random theta variable 
#It is an array ranging form -Q to Q
z = np.random.normal(0., 1., N) #initial conditions
np.savetxt('initial_neuron_dyn_'+str(llavor)+'.txt',z)

r = np.tanh(z)
e = np.zeros(m)
xhat = np.zeros(m)


#"""initialize decoders & RLS method"""
d = np.zeros((N,m)) #one decoder for each supervisor
q = np.zeros((N,m))
la = 1
Pinv = np.zeros((N,N))
Pinv += np.eye(N)/la

imin = int(500/dt) #start RLS
imax = int(0.5*T/dt) #stop RLS
step = 2
iplot = 0
ntplot = int(nt/step)+1
omega_in = np.random.rand(N)-1
#initialize storage vectors
store_r = np.zeros((ntplot, 10))
store_x = np.zeros((ntplot, m))
store_d_1 = np.zeros((ntplot, 10))

cvec = []

for i in range(nt):
    c = 0
    if (i%imin == 0):
        c = int(np.random.randint(1,3))
        if (c==2):
            c = -1
    cvec.append(c)
    z = z + dt*(-z + np.dot(omega,r) + np.dot(eta,xhat) + omega_in*c)
    r = np.tanh(z) #compute firing rates
    xhat = np.dot(np.transpose(d),r)

    ##RLS
    if (i >= imin):
        if (i < imax):
            if (i%imin == 0):
                if (c == 1):
                    sup = sup_A
                if (c == -1):
                    sup = sup_B
            if (i%step == 0):
                e = xhat - sup[:,i]
                q = np.dot(Pinv,r)
                Pinv = Pinv - (np.outer(q,np.transpose(q)))/(1+np.dot(np.transpose(r),q))
                d = d - np.outer(np.dot(Pinv,r),e)
    #store network output and firing rates
    if (i%step == 0):
        store_r[iplot, :] = r[:10]
        store_x[iplot,:] = xhat
        store_d_1[iplot, :] = d[:10,1]
        iplot +=1

#Saving data on a file
np.savetxt('output_plot_'+str(llavor)+'_Q_'+str(Q)+'_t_1e5.txt',store_x)
np.savetxt('d_'+str(llavor)+'_Q_'+str(Q)+'_t_1e5.txt',d)
np.savetxt('cvec_'+str(llavor)+'_t_1e5.txt',cvec)
np.savetxt('eta_'+str(llavor)+'.txt',eta)
np.savetxt('omega_'+str(llavor)+'.txt',omega)
np.savetxt('omega_in'+str(llavor)+'.txt',omega_in)
np.savetxt('final_neuron_dyn_'+str(llavor)+'.txt',z)


