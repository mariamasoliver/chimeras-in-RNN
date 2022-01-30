#This is a force code. 
#It reads from a file the supervisors. Which are two small populations
#of N Kuramoto oscillators each (nodes) displaying a chimera state.


import numpy as np 

#supervisor (KRM)
nodes = 3
#We have 2 populations with 3 nodes each. 
#Each node will result in 2 supervisors: cosine and sine.
#For each population we will have as a supervisor the order parameter
# m = 2*2*nodes+2
# Training the network without the order parameters as supervisors 
m = 2*2*nodes
K = 0.1
beta = 0.025
TC = 1
ic = 7
data = np.loadtxt('theta_A_'+str(K)+'_N_'+str(nodes)+'_beta_'+str(beta)+'_ic_'+str(ic)+'_TC_'+str(TC)+'.txt')
data2 = np.loadtxt('phi_A_'+str(K)+'_N_'+str(nodes)+'_beta_'+str(beta)+'_ic_'+str(ic)+'_TC_'+str(TC)+'.txt')
   
time = data[:,0]
dt = time[4] - time[3]

#Removing transients
ti_file = 2000
nti_file = int(ti_file/dt)
theta = np.transpose(data[nti_file:,1:])
phi = np.transpose(data2[nti_file:,1:])

ntf_file = len(phi[0,:])

sup = np.zeros((m,ntf_file))
sup[0:nodes] = np.cos(theta)
sup[nodes:2*nodes] = np.cos(phi)
sup[2*nodes:3*nodes] = np.sin(theta)
sup[3*nodes:4*nodes] = np.sin(phi)

N = 1500 #Number of neurons
T = 1e5 #Total simulation time
nt = int(T/dt)
p = 0.1 #
g = 1.5 #initial coupling strength on static weight matrix

llavor = 3 #seed
np.random.seed(llavor)

A = np.random.normal(0., 1.0, (N,N)) #NxN Matrix with a normal distribution
B = np.random.rand(N,N)<p #NxN Matrix with a uniform distribution. 
                          #It will return true and false, depending on p. 
omega = g*np.multiply(A,B)/np.sqrt(N*p) #Initial weight matrix
Q = 1
eta = Q*(2*np.random.rand(N,m)-1) #random theta variable 
#It is an array ranging form -Q to Q

z = np.random.normal(0., 1., N) #initial conditions
r = np.tanh(z)
e = np.zeros(m)
xhat = np.zeros(m)


#"""initialize decoders & RLS method"""
phi = np.zeros((N,m)) #one decoder for each supervisor
q = np.zeros((N,m))
la = 1
Pinv = np.zeros((N,N))
Pinv += np.eye(N)/la

imin = int(600/dt) #start RLS
imax = int(0.5*T/dt) #stop RLS
step = 2
iplot = 0
ntplot = int(nt/step)+1

#initialize storage vectors
store_r = np.zeros((ntplot, 10))
store_x = np.zeros((ntplot, m))
store_phi_1 = np.zeros((ntplot, 10))
for i in range(nt):
    z = z + dt*(-z + np.dot(omega,r) + np.dot(eta,xhat)) #integrating with Euler method
    r = np.tanh(z) #compute firing rates
    xhat = np.dot(np.transpose(phi),r)
    
    ##RLS
    if (i > imin):
        if (i < imax):
            if (i%step == 0):
                e = xhat - sup[:,i]
                q = np.dot(Pinv,r)
                Pinv = Pinv - (np.outer(q,np.transpose(q)))/(1+np.dot(np.transpose(r),q))
                phi = phi - np.outer(np.dot(Pinv,r),e)
                
    #store network output and firing rates
    if (i%step == 0):
        store_r[iplot, :] = r[:10]
        store_x[iplot,:] = xhat
        store_phi_1[iplot, :] = phi[:10,1]
        iplot +=1

#Saving data on a file

np.savetxt('firing_rates_plot_'+str(llavor)+'_Q_'+str(Q)+'.txt',store_r)
np.savetxt('output_plot_'+str(llavor)+'_Q_'+str(Q)+'.txt',store_x)
np.savetxt('phi_'+str(llavor)+'_Q_'+str(Q)+'.txt',phi)



