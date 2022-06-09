#This is a force code
#It reads from a file the supervisors. Which are two small populations
#of N Kuramoto oscillators each (nodes) displaying a chimera state.
# Eta and Phi are kept sparse

import numpy as np 

N = 3000 
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
ti_file = 3000
nti_file = int(ti_file/dt)
theta = np.transpose(data[nti_file:,1:])
phis = np.transpose(data2[nti_file:,1:])

ntf_file = len(phis[0,:])

sup = np.zeros((m,ntf_file))
sup[0:nodes] = np.cos(theta)
sup[nodes:2*nodes] = np.cos(phis)
sup[2*nodes:3*nodes] = np.sin(theta)
sup[3*nodes:4*nodes] = np.sin(phis)

T = 1e5 #Total simulation time
nt = int(T/dt)
p = 0.1 #
g = 1.5 #initial coupling strength on static weight matrix
llavor = 3
np.random.seed(llavor)
A = np.random.normal(0., 1.0, (N,N)) #NxN Matrix with a normal distribution
np.random.seed(llavor+1)
B = np.random.rand(N,N)<p #NxN Matrix with a uniform distribution. 
                          #It will return true and false, depending on p. 
omega = g*np.multiply(A,B)/np.sqrt(N*p) #Initial weight matrix

#sparsity level
c = int(N*p)

#sparsity level for eta
p_eta = p
c_eta = int(N*p_eta)

#Let's make eta sparse (here is set to 10% sparse, change p_eta if you want to increase/decrease it)
Q = 1
np.random.seed(llavor+2)
s1 = np.random.randint(0,N,size=c_eta)

np.random.seed(llavor+3)
sparse_eta = Q*(2*np.random.rand(N,m)-1)
sparse_eta[s1,:] = 0

#Let's make phi sparse (90%) and all the 
#other auxiliary matrices to perform RLS
d = np.zeros((N,m))
sparse_d = np.zeros((c,m))
np.random.seed(llavor+4)
s2 = np.random.randint(0,N,size=c)

q = np.zeros((c,m))
la = 1
Pinv = np.zeros((c,c))
Pinv += np.eye(c)/la

z = np.random.normal(0., 1., N) #initial conditions
r = np.tanh(z)
e = np.zeros(m)
xhat = np.zeros(m)
tau = 1

imin = int(50/dt) #start RLS
imax = int(0.5*T/dt) #stop RLS
step = 2
iplot = 0
ntplot = int(nt/step)+1

#initialize storage vectors
store_r = np.zeros((ntplot, 10))
store_x = np.zeros((ntplot, m))

for i in range(nt):
    z = 1./tau*(z + dt*(-z + np.dot(omega,r) + np.dot(sparse_eta,xhat))) #integrating with Euler method
    r = np.tanh(z) #compute firing rates
    xhat = np.dot(np.transpose(sparse_d),r[s2])
    
    ##RLS
    if (i > imin):
        if (i < imax):
            if (i%step == 0):
                e = xhat - sup[:,i]
                q = np.dot(Pinv,r[s2])
                Pinv = Pinv - (np.outer(q,np.transpose(q)))/(1+np.dot(np.transpose(r[s2]),q))
                sparse_d = sparse_d - np.outer(np.dot(Pinv,r[s2]),e)
    #store network output and firing rates
    if (i%step == 0):
        store_r[iplot, :] = r[:10]
        store_x[iplot,:] = xhat
        iplot +=1

d[s2,:] = sparse_d
omega_learned = np.dot(sparse_eta,np.transpose(d))


np.savetxt('firing_rates_seed_'+str(llavor)+'_N_'+str(N)+'_Q_'+str(Q)+'.txt',store_r)
np.savetxt('output_seed_'+str(llavor)+'_N_'+str(N)+'_Q_'+str(Q)+'.txt',store_x)
np.savetxt('d_sparse_seed_'+str(llavor)+'_N_'+str(N)+str(c)+'_Q_'+str(Q)+'.txt',sparse_d)
np.savetxt('omega_learned_'+str(llavor)+'_N_'+str(N)+str(c)+'_Q_'+str(Q)+'.txt',omega_learned)
  
