# This is a force code with the dale's law implemented. 
#It reads from a file the supervisors. Which are two small populations
#of N Kuramoto oscillators each (nodes) displaying a chimera state.


import numpy as np 

llavor = 3 #seed
nodes = 3
m = 2*2*nodes
K = 0.1
beta = 0.025
TC = 1
ic = 7

data = np.loadtxt('theta_A_'+str(K)+'_N_'+str(nodes)+'_beta_'+str(beta)+'_ic_'+str(ic)+'_TC_'+str(TC)+'.txt')
data2 = np.loadtxt('phi_A_'+str(K)+'_N_'+str(nodes)+'_beta_'+str(beta)+'_ic_'+str(ic)+'_TC_'+str(TC)+'.txt')

time = data[:,0]
dt = time[4] - time[3]

"""Removing transients"""
# ti_file = 18000 #for N = 25
ti_file = 3000 #for N = 3
nti_file = int(ti_file/dt)
theta_c = np.transpose(data[nti_file:,1:])
phi_c = np.transpose(data2[nti_file:,1:])
ntf_file = len(phi_c[0,:])

sup = np.zeros((m,ntf_file))
sup[0:nodes] = np.cos(theta_c)
sup[nodes:2*nodes] = np.cos(phi_c)
sup[2*nodes:3*nodes] = np.sin(theta_c)
sup[3*nodes:4*nodes] = np.sin(phi_c)


np.random.seed(llavor)

N = 3500 #Total number of neurons
mu = 0.5 #ratio of E to total 
NE = int(mu*N)
# NI = N - NE


T = 1e5 #1e5 #Total simulation time
dt = 0.1 #integration time step
nt = int(T/dt)
p = 0.1 #network sparsity
p_temp = p/mu

# Initializing Weight Matrix
omega = np.zeros((N,N))
g = 1.5  #Static Weight Magnitude 

for i in range(0,N):
    A = np.random.normal(0., 1.0, (N))
    omega[i,0:NE]=np.multiply(A[0:NE],A[0:NE]>0)
    omega[i,NE:N]=np.multiply(A[NE:N],A[NE:N]<0)


B = np.random.rand(N,N)<p_temp #NxN Matrix with a uniform distribution. 
                          #It will return true and false, depending on p. 
omega = g*np.multiply(omega,B)/np.sqrt(N*p) #Initial weight matrix

eta = (2*np.random.rand(N,m)-1) #random theta variable ranging from -1,1
etaminus = np.zeros((N,m))
etaplus = eta + etaminus
etaplus[etaplus<0] = 0
etaminus = eta - etaplus

z = np.random.normal(0., 1., N) #initial conditions
r = np.tanh(z)
e = np.zeros(m)

xhat = np.zeros(m)
xhat1 = np.zeros(m)
xhat2 = np.zeros(m)

#initialize decoders & RLS method
d = np.zeros((N,m)) #one decoder for each supervisor
d1 = np.zeros((N,m))
d2 = np.zeros((N,m))

q = np.zeros((N,m))
la = 1
Pinv = np.zeros((N,N))
Pinv += np.eye(N)/la

imin = int(100/dt) #start RLS
imax = int(0.8*T/dt) #stop RLS
step = 2
iplot = 0
ntplot = int(nt/step)+1
#initialize storage vectors
store_r = np.zeros((ntplot, 10))
store_x = np.zeros((ntplot, m))
store_d = np.zeros((ntplot, 10, m))

for i in range(nt):
    
    z = z + dt*(-z + np.dot(omega,r) + np.dot(etaplus,xhat1) + np.dot(etaminus,xhat2)) #integrating with Euler method
    r = np.tanh(z) #compute firing rates
    xhat = np.dot(np.transpose(d),r)
    xhat1 = np.dot(np.transpose(d1),r)
    xhat2 = np.dot(np.transpose(d2),r)
    
    ##RLS
    if (i > imin):
        if (i < imax):
            if (i%step == 0):
                e = xhat - sup[:,i]
                q = np.dot(Pinv,r)               
                Pinv = Pinv - (np.outer(q,np.transpose(q)))/(1+np.dot(np.transpose(r),q))
                d = d - np.outer(np.dot(Pinv,r),e)
                
                dep = np.multiply(d[0:NE,:],d[0:NE,:]>0)
                dem = np.multiply(d[0:NE,:],d[0:NE,:]<0)
                dip = np.multiply(d[NE:N,:],d[NE:N,:]>0)
                dim = np.multiply(d[NE:N,:],d[NE:N,:]<0)
                
                d1[0:NE] = dep
                d1[NE:N] = dim
                d2[0:NE] = dem*mu/(1-mu)
                d2[NE:N] = dip*mu/(1-mu)
                


    #store network output and firing rates
    if (i%step == 0):
        iplot +=1
        store_r[iplot, :] = r[:10]
        store_x[iplot,:] = xhat
        store_d[iplot, :, :] = d[:10,:]



    #Saving data on a file
    np.savetxt('firing_rates_seed_'+str(llavor)+'_N_'+str(N)+'_T_'+str(T)+'.txt',store_r)
    np.savetxt('output_seed_'+str(llavor)+'_N_'+str(N)+'_T_'+str(T)+'.txt',store_x)
    np.savetxt('d_seed_'+str(llavor)+'_N_'+str(N)+'_T_'+str(T)+'.txt',d)



