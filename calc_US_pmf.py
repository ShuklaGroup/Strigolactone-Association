import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rc('savefig',dpi=500)

import pymbar

from pymbar import timeseries

GAS_CONSTANT = 0.00198588

#name = "AtD14_OsD3_apo"
#name = "ShHTL7_OsD3_apo"
#name = "ShHTL7_OsD3_mut"
name = "ShHTL6_OsD3_apo"

temperature = 300
K = 64
n_reps = 5
N_max = 10000*n_reps
#N_max = 200*n_reps

N_k = np.zeros(K, np.int32) #Number of snapshots per window
K_k = np.zeros(K) #Spring constants
r0_k = np.zeros(K) #Centers
r_kn = np.zeros((K, N_max)) #Sep distance for each snapshot
u_kn = np.zeros((K, N_max)) #Reduced potential energy (to be calculated)
g_k = np.zeros(K) #Timeseries for subsampling
nbins = 100 #Number of bins for pmf

##Centers and spring constants
K_k[:] = 21.384000
r0_k[:] = np.linspace(26, 47, K)

##Load colvar values from simulations
for i in range(K):
    for j in range(n_reps):
        try:
            #sep_dist_data = np.loadtxt("image_%d/rep_%d/%s_im%d_%d.colvars.traj"%(i,j,name,i,j), usecols=(3)) #Read separation distances from colvars file
            sep_dist_data = np.loadtxt("image_%d/rep_%d/%s_im%d_%d-1.colvars.traj"%(i,j,name,i,j), usecols=(3)) #Read separation distances from colvars file
            sep_dist_data2 = np.loadtxt("image_%d/rep_%d/%s_im%d_%d-2.colvars.traj"%(i,j,name,i,j), usecols=(3)) #Read separation distances from colvars file
            #sep_dist_data3 = np.loadtxt("image_%d/rep_%d/%s_im%d_%d-3.colvars.traj"%(i,j,name,i,j), usecols=(3)) #Read separation distances from colvars file
            #sep_dist_data4 = np.loadtxt("image_%d/rep_%d/%s_im%d_%d-4.colvars.traj"%(i,j,name,i,j), usecols=(3)) #Read separation distances from colvars file
            #sep_dist_data5 = np.loadtxt("image_%d/rep_%d/%s_im%d_%d-5.colvars.traj"%(i,j,name,i,j), usecols=(3)) #Read separation distances from colvars file
            #sep_dist_data6 = np.loadtxt("image_%d/rep_%d/%s_im%d_%d-6.colvars.traj"%(i,j,name,i,j), usecols=(3)) #Read separation distances from colvars file
            #sep_dist_data7 = np.loadtxt("image_%d/rep_%d/%s_im%d_%d-7.colvars.traj"%(i,j,name,i,j), usecols=(3)) #Read separation distances from colvars file
            #sep_dist_data = np.hstack((sep_dist_data1, sep_dist_data2, sep_dist_data3, sep_dist_data4, sep_dist_data5, sep_dist_data6))
            #sep_dist_data = np.hstack((sep_dist_data1, sep_dist_data2, sep_dist_data3, sep_dist_data4, sep_dist_data5, sep_dist_data6, sep_dist_data7))
            #sep_dist_data = np.loadtxt("image_%d/rep_%d/%s_im%d_%d.colvars.traj"%(i,j,name,i,j), usecols=(1)) #Read separation distances from colvars file
            n_k = np.size(sep_dist_data)
            r_kn[i, N_k[i]:(N_k[i] + n_k)] = sep_dist_data
            N_k[i] += n_k
        except:
            pass

    #g_k[i] = timeseries.statisticalInefficiency(r_kn[i,:])
    #print(g_k[i])
    #indices = timeseries.subsampleCorrelatedData(r_kn[i,:], g=g_k[i])

    indices = list(range(N_k[i]))[::2]

    N_k[i] = len(indices)
    r_kn[i,0:N_k[i]] = r_kn[i,indices]

##Calculate u_kln from centers and actual sep distances
#for i in range(K):
#    for j in range(N_max):
#        if r_kn[i,j] != 0:
#            u_kn[i,j] = 1.0/(GAS_CONSTANT*temperature) * (K_k[i]/2.0) * (r_kn[i,j] - r0_k[i])**2



u_kln = np.zeros((K,K,N_max))
for k in range(K):
    for n in range(N_k[k]):
        dr = r_kn[k,n] - r0_k
        
        #print(np.shape(u_kln[k,:,n]))
        #print(np.shape(dr**2))
        #print(np.shape(K_k))
        #print(np.shape((1.0/(GAS_CONSTANT*temperature)) * (K_k/2.0) * dr**2))
        u_kln[k,:,n] = (1.0/(GAS_CONSTANT*temperature)) * (K_k/2.0) * dr**2

print(u_kln)

##Calculate bins
r_min = np.min(r_kn[np.nonzero(r_kn)])
r_max = np.max(r_kn[np.nonzero(r_kn)])
delta = (r_max - r_min)/nbins
bin_center_i = np.zeros(nbins)
for i in range(nbins):
    bin_center_i[i] = r_min + delta/2 + delta*i

#Bin data
bin_kn = np.zeros((K, N_max))
for k in range(K):
    for n in range(N_k[k]):
        bin_kn[k,n] = int((r_kn[k,n] - r_min) / delta)

##Run MBAR and calculate pmf
mbar = pymbar.MBAR(u_kln, N_k, verbose = True, maximum_iterations=10000)

(f_i, df_i) = mbar.computePMF(u_kn, bin_kn, nbins)

print(bin_center_i)
print(f_i*GAS_CONSTANT*temperature)
print(df_i)

np.save("bins.npy", bin_center_i)
np.save("pmf_kcal_mol.npy", f_i*GAS_CONSTANT*temperature)
np.save("error_kcal_mol.npy", df_i*GAS_CONSTANT*temperature)

matplotlib.rc('font',family='Helvetica-Normal',size=14)
plt.errorbar(bin_center_i, f_i*GAS_CONSTANT*300, yerr=df_i*GAS_CONSTANT*temperature)
#plt.savefig("D14-US_pmf.png", transparent=True)
plt.savefig("HTL6_apo-US_pmf.png", transparent=True)
