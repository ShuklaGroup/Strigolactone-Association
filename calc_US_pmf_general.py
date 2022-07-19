import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rc('savefig',dpi=500)
import pymbar
from pymbar import timeseries
import argparse
import os

GAS_CONSTANT = 0.00198588

def calc_pmf(name, K, n_reps, n_restarts, spring_constant, min_center, max_center, N_max, colvar_col, temperature=300, bin0=0):

    N_k = np.zeros(K, np.int32) #Number of snapshots per window
    K_k = np.zeros(K) #Spring constants
    r0_k = np.zeros(K) #Centers
    r_kn = np.zeros((K, N_max)) #Sep distance for each snapshot
    u_kn = np.zeros((K, N_max)) #Reduced potential energy (to be calculated)
    g_k = np.zeros(K) #Timeseries for subsampling
    nbins = 100 #Number of bins for pmf
    #nbins = 10 #Number of bins for pmf

    K_k[:] = spring_constant
    r0_k[:] = np.linspace(min_center, max_center, K)

    ##Load colvar values from simulations
    for i in range(bin0,K+bin0):
        for j in range(n_reps):
            try:
                colvars_data = []
                #colvars_data.append(np.loadtxt("image_%d/rep_%d/%s_im%d_%d.colvars.traj"%(i,j,name,i,j), usecols=(colvar_col))) #Remove this line later
                for k in range(1, n_restarts+1):
                    cv_file = "image_%d/rep_%d/%s_im%d_%d-%d.colvars.traj"%(i,j,name,i,j,k)
                    print(cv_file)
                    if os.path.isfile(cv_file):
                        colvars_data.append(np.loadtxt(cv_file, usecols=(colvar_col)))
                        #colvars_data.append(np.loadtxt("image_%d/rep_%d/%s_im%d_%d-%d.colvars.traj"%(i,j,name,i,j,k), usecols=(colvar_col)))

                        print("np.loadtxt passed")

                colvars_data = np.hstack(colvars_data)

                print("hstack passed")
                n_k = np.size(colvars_data)

                print(n_k)
                print(N_k[i-bin0])
                print(N_k[i-bin0] + n_k)
                print(np.shape(r_kn[i-bin0, N_k[i-bin0]:(N_k[i-bin0] + n_k)]))

                r_kn[i-bin0, N_k[i-bin0]:(N_k[i-bin0] + n_k)] = colvars_data

                print("r_kn updated")

                N_k[i-bin0] += n_k
            except:
                print("WARNING: One or more data files failed to load")
                pass

        #g_k[i] = timeseries.statisticalInefficiency(r_kn[i,:])
        #print(g_k[i])
        #indices = timeseries.subsampleCorrelatedData(r_kn[i,:], g=g_k[i])

        indices = list(range(N_k[i-bin0]))[::1]

        N_k[i-bin0] = len(indices)
        r_kn[i-bin0,0:N_k[i-bin0]] = r_kn[i-bin0,indices]

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

    print(r_kn)
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

    return bin_center_i, f_i, df_i

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--n_windows", type=int, nargs=1)
    parser.add_argument("--n_reps", type=int, nargs=1)
    parser.add_argument("--n_restarts", type=int, nargs=1)
    parser.add_argument("--k", type=float, nargs=1)
    parser.add_argument("--center_range", type=float, nargs=2)
    parser.add_argument("--colvar_col", type=int)
    parser.add_argument("--N_max", type=int, default=3000)
    parser.add_argument("--temp", type=float, default=300)
    args = parser.parse_args()

    return args

if __name__=="__main__":

    args = get_args()

    #calc_pmf(name, K, n_reps, n_restarts, spring_constant, min_center, max_center, N_max, colvar_col, temperature=300)

    bin_center_i, f_i, df_i = calc_pmf(args.name, args.n_windows[0], args.n_reps[0], args.n_restarts[0], args.k[0], args.center_range[0], args.center_range[1], args.N_max, args.colvar_col)

    print(bin_center_i)
    print(f_i*GAS_CONSTANT*args.temp)
    print(df_i)

    np.save("bins.npy", bin_center_i)
    np.save("pmf_kcal_mol.npy", f_i*GAS_CONSTANT*args.temp)
    np.save("error_kcal_mol.npy", df_i*GAS_CONSTANT*args.temp)

    matplotlib.rc('font',family='Helvetica-Normal',size=14)
    plt.errorbar(bin_center_i, f_i*GAS_CONSTANT*args.temp, yerr=df_i*GAS_CONSTANT*args.temp)
    plt.savefig("%s_pmf.png"%args.name, transparent=True)
