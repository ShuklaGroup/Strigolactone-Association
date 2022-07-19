import numpy as np
from scipy import integrate
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import os

global GAS_CONSTANT
GAS_CONSTANT = 0.001987204 #Units kcal/(mol*K)

global RAD_CONVERT
RAD_CONVERT = np.pi/180 #Degrees to radians conversion

class FreeEnergyCalc(object):
    def __init__(self, temperature, r_cut=None):
        self.temperature = temperature
        self.beta = 1.0/(GAS_CONSTANT*self.temperature)
        self.r_cut = None

    def calc_all(self, restr_file, bound_restr_sim, unbound_restr_sim):

        print(restr_file)
        print(bound_restr_sim)
        print(unbound_restr_sim)

        #Load restraints
        restraint_dict = self._parse_restr_file(restr_file)

        print(restraint_dict)

        #Separation term
        r_bins = np.load("./bins.npy")
        r_pmf = np.load("./pmf_kcal_mol.npy")

        if self.r_cut is None:
            self.r_cut = self._calc_r_cut(r_bins, r_pmf)
        
        print("Bound cutoff in Angstroms is %f"%self.r_cut)

        I_star = self.assoc_pmf_integral(r_bins, r_pmf)

        print("I_star term is %e"%I_star)

        #Bulk orientation term
        O_star = self.bulk_orientation(restraint_dict['theta2'][1], restraint_dict['phi2'][1], restraint_dict['theta2'][0])

        print("Bulk orientation term in Angstroms^2 is %f"%O_star)

        #Bulk angular term
        dGB0 = self.bulk_angular(restraint_dict['theta1'][1], restraint_dict['phi1'][1], restraint_dict['psi'][1], restraint_dict['theta1'][0])

        print("Bulk angular term in kcal/mol is %f"%dGB0)

        #Restraint simulations, bound state
        bound_restr_dG = {}
        for restr in bound_restr_sim:
            restr_bins = np.load("./restraint_release/%s/bins.npy"%restr)
            restr_pmf = np.load("./restraint_release/%s/pmf_kcal_mol.npy"%restr)
            dG = self.restraint_dG(restr_bins, restr_pmf, restraint_dict[restr][0], restraint_dict[restr][1])
            bound_restr_dG[restr] = dG

        print(bound_restr_dG)

        #Restraint simulations, bound state
        unbound_restr_dG = {}
        for restr in unbound_restr_sim:
            restr_bins = np.load("./restraint_release_unbound/%s/bins.npy"%restr)
            restr_pmf = np.load("./restraint_release_unbound/%s/pmf_kcal_mol.npy"%restr)
            dG = self.restraint_dG(restr_bins, restr_pmf, restraint_dict[restr][0], restraint_dict[restr][1])
            unbound_restr_dG[restr] = dG

        print(unbound_restr_dG)

        #Site angular/orientation term
        dGS0 = 0
        for restr in bound_restr_sim:
            if 'rmsd' not in restr:
                dGS0 += bound_restr_dG[restr]
                print(restr)
                print(bound_restr_dG[restr])

        print(dGS0)

        #Site and bulk RMSD terms
        dG_rmsd_site_all = np.zeros(len(unbound_restr_sim))
        dG_rmsd_bulk_all = np.zeros(len(unbound_restr_sim))
        for i in range(len(unbound_restr_sim)):
            restr = unbound_restr_sim[i]
            dG_rmsd_site_all[i] = bound_restr_dG[restr]
            dG_rmsd_bulk_all[i] = unbound_restr_dG[restr]
            print(restr)
            print(dG_rmsd_site_all[i])
            print(dG_rmsd_bulk_all[i])

        #Putting everything togther
        KA = O_star*I_star*np.exp(-self.beta*(np.sum(dG_rmsd_bulk_all - dG_rmsd_site_all) + dGB0 - dGS0))

        print(KA)

        standard_dG_binding = -(1.0/self.beta)*np.log(KA/1661)

        print(standard_dG_binding)

        return standard_dG_binding

    def assoc_pmf_integral(self, r, pmf):
        """I* term"""

        W_ref = np.mean(pmf[r>self.r_cut]) #Unbound pmf value

        pmf_exp_func = np.exp(-self.beta*(pmf[r<self.r_cut] - W_ref))
        pmf_integral = self._compute_integral(r[r<self.r_cut], pmf_exp_func)

        return pmf_integral

    def bulk_orientation(self, theta2_center, phi2_center, k_deg):
        """O* term, output in Angstroms^2
           Inputs: r_cut (Angstroms) - cutoff value for bound state definition
                   theta2_center, phi2_center: reference value for theta2
                   k_deg: Force constant of restraint
        """

        k_rad = k_deg/(RAD_CONVERT**2)

        #Check for and convert negative angles to coterminal positive angles
        if theta2_center < 0:
            theta2_center += 360
        if phi2_center < 0:
            phi2_center += 360

        phi_range = np.linspace(0, 2*np.pi, 2000)
        phi_exp_func = np.exp(-self.beta*0.5*k_rad*(phi_range - phi2_center*RAD_CONVERT)**2)
        phi_integral = self._compute_integral(phi_range, phi_exp_func)

        theta_range = np.linspace(0, np.pi, 1000)
        theta_exp_func = np.sin(theta_range)*np.exp(-self.beta*0.5*k_rad*(theta_range - theta2_center*RAD_CONVERT)**2)
        theta_integral = self._compute_integral(theta_range, theta_exp_func)

        O_term = (self.r_cut**2)*phi_integral*theta_integral

        return O_term

    def bulk_angular(self, theta1_center, phi1_center, psi_center, k_deg):

        k_rad = k_deg/(RAD_CONVERT**2)

        #Check for and convert negative angles to coterminal positive angles
        if theta1_center < 0:
            theta1_center += 360
        if phi1_center < 0:
            phi1_center += 360
        if psi_center < 0:
            psi_center += 360
          
        theta_range = np.linspace(0, np.pi, 1000)
        theta1_exp_func = np.sin(theta_range)*np.exp(-self.beta*0.5*k_rad*(theta_range - theta1_center*RAD_CONVERT)**2)
        theta1_integral = self._compute_integral(theta_range, theta1_exp_func)

        phi_range = np.linspace(0, 2*np.pi, 2000)
        phi1_exp_func = np.exp(-self.beta*0.5*k_rad*(phi_range - phi1_center*RAD_CONVERT)**2)
        phi1_integral = self._compute_integral(phi_range, phi1_exp_func)

        psi_range = np.linspace(0, 2*np.pi, 2000)
        psi_exp_func = np.exp(-self.beta*0.5*k_rad*(psi_range - psi_center*RAD_CONVERT)**2)
        psi_integral = self._compute_integral(psi_range, psi_exp_func)

        dG_bulk = -(1.0/self.beta)*np.log((1.0/(8.0*np.pi))*theta1_integral*phi1_integral*psi_integral)

        return dG_bulk

    def restraint_dG(self, xi, pmf, k, center):
        """Calculate contribution of restrained variable xi"""

        num_func = np.exp(-self.beta*(pmf + 0.5*k*(xi - center)**2))
        num_integral = self._compute_integral(xi, num_func)

        denom_func = np.exp(-self.beta*pmf)
        denom_integral = self._compute_integral(xi, denom_func)

        restraint_avg = num_integral/denom_integral

        restraint_dG = (1.0/self.beta)*np.log(restraint_avg)

        return restraint_dG

    def _parse_restr_file(self, restr_file):

        restraint_dict = {}

        with open(restr_file) as f:
            for line in f:
                restr_name = line.split()[0]
                k = float(line.split()[1])
                center = float(line.split()[2])
                restraint_dict[restr_name] = np.array((k, center))

        return restraint_dict

    def _calc_r_cut(self, r, pmf):
        """Calculate r cutoff value for unbound state"""

        pmf_end = pmf[-3]
        d_pmf = pmf[1:] - pmf[:-1] #Difference between subsequent pmf values
        r_inter = 0.5*(r[1:] + r[:-1]) #Values of r corresponding to d_pmf values

        for i in range(len(d_pmf)):
            if d_pmf[i] < 0.1 and np.abs(pmf_end - np.mean((pmf[i], pmf[i+1]))) < 0.1:
                r_cut = r_inter[i]
                break

        return r_cut

    def _compute_integral(self, x, y):
        """Compute an integral"""

        heights = x[1:] - x[:-1]
        y_avg = 0.5*(y[1:] + y[:-1])
        integral = np.sum(heights*y_avg)

        return integral

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--restr_file", type=str)
    parser.add_argument("--bound_restr_sim", type=str, nargs='+')
    parser.add_argument("--unbound_restr_sim", type=str, nargs='+')
    parser.add_argument("--temperature", type=float, default=300)
    args = parser.parse_args()

    return args

if __name__=="__main__":

    args = get_args()

    print(args.temperature)
    dG_calc = FreeEnergyCalc(args.temperature)
    dG_calc.calc_all(args.restr_file, args.bound_restr_sim, args.unbound_restr_sim)

