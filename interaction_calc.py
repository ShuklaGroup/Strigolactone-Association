import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rc('savefig', dpi=500)
matplotlib.rc('font',family='Helvetica-Normal',size=24)

import itertools
import mdtraj as md
import os

class PairInteraction(object):
    def __init__(self, pdb=None, proteinA_res=None, proteinB_res=None, pairs_file=None):
        self.pdb = pdb
        self.structure = md.load(pdb)
        self.proteinA_res = proteinA_res
        self.proteinB_res = proteinB_res
        if pairs_file is None:
            self.pairs = self.find_interface()
        else: 
            self.pairs = np.load("%s"%pairs_file)
        self.interaction_matrix = None
        self.masked_interaction_matrix = None

    def namd_setup(self, traj, template_config):
        """Write config files"""

        print(self.pairs)
        for i in range(len(self.pairs)):
            new_pdb_name = "%d-%d.pdb"%(self.pairs[i,0], self.pairs[i,1])
            self.write_pdb(new_pdb_name, self.pairs[i,0], self.pairs[i,1])
            self.write_conf(self.pairs[i], template_config, traj)

    def plot_cmesh(self, savename="pair_interaction.png", resshiftA=None, resshiftB=None):

        if self.interaction_matrix is None:
            self.construct_interaction_matrix()

        plt.pcolormesh(range(xmin, xmax), range(ymin, ymax), self.masked_interaction_matrix[xmin:xmax, ymin:ymax].T, cmap='jet')
        plt.colorbar()
        plt.savefig("%s"%savename)

    def plot_bars(self, savename="top_interactions.png", top_pairs=None, top_interactions=None):

        if top_interactions == None or top_pairs == None:
            top_pairs, top_interactions = self.interaction_sorter(n_top=20)

        labels = []
        for i in range(len(top_interactions)):
            labels.append("%d-%d"%(top_pairs[i][0], top_pairs[i][1]))

        plt.figure()
        fig, ax = plt.subplots()

        ax.bar(labels, top_interactions)
        plt.savefig("%s"%savename)

    def interaction_sorter(self, n_top=100):

        if self.interaction_matrix is None:
            self.construct_interaction_matrix()

        n_cols = np.shape(self.interaction_matrix)[1]

        flattened_sort_indices = np.argsort(self.interaction_matrix, axis=None)
        print(np.shape(flattened_sort_indices))

        top_pairs = []
        top_interactions = []

        for i in range(n_top):
            row = flattened_sort_indices[-(i+1)]/n_cols

            row = int(flattened_sort_indices[-(i+1)]/n_cols)
            col = flattened_sort_indices[-(i+1)] % n_cols

            top_pairs.append((row, col))
            top_interactions.append(self.interaction_matrix[row, col])

        return top_pairs, top_interactions

    def find_interface(self, savename="pairs.npy"):
        """Determine which residues are interfacial"""

        ##Contact list
        contact_list = np.zeros((len(self.proteinA_res)*len(self.proteinB_res), 2))

        pairs = itertools.product(self.proteinA_res, self.proteinB_res)

        i = 0
        for pair in pairs:
            contact_list[i,0] = pair[0]
            contact_list[i,1] = pair[1]
            i += 1

        distances = md.compute_contacts(self.structure, contacts=contact_list, scheme='closest-heavy')[0]

        use_contact = np.zeros(np.shape(distances))
        use_contact[distances < 1.2] = 1

        interface_contacts = []

        for i in range(np.size(use_contact)):
            if use_contact[0,i] == 1:
                interface_contacts.append(contact_list[i,:])        

        interface_contacts = np.vstack(interface_contacts)

        if savename is not None:
            np.save("%s"%savename, interface_contacts)

        return interface_contacts

    def write_pdb(self, new_pdb, resid1, resid2):
        
        original_file = open(self.pdb, 'r')
        new_file = open(new_pdb,'w')
        for line in original_file:
            if line.split()[0] == "ATOM":
                resid = int(line.split()[5]) #ASSUMES 6TH COLUMN IS RESID
                if resid == resid1+1:
                    new_file.write(line.replace("1.00  0.00","1.00  1.00"))
                elif resid == resid2+1:
                    new_file.write(line.replace("1.00  0.00","1.00  2.00"))
                else:
                    new_file.write(line)
            else:
                new_file.write(line)

    def write_conf(self, pair, template_config, traj):

        original_file = open(template_config, 'r')
        new_file = open("%d-%d.conf"%(pair[0], pair[1]),'w')

        for line in original_file:
            if len(line.split()) == 0:
                new_file.write(line)
            elif line.split()[0] == "outputname":
                new_file.write("outputname %d-%d \n"%(pair[0], pair[1]))
            elif line.split()[0] == "pairInteractionFile":
                new_file.write("pairInteractionFile %d-%d.pdb \n"%(pair[0], pair[1]))
            elif line.split()[0] == "coorfile" and line.split()[1] == "open":
                new_file.write("coorfile open dcd %s \n"%traj)
            else:
                new_file.write(line)

        original_file.close()
        new_file.close()

    def get_avg_interaction_energy(self, pair):

        logfile = open("%d-%d.log"%(pair[0], pair[1]),'r')

        energies = []

        for line in logfile.readlines():
            if len(line.split()) == 0:
                pass
            elif line.split()[0] == "ENERGY:":
                energies.append(float(line.split()[11]))
            else:
                pass

        avg_interaction_energy = np.mean(np.array(energies))

        return avg_interaction_energy

    def construct_interaction_matrix(self):

        interaction_matrix = np.zeros((np.int(np.max(self.pairs[:,0])+1), np.int(np.max(self.pairs[:,1])+1))) #pcolormesh matrix

        for i in range(len(self.pairs)):
            try:
                interaction_matrix[int(self.pairs[i,0]), int(self.pairs[i,1])] = self.get_avg_interaction_energy(self.pairs[i,:])
            except:
                interaction_matrix[int(self.pairs[i,0]), int(self.pairs[i,1])] = 0

        #Normalize as described in Sercinoglu and Ozbek, Nucleic Acids Res. 2018
        #interaction_energy[interaction_energy > 0] = 0
        #max_att = np.min(interaction_energy)
        #interaction_energy[interaction_energy < 0] /= max_att
        #interaction_matrix[interaction_matrix > 0] = 0
        #max_att = np.min(interaction_matrix)
        #max_att = -1
        #interaction_matrix[interaction_matrix < 0] /= max_att
            
        xmin = int(np.min(self.pairs[:,0]))
        xmax = int(np.max(self.pairs[:,0]))
        ymin = int(np.min(self.pairs[:,1]))
        ymax = int(np.max(self.pairs[:,1]))

        masked_interaction_matrix = np.ma.masked_equal(interaction_matrix, 0, copy=True)

        self.interaction_matrix = interaction_matrix
        self.masked_interaction_matrix = masked_interaction_matrix

