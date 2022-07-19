import numpy as np
import scipy
import mdtraj as md

import argparse
import os

class UmbrellaSim(object):

    def __init__(self, name, top, min_center=None, max_center=None, n_images = None, n_reps=None, temperature=300):
        self.name = name
        self.topfile = top
        self.n_images = n_images
        self.n_reps = n_reps
        self.temperature = 300
        self.min_center = min_center
        self.max_center = max_center

    def sim_setup(self, smd_traj="smd.dcd", colvar_template=None, sim_template=None, job_template=None, init_type=None):

        cwd = os.getcwd()

        for i in range(self.n_images):
            if not os.path.isdir("image_%d"%i):
                os.mkdir("image_%d"%i)

            os.chdir("image_%d"%i)

            for j in range(self.n_reps):
                if not os.path.isdir("rep_%d"%j):
                    os.mkdir("rep_%d"%j)

            os.chdir(cwd)

        if init_type == 'smd':
            centers, box_dim = self.get_structures_smd("./%s"%smd_traj)
        else:
            centers, box_dim = self.get_structures_general("./%s"%smd_traj)

        force_constants = self.compute_force_constants(centers)

        #Write colvars files
        self.write_colvars(centers, force_constants, colvar_template)

        #Write sim config files
        self.write_sim_config(10*box_dim, sim_template)
        self.write_restart_sim_config("sim_restart.conf", 1)

        #Write job scripts
        #self.write_job_scripts_bundled(job_template)
        self.write_job_scripts_combined(job_template)

    def sim_restart(self, sim_template=None, job_template=None, restart_number=1):

        cwd = os.getcwd()
        self.write_restart_sim_config(sim_template, restart_number)
        #self.write_restart_job_scripts(job_template, restart_number)
        self.write_restart_job_scripts_bundled(job_template, restart_number)

    def get_structures_general(self, traj, indices_file="indices.txt"):
        """Extract initial images, compute initial centers"""
        
        traj = md.load(traj, top="./sys/%s"%self.topfile)
        centers = np.linspace(self.min_center, self.max_center, self.n_images)

        np.savetxt("centers.txt", centers)

        indices = np.loadtxt(indices_file, dtype=int)
        frames = traj[indices]
        box_sizes = frames.unitcell_lengths

        for i in range(len(indices)):
            #Get restarts
            frames[i].save_amberrst7("image_%d/%s_image%d.rst"%(i, self.name, i))

        return centers, box_sizes

    def get_structures_smd(self, smd_traj):
        """Extract initial images from SMD trajectory and compute initial centers"""

        traj = md.load(smd_traj, top="./sys/%s"%self.topfile)
        indices = np.round(np.linspace(0, len(traj)-1, self.n_images)).astype(int)
        frames = traj[indices]
        box_sizes = frames.unitcell_lengths #Box dims for sim files

        for i in range(len(indices)):
            #Get restart files
            frames[i].save_amberrst7("image_%d/%s_image%d.rst"%(i, self.name, i))

        #Compute initial restraint centers (1D)
        centers = np.linspace(self.min_center, self.max_center, self.n_images)

        np.savetxt("centers.txt", centers)

        return centers, box_sizes

    def compute_force_constants(self, centers):

        force_constants = np.zeros(self.n_images)

        spacing = centers[1]-centers[0]
        force_constants[:] = 0.00198*self.temperature/(0.5*spacing)**2 

        return force_constants

    def write_colvars(self, centers, force_constants, template_config):
        """Replace separation distance colvar center for each image"""

        for image in range(self.n_images):

            original_file = open(template_config, 'r')
            new_file = open("image_%d/%s_image%d.colvar"%(image, self.name, image),'w')

            for line in original_file:

                if len(line.split()) == 0:
                    new_file.write(line)
                elif line.split()[-1] == "SETFORCECONSTANT":
                    new_file.write("  forceConstant %f \n"%force_constants[image])
                elif line.split()[-1] == "SETCENTER":
                    new_file.write("  centers %f \n"%centers[image])
                else:
                    new_file.write(line)

            original_file.close()
            new_file.close()
 
    def write_sim_config(self, box_dim, template_config):

        for image in range(self.n_images):
            for rep in range(self.n_reps):

                original_file = open(template_config, 'r')
                new_file = open("image_%d/rep_%d/%s_im%d_%d.conf"%(image, rep, self.name, image, rep),'w')

                for line in original_file:
                    if len(line.split()) == 0:
                        new_file.write(line)
                    elif line.split()[0] == "ambercoor":
                        new_file.write("ambercoor ../%s_image%d.rst \n"%(self.name, image))
                    elif line.split()[-1] == "SETOUTPUTNAME":
                        new_file.write("set outputname %s_im%d_%d \n"%(self.name, image, rep))
                    elif line.split()[0] == "cellBasisVector1":
                        new_file.write("cellBasisVector1 %f 0.0 0.0 \n"%box_dim[image, 0])
                    elif line.split()[0] == "cellBasisVector2":
                        new_file.write("cellBasisVector2 0.0 %f 0.0 \n"%box_dim[image, 1])
                    elif line.split()[0] == "cellBasisVector3":
                        new_file.write("cellBasisVector3 0.0 0.0 %f \n"%box_dim[image, 2])
                    elif line.split()[0] == "colvarsConfig":
                        new_file.write("colvarsConfig ../%s_image%d.colvar \n"%(self.name, image))
                    else:
                        new_file.write(line)

                original_file.close()
                new_file.close()

    def write_restart_sim_config(self, template_config, restart_number):

        for image in range(self.n_images):
            for rep in range(self.n_reps):

                original_file = open(template_config, 'r')
                new_file = open("image_%d/rep_%d/%s_im%d_%d_restart-%d.conf"%(image, rep, self.name, image, rep, restart_number),'w')

                if restart_number == 1:
                    for line in original_file:
                        if len(line.split()) == 0:
                            new_file.write(line)
                        elif line.split()[0] == "ambercoor":
                            new_file.write("ambercoor ../%s_image%d.rst \n"%(self.name, image))
                        elif line.split()[0] == "binCoordinates":
                            new_file.write("binCoordinates %s_im%d_%d.restart.coor \n"%(self.name, image, rep))
                        elif line.split()[0] == "binVelocities":
                            new_file.write("binVelocities %s_im%d_%d.restart.vel \n"%(self.name, image, rep))
                        elif line.split()[0] == "extendedSystem":
                            new_file.write("extendedSystem %s_im%d_%d.restart.xsc \n"%(self.name, image, rep)) 
                        elif line.split()[-1] == "SETOUTPUTNAME":
                            new_file.write("set outputname %s_im%d_%d-%d \n"%(self.name, image, rep, restart_number))
                        elif line.split()[0] == "colvarsConfig":
                            new_file.write("colvarsConfig ../%s_image%d.colvar \n"%(self.name, image))
                        else:
                            new_file.write(line)

                else:
                    for line in original_file:
                        if len(line.split()) == 0:
                            new_file.write(line)
                        elif line.split()[0] == "ambercoor":
                            new_file.write("ambercoor ../%s_image%d.rst \n"%(self.name, image))
                        elif line.split()[0] == "binCoordinates":
                            new_file.write("binCoordinates %s_im%d_%d-%d.restart.coor \n"%(self.name, image, rep, restart_number-1))
                        elif line.split()[0] == "binVelocities":
                            new_file.write("binVelocities %s_im%d_%d-%d.restart.vel \n"%(self.name, image, rep, restart_number-1))
                        elif line.split()[0] == "extendedSystem":
                            new_file.write("extendedSystem %s_im%d_%d-%d.restart.xsc \n"%(self.name, image, rep, restart_number-1)) 
                        elif line.split()[-1] == "SETOUTPUTNAME":
                            new_file.write("set outputname %s_im%d_%d-%d \n"%(self.name, image, rep, restart_number))
                        elif line.split()[0] == "colvarsConfig":
                            new_file.write("colvarsConfig ../%s_image%d.colvar \n"%(self.name, image))
                        else:
                            new_file.write(line)

                original_file.close()
                new_file.close()
    
    def write_job_scripts(self, template_config):

        for image in range(self.n_images):
            for rep in range(self.n_reps):

                original_file = open(template_config, 'r')
                new_file = open("image_%d/rep_%d/run_US.pbs"%(image, rep),'w')

                for line in original_file:
                    if len(line.split()) == 0:
                        new_file.write(line)
                    elif line.split()[0] == "aprun":
                        new_file.write("aprun -n 1 ~/NAMD_2.13_Linux-x86_64-multicore-CUDA/namd2 %s_im%d_%d.conf &> %s_im%d_%d.log \n"%(self.name, image, rep, self.name, image, rep))
                    else:
                        new_file.write(line)

                original_file.close()
                new_file.close()

    def write_job_scripts_bundled(self, template_config):

        for image in range(self.n_images):

            original_file = open(template_config, 'r')
            new_file = open("image_%d/run_US.pbs"%image,'w')

            for line in original_file:
                if len(line.split()) == 0:
                    new_file.write(line)
                elif line.split()[0] == "aprun":
                    for rep in range(self.n_reps):
                        new_file.write("aprun -n 1 ~/NAMD_2.13_Linux-x86_64-multicore-CUDA/namd2 rep_%d/%s_im%d_%d.conf &> rep_%d/%s_im%d_%d.log &\n"%(rep, self.name, image, rep, rep, self.name, image, rep))
                else:
                    new_file.write(line)
 
            new_file.write("\n")
            new_file.write("wait")

            original_file.close()
            new_file.close()

    def write_job_scripts_combined(self, template_config):

        for image in range(self.n_images):

            original_file = open(template_config, 'r')
            new_file = open("image_%d/run_US.pbs"%image,'w')

            for line in original_file:
                if len(line.split()) == 0:
                    new_file.write(line)
                elif line.split()[0] == "aprun":
                    for rep in range(self.n_reps):
                        new_file.write("aprun -n 1 ~/NAMD_2.13_Linux-x86_64-multicore-CUDA/namd2 rep_%d/%s_im%d_%d.conf &> rep_%d/%s_im%d_%d.log &\n"%(rep, self.name, image, rep, rep, self.name, image, rep))
                    new_file.write("\n")
                    new_file.write("wait\n")
                    new_file.write("\n")

                    for rep in range(self.n_reps):
                        new_file.write("aprun -n 1 ~/NAMD_2.13_Linux-x86_64-multicore-CUDA/namd2 rep_%d/%s_im%d_%d_restart-1.conf &> rep_%d/%s_im%d_%d-restart-1.log &\n"%(rep, self.name, image, rep, rep, self.name, image, rep))
                else:
                    new_file.write(line)
 
            new_file.write("\n")
            new_file.write("wait")

            original_file.close()
            new_file.close()

    def write_restart_job_scripts_bundled(self, template_config, restart_number):

        for image in range(self.n_images):

            original_file = open(template_config, 'r')
            new_file = open("image_%d/run_US_restart-%d.pbs"%(image, restart_number),'w')

            for line in original_file:
                if len(line.split()) == 0:
                    new_file.write(line)
                elif line.split()[0] == "aprun":
                    for rep in range(self.n_reps):
                        new_file.write("aprun -n 1 ~/NAMD_2.13_Linux-x86_64-multicore-CUDA/namd2 rep_%d/%s_im%d_%d_restart-%d.conf &> rep_%d/%s_im%d_%d-restart-%d.log &\n"%(rep, self.name, image, rep, restart_number, rep, self.name, image, rep, restart_number))
                else:
                    new_file.write(line)
 
            new_file.write("\n")
            new_file.write("wait")

            original_file.close()
            new_file.close()

    def write_restart_job_scripts(self, template_config, restart_number):

        for image in range(self.n_images):
            for rep in range(self.n_reps):

                original_file = open(template_config, 'r')
                new_file = open("image_%d/rep_%d/run_US_restart-7.pbs"%(image, rep),'w')

                for line in original_file:
                    if len(line.split()) == 0:
                        new_file.write(line)
                    elif line.split()[0] == "aprun":
                        new_file.write("aprun -n 1 ~/NAMD_2.13_Linux-x86_64-multicore-CUDA/namd2 %s_im%d_%d_restart-%d.conf &> %s_im%d_%d.log \n"%(self.name, image, rep, restart_number, self.name, image, rep))
                    else:
                        new_file.write(line)

                original_file.close()
                new_file.close()

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--prmtop", type=str, required=True)
    parser.add_argument("--center_range", type=float, nargs=2, required=False)
    parser.add_argument("--n_images",type=int, required=True)
    parser.add_argument("--n_reps", type=int, default=5, required=True)
    parser.add_argument("--temperature", type=float, default=300, required=True)
    parser.add_argument("--restart_number", type=int, default=None)
    parser.add_argument("--smd_traj", type=str)
    parser.add_argument("--colvar_template", type=str)
    args = parser.parse_args()

    return args

if __name__=="__main__":

    args = get_args()

    if args.restart_number is not None:
        us_sim = UmbrellaSim(args.name, args.prmtop, n_images=args.n_images, n_reps=args.n_reps, temperature=args.temperature)
        us_sim.sim_restart(sim_template="sim_restart.conf", job_template="md_run_restart.pbs", restart_number=args.restart_number)
    else:
        us_sim = UmbrellaSim(args.name, args.prmtop, args.center_range[0], args.center_range[1], n_images=args.n_images, n_reps=args.n_reps, temperature=args.temperature)
        us_sim.sim_setup(smd_traj = args.smd_traj, colvar_template=args.colvar_template, sim_template="smd.conf", job_template="md_run.pbs")


