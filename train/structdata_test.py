#%%writefile structdata.py

import torch
from torch.utils.data import Dataset, DataLoader
import generatecoeffs as gc
import pickle
import numpy as np


class Structures(Dataset):
    #@profile
    def __init__(self, rdf_means, adf_means, rdf_widths, adf_widths , num_rdf, num_adf, cutoff,net_ind):
        self.num_rdf =num_rdf
        self.num_adf =num_adf

        """
        Load all the files needed to generate the dataset and generate the coeffs
        """

        #Loading files...
        root = '/global/home/users/jwyoon/NN_Mo/data/vacancy_test/'
        with open(root+'num_atoms_categories.pickle', 'rb') as handle:
            categorized_num_atoms = pickle.load(handle)
        with open(root+'categorized_vacancy_energy.pickle', 'rb') as handle:
            target_energies = pickle.load(handle)
        with open(root+'categorized_vacancy_forces.pickle', 'rb') as handle:
            target_forces = pickle.load(handle)
        with open(root+'categorized_cutoff_vacancy_4A_N_all_xyzIncluded_index_image.pickle', 'rb') as handle:
            structs = pickle.load(handle)

        data_normalization = (np.load('myNet'+str(net_ind)+'_normalization_data.npy')).tolist()
        self.energies_mean = data_normalization[0]
        self.energies_std = data_normalization[1]
        self.rdf_means = data_normalization[2:num_rdf+2]
        self.rdf_stds = data_normalization[num_rdf+2:2*num_rdf+2]
        self.adf_means = data_normalization[2*num_rdf+2:2*num_rdf+num_adf+2]
        self.adf_stds = data_normalization[2*num_rdf+num_adf+2:2*num_rdf+2*num_adf+2]


        #generate rdf,adf,coeffs

        print('Generating RDF and ADF coeffs nowwww!!! Pay attention to cpu usage :) ',flush=True)


        categorized_rdf_adf =\
        [gc.generateRDFADFcoeffs_gaussians(category,
                                           cutoff,
                                           categorized_num_atoms[ind],
                                           rdf_means,
                                           rdf_widths,
                                           adf_means,
                                           adf_widths)
        for ind,category in enumerate(structs) ]

        print('Done with calculating rdf,adf coefficients!',flush=True)

        #flatten the data
        self.central_atom_index = []
        self.neigh_atom_index = []
        self.energies = []
        self.forces_x = []
        self.forces_y = []
        self.forces_z = []
        self.rdf = []
        self.rdf_dx = []
        self.rdf_dy = []
        self.rdf_dz = []
        self.adf = []
        self.adf_dx = []
        self.adf_dy = []
        self.adf_dz = []

        for ind in range(len(target_energies)):
            self.central_atom_index += [struct for struct in categorized_rdf_adf[ind]['central_atom_index']]
            self.neigh_atom_index += [struct for struct in categorized_rdf_adf[ind]['neigh_atom_index']]

            self.energies += target_energies[ind]
            #print('self.energies')
            #print(self.energies)
            self.forces_x += [ [atom_forces[0] for atom_forces in struct] for struct in target_forces[ind]]
            #print('self.forces_x')
            #print(self.forces_x)
            self.forces_y += [ [atom_forces[1] for atom_forces in struct] for struct in target_forces[ind]]
            self.forces_z += [ [atom_forces[2] for atom_forces in struct] for struct in target_forces[ind]]
            self.rdf += [struct for struct in categorized_rdf_adf[ind]['rdf']]
            #print('self.rdf before')
            #print(categorized_rdf_adf[ind]['rdf'])
            #print('self.rdf after')
            #print(self.rdf)
            self.rdf_dx += [struct for struct in categorized_rdf_adf[ind]['rdf_dx']]
            #print('self.rdf_dx')
            #print(self.rdf_dx)
            self.rdf_dy += [struct for struct in categorized_rdf_adf[ind]['rdf_dy']]
            self.rdf_dz += [struct for struct in categorized_rdf_adf[ind]['rdf_dz']]
            self.adf += [struct for struct in categorized_rdf_adf[ind]['adf']]
            #print('self.adf')
            #print(self.adf)
            self.adf_dx += [struct for struct in categorized_rdf_adf[ind]['adf_dx']]
            self.adf_dy += [struct for struct in categorized_rdf_adf[ind]['adf_dy']]
            self.adf_dz += [struct for struct in categorized_rdf_adf[ind]['adf_dz']]
        #print(self.rdf)

        self.normalize_all()

    def normalize_all(self):

        #we're using the mean and std from the training data as loaded in init()
        self.energies = [(i-self.energies_mean)/self.energies_std for i in self.energies]

        #self.forces_mean,self.forces_std,self.forces_x,self.forces_y,self.forces_z =\
        #self.normalize_forces(self.forces_x, self.forces_y, self.forces_z)
        self.forces_x = [ [(atom_force)/self.energies_std for atom_force in struct] for struct in self.forces_x]
        self.forces_y = [ [(atom_force)/self.energies_std for atom_force in struct] for struct in self.forces_y]
        self.forces_z = [ [(atom_force)/self.energies_std for atom_force in struct] for struct in self.forces_z]

        #self.rdf_means,self.rdf_stds,self.rdf = self.normalize_rdf(self.rdf)
        #self.rdf_dx_means,self.rdf_dx_stds,self.rdf_dx = self.normalize_rdf(self.rdf_dx)
        #self.rdf_dy_means,self.rdf_dy_stds,self.rdf_dy = self.normalize_rdf(self.rdf_dy)
        #self.rdf_dz_means,self.rdf_dz_stds,self.rdf_dz = self.normalize_rdf(self.rdf_dz)

        #we're using the means and stds of the training sample as loaded in init()
        self.rdf,self.rdf_dx,self.rdf_dy,self.rdf_dz =\
        self.normalize_coeffsNderivatives(self.rdf, self.rdf_dx, self.rdf_dy, self.rdf_dz, self.num_rdf,self.rdf_means,self.rdf_stds)

        #self.adf_means,self.adf_stds,self.adf = self.normalize_adf(self.adf)
        #self.adf_dx_means,self.adf_dx_stds,self.adf_dx = self.normalize_adf(self.adf_dx)
        #self.adf_dy_means,self.adf_dy_stds,self.adf_dy = self.normalize_adf(self.adf_dy)
        #self.adf_dz_means,self.adf_dz_stds,self.adf_dz = self.normalize_adf(self.adf_dz)
        self.adf,self.adf_dx,self.adf_dy,self.adf_dz =\
        self.normalize_coeffsNderivatives(self.adf, self.adf_dx, self.adf_dy, self.adf_dz, self.num_adf,self.adf_means,self.adf_stds)


    def normalize_forces(self, forces_x, forces_y, forces_z):
        flat = [ atom_force for struct in forces_x for atom_force in struct ]+\
        [ atom_force for struct in forces_y for atom_force in struct ]+\
        [ atom_force for struct in forces_z for atom_force in struct ]
        #print(flat)
        forces_mean = np.mean(flat)
        forces_std = np.std(flat)
        #print(forces_mean)
        #print(forces_std)
        #dealing with std==0 cases
        if forces_std == 0:
            forces_std =1
        normalized_forces_x = [ np.array([(atom_force-forces_mean)/forces_std for atom_force in struct]) for struct in forces_x]
        normalized_forces_y = [ np.array([(atom_force-forces_mean)/forces_std for atom_force in struct]) for struct in forces_y]
        normalized_forces_z = [ np.array([(atom_force-forces_mean)/forces_std for atom_force in struct]) for struct in forces_z]
        #print(normalized_forces)
        return forces_mean,forces_std,normalized_forces_x,normalized_forces_y,normalized_forces_z

    def normalize_rdf(self, rdf):
        rdf_means = []
        rdf_stds = []
        #print(rdf)
        #calculate means and stds
        for i in range(self.num_rdf):
            #rdf of a particular i-th kernel
            flat_rdf = [ atom_rdf[i] for struct in rdf for atom_rdf in struct]
            rdf_means += [np.mean(flat_rdf)]
            std = np.std(flat_rdf)
            #dealing with std==0 cases
            if std == 0:
                rdf_stds += [1]
            else:
                rdf_stds += [std]
        #print(rdf_means)
        #print(rdf_stds)
        #normalize now
        norm_rdf = [ [[(rdf-rdf_means[i])/rdf_stds[i] for i,rdf in enumerate(atom_rdf) ] for atom_rdf in struct ]
                    for struct in rdf]
        #print(norm_rdf)
        #print('******')
        return rdf_means, rdf_stds, norm_rdf

    def normalize_adf(self, adf):
        adf_means = []
        adf_stds = []

        #calculate means and stds
        for i in range(self.num_adf):
            #rdf of a particular i-th kernel
            flat_adf = [ atom_adf[i] for struct in adf for atom_adf in struct]
            adf_means += [np.mean(flat_adf)]
            std = np.std(flat_adf)
            #dealing with std==0 cases
            if std == 0:
                adf_stds += [1]
            else:
                adf_stds += [std]

        #normalize now
        norm_adf = [ [[(adf-adf_means[i])/adf_stds[i] for i,adf in enumerate(atom_adf)] for atom_adf in struct]
                    for struct in adf]

        return adf_means, adf_stds, norm_adf

    def normalize_coeffsNderivatives(self, coeffs, coeffs_x, coeffs_y, coeffs_z, num_coeffs,coeffs_means,coeffs_stds):

        #normalize now
        norm_coeffs = [ [[(coeff-coeffs_means[i])/coeffs_stds[i]
                          for i,coeff in enumerate(atom_coeffs)]
                         for atom_coeffs in struct]
                       for struct in coeffs]
        norm_coeffs_x = [ [[(coeff)/coeffs_stds[i]
                            for i,coeff in enumerate(atom_coeffs)]
                           for atom_coeffs in struct]
                         for struct in coeffs_x]
        norm_coeffs_y = [ [[(coeff)/coeffs_stds[i]
                            for i,coeff in enumerate(atom_coeffs)]
                           for atom_coeffs in struct]
                         for struct in coeffs_y]
        norm_coeffs_z = [ [[(coeff)/coeffs_stds[i]
                            for i,coeff in enumerate(atom_coeffs)]
                           for atom_coeffs in struct]
                         for struct in coeffs_z]

        return norm_coeffs, norm_coeffs_x, norm_coeffs_y, norm_coeffs_z

    def __len__(self):
        return len(self.energies)

    #@profile
    def __getitem__(self, idx):
        sample = {'energy_pa': np.array([self.energies[idx]]),
                  'force_x': np.array(self.forces_x[idx]),
                  'force_y': np.array(self.forces_y[idx]),
                  'force_z': np.array(self.forces_z[idx]),
                  'coeffs': np.array([ self.rdf[idx][ind]+self.adf[idx][ind] for ind in range(len(self.rdf[idx]))]),
                 'coeffs_x': np.array([ self.rdf_dx[idx][ind]+self.adf_dx[idx][ind] for ind in range(len(self.rdf_dx[idx]))]),
                 'coeffs_y': np.array([ self.rdf_dy[idx][ind]+self.adf_dy[idx][ind] for ind in range(len(self.rdf_dy[idx]))]),
                 'coeffs_z': np.array([ self.rdf_dz[idx][ind]+self.adf_dz[idx][ind] for ind in range(len(self.rdf_dz[idx]))]),
                 'central_atom_index': np.array(self.central_atom_index[idx]),
                 'neigh_atom_index': np.array(self.neigh_atom_index[idx])}
        #print('hey')
        #print([ self.rdf[idx][ind]+self.adf[idx][ind] for ind in range(len(self.rdf[idx]))])
        #print('hey')
        #print(self.adf[idx])
        #print(self.energies[idx])

        return sample
