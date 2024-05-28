#%%writefile structdata.py

import torch
from torch.utils.data import Dataset, DataLoader
import generatecoeffs as gc
import pickle
import numpy as np


class Structures(Dataset):
    #@profile
    def __init__(self, data,num_rdf,num_adf):
        """
        Save all the data
        """

        self.num_rdf = num_rdf
        self.num_adf = num_adf

        #data is the flatten data! can use the flatten_categorized_rdf_adf outside of this class to flatten the categorized data!
        self.central_atom_species_coeff,self.central_atom_index,self.neigh_atom_index,self.energies,self.forces_x,self.forces_y,self.forces_z,\
        self.rdf,self.rdf_dx,self.rdf_dy,self.rdf_dz,self.adf,self.adf_dx,self.adf_dy,self.adf_dz = data
        #data is the flatten data! can use the flatten_categorized_rdf_adf outside of this class to generate this


    def flatten_categorized_rdf_adf(categorized_rdf_adf,target_energies,target_forces):
        #helper function to flatten the categorized_rdf_adf data
        central_atom_species_coeff = []
        central_atom_index = []
        neigh_atom_index = []
        energies = []
        forces_x = []
        forces_y = []
        forces_z = []
        rdf = []
        rdf_dx = []
        rdf_dy = []
        rdf_dz = []
        adf = []
        adf_dx = []
        adf_dy = []
        adf_dz = []

        for ind in range(len(target_energies)):
            central_atom_species_coeff += [struct for struct in categorized_rdf_adf[ind]['central_atom_species_coeff']]
            central_atom_index += [struct for struct in categorized_rdf_adf[ind]['central_atom_index']]
            neigh_atom_index += [struct for struct in categorized_rdf_adf[ind]['neigh_atom_index']]

            energies += target_energies[ind]
            #print('self.energies')
            #print(self.energies)
            forces_x += [ [atom_forces[0] for atom_forces in struct] for struct in target_forces[ind]]
            #print('self.forces_x')
            #print(self.forces_x)
            forces_y += [ [atom_forces[1] for atom_forces in struct] for struct in target_forces[ind]]
            forces_z += [ [atom_forces[2] for atom_forces in struct] for struct in target_forces[ind]]
            rdf += [struct for struct in categorized_rdf_adf[ind]['rdf']]
            #print('self.rdf before')
            #print(categorized_rdf_adf[ind]['rdf'])
            #print('self.rdf after')
            #print(self.rdf)
            rdf_dx += [struct for struct in categorized_rdf_adf[ind]['rdf_dx']]
            #print('self.rdf_dx')
            #print(self.rdf_dx)
            rdf_dy += [struct for struct in categorized_rdf_adf[ind]['rdf_dy']]
            rdf_dz += [struct for struct in categorized_rdf_adf[ind]['rdf_dz']]
            adf += [struct for struct in categorized_rdf_adf[ind]['adf']]
            #print('self.adf')
            #print(self.adf)
            adf_dx += [struct for struct in categorized_rdf_adf[ind]['adf_dx']]
            adf_dy += [struct for struct in categorized_rdf_adf[ind]['adf_dy']]
            adf_dz += [struct for struct in categorized_rdf_adf[ind]['adf_dz']]
        #print(self.rdf)

        #print('central_atom_species_coeff',central_atom_species_coeff,flush=True)
        return central_atom_species_coeff,central_atom_index,neigh_atom_index,energies,forces_x,forces_y,forces_z,\
        rdf,rdf_dx,rdf_dy,rdf_dz,adf,adf_dx,adf_dy,adf_dz



    def normalize_all(self,energies_mean=None,energies_std=None,rdf_means=None,rdf_stds=None,adf_means=None,adf_stds=None):

        if (energies_mean!=None) and (energies_std!=None):
            self.energies_mean = energies_mean
            self.energies_std = energies_std

        else:
            self.energies_mean = np.mean(self.energies)
            self.energies_std = np.std(self.energies)

            #prevent division by zero
            if self.energies_std == 0:
                self.energies_std=1

        self.energies = [(i-self.energies_mean)/self.energies_std for i in self.energies]


        self.forces_x = [ [(atom_force)/self.energies_std for atom_force in struct] for struct in self.forces_x]
        self.forces_y = [ [(atom_force)/self.energies_std for atom_force in struct] for struct in self.forces_y]
        self.forces_z = [ [(atom_force)/self.energies_std for atom_force in struct] for struct in self.forces_z]

        self.rdf_means,self.rdf_stds,self.rdf,self.rdf_dx,self.rdf_dy,self.rdf_dz =\
        self.normalize_coeffsNderivatives(self.rdf, self.rdf_dx, self.rdf_dy, self.rdf_dz, self.num_rdf,rdf_means,rdf_stds)

        self.adf_means,self.adf_stds,self.adf,self.adf_dx,self.adf_dy,self.adf_dz =\
        self.normalize_coeffsNderivatives(self.adf, self.adf_dx, self.adf_dy, self.adf_dz, self.num_adf,adf_means,adf_stds)

        self.normalization_data = [self.energies_mean]+[self.energies_std]+\
                                  self.rdf_means+self.rdf_stds+\
                                  self.adf_means+self.adf_stds


    def normalize_forces(self, forces_x, forces_y, forces_z):
        flat = [ atom_force for struct in forces_x for atom_force in struct ]+\
        [ atom_force for struct in forces_y for atom_force in struct ]+\
        [ atom_force for struct in forces_z for atom_force in struct ]

        forces_mean = np.mean(flat)
        forces_std = np.std(flat)

        #dealing with std==0 cases
        if forces_std == 0:
            forces_std =1
        normalized_forces_x = [ np.array([(atom_force-forces_mean)/forces_std for atom_force in struct]) for struct in forces_x]
        normalized_forces_y = [ np.array([(atom_force-forces_mean)/forces_std for atom_force in struct]) for struct in forces_y]
        normalized_forces_z = [ np.array([(atom_force-forces_mean)/forces_std for atom_force in struct]) for struct in forces_z]

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

    def normalize_coeffsNderivatives(self, coeffs, coeffs_x, coeffs_y, coeffs_z, num_coeffs,coeffs_means=None,coeffs_stds=None):

        if (coeffs_means!=None)and(coeffs_stds!=None):
            pass #because we would have the info in the right variable already lol
        else:
            coeffs_means = []
            coeffs_stds = []

            #calculate means and stds
            for i in range(num_coeffs):
                #coeffs of a particular i-th kernel
                flat_coeff = [ atom_coeffs[i] for struct in coeffs for atom_coeffs in struct]
                coeffs_means += [np.mean(flat_coeff)]
                std = np.std(flat_coeff)
                #dealing with std==0 cases
                if std == 0:
                    coeffs_stds += [1]
                else:
                    coeffs_stds += [std]

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


        return coeffs_means, coeffs_stds, norm_coeffs, norm_coeffs_x, norm_coeffs_y, norm_coeffs_z

    def formatOutput(self):
        self.energies = [np.array([self.energies[i]]) for i in range(len(self.energies))]
        self.forces = [np.array([self.forces_x[i],self.forces_y[i],self.forces_z[i]]) for i in range(len(self.energies))]
        del self.forces_x,self.forces_y,self.forces_z
        self.coeffs = [np.array([self.rdf[i][ind]+self.adf[i][ind]+list(self.central_atom_species_coeff[i][ind]) for ind in range(len(self.rdf[i]))]) for i in range(len(self.energies))]
        del self.rdf,self.adf,self.central_atom_species_coeff
        self.coeffs_derivs = [np.array([ [self.rdf_dx[i][ind]+self.adf_dx[i][ind] for ind in range(len(self.rdf_dx[i]))],
                                         [self.rdf_dy[i][ind]+self.adf_dy[i][ind] for ind in range(len(self.rdf_dy[i]))],
                                         [self.rdf_dz[i][ind]+self.adf_dz[i][ind] for ind in range(len(self.rdf_dz[i]))]
                                       ]) for i in range(len(self.energies))]
        del self.rdf_dx,self.adf_dx,self.rdf_dy,self.adf_dy,self.rdf_dz,self.adf_dz
        self.central_atom_index= [ np.array(self.central_atom_index[i])for i in range(len(self.energies))]
        self.neigh_atom_index = [ np.array(self.neigh_atom_index[i]) for i in range(len(self.energies))]




    def __len__(self):
        return len(self.energies)

    #@profile
    def __getitem__(self, idx):
#         sample = {'energy_pa': np.array([self.energies[idx]]),
#                   'force_x': np.array(self.forces_x[idx]),
#                   'force_y': np.array(self.forces_y[idx]),
#                   'force_z': np.array(self.forces_z[idx]),
#                   'coeffs': np.array([ self.rdf[idx][ind]+self.adf[idx][ind] for ind in range(len(self.rdf[idx]))]),
#                  'coeffs_x': np.array([ self.rdf_dx[idx][ind]+self.adf_dx[idx][ind] for ind in range(len(self.rdf_dx[idx]))]),
#                  'coeffs_y': np.array([ self.rdf_dy[idx][ind]+self.adf_dy[idx][ind] for ind in range(len(self.rdf_dy[idx]))]),
#                  'coeffs_z': np.array([ self.rdf_dz[idx][ind]+self.adf_dz[idx][ind] for ind in range(len(self.rdf_dz[idx]))]),
#                  'central_atom_index': np.array(self.central_atom_index[idx]),
#                  'neigh_atom_index': np.array(self.neigh_atom_index[idx])}
        sample = {'energy_pa': self.energies[idx],
                  'forces': self.forces[idx],
                  'coeffs': self.coeffs[idx],
                 'coeffs_derivs': self.coeffs_derivs[idx],
                 'central_atom_index': self.central_atom_index[idx],
                 'neigh_atom_index': self.neigh_atom_index[idx]}
        return sample
