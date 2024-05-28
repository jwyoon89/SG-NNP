import pickle
import itertools
import numpy as np
import os
import torch
from torch.utils.data import random_split, ConcatDataset
from torch.autograd import Variable
import structdata
import random
import math
import gc as garbage
import generatecoeffs as gc
import multiprocessing
import pymatgen as pmg
import blackbox_latin as bb
import textwrap
import glob
from mpi4py import MPI
from dask.distributed import wait

import time

#Root of training files... see train() function for the exact files being read
train_root = '/processed/Ni-Mo/test'
train_data_lower_category = 0 #use None or 0 to include the first element of the list
train_data_upper_category = None #use only None to include the last element of the list

#Root of training files... see train() function for the exact files being read
test_root = '/processed/Ni-Mo/train'
test_data_lower_category = 0 #use None or 0 to include the first element of the list
test_data_upper_category = None #use only None to include the last element of the list

#elements (could be specie in the future with more information beyond atomic #)
elements_scale_factors = [('Mo',1.0),('Ni',3.5)] #scale the length of rdf and adf channels by specific constituent elements. can use stoichiometric ratios.
converted_elems_scale_factor = {pmg.Element(i[0]).Z:i[1] for i in elements_scale_factors}
adf_gap=0.05 #set this gap in the channels in ADF space relative to the values in "elements_scale_factor"

pmg_elements = [pmg.Element(i[0]).Z for i in elements_scale_factors]
pmg_elements.sort()

#generate pair indices
distinct_singlet_index = {ind:elem for ind,elem in enumerate(pmg_elements)}
distinct_pairs_index = {ind:pair for ind,pair in enumerate(itertools.combinations_with_replacement(pmg_elements,2)) }

print('distinct_singlet_index: '+str(distinct_singlet_index),flush=True)
print('distinct_pairs_index: '+str(distinct_pairs_index),flush=True)

#descriptor parameters
descriptor_mode = 'separate' #separate (i.e. define num_rdf, num_adf) or 'fixed' (i.e. define num_rdf and num_adf will be defined)
num_rdf = 7 #works in 'separate' and 'fixed' descriptor_mode
num_adf = 7 #only works in 'separate' descriptor_mode
rdf_mean_distribution = 'uniform' #residual eating,uniform,none
rdf_width_distribution = 'adaptive'#adaptive, none
adf_mean_distribution = 'uniform' #residual eating,uniform,none
adf_width_distribution = 'adaptive'#adaptive, none
range_rdf_means = [0, 1.]
range_rdf_widths = [0.8, num_rdf-1] #if rdf_width_dist = none this is in units of the whole rdf, if 'adaptive' this is in units of inter-gauss spacing. if spacing = 0 it is the previous spacing that is non-zero
range_adf_means = [0, 1.]
range_adf_widths = [0.8, num_adf-1] #if adf_width_dist = none this is in units of the whole adf, if 'adaptive' this is in units of inter-gauss spacing. if spacing = 0 it is the previous spacing that is non-zero
range_rdf_cutoff= [4.6,4.6]
range_adf_cutoff= [4.6,4.6]

#blackbox optimization parameters
n=20
m=10
batch=10
bb_num_workers = batch #total number of workers

#Neural Network Hyperparameters
NN_type = 'non-linear' # options: 'linear', 'slightly non-linear', 'non-linear'
nn_mode = 'N=total_num_gaussians' #options: 'fixed_N', 'N=total_num_gaussians'
N=14 #number of nodes in each intermediate layer in non-linear type. Only works for 'fixed_N' nn_mode
L=1 #number of intermediate layers in non-linear type

if NN_type == 'linear':
    import linear_network_pytorch as mynetwork
elif NN_type == 'slightly non-linear':
    import slightly_nonlinear_network_pytorch as mynetwork
elif NN_type == 'non-linear':
    import nonlinear_network_pytorch as mynetwork

torch_num_threads = 24 #for torch multithreading training
dl_num_workers = 0 #for dataloader num of subproccesses to load. should keep this at 0 as it slows down too much when > 0.


#run mode
run_mode = 'Hyperparameter Tuning' #'Hyperparameter Tuning' or 'Generate Model'
save_model_interval = 200
num_folds = 2 #k-fold cross validation
gen_parameters = [+8.00000000000e-01, +8.00000000000e-01, +6.00000000000e+00, +8.00000000000e-01, +8.00000000000e-01, +4.92945742300e+00, +6.00000000000e+00, +8.00000000000e-01, +6.00000000000e+00, +8.00000000000e-01, +6.00000000000e+00, +6.00000000000e+00, +8.00000000000e-01, +9.77894904017e-01, +3.90000000000e+00, -1.00000000000e+07, +0.00000000000e+00] #only works in 'Generate Model' run_mode. Copy from output of 'Hyperparameter Tuning' runs!!
range_regularizer_lambda_power = [-10000000,-10000000] #power of base 10
range_force_beta = [0,2.0] #power of base 10

#saving processed train data and test data
save_data_mode = 'off' #'on' or 'off' (only works when you're not using dask workers because they cant write data!)

#fitting run parameter
optimizer = 'LBFGS' #LBFGS or RMSprop
num_epochs = 50

#RMSprop optimizer params (ignore for others)
base_learning_rate = 0.01
momentum = 0.01 #
training_batch_size = 3 #only for nonLBGFS methods, BFGS usually all data for each iteration (i.e. full-batch)

#LBFGS optimizer params
lbfgs_lr = 0.01
lbfgs_max_iter=20
lbfgs_max_eval=lbfgs_max_iter * 1.25
#lbfgs_tolerance_grad=
#lbfgs_tolerance_change=
lbfgs_history_size=100
lbfgs_line_search_fn='strong_wolfe' #'strong_wolfe' or None

#decay scheduler params
sched = 'CyclicLR' # 'StepLR' or 'CyclicLR'
scheduler_iteration = 1 #how many iterations (i.e. mini-batches) before scheduler is advanced by one step

##Params for StepLR:
#step_size = 50
#decay_factor = 0.5

##Params for CyclicLR:
base_lr = 0.0001 #this has been determined in the simulations with stepLR. Has to reach this level to converge!
max_lr = base_learning_rate #this is the maximum, which was our original definition of 'base' in stepLR
step_size_up=1 # num of mini-batches to go up max. it will start from minimum then go up.
step_size_down=2700 # num of mini-batches/iteration to go down to min
cyclic_mode='triangular2'
#gamma=1.0

#Parameters for parallel generation of coefficients
dist_input_chunksize=100
angle_input_chunksize=100
gc_num_workers = torch_num_threads #same level of parallelization with pytorch training runs

#Parameters for dask cluster
cluster = 'slurm_dask_mpi_savio' #choices: savio, savio_bigmem, savio2,savio2_adaptive, savio2_knl, savio2_bigmem, cori_haswell, cori_knl, slurm_dask_mpi_savio, slurm_dask_mpi_stampede, slurm_dask_mpi_stampede_test, local
qos = ['--qos="savio_normal"'] #savio: savio_normal, savio_debug; cori: regular, premium, debug
processes=1 #per node
cores=processes
job_cpu = 24 #number of cpus to state in the jobscript. Gotta set for savio2 because there are nodes with 24 and 28 cores.
walltime="20:00:00"

#------

def generate_scale_shift_factors(elems_scale_factor,adf_gap,singlet_index,pairs_index,rdf_cutoff):
    """
    This functions takes in :
    (1) elems_scale_factor:  a dictionary that maps from atomic number to the scale factor associated with the particular atomic number
    (2) adf_gap:             a constant to spread out the channels in the ADF to avoid informational degeneracy
    (3) singlet_index:       a dict that maps from the rank position of an element in the RDF space to its atomic number
    (4) pairs_index:         a dict that maps from the rank position of a pair of elements in the ADF space to their sorted atomic numbers stored in a tuple

    And returns:
    (1) rdf_transformation_factors: a dict that maps from the atomic number to a tuple that contains (rdf shift factor, rdf scale factor)
                                    for an element with that particular atomic number
    (2) adf_transformation_factors: a dict that maps from the sorted tuple of pair atomic numbers to a tuple that contains (adf shift factor, adf scale factor)
                                    for a pair of elements with the particular sorted atomic numbers
    """

    sum_rdf_factors = sum([ elems_scale_factor[singlet_index[i]] for i in singlet_index.keys()])
    rdf_cumu_shift_factor = 0
    rdf_transformation_factors = dict()
    for i in range(max(singlet_index.keys())+1):
        rdf_scale_factor = elems_scale_factor[singlet_index[i]]/sum_rdf_factors
        rdf_transformation_factors[singlet_index[i]] = (rdf_cumu_shift_factor,rdf_scale_factor)
        rdf_cumu_shift_factor += rdf_scale_factor*rdf_cutoff


    sum_adf_factors = sum([ elems_scale_factor[pairs_index[i][0]]*elems_scale_factor[pairs_index[i][1]]+adf_gap for i in pairs_index.keys()])
    adf_cumu_shift_factor = 0
    adf_transformation_factors = dict()
    for i in range(max(pairs_index.keys())+1):
        adf_scale_factor = (elems_scale_factor[pairs_index[i][0]]*elems_scale_factor[pairs_index[i][1]])/sum_adf_factors
        adf_transformation_factors[pairs_index[i]] = (adf_cumu_shift_factor,adf_scale_factor)
        adf_cumu_shift_factor += adf_scale_factor + adf_gap/sum_adf_factors

    return rdf_transformation_factors,adf_transformation_factors


#@profile
def prepareData(par,data):
    train_categorized_num_atoms,train_target_energies,train_target_forces,train_structs,test_categorized_num_atoms,test_target_energies,test_target_forces,test_structs = data

    #generate means of adf and rdf from par
    #----

    if rdf_mean_distribution == 'residual eating':
        rdf_means =[]
        for ind,i in enumerate(par[0:int(num_rdf)]):
            if ind ==0:
                rdf_means.append(i)
            elif ind == (len(par[0:int(num_rdf)])-1): #last gaussian has to be at the end of the interval
                rdf_means.append(1)
            else:
                rdf_means.append(i*(1-rdf_means[-1])+rdf_means[-1])
        rdf_means = np.array(rdf_means)
    else:
        rdf_means = par[0:int(num_rdf)]

    if adf_mean_distribution == 'residual eating':
        adf_means =[]
        for ind,i in enumerate(par[2*int(num_rdf):2*int(num_rdf)+int(num_adf)]):
            if ind ==0:
                adf_means.append(i)
            elif ind == (len(par[2*int(num_rdf):2*int(num_rdf)+int(num_adf)])-1): #last gaussian has to be at the end of the interval
                adf_means.append(1)
            else:
                adf_means.append(i*(1-adf_means[-1])+adf_means[-1])
        adf_means = np.array(adf_means)
    else:
        adf_means = par[2*int(num_rdf):2*int(num_rdf)+int(num_adf)]
    #----
    #generate widths of adf and rdf from par, which is a "adaptive" representation (adapted to the inter-gaussian spacing)
    #----

    if rdf_width_distribution == 'adaptive':
        rdf_widths =[]
        for ind,i in enumerate(par[int(num_rdf):2*int(num_rdf)]):
            if ind ==0:
                temp_spacing = rdf_means[ind]
                if temp_spacing != 0:
                    save_spacing = temp_spacing
                elif rdf_mean_distribution == 'uniform':
                    save_spacing = rdf_means[1] - rdf_means[0]
                else:
                    save_spacing = 1

                rdf_widths.append(i*save_spacing)
            else:
                temp_spacing = rdf_means[ind]-rdf_means[ind-1]
                if temp_spacing != 0:
                    save_spacing = temp_spacing
                rdf_widths.append(i*save_spacing)


        rdf_widths = np.array(rdf_widths)
    else:
        rdf_widths = par[int(num_rdf):2*int(num_rdf)]

    if adf_width_distribution == 'adaptive':
        adf_widths =[]
        for ind,i in enumerate(par[2*int(num_rdf)+int(num_adf):2*int(num_rdf)+2*int(num_adf)]):
            if ind ==0:
                temp_spacing = adf_means[ind]
                if temp_spacing != 0:
                    save_spacing = temp_spacing

                elif adf_mean_distribution == 'uniform':
                    save_spacing = adf_means[1] - adf_means[0]

                else:
                    save_spacing = 1

                adf_widths.append(i*save_spacing)
            else:
                temp_spacing = adf_means[ind]-adf_means[ind-1]
                if temp_spacing != 0:
                    save_spacing = temp_spacing
                adf_widths.append(i*save_spacing)


        adf_widths = np.array(adf_widths)
    else:
        adf_widths = par[2*int(num_rdf)+int(num_adf):2*int(num_rdf)+2*int(num_adf)]


    if nn_mode == 'N=total_num_gaussians':
        global N
        N = num_rdf+num_adf

    rdf_cutoff = par[2*int(num_rdf)+2*int(num_adf)]

    adf_cutoff = par[2*int(num_rdf)+2*int(num_adf)+1]

    regularizer_lambda = 10**(par[2*int(num_rdf)+2*int(num_adf)+2]) #powers of 10

    force_beta = 10**(par[2*int(num_rdf)+2*int(num_adf)+3]) #powers of 10

    #generate rdf,adf,coeffs

    distinct_singlet_index_inverted = {distinct_singlet_index[ind]:ind for ind in distinct_singlet_index}

    rdf_transformation_factors,adf_transformation_factors=generate_scale_shift_factors(converted_elems_scale_factor,
                                                                                        adf_gap,
                                                                                        distinct_singlet_index,
                                                                                        distinct_pairs_index,
                                                                                        rdf_cutoff)
    print('rdf_tranformation_factor: ',rdf_transformation_factors,flush=True)
    print('adf_tranformation_factor: ',adf_transformation_factors,flush=True)

    print('Generating RDF and ADF coeffs for train nowwww!!! Pay attention to cpu usage :) ',flush=True)

    program_starts = time.time()

    if run_mode == 'Generate Model':
        train_categorized_rdf_adf,test_categorized_rdf_adf = gc.generateRDFADFcoeffs(train_structs,test_structs,
                                           rdf_cutoff,
                                           adf_cutoff,
                                           train_categorized_num_atoms,test_categorized_num_atoms,
                                           rdf_means,
                                           rdf_widths,
                                           adf_means,
                                           adf_widths,
                                           distinct_singlet_index_inverted,
                                           rdf_transformation_factors,
                                           adf_transformation_factors,
                                           dist_input_chunksize,
                                           angle_input_chunksize,
                                           gc_num_workers)
    else:
        train_categorized_rdf_adf,test_categorized_rdf_adf = gc.generateRDFADFcoeffs(train_structs,None,
                                       rdf_cutoff,
                                       adf_cutoff,
                                       train_categorized_num_atoms,None,
                                       rdf_means,
                                       rdf_widths,
                                       adf_means,
                                       adf_widths,
                                       distinct_singlet_index_inverted,
                                       rdf_transformation_factors,
                                       adf_transformation_factors,
                                       dist_input_chunksize,
                                       angle_input_chunksize,
                                       gc_num_workers)

    now = time.time()
    print("It has been {0} seconds since the loop started".format(now - program_starts),flush=True)
    print('Done with calculating rdf,adf coefficients!',flush=True)

    return train_categorized_rdf_adf,test_categorized_rdf_adf,regularizer_lambda,force_beta

# --------

def train_nonLBFGS(train_categorized_rdf_adf):
    torch.set_num_threads(torch_num_threads)
    current_process = multiprocessing.current_process()


    #flatten and normalize all data
    data =\
    structdata.Structures.flatten_categorized_rdf_adf(train_categorized_rdf_adf,train_target_energies,train_target_forces)
    del train_categorized_rdf_adf
    data = structdata.Structures(data,num_rdf,num_adf)
    data.normalize_all()

    if num_epochs>=save_model_interval: #save normalization data only if we'll need it for testing later
        #np.save('myNet'+str(current_process._identity[0])+'_normalization_data', np.array(data.normalization_data))
        print('\nmyNet'+str(current_process._identity[0])+'_normalization_data:\n',data.normalization_data,'\n',flush=True)

    #initializing neural net

    if NN_type == 'linear' or NN_type == 'slightly non-linear':
        net = mynetwork.Net(num_rdf+num_adf+len(distinct_singlet_index)).double()
    elif NN_type == 'non-linear':
        net = mynetwork.Net(num_rdf+num_adf+len(distinct_singlet_index),N,L).double() #+len(distinct_singlet_index) because of one-hot encoding of central species


    optimizer = torch.optim.RMSprop(net.parameters(), lr=base_learning_rate, momentum=momentum) #this works only if optimizer isn't 'LBFGS'

    if sched == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay_factor)
    elif sched == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up,
                         step_size_down=step_size_down,mode=cyclic_mode)

    mae_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()

    dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=dl_num_workers)#, collate_fn=my_collate)
    #iteration counts
    iteration = 0
    #ae_force_batch = []
    #ae_energy_batch = []
    se_force_batch = []
    se_energy_batch = []
    num_energies_batch = 0
    num_particles_batch = 0

    for epoch in range(num_epochs):

        garbage.collect()

        #statistics to save for each epoch
        ae_energy_save = 0
        ae_force_save = 0
        se_energy_save = 0
        se_force_save = 0
        ae_energy = 0
        ae_force_x = 0
        ae_force_y = 0
        ae_force_z = 0
        ae_force_mag_old = 0
        ae_force_mag_new = 0
        num_energies = 0
        num_particles = 0

        for i_batch, sample_batched in enumerate(dataloader):
            #iteration counter
            iteration += 1

            central_atom_index = sample_batched['central_atom_index'].tolist()[0]
            neigh_atom_index = sample_batched['neigh_atom_index'].tolist()[0]

            target_e_pa = Variable(sample_batched['energy_pa'])
            target_fx = Variable(sample_batched['force_x'])
            target_fy = Variable(sample_batched['force_y'])
            target_fz = Variable(sample_batched['force_z'])

            target_f = torch.stack([target_fx, target_fy, target_fz], dim=1)

            input_coeffs = Variable(sample_batched['coeffs'])
            input_coeff_x = Variable(sample_batched['coeffs_x'])
            input_coeff_y = Variable(sample_batched['coeffs_y'])
            input_coeff_z = Variable(sample_batched['coeffs_z'])


            num_energies += target_e_pa.shape[1] #first dim is batch
            num_particles += target_fx.shape[1]

            num_energies_batch += target_e_pa.shape[1] #first dim is batch
            num_particles_batch += target_fx.shape[1]

            e_pa,f = net(input_coeffs,
                                input_coeff_x,
                                input_coeff_y,
                                input_coeff_z,
                                central_atom_index,
                                neigh_atom_index)


            #save iteration loss

            se_energy_iteration = mse_loss(e_pa, target_e_pa)*target_e_pa.shape[1]/(0.009**2)
            se_force_iteration = mse_loss(f, target_f)*target_fx.shape[1]/(0.3**2)

            se_energy_save += se_energy_iteration.data
            se_force_save += se_force_iteration.data

            se_energy_batch += [se_energy_iteration]
            se_force_batch += [se_force_iteration]

            #propagate gradient for batch loss
            if iteration%training_batch_size ==0:
                loss = sum(se_energy_batch)/num_energies_batch+sum(se_force_batch)/num_particles_batch

                # zeroes the gradient buffers of all parameters
                net.zero_grad()
                #propagrate gradients back
                loss.backward()
                #update weights (parameters) of the model
                optimizer.step()
                #reset batch loss
                num_energies_batch = 0
                num_particles_batch = 0
                se_energy_batch = []
                se_force_batch = []

                if int(iteration/training_batch_size) % scheduler_iteration == 0:
                    scheduler.step()

            #accumulate statistics of interest
            ae_energy += (torch.sum(torch.abs(e_pa-target_e_pa))).data* data.energies_std
            ae_forces = torch.sum(torch.abs(f-target_f),2)* data.energies_std

            ae_force_x += ae_forces[0][0].data
            ae_force_y += ae_forces[0][1].data
            ae_force_z += ae_forces[0][2].data

            fxyz_sq_summed_sqrt = torch.sum(f**2,1).sqrt()
            target_fxyz_sq_summed_sqrt = torch.sum(target_f**2,1).sqrt()

            ae_force_mag_new += torch.sum(torch.sum((f-target_f)**2,1).sqrt()).data* data.energies_std


        loss_save = se_energy_save/num_energies + se_force_save/num_particles

        print('\nworker identity:', current_process._identity,
              '\n'+str(epoch)+'-th epoch \nRMSE loss: '+str((loss_save.item())**0.5),
              '\nnum_energies: '+str(num_energies),
              '\nnum_atoms: '+str(num_particles),
              '\nmae_energy: '+str(ae_energy.item()/num_energies),
              '\nmae_force_x: '+str(ae_force_x.item()/num_particles),
              '\nmae_force_y: '+str(ae_force_y.item()/num_particles),
              '\nmae_force_z: '+str(ae_force_z.item()/num_particles),
              #'\nmae_force_mag_old: '+str(ae_force_mag_old.item()/num_particles),
              '\nmae_force_mag_new: '+str(ae_force_mag_new.item()/num_particles),
              '\n******\n',flush=True)
        #save model along the way
        if (epoch+1)%save_model_interval==0:
            #torch.save(net.state_dict(), 'myNet'+str(current_process._identity[0])+'.pt')
            print('\nmyNet'+str(current_process._identity[0])+':\n',net.state_dict(),'\n',flush=True)

    return loss_save

def train_LBFGS(train_categorized_rdf_adf,test_categorized_rdf_adf,data,regularizer_lambda,force_beta):
    train_categorized_num_atoms,train_target_energies,train_target_forces,train_structs,test_categorized_num_atoms,test_target_energies,test_target_forces,test_structs = data

    torch.set_num_threads(torch_num_threads)
    if cluster[0:14] == 'slurm_dask_mpi':
        worker_identity = MPI.COMM_WORLD.rank
    else:
        worker_identity = multiprocessing.current_process()._identity[0]

    mae_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()

    #flatten and normalize all data
    data =\
    structdata.Structures.flatten_categorized_rdf_adf(train_categorized_rdf_adf,train_target_energies,train_target_forces)
    del train_categorized_rdf_adf

    if run_mode == 'Generate Model':
        data_test =\
        structdata.Structures.flatten_categorized_rdf_adf(test_categorized_rdf_adf,test_target_energies,test_target_forces)
        del test_categorized_rdf_adf

    #saving data (only works when you're not using dask workers because they cant write data!)

    if save_data_mode == 'on':
        with open('train_data.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if run_mode == 'Generate Model':
            with open('test_data.pickle', 'wb') as handle:
                pickle.dump(data_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #generate folds labels
    fold_labels = list(range(num_folds))*math.ceil(len(data[2])/num_folds)
    fold_labels = fold_labels[:len(data[2])]
    random.shuffle(fold_labels) #in-place shuffling of labels

    #store val minimum of folds
    folds_val_min = []

    for fold_ind in range(num_folds):

        #prepare data
        if run_mode == 'Hyperparameter Tuning':

            data_train = structdata.Structures([[i for ind, i in enumerate(entry) if fold_labels[ind] != fold_ind]
                                                for entry in data],
                                               num_rdf,
                                               num_adf)

            data_val = structdata.Structures([[i for ind, i in enumerate(entry) if fold_labels[ind] == fold_ind]
                                                for entry in data],
                                               num_rdf,
                                               num_adf)
        elif run_mode == 'Generate Model':
            data_train = structdata.Structures(data,
                                               num_rdf,
                                               num_adf)

            data_val = structdata.Structures(data_test,
                                               num_rdf,
                                               num_adf)

        #normalize data

        data_train.normalize_all() #normalize train
        data_val.normalize_all(data_train.energies_mean,
                               data_train.energies_std,
                               data_train.rdf_means,
                               data_train.rdf_stds,
                               data_train.adf_means,
                               data_train.adf_stds) #normalize val with train normalization values

        #format data

        data_train.formatOutput()
        data_val.formatOutput()

        dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=False, num_workers=dl_num_workers)#, collate_fn=my_collate)
        dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=1, shuffle=False, num_workers=dl_num_workers)#, collate_fn=my_collate)


        if num_epochs>=save_model_interval: #save normalization data only if we'll need it for testing later
            #np.save('myNet'+str(current_process._identity[0])+'_normalization_data', np.array(data.normalization_data))

            print('\n***Normalization Data***',
                  '\nWorker identity:', worker_identity,
                  '\nFold: '+str(fold_ind+1)+' / '+str(num_folds),
                  '\nNormalization_data:\n',
                  data_train.normalization_data,'\n',flush=True)

            if run_mode == 'Generate Model':
                np.save('normalization_data', np.array(data_train.normalization_data))


        #initializing neural net
        if NN_type == 'linear' or NN_type == 'slightly non-linear':
            net = mynetwork.Net(num_rdf+num_adf+len(distinct_singlet_index)).double()
        elif NN_type == 'non-linear':
            net = mynetwork.Net(num_rdf+num_adf+len(distinct_singlet_index),N,L).double()


        optimizer = torch.optim.LBFGS(net.parameters(),lr=lbfgs_lr,max_iter=lbfgs_max_iter,max_eval=lbfgs_max_eval,
                            history_size=lbfgs_history_size,line_search_fn=lbfgs_line_search_fn)

        #get stats:

        #training stats:
        num_energies_train = 0
        num_particles_train = 0
        for i_batch, sample_batched in enumerate(dataloader_train):
            target_e_pa = Variable(sample_batched['energy_pa'])
            target_fx = Variable(sample_batched['forces'])
            num_energies_train += target_e_pa.shape[1] #first dim is batch
            num_particles_train += target_fx.shape[2] #first dim is batch, second is force directions

        #validation stats:
        num_energies_val = 0
        num_particles_val = 0
        for i_batch, sample_batched in enumerate(dataloader_val):
            target_e_pa = Variable(sample_batched['energy_pa'])
            target_fx = Variable(sample_batched['forces'])
            num_energies_val += target_e_pa.shape[1] #first dim is batch
            num_particles_val += target_fx.shape[2] #first dim is batch, second is force directions

        #this list is to record the list of validation losses and then we will use this at the end of the training run
        #to pick the best (min?) RMSE to report back to blackbox optimization algo
        loss_list_val = []

        #this is to save the best model. None is used as the initializer first element
        saved_model = None

        for epoch in range(num_epochs):

            garbage.collect()

            def closure():
                se_force = []
                se_energy = []
                optimizer.zero_grad()
                for i_batch, sample_batched in enumerate(dataloader_train):
                    #iteration counter
                    #iteration += 1

                    central_atom_index = sample_batched['central_atom_index'].tolist()[0]
                    neigh_atom_index = sample_batched['neigh_atom_index'].tolist()[0]

                    target_e_pa = Variable(sample_batched['energy_pa'])
                    target_f = Variable(sample_batched['forces'])
                    input_coeffs = sample_batched['coeffs']
                    input_coeff_derivs = sample_batched['coeffs_derivs']

                    e_pa,f = net(input_coeffs,
                                        input_coeff_derivs,
                                        central_atom_index,
                                        neigh_atom_index)

                    #save iteration loss

                    se_energy_iteration = mse_loss(e_pa, target_e_pa)*target_e_pa.shape[1]* data_train.energies_std
                    se_force_iteration = mse_loss(f, target_f)*target_f.shape[2]* data_train.energies_std

                    se_energy += [se_energy_iteration]
                    se_force += [se_force_iteration]

                regularizer = 0
                for p in net.parameters():
                    regularizer += p.norm(2)**2

                energy_loss = ((sum(se_energy)/num_energies_train)**0.5)/0.009
                force_loss = ((sum(se_force)/num_particles_train)**0.5)/0.3
                loss_prev = energy_loss+force_loss
                loss = energy_loss + force_beta*force_loss + regularizer_lambda*regularizer
                loss.backward()

                return loss

            optimizer.step(closure)

            #check training performance:

            ae_energy = 0
            ae_force_x = 0
            ae_force_y = 0
            ae_force_z = 0
            ae_force_mag_old = 0
            ae_force_mag_new = 0

            se_energy_save = 0
            se_force_save = 0


            with torch.no_grad():

                for i_batch, sample_batched in enumerate(dataloader_train):
                    #iteration counter
                    #iteration += 1

                    central_atom_index = sample_batched['central_atom_index'].tolist()[0]
                    neigh_atom_index = sample_batched['neigh_atom_index'].tolist()[0]

                    target_e_pa = Variable(sample_batched['energy_pa'])

                    target_f = Variable(sample_batched['forces'])

                    input_coeffs = sample_batched['coeffs']
                    input_coeff_derivs = sample_batched['coeffs_derivs']

                    e_pa,f = net(input_coeffs,
                                        input_coeff_derivs,
                                        central_atom_index,
                                        neigh_atom_index)

                    #save iteration loss

                    se_energy_iteration = mse_loss(e_pa, target_e_pa)*target_e_pa.shape[1]* data_train.energies_std
                    se_force_iteration = mse_loss(f, target_f)*target_f.shape[2]* data_train.energies_std

                    se_energy_save += se_energy_iteration.data
                    se_force_save += se_force_iteration.data

                    #accumulate statistics of interest
                    ae_energy += (torch.sum(torch.abs(e_pa-target_e_pa))).data* data_train.energies_std
                    ae_forces = torch.sum(torch.abs(f-target_f),2)* data_train.energies_std

                    ae_force_x += ae_forces[0][0].data
                    ae_force_y += ae_forces[0][1].data
                    ae_force_z += ae_forces[0][2].data

                    fxyz_sq_summed_sqrt = torch.sum(f**2,1).sqrt()
                    target_fxyz_sq_summed_sqrt = torch.sum(target_f**2,1).sqrt()

                    ae_force_mag_new += torch.sum(torch.sum((f-target_f)**2,1).sqrt()).data* data_train.energies_std

            regularizer = 0
            for p in net.parameters():
                regularizer += p.norm(2)**2

            energy_loss = ((se_energy_save/num_energies_train)**0.5)/0.009
            force_loss = ((se_force_save/num_particles_train)**0.5)/0.3
            loss_save_prev = energy_loss + force_loss
            loss_save = energy_loss + force_beta*force_loss + regularizer_lambda*regularizer


            #check validation performance:

            #statistics to save for each epoch

            ae_energy_val = 0
            ae_force_x_val = 0
            ae_force_y_val = 0
            ae_force_z_val = 0
            ae_force_mag_new_val = 0

            se_energy_save_val = 0
            se_force_save_val = 0


            with torch.no_grad():

                for i_batch, sample_batched in enumerate(dataloader_val):
                    #iteration counter
                    #iteration += 1

                    central_atom_index = sample_batched['central_atom_index'].tolist()[0]
                    neigh_atom_index = sample_batched['neigh_atom_index'].tolist()[0]

                    target_e_pa = Variable(sample_batched['energy_pa'])
    
                    target_f = Variable(sample_batched['forces'])

                    input_coeffs = sample_batched['coeffs']
                    input_coeff_derivs = sample_batched['coeffs_derivs']

                    e_pa,f = net(input_coeffs,
                                        input_coeff_derivs,
                                        central_atom_index,
                                        neigh_atom_index)


                    #save iteration loss

                    se_energy_iteration = mse_loss(e_pa, target_e_pa)*target_e_pa.shape[1]* data_train.energies_std
                    se_force_iteration = mse_loss(f, target_f)*target_f.shape[2]* data_train.energies_std

                    se_energy_save_val += se_energy_iteration.data
                    se_force_save_val += se_force_iteration.data

                    #accumulate statistics of interest
                    ae_energy_val += (torch.sum(torch.abs(e_pa-target_e_pa))).data* data_train.energies_std
                    ae_forces = torch.sum(torch.abs(f-target_f),2)* data_train.energies_std

                    ae_force_x_val += ae_forces[0][0].data
                    ae_force_y_val += ae_forces[0][1].data
                    ae_force_z_val += ae_forces[0][2].data

                    fxyz_sq_summed_sqrt = torch.sum(f**2,1).sqrt()
                    target_fxyz_sq_summed_sqrt = torch.sum(target_f**2,1).sqrt()

                    ae_force_mag_new_val += torch.sum(torch.sum((f-target_f)**2,1).sqrt()).data* data_train.energies_std


            regularizer = 0
            for p in net.parameters():
                regularizer += p.norm(2)**2

            energy_loss = ((se_energy_save_val/num_energies_val)**0.5)/0.009
            force_loss = ((se_force_save_val/num_particles_val)**0.5)/0.3
            loss_save_val_prev = energy_loss+force_loss
            loss_save_val = energy_loss + force_beta*force_loss + regularizer_lambda*regularizer


            loss_list_val += [loss_save_val_prev] #using the val performance and not the cost because cost is different for different regularization parameter values

            print('\n***Training Run***',
                  '\nWorker identity:', worker_identity,
                  '\nFold: '+str(fold_ind+1)+' / '+str(num_folds),
                  '\nEpoch: '+str(epoch+1)+' / '+str(num_epochs),
                  '\n***Training Statistics***',
                  '\nRMSE Cost: '+str(loss_save.item()),
                  '\nRMSE Performance: '+str(loss_save_prev.item()),
                  '\nnum_energies: '+str(num_energies_train),
                  '\nnum_atoms: '+str(num_particles_train),
                  '\nmae_energy: '+str(ae_energy.item()/num_energies_train),
                  '\nmae_force_x: '+str(ae_force_x.item()/num_particles_train),
                  '\nmae_force_y: '+str(ae_force_y.item()/num_particles_train),
                  '\nmae_force_z: '+str(ae_force_z.item()/num_particles_train),
                  #'\nmae_force_mag_old: '+str(ae_force_mag_old.item()/num_particles),
                  '\nmae_force_mag: '+str(ae_force_mag_new.item()/num_particles_train),
                  '\n***Validation Statistics***',
                  '\nRMSE Cost: '+str(loss_save_val.item()),
                  '\nRMSE Performance: '+str(loss_save_val_prev.item()),
                  '\nnum_energies: '+str(num_energies_val),
                  '\nnum_atoms: '+str(num_particles_val),
                  '\nmae_energy: '+str(ae_energy_val.item()/num_energies_val),
                  '\nmae_force_x: '+str(ae_force_x_val.item()/num_particles_val),
                  '\nmae_force_y: '+str(ae_force_y_val.item()/num_particles_val),
                  '\nmae_force_z: '+str(ae_force_z_val.item()/num_particles_val),
                  #'\nmae_force_mag_old: '+str(ae_force_mag_old.item()/num_particles),
                  '\nmae_force_mag: '+str(ae_force_mag_new_val.item()/num_particles_val),
                  '\nmin Validation RMSE Performance so far: '+str((min(loss_list_val)).item()),
                  '\nEpoch at min Validation RMSE Performance: '+str(loss_list_val.index(min(loss_list_val))+1),
                  '\n******\n',flush=True)

            #save the minimum val loss model
            if (saved_model == None) or (loss_save_val <= loss_list_val[-1]):
                saved_model = net.state_dict()

            #save model along the way
            if (epoch+1)%save_model_interval==0:
                #torch.save(net.state_dict(), 'myNet'+str(current_process._identity[0])+'.pt')
                if (run_mode == 'Hyperparameter Tuning'):
                    print('\n***Print Neural Network Weights***',
                          '\nmyNet'+str(worker_identity)+':\n',saved_model,'\n',flush=True)

                if (run_mode == 'Generate Model'): #save the current model in 'Generate Model' run_mode
                    print('\n***Print Neural Network Weights***',
                          '\nmyNet:\n',saved_model,'\n',flush=True)
                    torch.save(saved_model,'nn_model_min_test_loss.pt')
                    torch.save(net.state_dict(),'nn_model_epoch'+str(epoch+1)+'.pt')

        folds_val_min += [(min(loss_list_val)).item()]


    mean_min_val = sum(folds_val_min)/len(folds_val_min) #return the average of minimums of validation loss across folds
    stdev_min_val = np.std(folds_val_min)

    #Print summary statistics

    if (run_mode == 'Hyperparameter Tuning'):

        print('\n***Summary Statistics***',
              '\nWorker: ',worker_identity,
              '\nmean_folds_val_min: ',mean_min_val,
              '\nstdev_folds_val_min: ',stdev_min_val,flush=True)

    return mean_min_val

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def nCr(n, r):

    return (fact(n) / (fact(r)
                * fact(n - r)))

# Returns factorial of n
def fact(n):

    res = 1

    for i in range(2, n+1):
        res = res * i

    return res


def main():

    if descriptor_mode == 'fixed':
            global num_adf
            num_adf = int(nCr(num_rdf,2))
            global range_adf_widths
            range_adf_widths = [0.8, num_adf-1]

    if run_mode == 'Hyperparameter Tuning':

        if rdf_mean_distribution == 'uniform':
            #just construct widths range here. means will be calculated later in the program since they will not be optimized.
            rdf_means_widths = [range_rdf_widths]*num_rdf
        else:
            rdf_means_widths = [range_rdf_means]*num_rdf+[range_rdf_widths]*num_rdf

        if adf_mean_distribution == 'uniform':
            #just construct widths range here. means will be calculated later in the program since they will not be optimized.
            adf_means_widths = [range_adf_widths]*num_adf
        else:
            adf_means_widths = [range_adf_means]*num_adf+[range_adf_widths]*num_adf


        print('Hyperparameter Tuning Mode!')

        ex = getExecutor()
        bb.search(f=train,  # given function
                          box=rdf_means_widths+adf_means_widths+[range_rdf_cutoff]+[range_adf_cutoff]+[range_regularizer_lambda_power]+[range_force_beta],  # range of values for each parameter (2D case)
                          n=n,  # number of function calls on initial stage (global search)
                          m=m,  # number of function calls on subsequent stage (local search)
                          batch=batch,  # number of calls that will be evaluated in parallel
                          resfile='output_hyperparameter_tuning.csv',
                          num_workers = bb_num_workers,
                          executor=ex)  # text file where results will be saved
                          #nrand_frac=0.002,

    elif run_mode == 'Generate Model':
        global num_folds
        num_folds = 1

        print('Generate Model Mode!')
        train(gen_parameters)

def load_data():

    with open(train_root+'/num_atoms_categories.pickle', 'rb') as handle:
        train_categorized_num_atoms = pickle.load(handle)[train_data_lower_category:train_data_upper_category]
    with open(train_root+'/categorized_energy.pickle', 'rb') as handle:
        train_target_energies = pickle.load(handle)[train_data_lower_category:train_data_upper_category]
    with open(train_root+'/categorized_forces.pickle', 'rb') as handle:
        train_target_forces = pickle.load(handle)[train_data_lower_category:train_data_upper_category]
    with open(train_root+'/categorized_structs.pickle', 'rb') as handle:
        train_structs = pickle.load(handle)[train_data_lower_category:train_data_upper_category]

    with open(test_root+'/num_atoms_categories.pickle', 'rb') as handle:
        test_categorized_num_atoms = pickle.load(handle)[test_data_lower_category:test_data_upper_category]
    with open(test_root+'/categorized_energy.pickle', 'rb') as handle:
        test_target_energies = pickle.load(handle)[test_data_lower_category:test_data_upper_category]
    with open(test_root+'/categorized_forces.pickle', 'rb') as handle:
        test_target_forces = pickle.load(handle)[test_data_lower_category:test_data_upper_category]
    with open(test_root+'/categorized_structs.pickle', 'rb') as handle:
        test_structs = pickle.load(handle)[test_data_lower_category:test_data_upper_category]

    return train_categorized_num_atoms,train_target_energies,train_target_forces,train_structs, test_categorized_num_atoms,test_target_energies,test_target_forces,test_structs

def train(par):

    data = load_data()

    torch.set_num_threads(torch_num_threads)
    if cluster[0:14] == 'slurm_dask_mpi':
        worker_identity = MPI.COMM_WORLD.rank
    else:
        worker_identity = multiprocessing.current_process()._identity[0]

    print('\n***Parameters Data***',
          '\nWorker identity:', worker_identity,
          '\nParameters data:\n',
          par,'\n',flush=True)

    if run_mode == 'Generate Model':
        np.save('generate_model_parameter_data', np.array(par))
    elif run_mode == 'Evaluate':
        np.save('evaluate_parameter_data', np.array(par))

    if rdf_mean_distribution == 'uniform':
        rdf_means = [ i*(range_rdf_means[1]-range_rdf_means[0])/(num_rdf-1)+range_rdf_means[0]
                            for i in range(num_rdf)]
        par = rdf_means+par #add back to par cos' I've taken the means out before the bb optimizer

    if adf_mean_distribution == 'uniform':
        adf_means = [ i*(range_adf_means[1]-range_adf_means[0])/(num_adf-1)+range_adf_means[0]
                            for i in range(num_adf)]
        par = par[:2*num_rdf]+adf_means+par[2*num_rdf:]

    torch.set_printoptions(precision=12,profile="full") #to print my model tensors to the right precision and all of the entries

    train_categorized_rdf_adf,test_categorized_rdf_adf,regularizer_lambda,force_beta = prepareData(par,data)

    if optimizer == 'RMSprop':
        loss_save = train_nonLBFGS(train_categorized_rdf_adf)
    elif optimizer == 'LBFGS':
        loss_save = train_LBFGS(train_categorized_rdf_adf,test_categorized_rdf_adf,data,regularizer_lambda,force_beta)

    return loss_save

def getExecutor():
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, LocalCluster

    if cluster == 'savio':
        def ex(*args, **kwargs):
            cluster = SLURMCluster(processes=processes,
                                   cores=cores,
                                   walltime=walltime,
                                   memory="64GB",
                                   job_cpu=job_cpu,
                                   project="",
                                   queue='savio',
                                   job_extra=qos,
                                   death_timeout=180)
            cluster.scale(*args, **kwargs)
            return Client(cluster,timeout=180)
    elif cluster == 'local':
        def ex(*args, **kwargs):
            cluster = LocalCluster(n_workers=bb_num_workers,
                                   processes=True,
                                   threads_per_worker=torch_num_threads,
                                   dashboard_address = None,)

            return Client(cluster,timeout=180)

    return ex


if __name__ == '__main__':
    #run mode
    #multiprocessing.set_start_method('spawn')
    main()

    #clean up mpi processes and files
    if  cluster == 'slurm_dask_mpi':
        # Remove scheduler's file
        os.system("rm myJob.sh scheduler.json")

        #cancel previous job
        outputs = glob.glob('./myjob.e*')
        outputs.sort(reverse=True)
        jobid = outputs[0].split('e')[1]
        os.system("scancel "+jobid)