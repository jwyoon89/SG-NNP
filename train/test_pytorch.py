#descriptor parameters
num_rdf = 5
num_adf = 5
cutoff= 4.5

#rdf and adf representations
rdf_distribution = '' #'residual eating'
adf_distribution = '' #'residual eating'

#Neural Network Dimensions
NN_type = 'non-linear' # options: 'linear', 'slightly non-linear', 'non-linear'
N=3 #number of nodes in each intermediate layer in non-linear type
L=1 #number of intermediate layers in non-linear type

net_index = 1

if NN_type == 'linear':
    import linear_network_pytorch as mynetwork
elif NN_type == 'slightly non-linear':
    import slightly_nonlinear_network_pytorch as mynetwork
elif NN_type == 'non-linear':
    import nonlinear_network_pytorch as mynetwork


#------
#@profile
def test():

    import pickle
    import numpy as np
    import torch
    from torch import Tensor
    from torch.autograd import Variable
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    import structdata_test as structdata
    import random
    import gc as garbage

    #initializing neural net
    torch.set_num_threads(10)
    
    par = (np.load('myNet'+str(net_index)+'_params.npy')).tolist()
    
    if NN_type == 'linear' or NN_type == 'slightly non-linear':
        net = mynetwork.Net(num_rdf+num_adf).double()
    elif NN_type == 'non-linear':
        net = mynetwork.Net(num_rdf+num_adf,N,L).double()
        
    net.load_state_dict(torch.load('myNet'+str(net_index)+'.pt'))
    net.eval()

    if rdf_distribution == 'residual eating':
        rdf_means =[]
        for ind,i in enumerate(par[0:int(num_rdf)]):
            if ind ==0:
                rdf_means.append(i)
            else:
                rdf_means.append(i*(1-rdf_means[-1])+rdf_means[-1])
        rdf_means = np.array(rdf_means)
    else:
        rdf_means = par[0:int(num_rdf)]
        
    if adf_distribution == 'residual eating':
        adf_means =[]
        for ind,i in enumerate(par[2*int(num_rdf):2*int(num_rdf)+int(num_adf)]):
            if ind ==0:
                adf_means.append(i)
            else:
                adf_means.append(i*(1-adf_means[-1])+adf_means[-1])
        adf_means = np.array(adf_means)
    else:
        adf_means = par[2*int(num_rdf):2*int(num_rdf)+int(num_adf)]
    #----

    #load, initialize, generate and normalize all data
    data = structdata.Structures(rdf_means, adf_means, par[int(num_rdf):2*int(num_rdf)],
              par[2*int(num_rdf)+int(num_adf):2*int(num_rdf)+2*int(num_adf)],num_rdf,num_adf, cutoff, net_index)


#---------
    #Here we will start to do training!

    dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=10)#, collate_fn=my_collate)
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    #statistics to save for each epoch
    # loss_save = 0
    # ae_energy = 0
    # ae_force_x = 0
    # ae_force_y = 0
    # ae_force_z = 0
    # ae_force_mag = 0
    # num_energies = 0
    # num_particles =0

    out_pred_energy = np.array([])
    out_target_energy = np.array([])
    out_pred_force_x = np.array([])
    out_target_force_x = np.array([])
    out_pred_force_y = np.array([])
    out_target_force_y = np.array([])
    out_pred_force_z = np.array([])
    out_target_force_z = np.array([])

    for i_batch, sample_batched in enumerate(dataloader):

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



        #num_energies += target_e_pa.shape[1] #first dim is batch
        #num_particles += target_fx.shape[1]

        #print(target_e_pa.shape)
        #print(target_fx.shape)
        #print(input_coeffs.shape)
        #print(input_coeff_x.shape)

        #e_pa,fx,fy,fz = net(input_coeffs,
        #                    input_coeff_x,
        #                    input_coeff_y,
        #                    input_coeff_z,
        #                    central_atom_index,
        #                    neigh_atom_index)
        e_pa,f = net(input_coeffs,
                            input_coeff_x,
                            input_coeff_y,
                            input_coeff_z,
                            central_atom_index,
                            neigh_atom_index)

        #print('target epa:')
        #print(target_e_pa)
        #print('predicted epa:')
        #print(e_pa)
        #print(target_e_pa[0])

        # print(e_pa.shape)
        # print(target_e_pa.shape)

        out_pred_energy = np.concatenate((out_pred_energy,e_pa.data.numpy()))
        out_target_energy = np.concatenate((out_target_energy,target_e_pa[0,:].data.numpy()))

        #accumulate statistics of interest
        #ae_energy += (torch.sum(torch.abs(e_pa-target_e_pa))).data[0]
#             ae_force_x += (torch.sum(torch.abs(fx-target_fx))).data[0]
#             ae_force_y += (torch.sum(torch.abs(fy-target_fy))).data[0]
#             ae_force_z += (torch.sum(torch.abs(fz-target_fz))).data[0]
#             ae_force_mag += (torch.sum(torch.abs((fx**2+fy**2+fz**2).sqrt()-\
#                                                  (target_fx**2+target_fy**2+target_fz**2).sqrt()))\
#                              ).data[0]

        out_pred_force_x = np.concatenate((out_pred_force_x,f[0,0,:].data.numpy()))
        out_target_force_x = np.concatenate((out_target_force_x,target_f[0,0,:].data.numpy()))
        out_pred_force_y = np.concatenate((out_pred_force_y,f[0,1,:].data.numpy()))
        out_target_force_y = np.concatenate((out_target_force_y,target_f[0,1,:].data.numpy()))
        out_pred_force_z = np.concatenate((out_pred_force_z,f[0,2,:].data.numpy()))
        out_target_force_z = np.concatenate((out_target_force_z,target_f[0,2,:].data.numpy()))

    np.save('out_pred_energy_net_'+str(net_index),out_pred_energy*data.energies_std + data.energies_mean)
    np.save('out_target_energy_'+str(net_index),out_target_energy*data.energies_std + data.energies_mean)

    np.save('out_pred_force_x_'+str(net_index), out_pred_force_x*data.energies_std)
    np.save('out_target_force_x_'+str(net_index), out_target_force_x*data.energies_std)
    np.save('out_pred_force_y_'+str(net_index), out_pred_force_y*data.energies_std)
    np.save('out_target_force_y_'+str(net_index), out_target_force_y*data.energies_std)
    np.save('out_pred_force_z_'+str(net_index), out_pred_force_z*data.energies_std)
    np.save('out_target_force_z_'+str(net_index), out_target_force_z*data.energies_std)

    # print(out_pred_energy[0:10])
    # print(out_pred_energy.shape)
    # print(out_pred_force_x[0:10])
    # print(out_pred_force_x.shape)
    # print(out_target_force_x[0:10])
    # print(out_target_force_x.shape)
    # print('******')



    return 1
# --------


def main():
    test()

if __name__ == '__main__':
    #run mode
    main()

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]
