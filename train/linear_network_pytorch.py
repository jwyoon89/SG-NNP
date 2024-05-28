import torch
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import math

class Net(nn.Module):

    def __init__(self,num_descriptors):
        super(Net, self).__init__()
        self.num_descriptors = num_descriptors
        #hidden layers
        self.weight1 = Parameter(torch.Tensor(1,self.num_descriptors))
        self.bias1 = Parameter(torch.zeros(1))

        #ordered lists of layers and biases
        self.all_weights = [self.weight1]#, self.weight2, self.weight3]
        self.all_biases = [self.bias1]#, self.bias2, self.bias3]

        self.reset_parameters()

    def reset_parameters(self):

        #xavier uniform
        for ind in range(len(self.all_weights)):
            stdv = math.sqrt(6/(self.all_weights[ind].size(0)+self.all_weights[ind].size(1)+1))
            self.all_weights[ind].data.uniform_(-stdv, stdv)
            #self.all_biases[ind].data.uniform_(-stdv, stdv)

    #@profile
    def forward(self, coeffs, coeffs_derivs, central_atom_index, neigh_atom_index):

        #assert len(coeffs_x)==len(coeffs_y) and len(coeffs_y)==len(coeffs_z)

        num_atoms = coeffs.shape[1]

        #initialize output
        e_pa = coeffs #(num_batch,num_atoms,num_rdf+num_adf)

        #f_x = coeffs_x #(num_batch,num_pairs,num_rdf+num_adf)
        #f_y = coeffs_y #(num_batch,num_pairs,num_rdf+num_adf)
        #f_z = coeffs_z #(num_batch,num_pairs,num_rdf+num_adf)

        f = coeffs_derivs #(num_batch,3,num_pairs,num_rdf+num_adf)

        #print('f shape:')
        #print(f.shape)


        #iterate through all layers except the last
        #for ind in range(len(self.all_weights)-1):
        #e_pa = F.linear(e_pa, self.all_weights[ind], self.all_biases[ind]) #(num_batch,num_atoms,N)
        e_pa = e_pa*self.weight1 #(num_batch,num_atoms,num_descriptors)
        #applying w into dfdw = 1-tanh^2(w)
        #dfdw = 1-(F.tanh(e_pa))**2
        #print(dfdw.shape)
        #print(central_atom_index)
        #dfdw = dfdw[:,central_atom_index] #pick out the central atoms of the rows of f_x, f_y, f_z
        #e_pa = F.tanh(e_pa)
        #print(dfdw.shape)
        #print(F.linear(f_x, self.all_weights[ind], None).shape)

        #f_x = dfdw*F.linear(f_x, self.all_weights[ind], None)#size of linear factor:(num_batch,num_pairs,N)
        #f_y = dfdw*F.linear(f_y, self.all_weights[ind], None)
        #f_z = dfdw*F.linear(f_z, self.all_weights[ind], None)

        f = f*self.weight1 #size of linear factor:(num_batch,3,num_pairs,num_descriptors)

        #last layer is slightly different as we're not applying the non-linearity on the output of energy
        e_pa = torch.sum(e_pa)/num_atoms + self.bias1
        #f_x = torch.squeeze(-F.linear(f_x, self.all_weights[-1], None)) #num_batch,num_pairs,1 -> #num_batch,num_pairs
        #f_y = torch.squeeze(-F.linear(f_y, self.all_weights[-1], None))
        #f_z = torch.squeeze(-F.linear(f_z, self.all_weights[-1], None))

        f = -torch.sum(f, 3)#num_batch,3,num_pairs,num_descriptors -> #num_batch,3,num_pairs

        #print('f shape:')
        #print(f.shape)

        #collate the forces
        #out_f_x = Variable(torch.DoubleTensor(1,num_atoms).zero_())
        #out_f_y = Variable(torch.DoubleTensor(1,num_atoms).zero_())
        #out_f_z = Variable(torch.DoubleTensor(1,num_atoms).zero_())

        out_f = Variable(torch.DoubleTensor(1,3,num_atoms).zero_())

        neigh_atom_index = torch.Tensor(neigh_atom_index) #1,num_pairs

        for i in range(num_atoms):
            #out_f_x[0,i] = torch.sum(f_x[neigh_atom_index==i])
            #out_f_y[0,i] = torch.sum(f_y[neigh_atom_index==i])
            #out_f_z[0,i] = torch.sum(f_z[neigh_atom_index==i])

            collated_forces = f*Variable((neigh_atom_index==i).double())

            #print('collated_forces shape:')
            #print(collated_forces.shape)

            #collated_forces = collated_forces.view(1,3,-1)
            out_f[:,:,i] = torch.sum(collated_forces,2)
        #print('IM OUTTA HERE!')

        #print('out_f shape:')
        #print(out_f.shape)

        return e_pa, out_f
