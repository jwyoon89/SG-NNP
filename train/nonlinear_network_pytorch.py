import torch
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import math

class Net(nn.Module):

    def __init__(self,num_descriptors,N,L=1):
        super(Net, self).__init__()
        self.N= N
        self.L =L

        assert(L<=10)

        self.num_descriptors = num_descriptors

        #ordered lists of intermediate layers and biases
        self.intermediate_weights = []
        self.intermediate_biases = []

        if L >= 1:
            self.intermediate_weight_1 = Parameter(torch.Tensor(self.num_descriptors,N).double())
            self.intermediate_weights += [self.intermediate_weight_1]

            self.intermediate_bias_1 = Parameter(torch.zeros(N).double())
            self.intermediate_biases += [self.intermediate_bias_1]
        if L >=2:
            self.intermediate_weight_2 = Parameter(torch.Tensor(N,N).double())
            self.intermediate_weights += [self.intermediate_weight_2]

            self.intermediate_bias_2 = Parameter(torch.zeros(N).double())
            self.intermediate_biases += [self.intermediate_bias_2]
        if L >=3:
            self.intermediate_weight_3 = Parameter(torch.Tensor(N,N).double())
            self.intermediate_weights += [self.intermediate_weight_3]

            self.intermediate_bias_3 = Parameter(torch.zeros(N).double())
            self.intermediate_biases += [self.intermediate_bias_3]
        if L >=4:
            self.intermediate_weight_4 = Parameter(torch.Tensor(N,N).double())
            self.intermediate_weights += [self.intermediate_weight_4]

            self.intermediate_bias_4 = Parameter(torch.zeros(N).double())
            self.intermediate_biases += [self.intermediate_bias_4]

        if L >=5:
            self.intermediate_weight_5 = Parameter(torch.Tensor(N,N).double())
            self.intermediate_weights += [self.intermediate_weight_5]

            self.intermediate_bias_5 = Parameter(torch.zeros(N).double())
            self.intermediate_biases += [self.intermediate_bias_5]

        if L >=6:
            self.intermediate_weight_6 = Parameter(torch.Tensor(N,N).double())
            self.intermediate_weights += [self.intermediate_weight_6]

            self.intermediate_bias_6 = Parameter(torch.zeros(N).double())
            self.intermediate_biases += [self.intermediate_bias_6]

        if L >=7:
            self.intermediate_weight_7 = Parameter(torch.Tensor(N,N).double())
            self.intermediate_weights += [self.intermediate_weight_7]

            self.intermediate_bias_7 = Parameter(torch.zeros(N).double())
            self.intermediate_biases += [self.intermediate_bias_7]
        if L >=8:
            self.intermediate_weight_8 = Parameter(torch.Tensor(N,N).double())
            self.intermediate_weights += [self.intermediate_weight_8]

            self.intermediate_bias_8 = Parameter(torch.zeros(N).double())
            self.intermediate_biases += [self.intermediate_bias_8]

        if L >=9:
            self.intermediate_weight_9 = Parameter(torch.Tensor(N,N).double())
            self.intermediate_weights += [self.intermediate_weight_9]

            self.intermediate_bias_9 = Parameter(torch.zeros(N).double())
            self.intermediate_biases += [self.intermediate_bias_9]

        if L >=10:
            self.intermediate_weight_10 = Parameter(torch.Tensor(N,N).double())
            self.intermediate_weights += [self.intermediate_weight_10]

            self.intermediate_bias_10 = Parameter(torch.zeros(N).double())
            self.intermediate_biases += [self.intermediate_bias_10]


        self.weight_last = Parameter(torch.Tensor(N,1).double())
        self.bias_last = Parameter(torch.zeros(1).double())



        self.reset_parameters()

    def reset_parameters(self):

        #xavier uniform
        for ind in range(len(self.intermediate_weights)):
            stdv = math.sqrt(6/(self.intermediate_weights[ind].size(0)+self.intermediate_weights[ind].size(1)+1))
            self.intermediate_weights[ind].data.uniform_(-stdv, stdv)
            #self.intermediate_biases[ind].data.uniform_(-stdv, stdv)

        stdv = math.sqrt(6/(self.intermediate_weights[-1].size(0)+1+1))
        self.weight_last.data.uniform_(-stdv, stdv)
        #self.bias_last.data.uniform_(-stdv, stdv)

    #@profile
    def forward(self, coeffs, coeffs_derivs, central_atom_index, neigh_atom_index):

        #assert len(coeffs_x)==len(coeffs_y) and len(coeffs_y)==len(coeffs_z)

        num_atoms = coeffs.shape[1]

        #initialize output
        e_pa = coeffs #(num_batch,num_atoms,num_rdf+num_adf+num_one_hot_encoding)
        #print('coeffs',coeffs,flush=True)
        #print(coeffs.shape)
        #print('coeffs_derivs',coeffs_derivs,flush=True)
        #print(coeffs_derivs.shape)
        #f_x = coeffs_x #(num_batch,num_pairs,num_rdf+num_adf)
        #f_y = coeffs_y #(num_batch,num_pairs,num_rdf+num_adf)
        #f_z = coeffs_z #(num_batch,num_pairs,num_rdf+num_adf)

        f = coeffs_derivs #(num_batch,3,num_pairs,num_rdf+num_adf) #note: no one-hot encoding as they are all zero-valued when differentiated

        #iterate through all intermediate layers
        for ind in range(len(self.intermediate_weights)):

            # e.g. to multiple high-dim tensors torch.matmul(A.view (10,3*4,5),B.view (10,5,6*7)).view (10,3,4,6,7)
            e_pa = torch.matmul(e_pa,
                                self.intermediate_weights[ind].view(1,-1,self.N))+\
            self.intermediate_biases[ind] #(num_batch,num_atoms,num_rdf+num_adf+num_one_hot_encoding) -> (num_batch,num_atoms,N)
            e_pa = torch.tanh(e_pa) #(num_batch,num_atoms,N)
            dfdw = 1-(e_pa)**2 #(num_batch,num_atoms,N)

            if ind == 0:
                #since i've not bothered to put all the zeros in the derivatives of the one-hot encoding into f,
                #I have to select only the part for the weights matrix that is relevant for the matrix product for going from the input to the first layer
                #hence, (self.intermediate_weights[ind][:f.shape[3],:])
                f = dfdw[:,central_atom_index,:].unsqueeze_(1).expand(-1,3,-1,-1)*\
                        (torch.matmul(f.view(1,-1,f.shape[3]),
                                  (self.intermediate_weights[0][:f.shape[3],:]).view(1,-1,self.N))).view(1,3,-1,self.N)# (num_batch,num_pairs,N) -> (num_batch,3,num_pairs,N) and dot with (num_batch,3,num_pairs,N)
            else:
                f = dfdw[:,central_atom_index,:].unsqueeze_(1).expand(-1,3,-1,-1)*\
                        (torch.matmul(f.view(1,-1,f.shape[3]),
                                  self.intermediate_weights[ind].view(1,-1,self.N))).view(1,3,-1,self.N)# (num_batch,num_pairs,N) -> (num_batch,3,num_pairs,N) and dot with (num_batch,3,num_pairs,N)

        #last layer is slightly different as we're not applying the non-linearity on the output of energy
        e_pa = torch.sum((torch.matmul(e_pa,self.weight_last)).squeeze(2),1,keepdim=True)/num_atoms+self.bias_last #(num_batch,num_atoms,1)

        f = -torch.matmul(f.view(1,-1,f.shape[3]),
                         self.weight_last.view(1,-1,1)).view(1,3,-1) #(num_batch,3,num_pairs,N) -> (num_batch,3,num_pairs,1) -> (num_batch,3,num_pairs)

        out_f = Variable(torch.DoubleTensor(1,3,num_atoms).zero_())

        neigh_atom_index = torch.Tensor(neigh_atom_index) #1,num_pairs

        for i in range(num_atoms):

            collated_forces = f*Variable((neigh_atom_index==i).double())
            out_f[:,:,i] = torch.sum(collated_forces,2)

        return e_pa, out_f
