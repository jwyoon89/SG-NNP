import torch
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import math

class Net(nn.Module):

    def __init__(self,num_descriptors):
        super(Net, self).__init__()
        
        self.num_descriptors = num_descriptors
        
        #ordered lists of intermediate layers and biases
        
        self.intermediate_weight_1 = Parameter(torch.Tensor(self.num_descriptors).double())
        self.intermediate_weights = [self.intermediate_weight_1]
        self.intermediate_bias_1 = Parameter(torch.zeros(self.num_descriptors).double())
        self.intermediate_biases = [self.intermediate_bias_1]
        
        self.weight_last = Parameter(torch.Tensor(self.num_descriptors).double())
        self.bias_last = Parameter(torch.zeros(1).double())
        

        self.reset_parameters()

    def reset_parameters(self):

        #xavier uniform
        for ind in range(len(self.intermediate_weights)):
            #stdv = math.sqrt(6/(self.intermediate_weights[ind].size(0)+self.intermediate_weights[ind].size(1)+1))
            stdv = math.sqrt(6/(1+1+1))
            self.intermediate_weights[ind].data.uniform_(-stdv, stdv)
            #self.intermediate_biases[ind].data.uniform_(-stdv, stdv)
        
        #stdv = math.sqrt(6/(self.intermediate_weights[-1].size(0)+1+1))
        stdv = math.sqrt(6/(1+1+1))
        self.weight_last.data.uniform_(-stdv, stdv)
        #self.bias_last.data.uniform_(-stdv, stdv)

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
        ###e_pa = e_pa*self.weight1 #(num_batch,num_atoms,num_descriptors)
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

        ###f = f*self.weight1 #size of linear factor:(num_batch,3,num_pairs,num_descriptors)
        
        
        #iterate through all intermediate layers
        for ind in range(len(self.intermediate_weights)):
            
            # e.g. to multiple high-dim tensors torch.matmul(A.view (10,3*4,5),B.view (10,5,6*7)).view (10,3,4,6,7)
            #e_pa = torch.matmul(e_pa,
            #                    self.intermediate_weights[ind].view(1,-1,self.N))+\
            #self.intermediate_biases[ind] #(num_batch,num_atoms,num_rdf+num_adf) -> (num_batch,num_atoms,N)
            
            e_pa = e_pa*self.intermediate_weights[ind] + self.intermediate_biases[ind] #(num_batch,num_atoms,num_rdf+num_adf)
            e_pa = torch.tanh(e_pa) #(num_batch,num_atoms,num_rdf+num_adf)
            dfdw = 1-(e_pa)**2 #(num_batch,num_atoms,num_rdf+num_adf)
            #f = torch.Tensor((dfdw[:,central_atom_index,:].unsqueeze_(1).expand(-1,3,-1,-1)*\
            #        (torch.matmul(f.view(1,-1,f.shape[3]),
            #                  self.intermediate_weights[ind].view(1,-1,self.N))).view(1,3,-1,self.N)).float()).double()# (num_batch,num_pairs,N) -> (num_batch,3,num_pairs,N) and dot with (num_batch,3,num_pairs,N)
            #f = dfdw[:,central_atom_index,:].unsqueeze_(1).expand(-1,3,-1,-1)*\
            #        (torch.matmul(f.view(1,-1,f.shape[3]),
            #                  self.intermediate_weights[ind].view(1,-1,self.N))).view(1,3,-1,self.N)# (num_batch,num_pairs,N) -> (num_batch,3,num_pairs,N) and dot with (num_batch,3,num_pairs,N)
            f = dfdw[:,central_atom_index,:].unsqueeze_(1).expand(-1,3,-1,-1)*\
                    (f*self.intermediate_weights[ind])# (num_batch,num_pairs,N) -> (num_batch,3,num_pairs,N) and dot with (num_batch,3,num_pairs,N)
                

        #last layer is slightly different as we're not applying the non-linearity on the output of energy
        ###e_pa = torch.sum(e_pa)/num_atoms + self.bias1
        #e_pa = torch.sum(torch.matmul(e_pa,self.weight_last))/num_atoms+self.bias_last #(num_batch,num_atoms,1)
        e_pa = torch.sum(e_pa*self.weight_last)/num_atoms+self.bias_last #(num_batch,num_atoms,1)
        #f = -torch.Tensor((torch.matmul(f.view(1,-1,f.shape[3]),
        #                 self.weight_last.view(1,-1,1)).view(1,3,-1)).float()).double() #(num_batch,3,num_pairs,N) -> (num_batch,3,num_pairs,1) -> (num_batch,3,num_pairs)
        #f = -torch.matmul(f.view(1,-1,f.shape[3]),
        #                 self.weight_last.view(1,-1,1)).view(1,3,-1) #(num_batch,3,num_pairs,N) -> (num_batch,3,num_pairs,1) -> (num_batch,3,num_pairs)
        
        f = -torch.sum(f*self.weight_last,dim=3) #(num_batch,3,num_pairs,N) -> (num_batch,3,num_pairs,1) -> (num_batch,3,num_pairs)
        
        #f_x = torch.squeeze(-F.linear(f_x, self.all_weights[-1], None)) #num_batch,num_pairs,1 -> #num_batch,num_pairs
        #f_y = torch.squeeze(-F.linear(f_y, self.all_weights[-1], None))
        #f_z = torch.squeeze(-F.linear(f_z, self.all_weights[-1], None))

        ###f = -torch.sum(f, 3)#num_batch,3,num_pairs,num_descriptors -> #num_batch,3,num_pairs

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
