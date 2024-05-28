import math
import itertools
import numpy as np
import gc as garbage
import pymatgen

import time

def poolInitializer(rdf_cutoff_in,adf_cutoff_in):
    global rdf_cutoff, adf_cutoff
    rdf_cutoff = rdf_cutoff_in
    adf_cutoff = adf_cutoff_in
    #print('rdf_cutoff,adf_cutoff: ',rdf_cutoff,adf_cutoff)

def generateRDFADFcoeffs(train_structs,test_structs, rdf_cutoff, adf_cutoff, train_num_atoms,test_num_atoms,
 rdf_means, rdf_widths,\
 adf_means, adf_widths,distinct_singlet_index_inverted,
 rdf_transformation_factors,adf_transformation_factors,
 dist_input_chunksize,angle_input_chunksize,
 gc_num_workers):
#setting up pool and quese for different number of workers
    if gc_num_workers ==1:
        from queue import Queue
        q=Queue
        p=None
    elif gc_num_workers >=2:
        from multiprocessing import Pool
        from multiprocessing import Manager
        manager = Manager()

        q=manager.Queue
        p = Pool(gc_num_workers,initializer=poolInitializer, initargs=(rdf_cutoff,adf_cutoff,)) #cutoff has to be passed in this way i.e. through the initializer or it won't work

    processed_train=[generateRDFADFcoeffs_gaussians(category,
                                       rdf_cutoff,
                                       adf_cutoff,
                                       train_num_atoms[ind],
                                       rdf_means,
                                       rdf_widths,
                                       adf_means,
                                       adf_widths,
                                       distinct_singlet_index_inverted,
                                       rdf_transformation_factors,
                                       adf_transformation_factors,
                                       dist_input_chunksize,
                                       angle_input_chunksize,
                                       gc_num_workers,
                                       [q() for _ in range(len(category))], #one for each struct in category: for rdf
                                       [q() for _ in range(len(category))], #one for each struct in category: for adf
                                       p)
    for ind,category in enumerate(train_structs) ]

    processed_test =[]
    if test_structs != None:
        processed_test = [generateRDFADFcoeffs_gaussians(category,
                                           rdf_cutoff,
                                           adf_cutoff,
                                           test_num_atoms[ind],
                                           rdf_means,
                                           rdf_widths,
                                           adf_means,
                                           adf_widths,
                                           distinct_singlet_index_inverted,
                                           rdf_transformation_factors,
                                           adf_transformation_factors,
                                           dist_input_chunksize,
                                           angle_input_chunksize,
                                           gc_num_workers,
                                           [q() for _ in range(len(category))], #one for each struct in category: for rdf
                                           [q() for _ in range(len(category))], #one for each struct in category: for adf
                                           p)
        for ind,category in enumerate(test_structs) ]

    return processed_train,processed_test

def generateRDFADFcoeffs_gaussians\
(structs, rdf_cutoff_in, adf_cutoff_in, num_atoms, \
 rdf_gaussians_means, rdf_gaussians_widths,\
 adf_gaussians_means, adf_gaussians_widths,
 distinct_singlet_index_inverted,
 rdf_transformation_factors,adf_transformation_factors,
 dist_input_chunksize,angle_input_chunksize,
 num_workers,qs_rdf,qs_adf,p=None):
    """
    Takes in a list of crystal_neighs, which contain all the structures on the first dimension, and the RDF and ADF
    info on the second dimension.

    means and width: For RDF, in units of rdf cutoff radius. For ADF, in units of 180 degrees.

    For this implementation, the number of atoms in a crystal is not kept at 4.

    Deleted: rdf_num...,adf_num..., factorForSig

    """

    global rdf_cutoff,adf_cutoff
    rdf_cutoff = rdf_cutoff_in
    adf_cutoff = adf_cutoff_in

    #supercell index
    central_atom_sc_index = []
    #cell index
    central_atom_cell_index = []
    neigh_atom_index = []

    #initialize numpy arrays to hold the coefficients separately
    rdf_coeff= np.zeros((len(structs),num_atoms,len(rdf_gaussians_means)), dtype=float)
    adf_coeff= np.zeros((len(structs),num_atoms,len(adf_gaussians_means)), dtype=float)

    central_atom_species_coeff = np.zeros((len(structs),num_atoms,len(distinct_singlet_index_inverted)), dtype=np.uint8)
    rdf_coeff_dx= []
    rdf_coeff_dy= []
    rdf_coeff_dz= []
    adf_coeff_dx= []
    adf_coeff_dy= []
    adf_coeff_dz= []

    m_rdf = []
    m_adf = []

    for struct_ind,struct in enumerate(structs):

#         program_starts = time.time()

        if num_workers == 1:
            struct_processed = struct2neighdistNangle_nearestN(struct,rdf_cutoff_in,adf_cutoff_in)
        elif num_workers >= 2:

            struct2neighdistNangle_input = struct_chunks(struct,rdf_cutoff_in,adf_cutoff_in, num_workers)


            multiple_results = p.starmap_async(struct2neighdistNangle_nearestN, struct2neighdistNangle_input)
            got_results = multiple_results.get()
            struct_processed = ([j for i in got_results for j in i[0]],[j for i in got_results for j in i[1]] )



        #Generating one-hot encoding of central atom species
        for ind,atom in enumerate(struct):
            central_atom_species_coeff[struct_ind][ind][distinct_singlet_index_inverted[atom.specie.Z]]=1

        pair_indices = [(i,i) for i in range(num_atoms)] #all the (i,i)'s

        # Note: we can actually separate these two pair indices out in to ADF and RDF components to avoid redundancy or memory output_stage
        #This is one area to decrease memory footprint if one should want to use a large rdf cutoff radius

        #add RDF pairs
        pair_indices = pair_indices+list(set([(i[0],i[7])
                                              for i in struct_processed[0]]+\
                                             [(i[7],i[0])
                                              for i in struct_processed[0]]
                                            )-set(pair_indices)) #adding in the (i,j) where i!=j
        #add ADF pairs
        pair_indices = pair_indices+list(set([(i[0],i[12])
                                              for i in struct_processed[1] if i[4] < adf_cutoff ]+\
                                             [(i[12],i[0])
                                              for i in struct_processed[1] if i[4] < adf_cutoff ]+\
                                              [(i[0],i[13])
                                                for i in struct_processed[1] if i[5] < adf_cutoff ]+\
                                               [(i[13],i[0])
                                                for i in struct_processed[1] if i[5] < adf_cutoff ]
                                            )-set(pair_indices)) #adding in the (i,j) where i!=j


        #supercell index
        central_atom_sc_index += [ [p[0] for p in pair_indices]]
        #cell index
        central_atom_cell_index += [ [p[0]%num_atoms for p in pair_indices] ]
        neigh_atom_index += [ [p[1] for p in pair_indices] ]
        #now adding the indices
        pair_indices = {p:i for i,p in enumerate(pair_indices)}

        #containers for adding results into
        rdf = np.zeros((num_atoms,len(rdf_gaussians_means)), dtype=float)
        rdf_dx = np.zeros((len(central_atom_sc_index[struct_ind]),len(rdf_gaussians_means)), dtype=float)
        rdf_dy = np.zeros((len(central_atom_sc_index[struct_ind]),len(rdf_gaussians_means)), dtype=float)
        rdf_dz = np.zeros((len(central_atom_sc_index[struct_ind]),len(rdf_gaussians_means)), dtype=float)
        adf = np.zeros((num_atoms,len(adf_gaussians_means)), dtype=float)
        adf_dx = np.zeros((len(central_atom_sc_index[struct_ind]),len(adf_gaussians_means)), dtype=float)
        adf_dy = np.zeros((len(central_atom_sc_index[struct_ind]),len(adf_gaussians_means)), dtype=float)
        adf_dz = np.zeros((len(central_atom_sc_index[struct_ind]),len(adf_gaussians_means)), dtype=float)


#         total_time3 += time.time()-program_starts

        q_rdf =  qs_rdf[struct_ind]
        q_adf =  qs_adf[struct_ind]

#         total_time4 += time.time()-program_starts

        q_rdf.put([rdf, rdf_dx, rdf_dy, rdf_dz])
        #print('Done with q_rdf put',flush=True)
        q_adf.put([adf, adf_dx, adf_dy, adf_dz])
        #print('Done with q_adf put',flush=True)

#         total_time5 += time.time()-program_starts

        dist_input = chunks(struct_processed[0],rdf_transformation_factors,rdf_cutoff,rdf_gaussians_means,rdf_gaussians_widths,
                           len(central_atom_sc_index[struct_ind]),pair_indices,num_atoms,q_rdf,
                            dist_input_chunksize)

        angle_input = chunks(struct_processed[1],adf_transformation_factors,adf_cutoff,adf_gaussians_means,adf_gaussians_widths,
                           len(central_atom_sc_index[struct_ind]),pair_indices,num_atoms,q_adf,
                            angle_input_chunksize)

#         total_time6 += time.time()-program_starts

        if num_workers == 1:
            #evaluate the starmap or else it will only remain as an iterator
            for i in itertools.starmap(loopoverallneighs_dist, dist_input):
                pass
            for i in itertools.starmap(loopoverallneighs_angle, angle_input):
                pass
        elif num_workers >= 2:

            m_rdf += [p.starmap_async(loopoverallneighs_dist, dist_input)]
            #print('Done with m_rdf',flush=True)
            m_adf += [p.starmap_async(loopoverallneighs_angle, angle_input)]
            #print('Done with m_adf',flush=True)

    if num_workers >=2:

#         program_starts = time.time()

        for i in range(len(m_rdf)):
            m_rdf[i].wait()
            #print('Done with m_rdf',flush=True)
            m_adf[i].wait()


#         total_time8 += time.time()-program_starts
    for struct_ind in range(len(structs)):

#         program_starts = time.time()

        #print('Done with mapping.')
        get_rdf = qs_rdf[struct_ind].get()
        #print('Done with q_rdf get',flush=True)
        get_adf = qs_adf[struct_ind].get()
        #print('Done with q_adf get',flush=True)

        rdf_coeff[struct_ind] += get_rdf[0]
        rdf_coeff_dx += [get_rdf[1]]
        rdf_coeff_dy += [get_rdf[2]]
        rdf_coeff_dz += [get_rdf[3]]

        #print('Done with updating rdf_coeff',flush=True)

        adf_coeff[struct_ind] += get_adf[0]
        adf_coeff_dx += [get_adf[1]]
        adf_coeff_dy += [get_adf[2]]
        adf_coeff_dz += [get_adf[3]]

    results = {'rdf': rdf_coeff,
               'rdf_dx': rdf_coeff_dx,
               'rdf_dy': rdf_coeff_dy,
               'rdf_dz': rdf_coeff_dz,
               'adf': adf_coeff,
               'adf_dx': adf_coeff_dx,
               'adf_dy': adf_coeff_dy,
               'adf_dz': adf_coeff_dz,
               'central_atom_species_coeff':central_atom_species_coeff,
               'central_atom_index':central_atom_cell_index,
               'neigh_atom_index':neigh_atom_index}

    print('\n'+str(num_atoms)+'-atom structures are done!!!!\n',flush=True)
    #print('results: ',results)
    return results

'Define Cutoff function'
def fc(r,cutoff):
    if r< cutoff and r>0:
        return 0.5*(np.cos(r*math.pi/cutoff)+1)
    else:
        return 0

'Define exponential function'
def exp(x,miu,sig):
    return np.exp(-((x-miu)**2)/(2*sig**2))

#component functions for rdf derivatives

#derivative of neighbor radial distance wrt to axis-a of neighbor atom
def drj_da(vec,a):
    '''
    Takes in the vector of a neigbor and axis of derivative.
    a= {0,1,2}
    0: x-axis
    1: y-axis
    2: z-axis
    '''
    norm = np.linalg.norm(vec)
    return vec[a]/norm

#derivative of cutoff value at neighbor radial distance wrt to axis-a of neighbor atom
def dfc_da(vec,a,cutoff):
    '''
    Takes in the vector of a neigbor and axis of derivative.
    a= {0,1,2}
    0: x-axis
    1: y-axis
    2: z-axis
    '''
    norm = np.linalg.norm(vec)
    return -drj_da(vec,a)*np.pi/(2*cutoff)*np.sin(norm*math.pi/cutoff)

#putting everything together into the derivative along axis-a of the rdf coefficient
def drdfcoeff_da(vec, a, miu, sig,cutoff,shift_factor,scale_factor):
    '''
    Takes in the vector of a neighbour and axis of differentiation and miu and sigma of gaussian kernel.
    Returns the derivative contribution of the particular neighbour to the particular gaussian kernel, moving the neighbor.
    a= {0,1,2}
    0: x-axis
    1: y-axis
    2: z-axis
    '''
    norm = np.linalg.norm(vec)
    exponential = exp(shift_factor+scale_factor*norm,miu,sig)

    if norm > cutoff:
        return 0

    first_term = -(shift_factor+scale_factor*norm-miu)/(sig**2)*scale_factor*exponential*drj_da(vec,a)*fc(norm,cutoff)
    second_term = dfc_da(vec,a,cutoff)*exponential
    return first_term+second_term

#derivative of theta with respect to axis-a of neighbor atom vec1
def dtheta_da(vec1, vec2, angle, a, move_central_atom):
    '''
    Takes in two neighbor vecs and the angle subtended by the vectors.
    Returns the derivative of the angle wrt to the axis-a.
    Angle is in degrees.
    '''
    #converts back to radians
    angle = angle/180.0*math.pi
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    #we're moving atom at vec 1
    if move_central_atom == False:

        #if implicit formula works
        if np.sin(angle)>10**-10:
            first_term =norm1**(-1)*drj_da(vec1,a)*np.dot(vec1,vec2)
            second_term = -vec2[a]

            return (np.sin(angle)*norm1*norm2)**(-1)*(first_term+second_term)
        #if angle is pi:
        elif angle > 3.1:
            unit_vec1 = vec1/norm1
            unit_vec1_in_a = np.abs(unit_vec1[a])

            return -(norm1+norm2)/(norm1*(norm1+norm2))*np.sin(np.arccos(unit_vec1_in_a))

        #if angle is zero:
        elif angle<0.01:
            unit_vec1 = vec1/norm1
            unit_vec1_in_a = np.abs(unit_vec1[a])

            return 1.0/norm1 *np.sin(np.arccos(unit_vec1_in_a))

        else:
            raise Exception('Check dtheta/da')


    #if neigh at vec1 isnt mirror of central atom but is that of the other neigh and we're moving atom at vec1
    elif move_central_atom == True:

        #if implicit formula works
        if np.sin(angle)>10**-10:

            first_term =np.dot(vec1,vec2)*((1.0/norm2)*(-drj_da(vec2,a))+(1.0/norm1)*(-drj_da(vec1,a)))
            second_term = (vec1[a]+vec2[a])

            return (norm1*norm2*np.sin(angle))**(-1)*(first_term+second_term)

        #if angle is pi:
        elif angle > 3.1:

            unit_vec1 = vec1/norm1
            unit_vec1_in_a = np.abs(unit_vec1[a])

            return -(norm1+norm2)/(norm1*norm2) * np.sin(np.arccos(unit_vec1_in_a))


        #if angle is zero:
        elif angle<0.01:

            unit_vec1 = vec1/norm1
            unit_vec1_in_a = np.abs(unit_vec1[a])

            return np.abs(norm1-norm2)/(norm1*norm2) * np.sin(np.arccos(unit_vec1_in_a))

def d_shifted_cos_angle_da(vec1, vec2, shifted_cos_angle, a, move_central_atom):
    '''
    Takes in two neighbor vecs and the shifted cosine angle subtended by the vectors.
    Returns the derivative of the shifted cosine angle wrt to the axis-a.
    '''
    #converts back to radians

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    #we're moving atom at vec 1
    if move_central_atom == False:

        first_term =norm1**(-1)*drj_da(vec1,a)*np.dot(vec1,vec2)
        second_term = -vec2[a]

        return -(2*norm1*norm2)**(-1)*(first_term+second_term)


    #if neigh at vec1 isnt mirror of central atom but is that of the other neigh and we're moving atom at vec1
    elif move_central_atom == True:

        first_term =np.dot(vec1,vec2)*((1.0/norm2)*(-drj_da(vec2,a))+(1.0/norm1)*(-drj_da(vec1,a)))
        second_term = (vec1[a]+vec2[a])

        return -(2*norm1*norm2)**(-1)*(first_term+second_term)


#putting everything together into the derivative along axis-a of the adf coefficient
#for neigh atom
def dadfcoeff_da(vec1, vec2, shifted_cos_angle, a, miu, sig, move_central_atom,cutoff,shift_factor,scale_factor):

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    exponential = exp(shift_factor+scale_factor*shifted_cos_angle,miu,sig)

    if ((norm1 > cutoff) or (norm2 > cutoff)) :

        return 0

    #we're moving atom at vec1
    if move_central_atom == False:
        first_term = -(shift_factor+scale_factor*shifted_cos_angle-miu)/(sig**2) * exponential *scale_factor* d_shifted_cos_angle_da(vec1, vec2, shifted_cos_angle, a, False) * fc(norm1,cutoff) * fc(norm2,cutoff)
        second_term = exponential * dfc_da(vec1,a,cutoff) * fc(norm2,cutoff)
        #third_term = exponential * dfc_da(vec2,a) * fc(norm1)

        return first_term + second_term #+ third_term

    #if neigh at vec1 isnt mirror of central atom but is that of the other neigh and we're moving atom at vec1
    elif move_central_atom == True:

        first_term = \
        -(shift_factor+scale_factor*shifted_cos_angle-miu)/(sig**2) * exponential *scale_factor* (d_shifted_cos_angle_da(vec1, vec2, shifted_cos_angle, a,True)) * fc(norm1,cutoff) * fc(norm2,cutoff)
        second_term = exponential * (-dfc_da(vec2,a,cutoff)) * fc(norm1,cutoff)
        third_term = exponential * (-dfc_da(vec1,a,cutoff)) * fc(norm2,cutoff)

        return first_term + second_term + third_term

def loopoverallneighs_dist(neigh_dists, rdf_transformation_factors, cutoff,
                           rdf_gaussians_means, rdf_gaussians_widths,
                           len_central_atom_index,  pair_indices, num_atoms,q_rdf):

    garbage.collect()

    #initialize numpy arrays to hold the cofficients separately
    rdf_coeff= np.zeros((num_atoms,len(rdf_gaussians_means)), dtype=float)
    rdf_coeff_dx= np.zeros((len_central_atom_index,len(rdf_gaussians_means)), dtype=float)
    rdf_coeff_dy= np.zeros((len_central_atom_index,len(rdf_gaussians_means)), dtype=float)
    rdf_coeff_dz= np.zeros((len_central_atom_index,len(rdf_gaussians_means)), dtype=float)

    for neigh_dist in neigh_dists:

        shift_factor = rdf_transformation_factors[neigh_dist[2]][0]
        scale_factor = rdf_transformation_factors[neigh_dist[2]][1]

        for coeff_ind in range(0,len(rdf_gaussians_means)):
            #one-specie case (gotta modify more more species):

            gaussian_mean = rdf_gaussians_means[coeff_ind]*cutoff
            gaussian_width = rdf_gaussians_widths[coeff_ind]*cutoff

            rdf_coeff[pair_indices[(neigh_dist[0],neigh_dist[0])]][coeff_ind] += \
            fc(neigh_dist[3],cutoff)*\
            exp(shift_factor+scale_factor*neigh_dist[3],gaussian_mean,gaussian_width)

            rdf_coeff_dx[pair_indices[(neigh_dist[0],neigh_dist[0])]][coeff_ind] += \
            -drdfcoeff_da(neigh_dist[4],0,gaussian_mean,gaussian_width,cutoff,shift_factor,scale_factor)

            rdf_coeff_dy[pair_indices[(neigh_dist[0],neigh_dist[0])]][coeff_ind] += \
            -drdfcoeff_da(neigh_dist[4],1,gaussian_mean,gaussian_width,cutoff,shift_factor,scale_factor)

            rdf_coeff_dz[pair_indices[(neigh_dist[0],neigh_dist[0])]][coeff_ind] += \
            -drdfcoeff_da(neigh_dist[4],2,gaussian_mean,gaussian_width,cutoff,shift_factor,scale_factor)

            if neigh_dist[6] == True: #if the neighbor atom is in the unit cell

                #moving neighbouring atom
                rdf_coeff_dx[pair_indices[(neigh_dist[0],neigh_dist[7])]][coeff_ind] += \
                drdfcoeff_da(neigh_dist[4],0,gaussian_mean,gaussian_width,cutoff,shift_factor,scale_factor)

                rdf_coeff_dy[pair_indices[(neigh_dist[0],neigh_dist[7])]][coeff_ind] += \
                drdfcoeff_da(neigh_dist[4],1,gaussian_mean,gaussian_width,cutoff,shift_factor,scale_factor)

                rdf_coeff_dz[pair_indices[(neigh_dist[0],neigh_dist[7])]][coeff_ind] += \
                drdfcoeff_da(neigh_dist[4],2,gaussian_mean,gaussian_width,cutoff,shift_factor,scale_factor)

            elif neigh_dist[6] == False: #if neigh atom is an image we would still want to account for the change in energy of the image due to moving the central atom

                rdf_coeff_dx[pair_indices[(neigh_dist[7],neigh_dist[0])]][coeff_ind] += \
                drdfcoeff_da(-neigh_dist[4],0,gaussian_mean,gaussian_width,cutoff,shift_factor,scale_factor)

                rdf_coeff_dy[pair_indices[(neigh_dist[7],neigh_dist[0])]][coeff_ind] += \
                drdfcoeff_da(-neigh_dist[4],1,gaussian_mean,gaussian_width,cutoff,shift_factor,scale_factor)

                rdf_coeff_dz[pair_indices[(neigh_dist[7],neigh_dist[0])]][coeff_ind] += \
                drdfcoeff_da(-neigh_dist[4],2,gaussian_mean,gaussian_width,cutoff,shift_factor,scale_factor)

    listofcoeffs = q_rdf.get()
    #print("rdf Get")
    listofcoeffs[0] += rdf_coeff
    listofcoeffs[1] += rdf_coeff_dx
    listofcoeffs[2] += rdf_coeff_dy
    listofcoeffs[3] += rdf_coeff_dz
    q_rdf.put(listofcoeffs)
    #print("rdf Put")

def loopoverallneighs_angle(neigh_angles,adf_transformation_factors, cutoff,
                            adf_gaussians_means, adf_gaussians_widths,
                            len_central_atom_index,  pair_indices, num_atoms,q_adf):

    garbage.collect()

    adf_coeff= np.zeros((num_atoms,len(adf_gaussians_means)), dtype=float)
    adf_coeff_dx= np.zeros((len_central_atom_index,len(adf_gaussians_means)), dtype=float)
    adf_coeff_dy= np.zeros((len_central_atom_index,len(adf_gaussians_means)), dtype=float)
    adf_coeff_dz= np.zeros((len_central_atom_index,len(adf_gaussians_means)), dtype=float)


    for neigh_angle in neigh_angles:

        shift_factor = adf_transformation_factors[neigh_angle[2]][0]
        scale_factor = adf_transformation_factors[neigh_angle[2]][1]

        for coeff_ind in range(0,len(adf_gaussians_means)):
            #one-specie case (gotta modify more more species):
            #print("\ni'm here")

            gaussian_mean = adf_gaussians_means[coeff_ind]*1.0
            gaussian_width = adf_gaussians_widths[coeff_ind]*1.0

            adf_coeff[pair_indices[(neigh_angle[0],neigh_angle[0])]][coeff_ind] += \
            fc(neigh_angle[4],cutoff)*fc(neigh_angle[5],cutoff)*\
            exp(shift_factor+scale_factor*neigh_angle[3],gaussian_mean,gaussian_width)

            adf_coeff_dx[pair_indices[(neigh_angle[0],neigh_angle[0])]][coeff_ind] += \
                    dadfcoeff_da(neigh_angle[6],neigh_angle[7],neigh_angle[3],0,gaussian_mean,gaussian_width,True,cutoff,shift_factor,scale_factor)

            adf_coeff_dy[pair_indices[(neigh_angle[0],neigh_angle[0])]][coeff_ind] += \
                    dadfcoeff_da(neigh_angle[6],neigh_angle[7],neigh_angle[3],1,gaussian_mean,gaussian_width,True,cutoff,shift_factor,scale_factor)

            adf_coeff_dz[pair_indices[(neigh_angle[0],neigh_angle[0])]][coeff_ind] += \
                    dadfcoeff_da(neigh_angle[6],neigh_angle[7],neigh_angle[3],2,gaussian_mean,gaussian_width,True,cutoff,shift_factor,scale_factor)

                #moving neighbouring atom, one at a time
            if neigh_angle[4] < cutoff: #only go on to the next part if there is a contribution to the central atom by neigh0
                    #mirrored_vec1 = neigh_angle[0] == neigh_angle[8]
                    #vec1_vec2_same = neigh_angle[8] == neigh_angle[9]
                if neigh_angle[10] == True: #if the neigh0 atom is in the unit cell
                    #move neigh0
                    adf_coeff_dx[pair_indices[(neigh_angle[0],neigh_angle[12])]][coeff_ind] += \
                        dadfcoeff_da(neigh_angle[6],neigh_angle[7],neigh_angle[3],0,gaussian_mean,gaussian_width,False,cutoff,shift_factor,scale_factor)

                    adf_coeff_dy[pair_indices[(neigh_angle[0],neigh_angle[12])]][coeff_ind] += \
                        dadfcoeff_da(neigh_angle[6],neigh_angle[7],neigh_angle[3],1,gaussian_mean,gaussian_width,False,cutoff,shift_factor,scale_factor)

                    adf_coeff_dz[pair_indices[(neigh_angle[0],neigh_angle[12])]][coeff_ind] += \
                        dadfcoeff_da(neigh_angle[6],neigh_angle[7],neigh_angle[3],2,gaussian_mean,gaussian_width,False,cutoff,shift_factor,scale_factor)

                elif neigh_angle[10] == False: #if neigh0 atom is an image we would still want to account for the change in energy of the image due to moving the central atom
                    #move neigh0
                    shifted_cos_angle_btw = adjusted_cos_angle_between(-neigh_angle[6],-neigh_angle[6]+neigh_angle[7])
                    adf_coeff_dx[pair_indices[(neigh_angle[12],neigh_angle[0])]][coeff_ind] += \
                        dadfcoeff_da(-neigh_angle[6],-neigh_angle[6]+neigh_angle[7],shifted_cos_angle_btw,0,gaussian_mean,gaussian_width,False,cutoff,shift_factor,scale_factor)

                    adf_coeff_dy[pair_indices[(neigh_angle[12],neigh_angle[0])]][coeff_ind] += \
                        dadfcoeff_da(-neigh_angle[6],-neigh_angle[6]+neigh_angle[7],shifted_cos_angle_btw,1,gaussian_mean,gaussian_width,False,cutoff,shift_factor,scale_factor)

                    adf_coeff_dz[pair_indices[(neigh_angle[12],neigh_angle[0])]][coeff_ind] += \
                        dadfcoeff_da(-neigh_angle[6],-neigh_angle[6]+neigh_angle[7],shifted_cos_angle_btw,2,gaussian_mean,gaussian_width,False,cutoff,shift_factor,scale_factor)


            if neigh_angle[5] < cutoff: #only go on to the next part if there is a contribution to the central atom by neigh1
                if neigh_angle[11] == True: #if the neigh1 atom is in the unit cell
                        #move neigh1
                    adf_coeff_dx[pair_indices[(neigh_angle[0],neigh_angle[13])]][coeff_ind] += \
                        dadfcoeff_da(neigh_angle[7],neigh_angle[6],neigh_angle[3],0,gaussian_mean,gaussian_width,False,cutoff,shift_factor,scale_factor)

                    adf_coeff_dy[pair_indices[(neigh_angle[0],neigh_angle[13])]][coeff_ind] += \
                        dadfcoeff_da(neigh_angle[7],neigh_angle[6],neigh_angle[3],1,gaussian_mean,gaussian_width,False,cutoff,shift_factor,scale_factor)

                    adf_coeff_dz[pair_indices[(neigh_angle[0],neigh_angle[13])]][coeff_ind] += \
                        dadfcoeff_da(neigh_angle[7],neigh_angle[6],neigh_angle[3],2,gaussian_mean,gaussian_width,False,cutoff,shift_factor,scale_factor)

                elif neigh_angle[11] == False: #if neigh1 atom is an image we would still want to account for the change in energy of the image due to moving the central atom
                        #move neigh1
                    shifted_cos_angle_btw = adjusted_cos_angle_between(-neigh_angle[7],-neigh_angle[7]+neigh_angle[6])
                    adf_coeff_dx[pair_indices[(neigh_angle[13],neigh_angle[0])]][coeff_ind] += \
                        dadfcoeff_da(-neigh_angle[7],-neigh_angle[7]+neigh_angle[6],shifted_cos_angle_btw,0,gaussian_mean,gaussian_width,False,cutoff,shift_factor,scale_factor)

                    adf_coeff_dy[pair_indices[(neigh_angle[13],neigh_angle[0])]][coeff_ind] += \
                        dadfcoeff_da(-neigh_angle[7],-neigh_angle[7]+neigh_angle[6],shifted_cos_angle_btw,1,gaussian_mean,gaussian_width,False,cutoff,shift_factor,scale_factor)

                    adf_coeff_dz[pair_indices[(neigh_angle[13],neigh_angle[0])]][coeff_ind] += \
                        dadfcoeff_da(-neigh_angle[7],-neigh_angle[7]+neigh_angle[6],shifted_cos_angle_btw,2,gaussian_mean,gaussian_width,False,cutoff,shift_factor,scale_factor)

    #get and update shared variable
    listofcoeffs = q_adf.get()
    #print("adf Get")
    listofcoeffs[0] += adf_coeff
    listofcoeffs[1] += adf_coeff_dx
    listofcoeffs[2] += adf_coeff_dy
    listofcoeffs[3] += adf_coeff_dz
    q_adf.put(listofcoeffs)


def chunks(l, indices, cutoff, means, widths, len_central_atom_index, pair_indices,num_atoms,queue, n):
    """Yield successive n-sized chunks from l."""

    for i in range(0, len(l), n):
        yield (l[i:i + n], indices, cutoff, means, widths, len_central_atom_index, pair_indices, num_atoms,queue)

def struct_chunks(struct,rdf_cutoff_in,adf_cutoff_in, num_chunks):
    """Yield successive n-sized chunks from l."""

    n = math.ceil(len(struct)/num_chunks) #ceil, so at most num_chunks
    for i in range(0, len(struct), n):
        yield (struct, rdf_cutoff_in, adf_cutoff_in, i,i+n)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def adjusted_cos_angle_between(v1, v2):
    """ Returns the shifted cos angle between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return 0.5*(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)+1)

def struct2neighdistNangle_nearestN(cry_struct,rdf_cutoff,adf_cutoff,start_ind=0,end_ind=9999999999):
    '''This function takes in a pymatgen structure and gives a list of list of center atom's
    index, center atom's atomic number, neighbor's atomic number, distance to the neighbor.

    It also gives a list of list of center atom's index, center atom's atomic number,
    sorted atomic numbers of two neighbours of center atom,  the angle the two neighbor atoms
    make with the center atom.

    '''

    rdf_neighs=cry_struct.get_all_neighbors(rdf_cutoff,include_index=True,include_image=True)
    adf_neighs=cry_struct.get_all_neighbors(adf_cutoff*2,include_index=True,include_image=True)

    num_atoms = cry_struct.num_sites

    max_index = 20 # the max abs index for cell as given by pymatgen. set it sufficiently large.

    cell_index = [(0,0,0)]
    for i in range(-max_index,max_index+1):
        for j in range(-max_index,max_index+1):
            for k in range(-max_index,max_index+1):
                if (i,j,k) != (0,0,0):
                    cell_index += [(i,j,k)]
    cell_index = {i:ind for ind,i in enumerate(cell_index)}


    center_atom_num = [atom.specie.Z for atom in cry_struct]


    res_dist = [[np.uint8(ind+start_ind), #0: central atom index
                 np.uint8(center_atom_num[ind+start_ind]), #1: atomic num
                 np.uint8(neigh[0].specie.Z),#2: neighbor's atomic num
                 neigh[1], #3: neigh's distance from central atom
                 neigh[0].coords-cry_struct[ind+start_ind].coords, #4: displacement from central atom
                 neigh[2], #5: neigh atom's unit cell index (i.e. not the supercell index)
                 neigh[3]==(0,0,0),#6: whether neigh atom is in the unit cell
                 num_atoms*cell_index[(neigh[3][0],neigh[3][1],neigh[3][2])]+neigh[2]] #7: neigh encoded supercell index number
               for ind,neighs_one_atom in enumerate(rdf_neighs[start_ind:end_ind])
               for neigh in neighs_one_atom #]
               if neigh[1] < rdf_cutoff]


    #program_starts = time.time()
    res_angle = [[np.uint8(ind+start_ind),  #0: central atom index
                  np.uint8(center_atom_num[ind+start_ind]), #1: central atom atomic num
                  tuple(sorted([np.uint8(pair_neighs[0][0].specie.Z),np.uint8(pair_neighs[1][0].specie.Z)])),#2: atomic numbers, sorted
                  adjusted_cos_angle_between(pair_neighs[0][0].coords-cry_struct[ind+start_ind].coords,
                                pair_neighs[1][0].coords-cry_struct[ind+start_ind].coords), #3: angle btw neigh0 and neigh1 by central atom
                  pair_neighs[0][1], #4:
                  pair_neighs[1][1], #5: distance of neigh0, distance of neigh1, from central atom
                  pair_neighs[0][0].coords-cry_struct[ind+start_ind].coords, #6: neigh0 displacemenet from central atom
                  pair_neighs[1][0].coords-cry_struct[ind+start_ind].coords, #7: neigh1 displacemenet from central atom
                  pair_neighs[0][2], #8: neigh0 index in cell for identification
                  pair_neighs[1][2], #9: neigh1 index in cell for identification
                  pair_neighs[0][3]==(0,0,0), #10: whether neigh0 atom is in the unit cell
                  pair_neighs[1][3]==(0,0,0), #11: whether neigh1 atom is in the unit cell
                  num_atoms*cell_index[(pair_neighs[0][3][0],pair_neighs[0][3][1],pair_neighs[0][3][2])]+\
                  pair_neighs[0][2], #12: neigh0 atom's encoded supercell index number
                  num_atoms*cell_index[(pair_neighs[1][3][0],pair_neighs[1][3][1],pair_neighs[1][3][2])]+\
                  pair_neighs[1][2]] #13: neigh1 atom's encoded supercell index number
                  #num_atoms*sum(pair_neighs[0][3])+pair_neighs[0][2], #neigh0 atom's supercell index number
                  #num_atoms*sum(pair_neighs[1][3])+pair_neighs[1][2]] #neigh1 atom's supercell index number
                 for ind,neighs_one_atom in enumerate(adf_neighs[start_ind:end_ind])
                 for pair_neighs in itertools.combinations(neighs_one_atom,2) #]
                if (int(pair_neighs[0][1] < adf_cutoff) +\
                 int(pair_neighs[1][1] < adf_cutoff) +\
                 int(np.linalg.norm(pair_neighs[0][0].coords-pair_neighs[1][0].coords)<adf_cutoff)) >=2 ]


    return res_dist,res_angle
