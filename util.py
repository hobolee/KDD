# Copyright 2022 Google LLC
# Copyright (c) 2020 Zonghan Wu

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

""" Primary utilities """
import pickle
import numpy as np
import os
import math
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import networkx as nx
import time
from sklearn.preprocessing import StandardScaler as SD # for standardizing the Data
from sklearn.decomposition import PCA # for PCA calculation

from fastdtw import fastdtw # dynamic time warping
from scipy.spatial.distance import euclidean

from tslearn.metrics import dtw
from tqdm import tqdm
def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.num_nodes = xs.shape[2]
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj



class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(args, dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    total_num_nodes = None
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

        print("Shape of ", category, " input = ", data['x_' + category].shape)

        total_num_nodes = data['x_' + category].shape[2]
        data['total_num_nodes'] = total_num_nodes

    if args.predefined_S:
        count = math.ceil(total_num_nodes * (args.predefined_S_frac / 100))
        oracle_idxs = np.random.choice( np.arange(total_num_nodes), size=count, replace=False )
        data['oracle_idxs'] = oracle_idxs
        for category in ['train', 'val', 'test']:
            data['x_' + category] = data['x_' + category][:, :, oracle_idxs, :]
            data['y_' + category] = data['y_' + category][:, :, oracle_idxs, :]

    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data



def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss), torch.mean(loss, dim=1)

def masked_rmse(preds, labels, null_val=np.nan):
    mse_loss, per_instance = masked_mse(preds=preds, labels=labels, null_val=null_val)
    return torch.sqrt(mse_loss), torch.sqrt(per_instance)


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss), torch.mean(loss, dim=1)


def metric(pred, real):
    mae, mae_per_instance = masked_mae(pred,real,0.0)[0].item(), masked_mae(pred,real,0.0)[1]
    rmse, rmse_per_instance = masked_rmse(pred,real,0.0)[0].item(), masked_rmse(pred,real,0.0)[1]
    return mae, rmse, mae_per_instance, rmse_per_instance



def get_node_random_idx_split(args, num_nodes, lb, ub):
    count_percent = np.random.choice( np.arange(lb, ub+1), size=1, replace=False )[0]
    count = math.ceil(num_nodes * (count_percent / 100))

    current_node_idxs = np.random.choice( np.arange(num_nodes), size=count, replace=False )
    return current_node_idxs


def zero_out_remaining_input(testx, idx_current_nodes, device):
    zero_val_mask = torch.ones_like(testx).bool()#.to(device)
    zero_val_mask[:, :, idx_current_nodes, :] = False
    inps = testx.masked_fill_(zero_val_mask, 0.0)
    return inps



def obtain_instance_prototypes(args, x_train):
    stride = x_train.shape[0] // args.num_prots
    prototypes = []
    for i in range(args.num_prots):
        a = x_train[ i*stride : (i+1)*stride ]
        prot = a[ np.random.randint(0, a.shape[0]) ] # randint will give a single interger here
        prot = np.expand_dims(prot, axis=0)
        prototypes.append(prot)

    prototypes = np.concatenate(tuple(prototypes), axis=0)
    print("\nShape of instance prototypes = ", prototypes.shape, "\n")
    prototypes = torch.FloatTensor(prototypes).to(args.device)
    return prototypes

def obtain_relevant_data(obtain_relevant_data_methods, args, testx, instance_prototypes,idx_current_nodes):
    if (obtain_relevant_data_methods == 1):
        return obtain_relevant_data_from_prototypes(args, testx, instance_prototypes, idx_current_nodes)
    elif (obtain_relevant_data_methods == 2):
        return obtain_relevant_data_with_DTW(args, testx, instance_prototypes, idx_current_nodes)
    elif (obtain_relevant_data_methods == 3):
        return obtain_relevant_data_with_euclidean_and_DTW(args, testx, instance_prototypes, idx_current_nodes)
    else:
        return 
def obtain_relevant_data_from_prototypes(args, testx, instance_prototypes, idx_current_nodes):
    rem_idx_subset = torch.LongTensor(np.setdiff1d(np.arange(args.num_nodes), idx_current_nodes)).to(args.device)
    idx_current_nodes = torch.LongTensor(idx_current_nodes).to(args.device)

    data_idx_train = instance_prototypes[:, :, idx_current_nodes, :].unsqueeze(0).repeat(testx.shape[0], 1, 1, 1, 1)
    a = testx[:, :, idx_current_nodes, :].transpose(3, 1).unsqueeze(1).repeat(1, instance_prototypes.shape[0], 1, 1, 1)
    assert data_idx_train.shape == a.shape

    raw_diff = data_idx_train - a
    diff = torch.pow( torch.absolute( raw_diff ), args.dist_exp_value ).view(testx.shape[0], instance_prototypes.shape[0], -1)
    diff = torch.mean(diff, dim=-1)

    min_values, topk_idxs = torch.topk(diff, args.num_neighbors_borrow, dim=-1, largest=False)
    b_size = testx.shape[0]
    original_instance = testx.clone()
    testx = testx.repeat(args.num_neighbors_borrow, 1, 1, 1)
    orig_neighs = []

    for j in range(args.num_neighbors_borrow):
        
        nbs = instance_prototypes[topk_idxs[:, j].view(-1)].transpose(3, 1)
        orig_neighs.append( nbs )
        desired_vals = nbs[:, :, rem_idx_subset, :]
        start, end = j*b_size, (j+1)*b_size
        _local = testx[start:end]
        _local[:, :, rem_idx_subset, :] = desired_vals
        testx[start:end] = _local

    orig_neighs = torch.cat(orig_neighs, dim=0)
    return testx, min_values, orig_neighs, topk_idxs, original_instance

# the function to replace the original KNN methods of retrieval with PCA methods
def obtain_relevant_data_with_euclidean_and_DTW(args, testx, instance_prototypes, idx_current_nodes):
    rem_idx_subset = torch.LongTensor(np.setdiff1d(np.arange(args.num_nodes), idx_current_nodes)).to(args.device)
    idx_current_nodes = torch.LongTensor(idx_current_nodes).to(args.device)

    data_idx_train = instance_prototypes[:, :, idx_current_nodes, :].unsqueeze(0).repeat(testx.shape[0], 1, 1, 1, 1)
    a = testx[:, :, idx_current_nodes, :].transpose(3, 1).unsqueeze(1).repeat(1, instance_prototypes.shape[0], 1, 1, 1)
    assert data_idx_train.shape == a.shape

    raw_diff = data_idx_train - a
    diff = torch.pow( torch.absolute( raw_diff ), args.dist_exp_value ).view(testx.shape[0], instance_prototypes.shape[0], -1)
    diff = torch.mean(diff, dim=-1)

    # here we use 10*args.num_neighbors_borrow as our first round selection
    min_values, topk_idxs = torch.topk(diff, 3*args.num_neighbors_borrow, dim=-1, largest=False)

    # after geting the top k diff as the first round rough filter, do DTW on it




    # data_idx_train: torch.Size([64, 36776, 12, 21, 1])
    data_train_instances = instance_prototypes[:, :, idx_current_nodes, :] #(36776, 12, 21, 1) 
    data_test_batch = testx[:, :, idx_current_nodes, :].transpose(3, 1)# one testing data batch with shape (64,12,21,1)

    full_batch_distance_list = [] # 64x (5*args.num_neighbors_borrow) list
    for k in range(3*args.num_neighbors_borrow):
        one_candidate_train_data_batch = data_train_instances[topk_idxs[:, k].view(-1)] # [64,12,21,1]
        # 64个testing data, 每个top 1 的vector
        # data_train_full = instance_prototypes[:, :, idx_current_nodes, :] # (36776,12,21,1)

        
        
        distance_list = [] # 64 x 1
        # print(data_test_batch.shape) #[64,12,21,1]

        for i in range(data_test_batch.shape[0]):

            distance = dtw(one_candidate_train_data_batch[i].squeeze(0).squeeze(2).cpu(), data_test_batch[i].squeeze(0).squeeze(2).cpu())
            # 对64个testing data，逐一计算其euclidean rank i 的distance
            distance_list.append(distance)
        full_batch_distance_list.append(np.array(distance_list)) # 1x64

    full_batch_distance_list = np.array(full_batch_distance_list).transpose()
    # print("shape of full_batch_distance_list:", full_batch_distance_list.shape) #(64, 25)

    min_values_dtw, topk_idxs_dtw = torch.topk(torch.from_numpy(full_batch_distance_list), args.num_neighbors_borrow, dim=-1, largest=False)
    # print("first topk_idxs_dtw:", topk_idxs_dtw[0]) # (64x5)
    # print(data_train_instances[topk_idxs[0][topk_idxs_dtw[0][0]]] ) #第一个testing data真正最接近的training data
    # print("min_values_dtw shape:",min_values_dtw.shape) # min_values_dtw shape: torch.Size([64, 5])
    # print(min_values_dtw)
    # print("2rd min_value for testing data2: ",min_values_dtw[63][4])

    # second_train = data_train_instances[topk_idxs[63][topk_idxs_dtw[63][4]]].squeeze(2)

    # print("second train",second_train) # first train torch.Size([12, 21])
    # print(dtw(second_train.cpu(), data_test_batch[63].squeeze(0).squeeze(2).cpu()))

    # final_index_list = topk_idxs[:,topk_idxs_dtw[0][0]] # 64x5

    final_full_list = [] # 64x5
    for j in range(full_batch_distance_list.shape[0]): #64
        final_list = [] # 1x5
        for i in range(args.num_neighbors_borrow):
            final_index = topk_idxs[j][topk_idxs_dtw[j][i]].cpu()
            final_list.append(final_index) 
        final_full_list.append(final_list)
    final_full_list_np = np.array(final_full_list)
    # print(data_train_instances[final_full_list[63][4]])

    # data_train_instances[topk_idxs[1][15]] 

    # data_train_instances[topk_idxs[:, k].view(-1)] # [64,12,21,1]
    # print("final_full_list_np[10]: ", final_full_list_np[10])
    # print("original full list[10]:", topk_idxs[10])

   
    b_size = testx.shape[0]
    original_instance = testx.clone()
    testx = testx.repeat(args.num_neighbors_borrow, 1, 1, 1)
    orig_neighs = []

    for j in range(args.num_neighbors_borrow):
         # instance_prototypes: (36776, 12, 137, 1) 
        nbs = instance_prototypes[torch.from_numpy(final_full_list_np)[:, j].view(-1)].transpose(3, 1)
        # print('nbs shape:', nbs.shape) # [64,1,137,12]
        orig_neighs.append( nbs )
        desired_vals = nbs[:, :, rem_idx_subset, :]
        start, end = j*b_size, (j+1)*b_size
        _local = testx[start:end]
        _local[:, :, rem_idx_subset, :] = desired_vals
        testx[start:end] = _local

    orig_neighs = torch.cat(orig_neighs, dim=0)
    return testx, min_values, orig_neighs, topk_idxs, original_instance


# the function to replace the original KNN methods of retrieval with dynamic time warping
def obtain_relevant_data_with_DTW(args, testx, instance_prototypes, idx_current_nodes):
    rem_idx_subset = torch.LongTensor(np.setdiff1d(np.arange(args.num_nodes), idx_current_nodes)).to(args.device)
    # index of variables remainded 

    idx_current_nodes = torch.LongTensor(idx_current_nodes).to(args.device)
    

    # since the input data are in shape like (36776, 12, 137, 1) : 1. number of instance 2. sequence length  3. The number of variables

    # unsqueeze: add a dimention at axis = 0, for batch_size = 64 (which is from testx.shape[0])
    # transpose(3,1): switch the axis 3 and 1
    # repeat: repeat t times at dimension s. 
    # Shape of instance prototypes =  (36776, 12, 137, 1)


    # try if we could calculate the dtw distance between a testing and a training data:

    data_train_full = instance_prototypes[:, :, idx_current_nodes, :] # (36776,12,21,1)
    # data_train_array = torch.split(data_train_full,1,dim=0) # split is too slow!
    # print("shape:",data_train_array)

    data_test_batch = testx[:, :, idx_current_nodes, :].transpose(3, 1)# one testing data batch with shape (64,12,21,1)

    distance_list_batch = []

    # for each testing data, we do a random selection with (10%) data from training set to do DTW


    

    np_data_train = data_train_full.cpu().numpy()
    # random index list for the 10% selected training data
    rand_list = np.random.choice(np_data_train.shape[0], int(np_data_train.shape[0]/20), replace=False) # [1x3677]
    # print("rand_list:",rand_list)
    # np form 10% training data
    np_data_train_rand = np_data_train[rand_list,:]

    # torch form 10% training data
    torch_data_train = torch.from_numpy(np_data_train_rand)
    # print("shape of 10% torch data:", torch_data_train.shape)
    # idx_array = np.random.randint(data_train_full.size(0), size = int(data_train_full.size(0)/10))
    # rand_data_train = data_train_full.cpu().numpy()[idx_array:] # (3677,12,21,1)

    for i in range(len(data_test_batch)):
        # random index list:

        distance_list = []
        
        for j in range(len(torch_data_train)):
            distance = dtw(torch_data_train[j].squeeze(0).squeeze(2).cpu(), data_test_batch[i].squeeze(0).squeeze(2).cpu()) # dtw distance between one testing data and all training data
            distance_list.append(distance)
        distance_list_batch.append(distance_list)

    # Convert the  list to numpy.ndarray then to tensor

    np_arr_distance_list_batch = np.array(distance_list_batch)
    tensor_distance_list_batch = torch.from_numpy(np_arr_distance_list_batch)
    # print("shape of distance: ",np_arr_distance_list_batch.shape) # (64, 36776)


    # here delete the last dimension 
    
    # data_idx_train = instance_prototypes[:, :, idx_current_nodes, :].unsqueeze(0).repeat(testx.shape[0], 1, 1, 1, 1).squeeze(4)
    # a = testx[:, :, idx_current_nodes, :].transpose(3, 1).unsqueeze(1).repeat(1, instance_prototypes.shape[0], 1, 1, 1).squeeze(4)
    # print('shape of a:', a.shape)

    # # try without batchsize
    # data_idx_train_wo_batch = instance_prototypes[:, :, idx_current_nodes, :].unsqueeze(0)
    # a_wo_batch = testx[:, :, idx_current_nodes, :].transpose(3, 1).unsqueeze(1)
    # assert data_idx_train.shape == a.shape
    # print("data_idx_train_wo_batch: ",data_idx_train_wo_batch.shape) #torch.Size([1, 36776, 12, 21, 1])
    # print('a_wo_batch:',a_wo_batch.shape) # torch.Size([64, 1, 12, 21, 1])
    # print(a_wo_batch[0]-a_wo_batch[1])
    # print('hh:', a_wo_batch[0])
    # assert data_idx_train_wo_batch.shape == a_wo_batch.shape
    # assert : if false than rise error, for SOLAR data, a.shape is: torch.Size([64, 36776, 12, 21, 1])

    # try PCA: do dimension reduction on the full variable training data. Find out how many significant dimension is in the remaining 15% and do distance calculation base on them.

    # print('idx_current_nodes: ', idx_current_nodes)

    # data_train_wo_b_switched = data_idx_train_wo_batch.cpu().numpy()
    # a_wo_b_switched = a_wo_batch.cpu().numpy()
    # # print("type:", type(a_type_switched))
    # # print("a_type_switched:", a_type_switched.shape)
    # distance_wb = dtw(data_train_wo_b_switched,a_wo_b_switched)

    # print('shape of a:', a.shape)
    # print('shape of data_idx_train:', data_idx_train.shape)
    # print('data_idx_train:',data_idx_train.cpu().numpy()[0])
    # distance, path = fastdtw(data_idx_train.cpu().numpy(), a.cpu().numpy(), dist=euclidean)
    # data_train_type_switched = data_idx_train.cpu()
    # a_type_switched = a.cpu()
    # # print("type:", type(a_type_switched))
    # # print("a_type_switched:", a_type_switched.shape)
    # distance = dtw(data_train_type_switched,a_type_switched)

    # print('distance:', distance)





    # raw_diff = data_idx_train - a
    # # dist_exp_value is the number of exponent value used for euclidean distance
    # # .view will reform the tensor to desired shape (here (testx.shape[0], instance_prototypes.shape[0], -1)) the last dimension will be automatically calculated
    # diff_test = torch.pow( torch.absolute( raw_diff ), args.dist_exp_value )
    # # print(' diff_test:', diff_test.shape) #([64, 36776, 12, 21, 1])
    # diff = torch.pow( torch.absolute( raw_diff ), args.dist_exp_value ).view(testx.shape[0], instance_prototypes.shape[0], -1)
    # # print('shape of diff before mean: ',diff.shape) # torch.Size([64, 36776, 252])
    # diff = torch.mean(diff, dim=-1)

    # print('shape of diff: ',diff.shape) # torch.Size([64, 36776])


    # number of neighbors to borrow from, during aggregation
    min_values, topk_idxs = torch.topk(tensor_distance_list_batch, args.num_neighbors_borrow, dim=-1, largest=False)
    # print("topk_idxs: ",topk_idxs.shape) #torch.Size([64, 5])
    # print("min_values", min_values.shape) #torch.Size([64, 5])
    b_size = testx.shape[0] # batch_size
    original_instance = testx.clone()
    testx = testx.repeat(args.num_neighbors_borrow, 1, 1, 1)
    orig_neighs = []

    # prepare the current 10% instance data then use it to generate nbs
    # torch_train_data_fullvar = instance_prototypes[]
    for j in range(args.num_neighbors_borrow):
        # here need to transform the nds index
        # print("index:", topk_idxs[:, j].view(-1)) # a 1x64 tensor
        # print("modified_index:", rand_list[topk_idxs[:, j].view(-1)])

        nbs = instance_prototypes[rand_list[topk_idxs[:, j].view(-1)]].transpose(3, 1) # get the top-k nearest neighbours one by one
        # print('nbs shape:', nbs.shape) # [64,1,137,12]
        orig_neighs.append( nbs )
        desired_vals = nbs[:, :, rem_idx_subset, :]
        start, end = j*b_size, (j+1)*b_size
        # start and end position of the current neighbor to be placed, than copy it to testx
        _local = testx[start:end]
        _local[:, :, rem_idx_subset, :] = desired_vals
        testx[start:end] = _local

    orig_neighs = torch.cat(orig_neighs, dim=0)
    return testx, min_values, orig_neighs, topk_idxs, original_instance

def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    z = torch.tensor((x-mean)/std,dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


def obtain_discrepancy_from_neighs(preds, orig_neighs_forecasts, args, idx_current_nodes):
    orig_neighs_forecasts = orig_neighs_forecasts.transpose(1, 3)
    orig_neighs_forecasts = orig_neighs_forecasts[:, 0, :, :]
    orig_neighs_forecasts = orig_neighs_forecasts[:, idx_current_nodes, :]

    orig_neighs_forecasts = torch.chunk(orig_neighs_forecasts, args.num_neighbors_borrow)
    orig_neighs_forecasts = [ a.unsqueeze(1) for a in orig_neighs_forecasts ]
    orig_neighs_forecasts = torch.cat(orig_neighs_forecasts, dim=1)

    len_tensor = torch.FloatTensor( np.arange(1, preds.shape[-1]+1) ).to(args.device).view(1, 1, 1, -1).repeat(
                              preds.shape[0], args.num_neighbors_borrow, preds.shape[2], 1) # tensor of time step indexes
    distance = torch.absolute( (preds - orig_neighs_forecasts) / len_tensor ).view(preds.shape[0], args.num_neighbors_borrow, -1)
    distance = torch.mean(distance, dim=-1)
    return distance, orig_neighs_forecasts
