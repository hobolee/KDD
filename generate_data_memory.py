import os
import random

import torch
import numpy as np
import argparse
import time
from util import *
from trainer import Trainer
from net import gtnet
import ast
from copy import deepcopy
from train_multi_step import str_to_bool

parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')

parser.add_argument('--adj_data', type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')

parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
parser.add_argument('--end_channels',type=int,default=128,help='end channels')


parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')

parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=float, default=5.0, help='clip')
parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')


parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--path_model_save', type=str, default=None)
parser.add_argument('--expid',type=int,default=1,help='experiment id')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')

parser.add_argument('--runs',type=int,default=10, help='number of runs')

parser.add_argument('--random_node_idx_split_runs', type=int, default=100, help='number of random node/variable split runs')
parser.add_argument('--lower_limit_random_node_selections', type=int, default=15, help='lower limit percent value for number of nodes in any given split')
parser.add_argument('--upper_limit_random_node_selections', type=int, default=15, help='upper limit percent value for number of nodes in any given split')

parser.add_argument('--model_name', type=str, default='mtgnn')

parser.add_argument('--mask_remaining', type=str_to_bool, default=False, help='the partial setting, subset S')

parser.add_argument('--predefined_S', type=str_to_bool, default=False, help='whether to use subset S selected apriori')
parser.add_argument('--predefined_S_frac', type=int, default=15, help='percent of nodes in subset S selected apriori setting')
parser.add_argument('--adj_identity_train_test', type=str_to_bool, default=False, help='whether to use identity matrix as adjacency during training and testing')

parser.add_argument('--do_full_set_oracle', type=str_to_bool, default=False, help='the oracle setting, where we have entire data for training and \
                            testing, but while computing the error metrics, we do on the subset S')
parser.add_argument('--full_set_oracle_lower_limit', type=int, default=15, help='percent of nodes in this setting')
parser.add_argument('--full_set_oracle_upper_limit', type=int, default=15, help='percent of nodes in this setting')

parser.add_argument('--borrow_from_train_data', type=str_to_bool, default=False, help="the Retrieval solution")
parser.add_argument('--num_neighbors_borrow', type=int, default=5, help="number of neighbors to borrow from, during aggregation")
parser.add_argument('--dist_exp_value', type=float, default=0.5, help="the exponent value")
parser.add_argument('--neighbor_temp', type=float, default=0.1, help="the temperature paramter")
parser.add_argument('--use_ewp', type=str_to_bool, default=False, help="whether to use ensemble weight predictor, ie, FDW")

parser.add_argument('--fraction_prots', type=float, default=1.0, help="fraction of the training data to be used as the Retrieval Set")

args = parser.parse_args()
torch.set_num_threads(3)


def main(runid):
    device = torch.device(args.device)
    dataloader = load_dataset(args, args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    args.num_nodes = dataloader['train_loader'].num_nodes
    print("Number of variables/nodes = ", args.num_nodes)

    dataset_name = args.data.strip().split('/')[-1].strip()

    if dataset_name == "METR-LA":
        args.in_dim = 2
    else:
        args.in_dim = 1

    args.runid = runid

    if dataset_name == "METR-LA":
        args.adj_data = "data/sensor_graph/adj_mx.pkl"

        predefined_A = load_adj(args.adj_data)
        predefined_A = torch.tensor(predefined_A) - torch.eye(dataloader['total_num_nodes'])
        predefined_A = predefined_A.to(device)

    else:
        predefined_A = None

    if args.adj_identity_train_test:
        if predefined_A is not None:
            print("\nUsing identity matrix during training as well as testing\n")
            predefined_A = torch.eye(predefined_A.shape[0]).to(args.device)

    if args.predefined_S and predefined_A is not None:
        oracle_idxs = dataloader['oracle_idxs']
        oracle_idxs = torch.tensor(oracle_idxs).to(args.device)
        predefined_A = predefined_A[oracle_idxs, :]
        predefined_A = predefined_A[:, oracle_idxs]
        assert predefined_A.shape[0] == predefined_A.shape[1] == oracle_idxs.shape[0]
        print("\nAdjacency matrix corresponding to oracle idxs obtained\n")

    # Retrieval set as the training data
    if args.borrow_from_train_data:
        num_prots = math.floor( args.fraction_prots * dataloader["x_train"].shape[0] )  # defines the number of training instances to be used in retrieval
        args.num_prots = num_prots
        print("\nNumber of Prototypes = ", args.num_prots)

        instance_prototypes = obtain_instance_prototypes(args, dataloader["x_train"])

    random_node_split_avg_mae = []
    random_node_split_avg_rmse = []

    for split_run in range(args.random_node_idx_split_runs):
        if args.predefined_S:
            pass
        else:
            print("running on random node idx split ", split_run)

            if args.do_full_set_oracle:
                idx_current_nodes = np.arange(args.num_nodes, dtype=int).reshape(-1)
                assert idx_current_nodes.shape[0] == args.num_nodes

            else:
                idx_current_nodes = get_node_random_idx_split(args, args.num_nodes, args.lower_limit_random_node_selections, args.upper_limit_random_node_selections)

            print("Number of nodes in current random split run = ", idx_current_nodes.shape)

        data_indice = []
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            p = random.random()
            if iter > 0 and p > 0.01:
                continue
            data_indice.append(iter)
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            if not args.predefined_S:
                if args.borrow_from_train_data:
                    testx, dist_prot, orig_neighs, neighbs_idxs, original_instances = obtain_relevant_data_from_prototypes(args, testx, instance_prototypes,
                                                                                            idx_current_nodes)
                else:
                    testx = zero_out_remaining_input(testx, idx_current_nodes, args.device) # Remove the data corresponding to the variables that are not a part of subset "S"

            if iter == 0:
                memory_train = testx
                memory_train_label = torch.from_numpy(x)
            else:
                memory_train = torch.cat((memory_train, testx))
                memory_train_label = torch.cat((memory_train_label, torch.from_numpy(x)))

        print("random nodes: ", idx_current_nodes)
        with open("./data/memory/random nodes log", "a") as f:
            f.write(str(idx_current_nodes))
        print("data indices: ", data_indice)
        with open("./data/memory/data indices log", "a") as f:
            f.write(str(data_indice))
    return memory_train, memory_train_label



if __name__ == "__main__":
    for i in range(10):
        memory_train, memory_train_label = main(i)
        if i == 0:
            memory_train_con = memory_train
            memory_train_label_con = memory_train_label
        else:
            memory_train_con = torch.cat((memory_train_con, memory_train))
            memory_train_label_con = torch.cat((memory_train_label_con, memory_train_label))

    torch.save(memory_train_con, f'./data/memory/memory_test.pt')
    torch.save(memory_train_label_con, f'./data/memory/memory_test_label.pt')
    print("Saving done.")