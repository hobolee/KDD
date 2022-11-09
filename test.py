import os

import torch
import numpy as np
import argparse
import time
from util import *
from trainer import Trainer
from net import gtnet
import ast
from copy import deepcopy
# from tqdm import tqdm


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/TRAFFIC',help='data path')

parser.add_argument('--adj_data', type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')

parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=137,help='number of nodes/variables')
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
parser.add_argument('--obtain_relevant_data_methods', type=int, default=1, help="1: Author-proposed; 2: DTW; 3: Euclidean+DTW")

args = parser.parse_args()
torch.set_num_threads(3)

def main(runid):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');　　
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device)

    dataloader = load_dataset(args, args.data, args.batch_size, args.batch_size, args.batch_size)
    # this function is in util.py
    scaler = dataloader['scaler']

    args.num_nodes = dataloader['train_loader'].num_nodes
    print("Number of variables/nodes = ", args.num_nodes)
    print(args.data.strip().split('/'))
    dataset_name = args.data.strip().split('/')[-1].strip()
    print('data set: ',dataset_name)
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

    # no need for training, so skip line 144 to line 156
    args.path_model_save = "./saved_models/" + args.model_name + "/" + dataset_name + "/"
    import os
    if not os.path.exists(args.path_model_save):
        os.makedirs(args.path_model_save)

    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, predefined_A=predefined_A,
                    dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim,
                    dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True, obtain_relevant_data_methods = args.obtain_relevant_data_methods)

    print('The recpetive field size is', model.receptive_field)

    print(args)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(args, model, args.model_name, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler, device, args.cl)

    # skip the training part (line 182 to 256)

    # torch.load_state_dict() loads the parameters to the new model
    # ! important ! if testing on a GPU, delete: map_location=torch.device('cpu')
    engine.model.load_state_dict(torch.load(args.path_model_save + "exp" + str(args.expid) + "_" + str(runid) +".pth"))
    
    print("\nModel loaded\n")
    

    engine.model.eval()

    # Retrieval set as the training data
    if args.borrow_from_train_data:
        num_prots = math.floor( args.fraction_prots * dataloader["x_train"].shape[0] )  # defines the number of training instances to be used in retrieval
        args.num_prots = num_prots
        print("\nNumber of Prototypes = ", args.num_prots)

        instance_prototypes = obtain_instance_prototypes(args, dataloader["x_train"])


    print("\n Performing test set run. To perform the following inference on validation data, simply adjust 'y_test' to 'y_val' and 'test_loader' to 'val_loader', which\
            has been commented out for faster execution \n")

    random_node_split_avg_mae = []
    random_node_split_avg_rmse = []

    for split_run in range(args.random_node_idx_split_runs):
        if args.predefined_S:
            pass
        else:
            print("running on random node idx split ", split_run)

            if args.do_full_set_oracle:
                idx_current_nodes = np.arange( args.num_nodes, dtype=int ).reshape(-1)
                assert idx_current_nodes.shape[0] == args.num_nodes

            else:
                idx_current_nodes = get_node_random_idx_split(args, args.num_nodes, args.lower_limit_random_node_selections, args.upper_limit_random_node_selections)

            print("Number of nodes in current random split run = ", idx_current_nodes.shape)


        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 3)[:, 0, :, :]
        if not args.predefined_S:
            realy = realy[:, idx_current_nodes, :]


        for iter, (x, y) in enumerate (dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)

            if not args.predefined_S:
                if args.borrow_from_train_data:  
                    # TODO Here derive kNN Neighbors. 
                    # Our proposal modify how to retrieve the K-neighbourhoods. 
                    # Change this obtain_relavant_data_from_prototypes to something else. The definition of it is from util.py
                    #
                    # the code should be like follows. 
                    # Eventually we need to return neighbours in a new method
                    #
                    # orig_neighs = proposed.retrieve_neighbors(testx, instance_prototypes, idx_current_nodes....) # proposed.py 
                    testx, dist_prot, orig_neighs, neighbs_idxs, original_instances = obtain_relevant_data(args.obtain_relevant_data_methods, args, testx, instance_prototypes,
                                                                                            idx_current_nodes)  
                    # testx, dist_prot, orig_neighs, neighbs_idxs, original_instances = obtain_relevant_data_from_prototypes(args, testx, instance_prototypes,
                    #                                                                         idx_current_nodes)  
                else:
                    testx = zero_out_remaining_input(testx, idx_current_nodes, args.device) # Remove the data corresponding to the variables that are not a part of subset "S"

            with torch.no_grad():
                if args.predefined_S:
                    idx_current_nodes = None
                preds = engine.model(testx, args=args, mask_remaining=args.mask_remaining, test_idx_subset=idx_current_nodes)

                preds = preds.transpose(1, 3)
                preds = preds[:, 0, :, :]
                if not args.predefined_S:
                    preds = preds[:, idx_current_nodes, :]

                # aggregating from multiple neighbors
                if args.borrow_from_train_data:
                    _split_preds = []
                    b_size = preds.shape[0] // args.num_neighbors_borrow
                    for jj in range(args.num_neighbors_borrow):
                        start, end = jj*b_size, (jj+1)*b_size
                        _split_preds.append(preds[start:end].unsqueeze(1))
                    preds = torch.cat(_split_preds, dim=1)

                    
                    # TODO Here is the main proposal part, where the neighbors will be aggregated in the proposed methods. 
                    # Our method will modify how to aggregate the neighbors. 
                    # 
                    # the code should be like follows. 
                    # Eventually we need to return weights (Distance to Prot = dist_prot) from the neighbourhood of original data point. 
                    #
                    # dist_prot = proposed.aggregation(orig_neighs,....) # proposed.py 

                    if args.use_ewp:    
                        # TODO here corresponds to 5.2.3 of the paper
                        orig_neighs_forecasts = engine.model(orig_neighs, args=args, mask_remaining=args.mask_remaining, test_idx_subset=idx_current_nodes)
                        dist_prot, orig_neighs_forecasts_reshaped = obtain_discrepancy_from_neighs(preds, orig_neighs_forecasts, args, idx_current_nodes)
                        dist_prot = torch.nn.functional.softmax(-dist_prot / args.neighbor_temp, dim=-1).view(b_size, args.num_neighbors_borrow, 1, 1)

                    else:
                        # TODO here corresponds to 5.2.1 and 5.2.2
                        # 
                        # DDW scheme
                        # dist_prot = torch.nn.functional.softmax(-dist_prot / args.neighbor_temp, dim=-1).view(b_size, args.num_neighbors_borrow, 1, 1) 
                        #
                        # UW scheme

                        uniform_tensor = torch.FloatTensor( np.ones(args.num_neighbors_borrow) / args.num_neighbors_borrow ).to(args.device).unsqueeze(0).repeat(b_size, 1)
                        dist_prot = uniform_tensor.view(b_size, args.num_neighbors_borrow, 1, 1)

                    preds = torch.sum( dist_prot * preds , dim=1)

            outputs.append(preds)

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        mae = []
        rmse = []

        is_plotted = False

        if args.do_full_set_oracle:
            full_set_oracle_idx = get_node_random_idx_split(args, args.num_nodes, args.full_set_oracle_lower_limit, args.full_set_oracle_upper_limit)

            print("Number of nodes in current oracle random split = ", full_set_oracle_idx.shape)

        for i in range(args.seq_out_len):   # this computes the metrics for multiple horizons lengths, individually, starting from 0 to args.seq_out_len. Here in the paper setting, we predict the 12 stpes.
            pred = scaler.inverse_transform(yhat[:, :, i])
            real = realy[:, :, i]

            if args.do_full_set_oracle:
                pred = pred[:, full_set_oracle_idx]
                real = real[:, full_set_oracle_idx]
                assert pred.shape[1] == real.shape[1] == full_set_oracle_idx.shape[0]

            metrics = metric(pred, real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}'
            mae.append(metrics[0])
            rmse.append(metrics[1])

        random_node_split_avg_mae.append(mae)
        random_node_split_avg_rmse.append(rmse)

    return random_node_split_avg_mae, random_node_split_avg_rmse

if __name__ == "__main__":
    mae = []
    rmse = []

    for i in range(args.runs):
        m1, m2 = main(i)
        mae.extend(m1)
        rmse.extend(m2)

    mae = np.array(mae)
    rmse = np.array(rmse)

    amae = np.mean(mae,0)
    armse = np.mean(rmse,0)

    smae = np.std(mae,0)
    srmse = np.std(rmse,0)

    print('\n\nResults for multiple runs\n\n')
    for i in range(args.seq_out_len):
        print("horizon {:d} ; MAE = {:.4f} +- {:.4f} ; RMSE = {:.4f} +- {:.4f}".format(
              i+1, amae[i], smae[i], armse[i], srmse[i]))
