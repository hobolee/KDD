import os

import torch
import numpy as np
import argparse
import time
from util import *
from trainer import Trainer, Trainer_memory
from net import gtnet, memory, s2s_gtnet
import ast
from copy import deepcopy
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convgru_encoder_params, convgru_decoder_params


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
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

    args.path_model_save = "./saved_models/" + args.model_name + "/" + dataset_name + "/"
    import os
    if not os.path.exists(args.path_model_save):
        os.makedirs(args.path_model_save)

parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/SOLAR',help='data path')

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

args = parser.parse_args()
torch.set_num_threads(3)

def main(runid):
    device = torch.device(args.device)
    dataloader = load_dataset_memory(args, args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    # args.num_nodes = dataloader['train_loader'].num_nodes
    # print("Number of variables/nodes = ", args.num_nodes)

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

    args.path_model_save = "./saved_models/" + args.model_name + "/" + dataset_name + "/"
    import os
    if not os.path.exists(args.path_model_save):
        os.makedirs(args.path_model_save)

    # model = memory(args.num_nodes)
    encoder = Encoder(convgru_encoder_params[0], convgru_encoder_params[1])
    decoder = Decoder(convgru_decoder_params[0], convgru_decoder_params[1])
    model = ED(encoder, decoder)

    print(args)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer_memory(args, model, args.model_name, args.learning_rate, args.weight_decay, args.clip, args.step_size1, scaler, device, args.cl)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    minl = 1e5

    for i in range(1, args.epochs+1):
        train_loss = []
        train_rmse = []
        t1 = time.time()
        # dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            # trainx = torch.squeeze(trainx[:, 0, :, -1])
            trainy = torch.Tensor(y).to(device)
            # trainy = torch.squeeze(trainy[:, -1, :, 0])
            if iter % args.step_size2 == 0:
                perm = np.random.permutation(range(args.num_nodes))

            trainx = trainx.unsqueeze(1)
            trainy = trainy.unsqueeze(1)
            trainx = torch.permute(trainx, [0, 3, 1, 2])
            trainy = torch.permute(trainy, [0, 2, 1, 3])
            metrics = engine.train(args, trainx, trainy, i, dataloader['train_loader'].num_batch, iter)

            train_loss.append(metrics[0])
            train_rmse.append(metrics[1])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_rmse[-1]), flush=True)

        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_rmse = []

        s1 = time.time()

        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testy = torch.Tensor(y).to(device)
            testx = testx.unsqueeze(1)
            testy = testy.unsqueeze(1)
            testx = torch.permute(testx, [0, 3, 1, 2])
            testy = torch.permute(testy, [0, 2, 1, 3])
            metrics = engine.eval(args, testx, testy)
            valid_loss.append(metrics[0])
            valid_rmse.append(metrics[1])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(np.array(train_loss))
        # mtrain_rmse = np.mean(np.array(train_rmse))

        mvalid_loss = np.mean(np.array(valid_loss))
        # mvalid_rmse = np.mean(np.array(valid_rmse))
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_loss, mvalid_loss, mvalid_loss, (t2 - t1)), flush=True)
        torch.save(engine.model.state_dict(),
                   args.path_model_save + "exp" + str(args.expid) + "_" + str(runid) + ".pth")

if __name__ == "__main__":
    main(0)
