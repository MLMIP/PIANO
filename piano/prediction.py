import numpy as np
import argparse
import os
import random
import math

import pandas as pd
import torch
import sys
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DataListLoader
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import scipy
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Model import RegressionModel
import warnings
warnings.filterwarnings("ignore")
data_path = 'piano/Data/'
data_seq_name = 'DSeqData' 

def data_create_loader_regression(data_name):

    atom_list = []
    res_list = []
    seq = []
    seq_mut = []
    j = 0

    data_list = 'pred_data_list.npy'  # Set a list of data sample names you want to predict（i.e. 1a22_CA221K）
    data_list = list(np.load(data_list))
    for e in data_list:

        pa = data_path + data_name + '/' + e + '_p_atom_A.pt'
        se1 = data_path + data_name + '/' + e + '_atom_E.pt'
        pr = data_path+ data_name + '/' + e + '_p_res_A.pt'
        se2 = data_path+ data_name + '/' + e + '_res_E.pt'
        index_d = data_path + data_name + '/' + e + '_index_dict.pt'
        pattr = data_path + data_name + '/' + e + '_p_atom_attr.pt'
        prttr = data_path + data_name + '/' + e + '_p_res_attr.pt'

        s_A = data_path+data_seq_name+'/'+e+'_p_res_A.pt'
        s_e = data_path+ data_seq_name + '/' + e + '_res_E.pt'
        s_A_mut = data_path+data_seq_name+'/'+e+'_mut_p_res_A.pt'
        s_e_mut = data_path+ data_seq_name + '/' + e + '_mut_res_E.pt'
        if os.path.exists(pa) and os.path.exists(index_d) and os.path.exists(s_A):
            index_list = [0]
            pre = 0
            index_dict = torch.load(index_d)

            for key in index_dict.keys():
                if index_dict[key] == 0:
                    continue
                index_list.append(pre + index_dict[key])
                pre += index_dict[key]
            
            a11 = torch.load(pa)
            a22 = torch.load(pr)
            e1 = torch.load(se1)
            e2 = torch.load(se2)
            sa_a = torch.load(pattr)
            sr_a = torch.load(prttr)
            s_A_a = torch.load(s_A)
            s_e_a = torch.load(s_e)
            s_A_mut_a = torch.load(s_A_mut)
            s_e_mut_a = torch.load(s_e_mut)

            # if len(index_list)-1 != len(a22.tolist()):
            #     continue
            # atom_list.append(Data(x=a11, edge_index=e1, edge_attr=sa_a, y=torch.tensor(index_list)))
            
            atom_list.append(Data(x=a11, edge_index=e1, edge_attr=sa_a, y=torch.tensor(index_list)))
            res_list.append(Data(x=a22, edge_index=e2, edge_attr=sr_a, y=torch.tensor(1)))
            seq.append(Data(x=s_A_a, edge_index=s_e_a, y=torch.tensor(1)))
            seq_mut.append(Data(x=s_A_mut_a, edge_index=s_e_mut_a, y=torch.tensor(1)))
            j += 1
            if j >= 100000:
                break
    atom_loader = DataLoader(dataset=atom_list, worker_init_fn=worker_init_fn)
    res_loader = DataLoader(dataset=res_list, worker_init_fn=worker_init_fn)
    seq_loader = DataLoader(dataset=seq, worker_init_fn=worker_init_fn)
    seq_mut_loader = DataLoader(dataset=seq_mut, worker_init_fn=worker_init_fn)

    return atom_loader, res_loader, j, seq_loader, seq_mut_loader

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  
    torch.backends.cudnn.benchmark = False  

def worker_init_fn(worked_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_y(path):
    f = pd.read_csv(path, sep=',')
    y = []
    for i in range(f.shape[0]):
        y.append(torch.tensor(float(f['ddG'][i])))
    return y


def build_args():
    parser = argparse.ArgumentParser(description='Pretrain')
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=300,
                        help="number of training epochs")

    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--a_num_feature", type=int, default=6)
    parser.add_argument("--r_num_feature", type=int, default=105)
    parser.add_argument("--a_e_dim", type=int, default=3)
    parser.add_argument("--r_e_dim", type=int, default=3)
    parser.add_argument("--output_dim", type=int, default=1024)
    parser.add_argument("--mid_dim", type=int, default=512)
    parser.add_argument("--d_output_dim", type=int, default=1600)
    parser.add_argument("--num_stacks", type=int, default=2,
                        help="number of stacks")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of layers")
    parser.add_argument("--num_hidden", type=int, default=64,
                        help="number of hidden units")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="weight decay")
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--re_mask_rate", type=float, default=0.5)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument("--loss_f", type=str, default="mse")
    parser.add_argument("--alpha_l", type=float, default=2, help="`pow`coefficient for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")

    parser.add_argument("--batch_size", type=int, default=1)

    argss = parser.parse_args()
    return argss

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data(batchs):
    tar = [data.y for data in batchs]
    x = [data.x for data in batchs]
    e = [data.edge_index for data in batchs]
    et = [data.edge_attr for data in batchs]
    et = torch.cat(et)
    x = torch.cat(x)
    e = torch.cat(e)
    return x, e, tar, et

def get_data_seq(batchs):
    x = [data.x for data in batchs]
    e = [data.edge_index for data in batchs]
    x = torch.cat(x)
    e = torch.cat(e)

    return x, e

def get_batch_data(batch):
    return batch.x, batch.edge_index, batch.y, batch.edge_attr

def stdd(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x*1.6-0.8


def predict(model, loader_atom, loader_res, loader_seq, loader_seq_mut):
    out_list = []
    loss_list = []
    count = 0
    for batch1, batch2, batch3, batch4 in zip(loader_atom, loader_res, loader_seq, loader_seq_mut):
        mut_num_list = []
        x_atom, e_atom, index, et1 = get_batch_data(batch1)
        x_res, e_res, targ, et2 = get_batch_data(batch2)
        x_res_arr = np.array(x_res[:, 40:41])
        for i in range(len(x_res_arr)):
            if x_res_arr[i][0] != 0:
                mut_num_list.append(i)
        x_seq, e_seq, _, _ = get_batch_data(batch3)
        x_seq_mut, e_seq_mut, _, _ = get_batch_data(batch4)
  

        et1 = et1.to(device)
        et2 = et2.to(device)
        x_atom = x_atom.to(device)
        e_atom = e_atom.to(device)
        x_res = x_res.to(device)
        e_res = e_res.to(device)

        x_seq = x_seq.to(device)
        x_seq_mut = x_seq_mut.to(device)
        
        pout = model(x_atom, e_atom, et1, x_res, e_res, et2, [index], mut_num_list, x_seq, x_seq_mut)

        pout = pout.to(device0)
        pout = pout.detach().numpy()

        pout = pout.tolist()[0]
        out_list.append(pout[0])

    # print(count)
    count += 1
    return out_list



args = build_args()
seeds = args.seeds
set_seed(123)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device0 = torch.device("cpu")

train_atom_loader, train_res_loader, ava_sample, train_seq_loader, train_seq_mut_loader = data_create_loader_regression('DStructData') 

model = RegressionModel(
    a_num_feature=args.a_num_feature,
    r_num_feature=args.r_num_feature,
    a_e_dim=args.a_e_dim,
    r_e_dim=args.r_e_dim,
    mid_dim=args.mid_dim,
    output_dim=args.output_dim,
    num_stacks=args.num_stacks,
    num_layers=args.num_layers,
    hidden_size=args.num_hidden,
    nheads=args.num_heads,
    d_output_dim=args.d_output_dim,

)
model.load_state_dict(torch.load('piano/Data/model_params/piano_model.pkl'))

model = model.to(device)

model.eval()
out_p = predict(model, train_atom_loader, train_res_loader, train_seq_loader, train_seq_mut_loader)
pred_data_list = 'pred_data_list.npy'  # Set a list of data sample names you want to predict（i.e. 1a22_CA221K）
pred_data_list = list(np.load(pred_data_list))

with open('PredictedResults.txt', 'w') as f:
    for k in range(len(out_p)):
        f.write(pred_data_list[k]+':'+str(out_p[k])+'\n')


