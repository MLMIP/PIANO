import numpy as np
import torch.nn
import torch_geometric.nn as gnn
import os,sys
import torch.nn.functional as F
import torch.nn.init as init
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from EncoderDecoder import *
from functools import partial
device0 = torch.device("cpu")


def get_std(x, num=-1):
    mean_list = [2.1175, 1.0587, 0.7058, 0.5294, 0.4235, 0.3529, 0.3025, 0.2391, 4.3047] 
    sigma_list = [1.0301, 1.0833, 1.0182, 0.8945, 0.7883, 0.6952, 0.6194, 0.0595, 44.9736]
    x = (x - torch.tensor(mean_list[num])) / torch.tensor(sigma_list[num])
    return x


def get_std_init(x, num):
    x = (x - torch.mean(x)) / torch.std(x)
    return x
def clean_coords(x_init, flag, isdual):
    if isdual:
        if flag == 'atom':
            x = torch.cat([x_init[:, :4], x_init[:, 7:]], 1)
            # x = torch.cat([x[:, :4], get_std(x[:, 4:5]), x[:, 5:]], 1)
        else:


            x = torch.cat([x_init[:, :79], x_init[:, 82:]], 1)
            x21 = get_std(x[:, 22:23], 0)
            x22 = get_std(x[:, 23:24], 1)
            x23 = get_std(x[:, 24:25], 2)
            x24 = get_std(x[:, 25:26], 3)
            x25 = get_std(x[:, 26:27], 4)
            x26 = get_std(x[:, 27:28], 5)
            x27 = get_std(x[:, 28:29], 6)

            x = torch.cat([x[:, :22], x21, x22, x23, x24, x25, x26, x27, x[:, 29:]], 1)
    else:
        if flag == 'atom':

            x = torch.cat([x_init[:, :5], x_init[:, 8:]], 1)
            x = torch.cat([x[:, :4], get_std_init(x[:, 4:5], 0), x[:, 5:]], 1)
        else:

            x = torch.cat([x_init[:, :104], x_init[:, 107:]], 1)
            x21 = get_std_init(x[:, 42:43], 0)
            x22 = get_std_init(x[:, 43:44], 1)
            x23 = get_std_init(x[:, 44:45], 2)
            x24 = get_std_init(x[:, 45:46], 3)
            x25 = get_std_init(x[:, 46:47], 4)
            x26 = get_std_init(x[:, 47:48], 5)
            x27 = get_std_init(x[:, 48:49], 6)
            x28 = get_std_init(x[:, 49:50], 7)
            x29 = get_std_init(x[:, 53:54], 8)
            x = torch.cat([x[:, :42], x21, x22, x23, x24, x25, x26, x27, x28, x[:, 50:53], x29, x[:, 54:]], 1)
    return x


def get_edge_std(x_init, flag):
    if flag == 'atom':
        x1 = get_std(x_init[:, 0:1])
        x2 = get_std(x_init[:, 1:2])
        x3 = get_std(x_init[:, 2:])
        x = torch.cat([x1, x2, x3], 1)
    else:
        x1 = get_std(x_init[:, 0:1])
        x2 = get_std(x_init[:, 1:2])
        x3 = get_std(x_init[:, 2:])
        x = torch.cat([x1, x2, x3], 1)
    return x


class RegressionModel(torch.nn.Module):
    def __init__(self,
                 a_num_feature,
                 r_num_feature,
                 a_e_dim,
                 r_e_dim,
                 mid_dim=512,
                 output_dim=1024,
                 num_stacks=2,
                 num_layers=2,
                 hidden_size=64,
                 nheads=8,
                 d_output_dim=256,
                 ):
        super(RegressionModel, self).__init__()

        self.atom_encoder = Encoder(a_num_feature, a_e_dim, int(hidden_size/2), hidden_size, num_stacks, num_layers)
        self.res_encoder = Encoder_transformer(r_num_feature, mid_dim, output_dim, nheads, e_dim=r_e_dim, droupt=0.2) # 512
        self.seq_encoder = SeqEncoder()
        self.seq_encoder_mut = SeqEncoder()
        # self.res_encoder.load_state_dict(torch.load('piano/Data/model_params/res_encoder_student_paramst.pkl'))
       
        self.pool_mid = AtomPooling_SA(64, 64)

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)

        self.FC1 = torch.nn.Linear(d_output_dim, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.FC2 = torch.nn.Linear(512, 256)
        self.FC3 = torch.nn.Linear(256, 1)

    def forward(self, x_atom, edge_index_atom, edge_attr_atom, x_res, edge_index_res, edge_attr_res,
                index, mut_list, x_seq, x_seq_mut):

        x_atom = clean_coords(x_atom, 'atom', False)
        x_res = clean_coords(x_res, 'res', False)

        x_atom = self.atom_encoder(x_atom, edge_index_atom, edge_attr_atom)
        x_res = self.res_encoder(x_res, edge_index_res, edge_attr_res)

        x_seq = self.seq_encoder(x_seq)
        x_seq_mut = self.seq_encoder_mut(x_seq_mut)

        x_atom_m = x_atom
        for i in range(len(index)):
            p_x_atom = self.pool_mid(x_atom, index[i])
            if i == 0:
                x_atom_m = p_x_atom
            else:
                x_atom_m = torch.cat((x_atom_m,  p_x_atom), dim=0)

        x = torch.cat((x_atom_m, x_res), dim=1)
        # print(x.size())
        # print(mut_list)
        x_mut = x[mut_list[0], :]
        x_mut = x_mut.view(1, 1088)
        for i in range(1, len(mut_list)):
            pre_x = x[mut_list[i], :]
            pre_x = pre_x.view(1, 1088)
            x_mut = torch.cat([x_mut, pre_x], 0)

        if len(mut_list) != 1:
            x_mut = self.avgpool(x_mut.T)
        x = x_mut
        x = x.view(1, 1088)

        x_seq = x_seq - x_seq_mut
        x = torch.cat([x, x_seq], 1)

        x = self.FC1(x)
        x = F.tanh(x)
        x = self.FC2(x)
        # x = F.tanh(x)
        x = self.FC3(x)

        return x

