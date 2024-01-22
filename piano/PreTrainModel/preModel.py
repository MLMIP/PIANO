import torch
import torch.nn.functional as F
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from loss_f import sce_loss
from Encoder_Decoder import *
# from ..SelfAttentionPooling import AtomPooling
from functools import partial

def get_std(x):
    x = (x - torch.mean(x)) / torch.std(x)+1e-5
    return x


def clean_coords(x_init, flag):

    if flag == 'atom':
        x = torch.cat([x_init[:, :5], x_init[:, 8:]], 1)
        x = torch.cat([x[:, :4], get_std(x[:, 4:5]), x[:, 5:]], 1)
    else:
        x = torch.cat([x_init[:, :104], x_init[:, 107:]], 1)
        x21 = get_std(x[:, 42:43])
        x22 = get_std(x[:, 43:44])
        x23 = get_std(x[:, 44:45])
        x24 = get_std(x[:, 45:46])
        x25 = get_std(x[:, 46:47])
        x26 = get_std(x[:, 47:48])
        x27 = get_std(x[:, 48:49])
        x28 = get_std(x[:, 49:50])
        x29 = get_std(x[:, 53:54])
        x = torch.cat([x[:, :42], x21, x22, x23, x24, x25, x26, x27, x28, x[:, 50:53], x29, x[:, 54:]], 1)
    return x

# 123
class PreModel_dual_momentum_trans(torch.nn.Module):
    def __init__(self,
                 r_num_feature,
                 r_e_dim,
                 mid_dim=512,
                 output_dim=1024,
                 de_num_stacks=8,
                 de_num_layers=4,
                 hidden_size=64,
                 nheads=8,
                 d_output_dim=256,
                 mask_rate=0.3,
                 replace_rate=0.1,
                 loss_f='sce',
                 alpha_l=2,
                 ):
        super(PreModel_dual_momentum_trans, self).__init__()
        self._mask_rate = mask_rate
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        self.res_encoder_student = Encoder_transformer(r_num_feature, mid_dim, output_dim, heads=nheads, e_dim=r_e_dim, droupt=0.2)
        self.res_decoder_student = Encoder(output_dim, r_e_dim, hidden_size, 256,  de_num_stacks, de_num_layers)
        # self.res_decoder_student = Encoder_transformer(output_dim, hidden_size, 256, heads=nheads, e_dim=r_e_dim, droupt=0.2)
        self.lin1_student = torch.nn.Linear(256, 128)
        self.lin2_student = torch.nn.Linear(128, d_output_dim)
        # self.gn1 = GroupNorm(16, d_output_dim)


        self.enc_mask_token = torch.nn.Parameter(torch.zeros(1, r_num_feature))
        self.enc_mask_token_mid = torch.nn.Parameter(torch.zeros(1, output_dim))

        self.res_encoder_teacher = Encoder_transformer(r_num_feature, mid_dim, output_dim, heads=nheads, e_dim=r_e_dim, droupt=0.2)
        self.res_decoder_teacher = Encoder(output_dim, r_e_dim, hidden_size, 256,  de_num_stacks, de_num_layers)
        # self.res_decoder_teacher = Encoder_transformer(output_dim, hidden_size, 256, heads=nheads, e_dim=r_e_dim, droupt=0.2)
        self.lin1_teacher = torch.nn.Linear(256, 128)
        self.lin2_teacher = torch.nn.Linear(128, d_output_dim)
        # self.gn2 = GroupNorm(16, d_output_dim)
  
        for param_q, param_k in zip(
            self.res_encoder_student.parameters(), self.res_encoder_teacher.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.criterion = self.setup_loss_fn(loss_f, alpha_l)

        self.m = 0.999
        # self.lamda = torch.nn.Parameter(torch.tensor(1.))
        self.bn = BatchNorm1d(d_output_dim, affine=False)


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.res_encoder_student.parameters(), self.res_encoder_teacher.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, x_res, edge_index_res, edge_attr_res):

        x_res = clean_coords(x_res, 'res')
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder() 
        # mask branch
        pre_x_res = x_res
        use_x_res, (mask_nodes_res, keep_nodes_res) = \
            self.encoding_mask_noise(x_res.shape[0], x_res, self._mask_rate)

        use_x_res = self.res_encoder_student(use_x_res, edge_index_res, edge_attr_res)
        x = use_x_res.clone()
        # re-mask

        x[mask_nodes_res] = 0.0
        x[mask_nodes_res] += self.enc_mask_token_mid
        x = self.res_decoder_student(x, edge_index_res, edge_attr_res)
        x = self.lin1_student(x)
        x = F.tanh(x)
        x = self.lin2_student(x)
        s_x = x
        x_init = pre_x_res[mask_nodes_res]
        x = x[mask_nodes_res]
        loss_mse = self.criterion(x, x_init)

        # no mask branch
        t_res = x_res

        t_res = self.res_encoder_teacher(t_res, edge_index_res, edge_attr_res)
        t = t_res.clone()
        # re-mask

        t[mask_nodes_res] = 0.0
        t[mask_nodes_res] += self.enc_mask_token_mid
        t = self.res_decoder_teacher(t, edge_index_res, edge_attr_res)
        t = self.lin1_teacher(t)
        t = F.tanh(t)
        t = self.lin2_teacher(t)

        s_x = self.bn(s_x)
        t = self.bn(t)
        loss_corr = self.cross_corr(s_x, t)

        return loss_mse + loss_corr

    def encoding_mask_noise(self, num_nodes, x, mask_rate=0.3):
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        # print(num_nodes)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
        # print(out_x[token_nodes].size())
        out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)
    
    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = torch.nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        return criterion
    
    def cross_corr(self, z1, z2):
        # print(z1.size())
        N = z1.size()[0]
        D = z1.size()[1]
        c = z1.T @ z2 / N

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum() / D
        # print(on_diag)
        off_diag = self.off_diagonal(c).pow_(2).sum()
        # print(on_diag)
        loss = on_diag + 0.0051 * off_diag
        # loss = off_diag.sum()
        return loss

    def off_diagonal(self, x):
    # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def save_model(self):

        torch.save(self.res_encoder_student.state_dict(),
                   'res_encoder_student_paramst.pkl')
        torch.save(self.res_encoder_teacher.state_dict(),
                   'res_encoder_teacher_paramst.pkl')
