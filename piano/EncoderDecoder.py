import os, sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from torch.nn import BatchNorm1d, GroupNorm
from torch_geometric.nn import BatchNorm, GATConv, TransformerConv
from torch_geometric.typing import OptTensor
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from SEM_ARMA import ARMAConv
from SelfAttentionPooling import SelfAttentionPooling


class Encoder(torch.nn.Module):
    def __init__(self, num_feature, e_dim, mid_dim=512, output_dim=1024, num_stacks=8, num_layers=4):
        super(Encoder, self).__init__()
        self.conv1 = ARMAConv(num_feature, mid_dim, num_stacks, num_layers, edge_dim=e_dim, act=None)
        self.bn1 = GroupNorm(16, mid_dim)

        self.conv2 = ARMAConv(mid_dim, output_dim, num_stacks, num_layers, edge_dim=e_dim, act=None)
        self.bn2 = GroupNorm(16, output_dim)
        self.activation1 = torch.nn.PReLU(mid_dim)
        self.activation2 = torch.nn.PReLU(output_dim)

    def forward(self, x, edge_index, edge_attr: OptTensor = None):

        x = self.conv1(x, edge_index.T, edge_attr)
        x = self.bn1(x)
        x = F.tanh(x)

        x = self.conv2(x, edge_index.T, edge_attr)
        x = self.bn2(x)
        x = F.tanh(x)

        return x
class Encoder_transformer(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, heads, e_dim, droupt=0.2):
        super(Encoder_transformer, self).__init__()
        self.conv1 = TransformerConv(in_channels, mid_channels, heads=heads, edge_dim=e_dim, dropout=droupt, concat=False)
        self.bn1 = GroupNorm(16, mid_channels)

        self.conv2 = TransformerConv(mid_channels, out_channels, heads=heads, edge_dim=e_dim, dropout=droupt, concat=False)
        self.bn2 = GroupNorm(16, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index.T, edge_attr)
        x = self.bn1(x)
        x = F.tanh(x)

        x = self.conv2(x, edge_index.T, edge_attr)
        x = self.bn2(x)
        x = F.tanh(x)

        return x 


class AtomPooling_SA(torch.nn.Module):
    def __init__(self, input_dim, out_dim):
        super(AtomPooling_SA, self).__init__()
        self.pool = SelfAttentionPooling(input_dim, out_dim)

    def forward(self, atom_features, index_list):
        x = []
        index = index_list.tolist()
        for i in range(len(index)-1):
            x.append(self.pool(atom_features[index[i]:index[i+1], :]))
        y = torch.cat((x[0], x[1]), dim=0)
        for j in range(2, len(x)):
            y = torch.cat((y, x[j]), dim=0)
        return y

class CNNConvLayersPre(nn.Module):
    def __init__(self, kernels):
        super(CNNConvLayersPre,self).__init__()
        
        self.cnn_conv = nn.ModuleList()
        for l in range(len(kernels)):
            self.cnn_conv.append(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernels[l],1), padding=(kernels[l]//2,0)))
            self.cnn_conv.append(nn.Tanh())

    def forward(self, x):
        for l , m in enumerate(self.cnn_conv):
            x = m(x)
        return x
    
class CNNConvLayersMid(nn.Module):
    def __init__(self, channels, kernels):
        super(CNNConvLayersMid,self).__init__()
        
        self.cnn_conv = nn.ModuleList()
        self.cnn_conv.append(nn.Conv2d(in_channels=1, out_channels=channels[0], kernel_size=(kernels[0],1), padding=(kernels[0]//2,0)))
        self.cnn_conv.append(nn.Tanh())
        self.cnn_conv.append(nn.AvgPool2d(kernel_size=(kernels[0],1), stride=(2,1), padding=(kernels[0]//2,0)))
        
        for l in range(1, len(kernels)-1):
            self.cnn_conv.append(nn.Conv2d(in_channels=channels[l-1], out_channels=channels[l], kernel_size=(kernels[l],1), padding=(kernels[l]//2,0)))
            self.cnn_conv.append(nn.Tanh())
            self.cnn_conv.append(nn.AvgPool2d(kernel_size=(kernels[l],1), stride=(2,1), padding=(kernels[l]//2,0)))
            
    def forward(self, x):
        for l , m in enumerate(self.cnn_conv):
            x = m(x)
        return x
    
class CNNConvLayersLast(nn.Module):
    def __init__(self, channels, kernels, feature_dim):
        super(CNNConvLayersLast,self).__init__()
        
        self.cnn_conv = nn.ModuleList()
        self.cnn_conv.append(nn.Conv2d(in_channels=channels[-2], out_channels=channels[-1], kernel_size=(kernels[-1],feature_dim), padding=(kernels[-1]//2,0)))
        self.cnn_conv.append(nn.Tanh())
        self.cnn_conv.append(nn.AdaptiveAvgPool2d(output_size=(1,1)))
        
    def forward(self, x):
        for l , m in enumerate(self.cnn_conv):
            x = m(x)
        shapes = x.data.shape
        x = x.view(1, 512)
        return x

class SeqEncoder(torch.nn.Module):
    def __init__(self):
        super(SeqEncoder, self).__init__()
        kernel = [3,5,7]
        channels=[128, 256, 512]
        self.conv1 = CNNConvLayersPre(kernel)
        self.conv2 = CNNConvLayersMid(channels, kernel)
        self.conv3 = CNNConvLayersLast(channels, kernel, 80)
    
    def forward(self, x):

        x = x.view(1, -1, 80)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
