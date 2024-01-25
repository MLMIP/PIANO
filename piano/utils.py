from Bio import PDB
import os
import copy
import numpy as np
import torch
import subprocess
from Bio.PDB.ResidueDepth import get_surface
from Bio.PDB.ResidueDepth import residue_depth
from Bio.PDB.DSSP import DSSP
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import warnings
warnings.filterwarnings("ignore")
from config import naccess_path

pre_path = 'piano/OtherFiles/'
psi_path = 'piano/Data/PSIAns/'
hhm_path = 'piano/Data/HHMAns/'

def get_pdb_list(filename):
    # get pdb file
    f = open(filename, 'r')
    pdb_list = []
    line = f.readline()
    while line:
        pdb_list.append(line[0:-1])
        line = f.readline()
    f.close()
    return pdb_list

def get_pdb_seq_bio(pdb_name, pdb_file):
    corr = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
            'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
            'GLY': 'G', 'HIS': 'H', 'HSD': 'H', 'HSE': 'H', 'LEU': 'L',
            'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL': 'V', 'GLU': 'E',
            'TYR': 'Y', 'MET': 'M', 'MSE': 'M', 'PTR': 'Y', 'TYS': 'Y',
            'SEP': 'S', 'TPO': 'T', 'HIP': 'H', 'F2F': 'F', 'BFD': 'D',
            'CSX': 'C', 'LLP': 'K', 'HIC': 'H', 'MEN': 'N', 'MLY': 'K',
            'KCX': 'K', 'CME': 'C', 'CYG': 'C', 'CSD': 'C', 'SME': 'M',
            'CGU': 'E', 'CAS': 'C', 'CSO': 'C', 'NEP': 'H', 'OCS': 'C',
            'CSA': 'C', 'MLZ': 'K', 'CYJ': 'K', 'SCS': 'C', 'CSS': 'C',
            'HS8': 'H', 'LED': 'L', 'R1A': 'C', 'HTR': 'W', 'NIY': 'Y',
            'M3L': 'K', 'GPL': 'K', 'AGT': 'C', 'NLE': 'L', 'YCM': 'C',
            'ABA': 'A', 'LET': 'K', 'LP6': 'K', 'HIQ': 'H', 'PAQ': 'Y',
            'XSN': 'N', 'DDE': 'H', 'HYP': 'P', 'DAL': 'A', 'ALY': 'K',
            'AAR': 'R', 'PHD': 'D', 'TYI': 'Y', 'MSO': 'M', 'MLE': 'L',
            'DYA': 'D', 'SMC': 'C', 'CCS': 'C', 'DLE': 'L', 'DPR': 'P',
            'DVA': 'V', 'DCY': 'C', 'DGL': 'E', 'DTH': 'T', 'DSG': 'N',
            'DSN': 'S', 'DTR': 'W', 'DAR': 'R', 'TRQ': 'W', 'DDZ': 'A',
            'SNC': 'C', 'ALO': 'T', 'TYQ': 'Y', 'KPI': 'K', 'FGP': 'S',
            'KYN': 'W', 'CMH': 'C', 'LCK': 'K', 'P1L': 'C', 'SVY': 'S',
            'TH6': 'T', 'PRS': 'P', 'TPQ': 'Y', '0AF': 'W', 'DHA': 'S',
            'LA2': 'K', 'SCH': 'C', 'DAS': 'D', 'DLY': 'K', 'DTY': 'Y',
            'NFA': 'F', 'GLZ': 'G', 'FTY': 'Y', 'MDO': 'X', 'LYZ': 'K',
            'DA': 'A', 'DG': 'G', 'SEB': 'S', 'OMT': 'M', 'IYR': 'Y',
            'HTI': 'C', 'OHI': 'H', 'SCY': 'C', 'ASA': 'D', 'MCS': 'C',
            'LYR': 'K', 'TRN': 'W', 'IAS': 'D', 'DAH': 'F', 'TY2': 'Y',
            'FGL': 'G', 'FHO': 'K', 'ALC': 'A', 'HZP': 'P', 'CSI': 'X',
            'PHA': 'F', 'MHO': 'M', 'APK': 'K', 'CYR': 'C', 'JJJ': 'C',
            'CSP': 'C', 'TRO': 'W', 'CAF': 'C', 'OAS': 'S', 'DT': 'T',
            'DC': 'C', 'CMT': 'C', 'UNK': 'X', 'GLX': 'Z'
            }
    parser = PDB.PDBParser()
    structure = parser.get_structure(pdb_name, pdb_file)

    model = structure[0]
    c_id = {}
    ci = 0

    ans = {}
    serial_dict = {}
    for c in structure.get_chains():
        c_id[c.get_id()] = ci
        ci += 1
    for c in c_id.keys():
        seq = ''
        num_list = []
        for res in model[c]:
            res_idx = str(res.get_id()[1]) + res.get_id()[2].lower()
            res_idx = res_idx.strip()
            res_name = res.get_resname()
            if res_name in corr.keys():
                seq += corr[res_name]
                num_list.append(res_idx)

        ans[c] = seq
        serial_dict[c] = num_list
    
    return ans, serial_dict
def readFa(fa):
    '''
    @msg: 读取一个fasta文件
    @param fa {str}  fasta 文件路径
    @return: {generator} 返回一个生成器，能迭代得到fasta文件的每一个序列名和序列
    '''
    with open(fa,'r') as FA:
        seqName,seq='',''
        while 1:
            line=FA.readline()
            line=line.strip('\n')
            if (line.startswith('>') or not line) and seqName:
                yield((seqName,seq))
            if line.startswith('>'):
                seqName = line[1:]
                seq=''
            else:
                seq+=line
            if not line:break

def process_hhm(path):
    with open(path,'r') as fin:
        fin_data = fin.readlines()
        hhm_begin_line = 0
        hhm_end_line = 0
        for i in range(len(fin_data)):
            if '#' in fin_data[i]:
                hhm_begin_line = i+5
            elif '//' in fin_data[i]:
                hhm_end_line = i
        feature = np.zeros([int((hhm_end_line-hhm_begin_line)/3),30])
        axis_x = 0
        for i in range(hhm_begin_line,hhm_end_line,3):
            line1 = fin_data[i].split()[2:-1]
            line2 = fin_data[i+1].split()
            axis_y = 0
            for j in line1:
                if j == '*':
                    feature[axis_x][axis_y]=9999/10000.0
                else:
                    feature[axis_x][axis_y]=float(j)/10000.0
                axis_y+=1
            for j in line2:
                if j == '*':
                    feature[axis_x][axis_y]=9999/10000.0
                else:
                    feature[axis_x][axis_y]=float(j)/10000.0
                axis_y+=1
            axis_x+=1
        feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))

        return feature
def process_pssm(path):
    import math
    with open(path,'r') as fin:
        fin_data = fin.readlines()
        pssm_begin_line = 3
        pssm_end_line = 0
        for i in range(1,len(fin_data)):
            if fin_data[i] == '\n':
                pssm_end_line = i
                break

        feature = np.zeros([(pssm_end_line-pssm_begin_line),20])
        axis_x = 0
        for i in range(pssm_begin_line,pssm_end_line):
            raw_pssm = fin_data[i].split()[2:22]
            axis_y = 0
            for j in raw_pssm:
                feature[axis_x][axis_y]= (1 / (1 + math.exp(-float(j))))
                axis_y+=1
            axis_x+=1

    return feature


def psi_pre(pdb, chain):
    f_name = pdb+'_'+chain
    pssm_path = psi_path+f_name+'.pssm'
    if not os.path.exists(pssm_path):
        print('no pssm file')
        return -1, []
    feature = process_pssm(pssm_path)
        
    return feature

def hhm_pre(pdb, chain):
    hhm_p = hhm_path+pdb+'_'+chain+'.hhm'
    if not os.path.exists(hhm_p):
        return []
    feature = process_hhm(hhm_p)

    return feature

def pssm_mut(pdb, chain, wild, mut, index):

    mut_detail = pdb+'_'+wild+chain+index+mut
    # pssm

    pssm_path = psi_path+mut_detail+'.pssm'
    if not os.path.exists(pssm_path):
        return []
    feature = process_pssm(pssm_path)

    return feature


def hhm_mut(pdb, chain, wild, mut, index):

    mut_detail = pdb+'_'+wild+chain+index+mut
    hhm_p = hhm_path+mut_detail+'.hhm'
    if not os.path.exists(hhm_p):
        return []
    feature = process_hhm(hhm_p)

    return feature

def get_pdb_list(filename):
    # get pdb file
    f = open(filename, 'r')
    pdb_list = []
    line = f.readline()
    while line:
        pdb_list.append(line[0:-1])
        line = f.readline()
    f.close()
    return pdb_list


def interface_features(if_info, tar_pdb, chainid):
    pdbfile = 'piano/Data/pdb/' + tar_pdb + '.pdb'

    workdir = 'temp'

    os.system("python piano/generate_interface.py {} {} {} > {}/pymol_{}.log".format(
        pdbfile, if_info, workdir, workdir, tar_pdb))
    interface_file = '{}/interface_{}.txt'.format(workdir, tar_pdb)

    interface_res = read_inter_result(interface_file, if_info, chainid)
    return interface_res


def read_inter_result(path, if_info=None, chainid=None, old2new=None):
    if if_info is not None:
        info1 = if_info.split('_')
        pA = info1[0]  
        pB = info1[1]  
        mappings = {}  
        for a in pA:
            for b in pB:
                if a not in mappings:
                    mappings[a] = [b]
                else:
                    mappings[a] += [b]
                if b not in mappings:
                    mappings[b] = [a]
                else:
                    mappings[b] += [a]

        target_chains = []
        for chainidx in chainid:
            if chainidx in mappings:
                target_chains += mappings[chainidx]

        target_inters = []
        for chainidx in chainid:
            target_inters += ['{}_{}'.format(chainidx, y) for y in target_chains] + \
                             ['{}_{}'.format(y, chainidx) for y in target_chains]

        target_inters = list(set(target_inters))
    else:
        target_inters = None
    inter = open(path)
    interlines = inter.read().splitlines()
    interface_res = []
    for line in interlines:
        iden = line[:3]
        if target_inters is None:
            if iden.split('_')[0] not in chainid and iden.split('_')[1] not in chainid:
                continue
        else:
            if iden not in target_inters:
                continue
        infor = line[4:].strip().split('_')  # chainid, resid
        assert len(infor) == 2
        interface_res.append('_'.join(infor))


    if old2new is not None:
        mapps = {x[:-4]: y[:-4] for x, y in old2new.items()}
        interface_res = [mapps[x] for x in interface_res if x in mapps]

    return interface_res


def get_interface_feature(tar_pdb):
    if_info = []
    chain = []
    pdb = open('piano/Data/pdb/' + tar_pdb + '.pdb')
    lines = pdb.read().splitlines()
    for line_pro in lines:
        if line_pro[0:4] == 'ATOM':
            chain_id = line_pro[21]
            if chain_id not in chain:
                chain.append(chain_id)

    for i in range(len(chain)):
        for j in range(i + 1, len(chain)):
            pre = chain[i] + '_' + chain[j]
            if_info.append(pre)

    interface_res_s = []
    interface_res = []
    for info in if_info:
        interface_res_s.append(interface_features(info, tar_pdb, chain))
    for i in range(len(interface_res_s)):
        for res in interface_res_s[i]:
            interface_res.append(res)
    return interface_res


def get_interface_feature_ds(tar_pdb, data_name, chainid):
    if_info = []
    # for i in range(p.shape[0]):
    #     if p['#PDB'][i].lower() == tar_pdb and p['Partners(A_B)'][i] not in if_info:
    #         if_info.append(p['Partners(A_B)'][i])
    s=chainid[0]+'_'
    for i in range(1, len(chainid)):
        s += chainid[i]
    if_info.append(s)
    interface_res_s = []
    interface_res = []
    for info in if_info:
        interface_res_s.append(interface_features(info, tar_pdb, chainid))
    for i in range(len(interface_res_s)):
        for res in interface_res_s[i]:
            interface_res.append(res)
    return interface_res


def bg_pos(features, cutoff):
    features = torch.tensor(features, dtype=torch.float)  # trans - torch
    N = features.size(0)

    pos = features[:, -4:-1]  # N,3 ,get coords

    # print(pos)
    row = pos[:, None, :].repeat(1, N, 1)
    col = pos[None, :, :].repeat(N, 1, 1)
    direction = row - col
    del row, col
    distance = torch.sqrt(torch.sum(direction ** 2, 2)) + 1e-10  # get distance different
    if cutoff < 10:
        distance1 = (1.0 / distance) * (distance < float(cutoff)).float()
        del distance
        diag = torch.diag(torch.ones(N))
        dist = diag + (1 - diag) * distance1
        del distance1, diag
        flag = (dist > 0).float()
        direction = direction * flag.unsqueeze(2)
        del direction, dist
        edge_index = torch.nonzero(flag)  # K,2

    else:
        edge_index = []
        t = copy.deepcopy(distance.tolist())
        length = len(t[0])
        res_num = 17
        if length <= 16:
            res_num = length
        for i in range(length):
            min_number = []
            min_index = []
            for _ in range(res_num):
                number = min(t[i])
                index = t[i].index(number)
                t[i][index] = 9999
                min_number.append(number)
                min_index.append(index)
            for e in min_index:
                edge_index.append([i, e])
        edge_index = torch.tensor(edge_index)
    edge_attr = []
    for k in range(len(edge_index)):
        point = edge_index[k]
        edge_attr_pre = []
        a = pos[point[0]]
        b = pos[point[1]]
        direction = a - b
        direction = np.array(direction)
        direction_x = np.array([0, 0, 0])
        direction_x[0] = direction[0]
        direction_x[1] = direction[1]
        dist = np.sqrt(direction.dot(direction))
        z_dir = np.array([0, 0, 1])
        x_dir = np.array([1, 0, 0])
        za = np.arccos(
            direction.dot(z_dir) / (np.sqrt(direction.dot(direction)) * np.sqrt(z_dir.dot(z_dir)) + 1e-10))
        xa = np.arccos(
            direction_x.dot(x_dir) / (np.sqrt(direction_x.dot(direction_x)) * np.sqrt(x_dir.dot(x_dir)) + 1e-10))
        edge_attr_pre.append(dist)
        edge_attr_pre.append(za)
        edge_attr_pre.append(xa)
        edge_attr.append(edge_attr_pre)

    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    return features, edge_index, edge_attr

def gb_pos_64(p_features_atom, p_features_res, data_name, k):

    sa1 =  data_name + '/' + str(k) + '_' + 'p_atom_A.pt'
    se1 =  data_name + '/' + str(k) + '_' + 'atom_E.pt'
    sa2 =  data_name + '/' + str(k) + '_' + 'p_res_A.pt'
    se2 =  data_name + '/' + str(k) + '_' + 'res_E.pt'
    edg1 = data_name + '/' + str(k) + '_' + 'p_atom_attr.pt'
    edg2 =  data_name + '/' + str(k) + '_' + 'p_res_attr.pt'
    a1, e1, ed1 = bg_pos(p_features_atom, 3)
    a2, e2, ed2 = bg_pos(p_features_res, 24)
    torch.save(a1, sa1)
    torch.save(e1, se1)
    torch.save(a2, sa2)
    torch.save(e2, se2)
    torch.save(ed1, edg1)
    torch.save(ed2, edg2)

def seq_graph(features, s_key):
    features = torch.tensor(features, dtype=torch.float)  # trans - torch
    N = features.size(0)

    edge_index = []
    for i in range(s_key, s_key+1):
        for j in range(1, N):
                edge_index.append([i, j])
                edge_index.append([j, i])
    # print(direction[0])
    edge_index = torch.tensor(edge_index)

    return features, edge_index


def seq_pos(p_features_res, data_name, k, s_key):
    sa2 =  data_name + '/' + str(k) +'_p_res_A.pt'
    se2 =  data_name + '/' + str(k) +'_res_E.pt'
    
    a2, e2 = seq_graph(p_features_res, s_key)
    torch.save(a2, sa2)
    torch.save(e2, se2)
def is_exist_file(path):
    if os.path.exists(path):
        return True
    else:
        return False


def get_naccess(pdb):
    pdb_file = 'piano/Data/pdb/' + pdb + '.pdb'
    
    # cwd = os.getcwd()
    # os.chdir('tempt')
    cwd = os.getcwd()
    os.chdir(os.path.dirname(pdb_file))
    pre_p = os.getcwd().split('/')
    pre_path = ''
    for i in range(len(pre_p)):
        if i < len(pre_p) -3:
            pre_path += pre_p[i] +'/'
    naccess_p = pre_path + naccess_path
    raw_naccess_output = []
    _, _ = subprocess.Popen([naccess_p+'naccess', pre_path+pdb_file, '-h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    try:
        raw_naccess_output += open(os.path.splitext(pre_path+pdb_file)[0]+'.asa', 'r').readlines()
    except IOError:
        raise IOError('ERROR: Naccess .asa file was not written. The following command was attempted: %s %s' %('naccess', pdb_file))

    os.chdir(cwd)
    return raw_naccess_output


def get_residue_depth(pdb_model, interface):
    model = pdb_model
    surface = get_surface(model)
    rd_dict = {}

    for inter in interface:
        s = inter.split('_')
        s[0] = s[0].strip()
        s[1] = s[1].strip()
        for res in model[s[0]]:
            idx = str(res.get_id()[1])+res.get_id()[2].lower()
            idx = idx.strip()
            if idx == s[1]:
                rd = residue_depth(res, surface)
                rd_dict[inter] = rd
    return rd_dict


def get_residue_dssp(pdb_model, pdb):
    fa = 'piano/Data/pdb/' + pdb + '.pdb'

    model = pdb_model
    dssp = DSSP(model, fa, dssp='piano/Software/mkdssp')

    dssp_key = []
    for a_key in list(dssp.keys()):
        s = a_key[0] + '_' + str(a_key[1][1]) + a_key[1][2].lower()
        s = s.strip()
        dssp_key.append(s)
    return dssp, dssp_key
