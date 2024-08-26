import os
import random
import sys
import threading
import torch
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils import psi_pre, hhm_pre, pssm_mut, hhm_mut
from utils import seq_pos

import warnings
warnings.filterwarnings("ignore")
import operator

def sort_dict_by_key(d, reverse=False):
    sorted_list = sorted(d.items(), key=operator.itemgetter(0), reverse=reverse)
    return sorted_list

def cmp(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    elif isinstance(a, str) and isinstance(b, str):
        if int(a[0:-1]) != int(b[0:-1]):
            return int(a[0:-1]) - int(b[0:-1])
        else:
            return ord(a[-1]) - ord(b[-1])
    elif isinstance(a, int) and isinstance(b, str):
        if a - int(b[0:-1]) == 0:
            return -1
        else:
            return a - int(b[0:-1])
    else:
        return 1

random_seed = 123
torch.manual_seed(random_seed)
num = 10
np.random.seed(10)
random.seed(num)

ddg_dict = {}
count = 0
num_pdb_dict = {}

res_corr = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
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
            'NFA': 'F', 'GLZ': 'G', 'FTY': 'Y', 'LYZ': 'K',
            'DA': 'A', 'DG': 'G', 'SEB': 'S', 'OMT': 'M', 'IYR': 'Y',
            'HTI': 'C', 'OHI': 'H', 'SCY': 'C', 'ASA': 'D', 'MCS': 'C',
            'LYR': 'K', 'TRN': 'W', 'IAS': 'D', 'DAH': 'F', 'TY2': 'Y',
            'FGL': 'G', 'FHO': 'K', 'ALC': 'A', 'HZP': 'P',
            'PHA': 'F', 'MHO': 'M', 'APK': 'K', 'CYR': 'C', 'JJJ': 'C',
            'CSP': 'C', 'TRO': 'W', 'CAF': 'C', 'OAS': 'S', 'DT': 'T',
            'DC': 'C', 'CMT': 'C'
            }
def choose_mut_residue(chain_dict: dict, mut_res):
    mut_list = []
    mut_res_coord = chain_dict[mut_res]
    mut_list.append(mut_res)
    ans_dist = {}
    for key in chain_dict.keys():
        if key != mut_res:
            dist = np.array(chain_dict[key]-mut_res_coord)
            dist = np.sqrt(dist.dot(dist))
            ans_dist[key] = dist
            # if dist <= 72:
            #     mut_list.append(key)
    ans = sorted(ans_dist.items(), key=lambda x: x[1])
    if len(ans) < 64:
        it = len(ans)
    else:
        it = 64
    for i in range(it):
        mut_list.append(ans[i][0])
    return mut_list

def get_pdb_seq(pdb_id):
    """
    Get the sequence of a pdb
    """
    import Bio.PDB
    pdb_file = 'piano/Data/pdb/{}.pdb'.format(pdb_id)
    parser = Bio.PDB.PDBParser()
    structure = parser.get_structure(pdb_id, pdb_file)
    model = structure[0]

    c_id = {}
    ci = 0
    idx_dict = {}
    for c in structure.get_chains():
        c_id[c.get_id()] = ci
        idx_dict[c.get_id()] = []
        ci += 1
    seq_dict = {}
    
    for c in c_id.keys():
        seq = ''
        for res in model[c]:
            res_idx = str(res.get_id()[1]) + res.get_id()[2].lower()
            res_idx = res_idx.strip()
            if res.get_resname() not in res_corr.keys():
                continue
            seq += res_corr[res.get_resname()]
            idx_dict[c].append(res_idx)
        seq_dict[c] = seq
    return seq_dict, idx_dict


def no_perturbation(tar_pdb, res_phi_chem,
                    residues, res_code, res_corr, pre_data, name_key):

    # dict - res
    global count

    residues_dict = {}
    res_code_dict = {}
    low_all = {}
    for i in range(len(residues)):
        residues_dict[residues[i]] = i
        res_code_dict[res_code[i]] = i
        low_all[res_code[i]] = residues[i]
    # get interface_res
    for i in range(len(residues)):
        residues_dict[residues[i]] = i
        res_code_dict[res_code[i]] = i
        low_all[res_code[i]] = residues[i]

    mut_csv = pre_data
    mut_info = []
    s = ['S_S', mut_csv.split('_')[1][1], mut_csv.split('_')[1][2:-1],
           mut_csv.split('_')[1][0], mut_csv.split('_')[1][-1], '0']
    mut_info.append(s)
    
    pdb_path = 'piano/Data/pdb/' + tar_pdb + '.pdb'
    if not os.path.exists(pdb_path):
        return False
    p = PDBParser()
    stru = p.get_structure(tar_pdb, pdb_path)
    pdb_model = stru[0]
    
    seq_dict, idx_dict = get_pdb_seq(tar_pdb)
    for i in range(len(mut_info)):

        mut_chain = mut_info[i][1]
        mut_ind = str(mut_info[i][2])
        mut_after_res = mut_info[i][4]
        mut_before_res = mut_info[i][3]
        mut_dict = {}
        mut_resi = mut_chain+'_'+ mut_ind
        mut_dict[mut_resi] = mut_after_res

        interface_res = []
        mid_ind = 0
        for j in range(len(idx_dict[mut_chain])):
            if str(idx_dict[mut_chain][j]) == str(mut_ind).lower():
                mid_ind = j
                break
        i_ind = mid_ind-1
        j_ind = mid_ind+1
        interface_res.append(mut_chain+'_'+str(mut_ind).lower())
        while i_ind >= 0 and j_ind < len(idx_dict[mut_chain]):
            interface_res.append(mut_chain+'_'+str(idx_dict[mut_chain][i_ind]))
            interface_res.append(mut_chain+'_'+str(idx_dict[mut_chain][j_ind]))
            if len(interface_res) == 128:
                break
            i_ind -= 1
            j_ind += 1
        if len(seq_dict[mut_chain[0]]) < 128:
            assert len(interface_res) < 128
        elif len(interface_res) < 128:
            if i_ind < 0:
                for k in range(128 - len(interface_res)):
                    interface_res.append(mut_chain+'_'+str(idx_dict[mut_chain][j_ind]))
                    j_ind += 1
            if j_ind >= len(idx_dict[mut_chain]):
                for k in range(128 - len(interface_res)):
                    interface_res.append(mut_chain+'_'+str(idx_dict[mut_chain][i_ind]))
                    i_ind -= 1
        sort_num_indcode = {}
        pre_ch = ''
        for ir in interface_res:
            pre_ch = ir.split('_')[0]
            idx = ir.split('_')[1]
            if idx[0] == '-':
                sort_num_indcode[int(ir.split('_')[1])] = 'A'
            elif idx.isdigit():
                if int(ir.split('_')[1]) in sort_num_indcode.keys():
                    sort_num_indcode[int(ir.split('_')[1])] += 'A'
                else:
                    sort_num_indcode[int(ir.split('_')[1])] = 'A'
            else:
                if int(ir.split('_')[1][0:-1]) in sort_num_indcode.keys():
                    sort_num_indcode[int(ir.split('_')[1][0:-1])] += idx[-1]
                else:
                    sort_num_indcode[int(ir.split('_')[1][0:-1])] = idx[-1]
        sort_num_indcode = sort_dict_by_key(sort_num_indcode)
        sort_inter = []
        for ir in sort_num_indcode:
            if len(ir[1]) > 1:
                pre_l = sorted(list(ir[1]))
                for l in pre_l:
                    if l == 'A':
                        ss = pre_ch+'_'+str(ir[0])
                        sort_inter.append(ss.strip())
                    else:
                        ss = pre_ch+'_'+str(ir[0])+l
                        sort_inter.append(ss.strip())
            else:
                ss = pre_ch+'_'+str(ir[0])
                sort_inter.append(ss.strip())
        interface_res = sort_inter

        mut_ind_num = 0
        for e in interface_res:
            if e == mut_chain+'_'+mut_ind:
                break
            else:
                mut_ind_num += 1
        s_key = mut_chain

        p_features_res = []
        m_features_res = []
        res_f1 = {}
        res_f2 = {}
        for c in [s_key]:
            res_index_num = 0
            pssm_p = psi_pre(tar_pdb, c)
            hhm_p = hhm_pre(tar_pdb, c)
            pssm_m = pssm_mut(tar_pdb, c, mut_before_res, mut_after_res, mut_ind)
            hhm_m = hhm_mut(tar_pdb, c, mut_before_res, mut_after_res, mut_ind)
            if len(pssm_p) == 0 or len(hhm_p) ==0 or len(pssm_m) == 0 or len(hhm_m) == 0:
                continue
            for res in pdb_model[c]:
                res_features = [0] * 80
                m_res_features = [0] * 80
                res_idx = str(res.get_id()[1]) + res.get_id()[2].lower()
                res_idx = res_idx.strip()
                res_name = res.get_resname()
                cr_token = '{}_{}'.format(c, res_idx)
                if res_name not in res_corr.keys():
                    continue
                res_id = res_code_dict[res_corr[res_name]]
                if cr_token in interface_res:
                    res_features[res_id] = 1  # one-hot-res 1000...00(GLN, ASP, ...)
                    if cr_token in mut_dict.keys():
                        mut_res_name = mut_dict[cr_token]
                        m_res_features[res_code_dict[mut_res_name]] = 1
                        res_features[20] = 1
                        m_res_features[20] = 1
                        m_res_features[22] = res_phi_chem[low_all[mut_res_name]][0]
                        m_res_features[23] = res_phi_chem[low_all[mut_res_name]][1]
                        m_res_features[24] = res_phi_chem[low_all[mut_res_name]][2]
                        m_res_features[25] = res_phi_chem[low_all[mut_res_name]][3]
                        m_res_features[26] = res_phi_chem[low_all[mut_res_name]][4]
                        m_res_features[27] = res_phi_chem[low_all[mut_res_name]][5]
                        m_res_features[28] = res_phi_chem[low_all[mut_res_name]][6]
                        if mut_res_name in ['R', 'D', 'N', 'Q', 'E', 'H', 'K', 'S', 'T', 'Y']:
                            m_res_features[79] = 1
                        
                        pssm_f = pssm_m[res_index_num]
                        hhm_f = hhm_m[res_index_num]
    
                        pssm_f = list(pssm_f)
                        for pf in range(len(pssm_f)):
                            m_res_features[29+pf] = float(pssm_f[pf])
                        hhm_f = list(hhm_f)
                        for pf in range(len(hhm_f)):
                            m_res_features[49+pf] = float(hhm_f[pf])


                        
                    else:
                        m_res_features[res_id] = 1
                        m_res_features[22] = res_phi_chem[low_all[res_corr[res_name]]][0]
                        m_res_features[23] = res_phi_chem[low_all[res_corr[res_name]]][1]
                        m_res_features[24] = res_phi_chem[low_all[res_corr[res_name]]][2]
                        m_res_features[25] = res_phi_chem[low_all[res_corr[res_name]]][3]
                        m_res_features[26] = res_phi_chem[low_all[res_corr[res_name]]][4]
                        m_res_features[27] = res_phi_chem[low_all[res_corr[res_name]]][5]
                        m_res_features[28] = res_phi_chem[low_all[res_corr[res_name]]][6]
                        if res_corr[res_name] in ['R', 'D', 'N', 'Q', 'E', 'H', 'K', 'S', 'T', 'Y']:
                            m_res_features[79] = 1
                        if c == mut_chain:
                            pssm_f = pssm_m[res_index_num]
                            hhm_f = hhm_m[res_index_num]
                        else:
                            pssm_f = pssm_p[res_index_num]
                            hhm_f = hhm_p[res_index_num]
      
                        pssm_f = list(pssm_f)
                        for pf in range(len(pssm_f)):
                            m_res_features[29+pf] = float(pssm_f[pf])
                        hhm_f = list(hhm_f)
                        for pf in range(len(hhm_f)):
                            m_res_features[49+pf] = float(hhm_f[pf])
                    #     res_features[20+res_id] = 1
                    if c == mut_chain:
                        res_features[21] = 1
                        m_res_features[21] = 1
                    res_features[22] = res_phi_chem[low_all[res_corr[res_name]]][0]
                    res_features[23] = res_phi_chem[low_all[res_corr[res_name]]][1]
                    res_features[24] = res_phi_chem[low_all[res_corr[res_name]]][2]
                    res_features[25] = res_phi_chem[low_all[res_corr[res_name]]][3]
                    res_features[26] = res_phi_chem[low_all[res_corr[res_name]]][4]
                    res_features[27] = res_phi_chem[low_all[res_corr[res_name]]][5]
                    res_features[28] = res_phi_chem[low_all[res_corr[res_name]]][6]
                    
                    if res_corr[res_name] in ['R', 'D', 'N', 'Q', 'E', 'H', 'K', 'S', 'T', 'Y']:
                        res_features[79] = 1
                    pssm_f = pssm_p[res_index_num]
                    hhm_f = hhm_p[res_index_num]
                    pssm_f = list(pssm_f)
                    for pf in range(len(pssm_f)):
                        res_features[29+pf] = float(pssm_f[pf])
                    hhm_f = list(hhm_f)
                    for pf in range(len(hhm_f)):
                        res_features[49+pf] = float(hhm_f[pf])

                    res_f1[cr_token] = res_features
                    res_f2[cr_token] = m_res_features
                    

                # print(atom_features)
        for j in range(128 - len(res_f1.keys())):
            p_data = [0] * 80
            res_f1[str(j)] = p_data
        for key in res_f1.keys():
            p_features_res.append(res_f1[key])
        for key in res_f2.keys():
            m_features_res.append(res_f2[key])
        # before
        mut_detail = mut_before_res+mut_chain+mut_ind+mut_after_res

        seq_pos(p_features_res, 'piano/Data/apo_seq', name_key, mut_ind_num)
        seq_pos(m_features_res, 'piano/Data/apo_seq', name_key+'_mut', mut_ind_num)
        count += 1

    return True


def main():
    
    # get xr angles
    residues = ['ARG', 'MET', 'VAL', 'ASN', 'PRO', 'THR', 'PHE', 'ASP', 'ILE', 'ALA', 'GLY', 'GLU', 'LEU', 'SER',
                'LYS', 'TYR', 'CYS', 'HIS', 'GLN', 'TRP']
    res_code = ['R', 'M', 'V', 'N', 'P', 'T', 'F', 'D', 'I', 'A', 'G', 'E', 'L', 'S', 'K', 'Y', 'C', 'H', 'Q', 'W']

    # a Steric parameter (graph shape index)
    # b Polarizability
    # c Volume (normalized van der Waals volume)
    # d Hydrophobicity
    # e Isoelectric point
    # f Helix probability
    # g Sheet probability
    res_phi_chem = {'ALA': [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
                    'GLY': [0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
                    'VAL': [3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
                    'LEU': [2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
                    'ILE': [4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
                    'PHE': [2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
                    'TYR': [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
                    'TRP': [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
                    'THR': [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
                    'SER': [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
                    'ARG': [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
                    'LYS': [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
                    'HIS': [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
                    'ASP': [1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
                    'GLU': [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
                    'ASN': [1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
                    'GLN': [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
                    'MET': [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
                    'PRO': [2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
                    'CYS': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41]}
    pdb_info = np.load('apo_dict.npy', allow_pickle=True).item()

    key_l = list(pdb_info.keys())
    for i in range(0, len(key_l)):
        key = key_l[i]
        pre_ans = pdb_info[key]
        if pre_ans[0] =='NA' or pre_ans[1] == 'NA':
            continue
        pdb = pre_ans[0].split('_')[0]
        had_pdb = pre_ans[0]+'_'+pre_ans[1]
        print(pdb + '_' + str(i))
        flag = no_perturbation(pdb.lower(), res_phi_chem, residues, res_code, res_corr, pre_ans[2][0], key)
        print(flag)
        print(count)


main()


