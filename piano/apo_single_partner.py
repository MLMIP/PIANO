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
from utils import gb_pos_64
from utils import get_residue_depth,psi_pre, hhm_pre
import warnings
warnings.filterwarnings("ignore")
import operator
 

num_res_c = 64

def sort_dict_by_key(d, reverse=False):
    sorted_list = sorted(d.items(), key=operator.itemgetter(0), reverse=reverse)
    return sorted_list
random_seed = 123
torch.manual_seed(random_seed)
num = 10
np.random.seed(10)
random.seed(num)

ddg_dict = {}
pdb2chain = []
def distance_matrix(points, length):
    return np.sqrt(((points[:length, None] - points) ** 2).sum(-1))

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
    if len(ans) < num_res_c:
        it = len(ans)
    else:
        it = num_res_c
    for i in range(it):
        mut_list.append(ans[i][0])
    return mut_list

def get_interface_res_surr(interface_res, chain_dict):
    
    all_coord = []
    all_res = []
    for key in chain_dict.keys():
        if key in interface_res:
            all_coord.append(chain_dict[key])
            all_res.append(key)
    for key in chain_dict.keys():
        if key not in interface_res:
            all_coord.append(chain_dict[key])
            all_res.append(key)
    all_coord = np.array(all_coord)
    dist = distance_matrix(all_coord, len(interface_res))
    res_list = interface_res
    sum_inter = num_res_c
    dist_cutoff = 10
    while sum_inter > num_res_c:
        res_list = interface_res
        for i in range(len(interface_res)):
            for j in range(dist[i]):
                if dist[i][j] <= dist_cutoff:
                    res_list.append(all_res[j])
        res_list = list(set(res_list))
        if len(res_list) < sum_inter:
            break
        else:
            dist_cutoff -= 1
    return res_list


def no_perturbation(tar_pdb, res_phi_chem, atom_dict,
                    residues, res_code, res_corr, pre_data,had_pdb, name_key):

    # dict - res
    residues_dict = {}
    res_code_dict = {}
    low_all = {}
    pre_ddg = []
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
    p = PDBParser()
    stru = p.get_structure(tar_pdb, pdb_path)
    pdb_model = stru[0]
    

    res_naccess = 'piano/Data/features/'+tar_pdb+'_residues_naccess.npy'
    dssp_path = 'piano/Data/features/'+tar_pdb+'_residues_dssp.npy'
    dssp_key_path = 'piano/Data/features/'+tar_pdb+'_dssp_key.npy'
    
    if not (os.path.exists(res_naccess) and os.path.exists(dssp_path) and os.path.exists(dssp_key_path)):
        return False

    residues_dssp = list(np.load(dssp_path))
    dssp_key = list(np.load(dssp_key_path))

    
    residues_naccess = list(np.load(res_naccess))
    
    for i in range(0, len(mut_info)):
        exi_flag = True
        mut_chain = mut_info[i][1]
        mut_ind = str(mut_info[i][2])
        mut_after_res = mut_info[i][4]
        mut_before_res = mut_info[i][3]
        mut_dict = {}
        mut_resi = mut_chain+'_'+ mut_ind
        mut_dict[mut_resi] = mut_after_res
        # part_chain = list(mut_info[i][0].replace('_', ''))
        # part_chain.remove('C')
        

        chain_dict = {}
        c_id = {}
        ci = 0

        for c in stru.get_chains():
            c_id[c.get_id()] = ci
            ci += 1

        if had_pdb[-1] not in c_id.keys():
            continue
        pre_interface  = []
        for res in pdb_model[had_pdb[-1]]:
            res_idx = str(res.get_id()[1]) + res.get_id()[2].lower()
            res_idx = res_idx.strip()
            res_name = res.get_resname()
            cr_token = '{}_{}'.format(had_pdb[-1], res_idx)
            
            if res_name not in res_corr.keys():
                continue
            pre_interface.append(cr_token)
            for atom in res:
                atom_name = atom.get_name()
                element_name = list(filter(lambda x: x.isalpha(), atom_name))[0]
                if element_name not in atom_dict.keys():
                    continue
                coords = torch.tensor(atom.get_coord())
                if atom_name == 'CA':
                    chain_dict[cr_token] = coords

        interface_res = pre_interface
        mut_isava = True
        ir_remove = []
        for ir in interface_res:
            ir_chain = ir.split('_')[0]
            if ir_chain  !=  had_pdb[-1]:
                ir_remove.append(ir)
        for ir in ir_remove:
            interface_res.remove(ir)

        interface_res = list(set(interface_res))
        clean_inter = []
        for ir in interface_res:
            ir_chain = ir.split('_')[0]
            if ir_chain == had_pdb[-1]:
                clean_inter.append(ir)
        interface_res = clean_inter
       
        # depth_dict
        residues_depth = get_residue_depth(pdb_model, interface_res)
        lo = []
        residues_naccess_dict = {}
        for rn in residues_naccess:
            pre_atom = rn[12:16].strip()
            pre_c = rn[21].strip()
            if pre_c not in lo:
                lo.append(pre_c)
            pre_idx = rn[22:28].strip()
            pre_cr_token = pre_c+'_'+pre_idx.lower()
            if pre_cr_token in interface_res:
                residues_naccess_dict[pre_atom+'_'+pre_cr_token] = float(rn[-15:-6].strip())
        if not residues_naccess_dict:
            continue
        res_atom_num = {}
        for res in interface_res:
            res_atom_num[res] = 0

        ca_coord = {}  # Ca coords = res coords
        for c in c_id.keys():
            for res in pdb_model[c]:
                res_idx = str(res.get_id()[1]) + res.get_id()[2].lower()
                res_idx = res_idx.strip()
                res_name = res.get_resname()
                cr_token = '{}_{}'.format(c, res_idx)
                if res_name not in res_corr.keys():
                    continue
                for atom in res:
                    atom_name = atom.get_name()
                    element_name = list(filter(lambda x: x.isalpha(), atom_name))[0]
                    if element_name not in atom_dict.keys():
                        continue
                    coords = torch.tensor(atom.get_coord())
                    if cr_token in interface_res:
                        if atom_name == 'CA':  # is Ca
                            ca_coord[cr_token] = coords

        p_features_atom = []
        p_features_res = []
        res_f1 = {}
        for c in [had_pdb[-1]]:
            res_index_num = 0
            pssm_p = psi_pre(tar_pdb, c)
            hhm_p = hhm_pre(tar_pdb, c)
            if len(pssm_p) == 0 or len(hhm_p) == 0:
                continue
            for res in pdb_model[c]:
                res_features = [0] * 108
                res_idx = str(res.get_id()[1]) + res.get_id()[2].lower()
                res_idx = res_idx.strip()
                res_name = res.get_resname()

                cr_token = '{}_{}'.format(c, res_idx)
                if res_name not in res_corr.keys():
                    continue
                res_id = res_code_dict[res_corr[res_name]]
                if cr_token in interface_res and cr_token in dssp_key:
                    res_features[res_id] = 1  # one-hot-res 1000...00(GLN, ASP, ...)
                    if cr_token in mut_dict.keys():
                        mut_res_name = mut_dict[cr_token]
                        res_features[20+res_code_dict[mut_res_name]] = 1
                        res_features[40] = 1
                    else:
                        res_features[20+res_id] = 1
                    if c in mut_chain:
                        res_features[41] = 1
                    res_features[42] = res_phi_chem[low_all[res_corr[res_name]]][0]
                    res_features[43] = res_phi_chem[low_all[res_corr[res_name]]][1]
                    res_features[44] = res_phi_chem[low_all[res_corr[res_name]]][2]
                    res_features[45] = res_phi_chem[low_all[res_corr[res_name]]][3]
                    res_features[46] = res_phi_chem[low_all[res_corr[res_name]]][4]
                    res_features[47] = res_phi_chem[low_all[res_corr[res_name]]][5]
                    res_features[48] = res_phi_chem[low_all[res_corr[res_name]]][6]

                    # depth
                    res_features[53] = residues_depth[cr_token]
                    # SASA
                    a_key = ''
                    for a_k in range(len(dssp_key)):
                        if cr_token == dssp_key[a_k]:
                            a_key = a_k
                            break
                    pre_ss = residues_dssp[a_key][2]
                    if pre_ss == 'H' or pre_ss == 'G' or pre_ss == 'I':
                        ans_sec = [1, 0, 0]  # 'H'
                    elif pre_ss == 'B' or pre_ss == 'E':
                        ans_sec = [0, 1, 0]  # 'E'
                    else:
                        assert pre_ss == 'T' or pre_ss == 'S' or pre_ss == '-'
                        ans_sec = [0, 0, 1]  # 'C'
                    if residues_dssp[a_key][3] == 'NA':
                        res_features[49] = 0
                    else:
                        res_features[49] = float(residues_dssp[a_key][3])
                    res_features[50] = ans_sec[0]
                    res_features[51] = ans_sec[1]
                    res_features[52] = ans_sec[2]
                    if res_corr[res_name] in ['R', 'D', 'N', 'Q', 'E', 'H', 'K', 'S', 'T', 'Y']:
                        res_features[107] = 1
                    if cr_token not in ca_coord.keys():
                        print(tar_pdb + '_' + cr_token)
                        return False
                    coords_res = ca_coord[cr_token]  # res coords
                    float_cd = [float(x) for x in coords_res]  # coords
                    cd_tensor = torch.tensor(float_cd)
                    if res_index_num >= len(pssm_p) or res_index_num >= len(hhm_p):
                        continue
                    pssm_f = pssm_p[res_index_num]
                    hhm_f = hhm_p[res_index_num]
                    
                    # res_features[42:45] = cd_tensor
                    pssm_f = list(pssm_f)
                    for pf in range(len(pssm_f)):
                        res_features[54+pf] = float(pssm_f[pf])
                    hhm_f = list(hhm_f)
                    for pf in range(len(hhm_f)):
                        res_features[74+pf] = float(hhm_f[pf])
                    res_features[104:107] = cd_tensor

                    res_f1[cr_token] = res_features

                    for atom in res:
                        atom_features = [0] * 9
                        atom_name = atom.get_name()
                        element_name = list(filter(lambda x: x.isalpha(), atom_name))[0]
                        if element_name not in atom_dict.keys():
                            continue
                        coords = torch.tensor(atom.get_coord())
                        if atom_name + '_' + cr_token not in residues_naccess_dict.keys():
                            continue
                        atom_id = atom_dict[element_name]
                        atom_features[atom_id] = 1  # one-hot-atom 1000,0100,0010,0001(C,N,O,S)
                        # atom_features[4 + res_id] = 1  # one-hot-res 1000...00(GLN, ASP, ...)
                        if c in mut_chain:  # is mut_chain->1 other is 0
                            atom_features[8] = 1
                            # naccess

                        atom_features[4] = residues_naccess_dict[atom_name + '_' + cr_token]
                        float_cd = [float(x) for x in coords]  # coords
                        cd_tensor = torch.tensor(float_cd)
                        atom_features[5:8] = cd_tensor
                        p_features_atom.append(atom_features)
                        res_atom_num[cr_token] += 1
                    if res_atom_num[cr_token] == 0:
                        del res_f1[cr_token]
                res_index_num += 1
        for ress in res_f1.keys():
            p_features_res.append(res_f1[ress])
        if not exi_flag:
            continue

        mut_detail = mut_before_res+mut_chain+mut_ind+mut_after_res
        index = 'piano/Data/apo_single2' + '/' + name_key + '_B_' + 'index_dict.pt'
        torch.save(res_atom_num, index)
        if len(p_features_atom) == 0 or len(p_features_res) == 0:
            return False
        gb_pos_64(p_features_atom, p_features_res, 'piano/Data/apo_single2', name_key+'_B')

        pdb2chain.append(tar_pdb+'_'+mut_detail)

        
    return True


def main():

    # get xr angles
    res2atom = {'ALA': ['C', 'N', 'O', 'CA', 'CB'],
                'ARG': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
                'ASN': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'ND2', 'OD1'],
                'ASP': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'OD1', 'OD2'],
                'CYS': ['C', 'N', 'O', 'CA', 'CB', 'SG'],
                'GLN': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD', 'NE2', 'OE1'],
                'GLU': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
                'GLY': ['C', 'N', 'O', 'CA'],
                'HIS': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD2', 'ND1', 'CE1', 'NE2'],
                'ILE': ['C', 'N', 'O', 'CA', 'CB', 'CG1', 'CG2', 'CD1'],
                'LEU': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD1', 'CD2'],
                'LYS': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD', 'CE', 'NZ'],
                'MET': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'SD', 'CE'],
                'PHE': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
                'PRO': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD'],
                'SER': ['C', 'N', 'O', 'CA', 'CB', 'OG'],
                'THR': ['C', 'N', 'O', 'CA', 'CB', 'CG2', 'OG1'],
                'TRP': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CZ2', 'CZ3', 'CH2'],
                'TYR': ['C', 'N', 'O', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
                'VAL': ['C', 'N', 'O', 'CA', 'CB', 'CG1', 'CG2']}

    res2atom_dict = {}
    num_s = 0
    for key in res2atom.keys():
        for atom in res2atom[key]:
            res2atom_dict[key+'_'+atom] = num_s
            num_s += 1
    atom_dict = {'C': 0, 'N': 1, 'O': 2, 'S': 3}  # atom dict type
    residues = ['ARG', 'MET', 'VAL', 'ASN', 'PRO', 'THR', 'PHE', 'ASP', 'ILE', 'ALA', 'GLY', 'GLU', 'LEU', 'SER',
                'LYS', 'TYR', 'CYS', 'HIS', 'GLN', 'TRP']
    res_code = ['R', 'M', 'V', 'N', 'P', 'T', 'F', 'D', 'I', 'A', 'G', 'E', 'L', 'S', 'K', 'Y', 'C', 'H', 'Q', 'W']
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
        pdb_l = []
        had_pdb_l = []
        for e in pre_ans[1].split(','):
            pdb_l.append(e[0:4])
            had_pdb_l.append(e)
        for j in range(len(pdb_l)):
            pdb = pdb_l[j]
            had_pdb = had_pdb_l[j]
            print(pdb + '_' + str(i))

            flag = no_perturbation(pdb.lower(), res_phi_chem, atom_dict, residues, res_code, res_corr, pre_ans[2][0], had_pdb, key+'_'+str(j))




main()


