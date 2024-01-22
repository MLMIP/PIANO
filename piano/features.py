import os
import sys
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils import get_pdb_seq_bio, readFa,get_interface_feature,get_residue_dssp, get_naccess
from config import psi_p, psi_db, hhm_db, hhm_p
import warnings
warnings.filterwarnings("ignore")

def get_f(tar_pdb):

    pdb_path = 'piano/Data/pdb/' + tar_pdb + '.pdb'
    p = PDBParser()
    stru = p.get_structure(tar_pdb, pdb_path)
    pdb_model = stru[0]
    pre_interface_res = get_interface_feature(tar_pdb)
    np.save('piano/Data/features/'+tar_pdb+'_interface.npy', pre_interface_res)

    residues_dssp, dssp_key = get_residue_dssp(pdb_model, tar_pdb)
    residues_naccess = get_naccess(tar_pdb)

    if  len(residues_naccess) > 10 and  len(residues_dssp) > 10:
        np.save('piano/Data/features/'+tar_pdb+'_residues_dssp.npy', residues_dssp)
        np.save('piano/Data/features/'+tar_pdb+'_dssp_key.npy', dssp_key)
        
        np.save('piano/Data/features/'+tar_pdb+'_residues_naccess.npy', residues_naccess)

def get_pssm(fa):
    for seqName, seq in readFa(fa):
        f = open('temp/p0.fasta', 'w')
        f.write('>'+seqName+'\n')
        f.write(seq+'\n')
        f.close()
        psi_comm = psi_p+' -db '+psi_db+' -show_gis  -evalue 100.0 -gapopen 11 -gapextend 1 -num_descriptions 5000 -num_alignments 5000 -matrix BLOSUM62 -query temp/p0.fasta -inclusion_ethresh 0.005 -num_iterations 3 -out_ascii_pssm piano/Data/PSIAns/'+seqName+'.pssm -num_threads 16 -comp_based_stats 2'
        os.system(psi_comm)


def get_hhm(fa):
    for seq_name, seq in readFa(fa):
        f = open('temp/a0.fasta', 'w')
        f.write('>' + seq_name + '\n')
        f.write(seq)
        f.close()
        hhblits_comm = hhm_p+'hhblits -cpu 8 -i {} -d {} -ohhm {}'.format('temp/a0.fasta', hhm_db, 'piano/Data/HHMAns/'+seq_name+'.hhm')
        os.system(hhblits_comm)

pre_data = pd.read_csv('pred_data.csv', sep=',')
pdb_l = []
data_list = []
for i in range(pre_data.shape[0]):
    pdb = pre_data['#PDB'][i].lower()
    part_c = list(pre_data['Partners(A_B)'][i].replace('_', ''))
    seq, ser = get_pdb_seq_bio(pdb, 'piano/Data/pdb/'+pdb+'.pdb')
    if pdb not in pdb_l:
        pdb_l.append(pdb)
        with open('temp/wt.fasta', 'w') as f1:
            for e in part_c:
                f1.write('>'+pdb+'_'+e+'\n')
                f1.write(seq[e]+'\n')
        try:
            get_f(pdb)
            get_pssm('temp/wt.fasta')
            get_hhm('temp/wt.fasta')
        except:
            print('Features error!')
    
    wt = pre_data['Mutation_wild'][i]
    mut = pre_data['Mutation_mut'][i]
    chain = pre_data['Mutation_chain'][i]
    ind = str(pre_data['Mutation_index'][i])

    mut_detail = pdb+'_'+wt+chain+ind+mut
    data_list.append(mut_detail)
    pre_seq = seq[chain]
    tar_id = 0
    for i in range(len(ser[e])):
        if ser[e][i] == ind:
            tar_id = i
    pre_seq = pre_seq[0:tar_id]+mut+pre_seq[tar_id+1:]
    with open('temp/mut.fasta', 'w') as f2:
        f2.write('>'+mut_detail+'\n')
        f2.write(pre_seq+'\n')
    try:
        get_pssm('temp/mut.fasta')
        get_hhm('temp/mut.fasta')
    except:
        print('Features error!')
np.save('pred_data_list.npy', data_list)
