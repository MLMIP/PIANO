import numpy as np
import os
import pandas as pd
import sys

os.system('mkdir -p temp')
flag = int(sys.argv[1])

if flag == 0:

    pdbname = sys.argv[2]
    chainid = sys.argv[3]
    wildtype = sys.argv[4]
    mutanttype = sys.argv[5]
    resid = sys.argv[6]
    part_c = sys.argv[7]
    print(pdbname)
    if len(pdbname) < 4:
        print('Please check your input!')
        sys.exit()
    
    if not os.path.exists('piano/Data/pdb/'+pdbname.lower()+'.pdb'):
        os.system('wget https://files.rcsb.org/download/'+pdbname+'.pdb')
        os.system('mv -u '+pdbname+'.pdb piano/Data/pdb/')
    ans_ske = pd.DataFrame(np.zeros((1, 6)),
                            columns=['#PDB', 'Partners(A_B)',
                                     'Mutation_chain', 'Mutation_index', 'Mutation_wild',
                                     'Mutation_mut',])
    ans_ske.loc[0, '#PDB'] = pdbname.lower()
    ans_ske.loc[0, 'Partners(A_B)'] = part_c
    ans_ske.loc[0, 'Mutation_chain'] = chainid
    ans_ske.loc[0, 'Mutation_index'] = resid
    ans_ske.loc[0, 'Mutation_wild'] = wildtype
    ans_ske.loc[0, 'Mutation_mut'] = mutanttype

    ans_ske.to_csv('pred_data.csv', sep=',', index=False, header=True)

elif flag == 1:
    pre_data = pd.read_csv('pred_data.csv', sep=',')
    pdb_l = []

    for i in range(pre_data.shape[0]):
        pdb = pre_data['#PDB'][i].lower()
        if pdb not in pdb_l:
            pdb_l.append(pdb)
            if not os.path.exists('piano/Data/pdb/'+pdb+'.pdb'):
                os.system('wget https://files.rcsb.org/download/'+pdb+'.pdb')
                os.system('mv -u '+pdb+'.pdb piano/Data/pdb/')
else:
    print('Please check your input')

print('FeatureExtraction......')
os.system('python piano/features.py')
print('FeatureExtraction Finished!')

print('Structural data sample generation......')
os.system('python piano/DownstreamDataGeneration_stru.py')
print('Structural data sample generation Finished!')

print('Sequence data sample generation......')
os.system('python piano/DownstreamDataGeneration_seq.py')
print('Sequence data sample generation Finished!')

print('Start prediction......')
os.system('python piano/prediction.py')
print('Prediction Finished!')
os.system('rm -rf temp')
