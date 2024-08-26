import numpy as np
import os
import pandas as pd
import sys

os.system('mkdir -p temp')
c_m = sys.argv[1]


if c_m == 'c':
    flag = int(sys.argv[2])
    if flag == 0:

        pdbname = sys.argv[3]
        chainid = sys.argv[4]
        wildtype = sys.argv[5]
        mutanttype = sys.argv[6]
        resid = sys.argv[7]
        part_c = sys.argv[8]
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
    os.system('python piano/features.py c')
    print('FeatureExtraction Finished!')

    print('Structural data sample generation......')
    os.system('python piano/DownstreamDataGeneration_struct.py')
    print('Structural data sample generation Finished!')

    print('Sequence data sample generation......')
    os.system('python piano/DownstreamDataGeneration_seq.py')
    print('Sequence data sample generation Finished!')

    print('Start prediction......')
    os.system('python piano/prediction.py')
    print('Prediction Finished!')
    os.system('rm -rf temp')

elif c_m == 'm':
    m1 = sys.argv[2]
    m2 = sys.argv[3]
    # print(m1)
    # print(m2)

    chainid = sys.argv[4]
    wildtype = sys.argv[5]
    mutanttype = sys.argv[6]
    resid = sys.argv[7]

    pdb_l = []
    pdb_l.append(m1.split('_')[0].lower())
    if m2.split('_')[0].lower() not in pdb_l:
        pdb_l.append(m2.split('_')[0].lower())
    for pdbname in pdb_l:
        if not os.path.exists('piano/Data/pdb/'+pdbname.lower()+'.pdb'):
            os.system('wget https://files.rcsb.org/download/'+pdbname+'.pdb')
            os.system('mv -u '+pdbname+'.pdb piano/Data/pdb/')
    apo_dict = {}
    key = m1[0:4].lower()+'_'+wildtype+chainid+resid+mutanttype
    apo_dict[key] = [m1, m2, [key]]
    np.save('apo_dict.npy', apo_dict)
    print('FeatureExtraction......')
    os.system('python piano/features.py m')
    print('FeatureExtraction Finished!')

    print('Structural data sample generation......')
    os.system('python piano/apo_single_mut.py')
    os.system('python piano/apo_single_partner.py')
    print('Structural data sample generation Finished!')

    print('Sequence data sample generation......')
    os.system('python piano/apo_seq.py')
    print('Sequence data sample generation Finished!')

    print('Start prediction......')
    os.system('python piano/apo_prediction.py')
    print('Prediction Finished!')
    os.system('rm -rf temp')
    
else:
    print('Please check your input')
