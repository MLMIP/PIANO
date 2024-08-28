# PIANO
## Graph masked self-distillation learning for prediction of mutation impact on protein-protein interactions.

This source code is tested with Python3.8 on Ubuntu20.04.

## Step 1\: Clone the GitHub repository

```bash
git clone https://github.com/MLMIP/PIANO.git
cd PIANO
```

## Step 2\: Build required dependencies
Before installing the relevant dependency packages, please make sure Anaconda3 has been installed. If not, please click [here](https://www.anaconda.com/download#downloads) to install it.

```bash
source install.sh
```
Running the above command will automatically set up the Anaconda virtual environment. Once completed, a virtual environment named 'piano' will be created. You can obtain the model weights from [Zenodo-PIANO](https://doi.org/10.5281/zenodo.13375314). After downloading, place the file in the piano/Data/model_params directory.

## Step 3\: Download required software

The download links for various software packages are provided below. After downloading, follow the official tutorial for installation.

[PSI-BLAST](https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html)

[PSI-BLAST Database\:Uniref90](https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/)

[HHblits](https://github.com/soedinglab/hh-suite)

[HHblits Database\:Uniref30](https://gwdu111.gwdg.de/\~compbiol/uniclust/2023_02/)

[NACCESS](http://www.bioinf.manchester.ac.uk/naccess/)

[MSMS](https://ccsb.scripps.edu/msms/downloads/)

Please install these software packages in the `piano/Software` directory. Additionally, [mkdssp](https://github.com/cmbi/hssp/releases) requires specific permissions, which can be set by executing the following command:
```bash
chmod a+x mkdssp
```

## Step 4\: Running PIANO

Activate the installed piano virtual environment and ensure that the current working directory is PIANO.

```bash
conda activate piano
```

To obtain the prediction results for a single sample, execute the following command in the terminal. The results will be saved in `PredictedResults.txt`:

```bash
python run.py c 0 [pdb name] [mut_chain] [wildtype] [mutant] [resid] [partnerA_partnerB]
```

where `0` indicates that the program will only perform predictions for individual samples. `c` refers the input of a complex structure. `[pdb_name]` is the name of the complex to be predicted (e.g., 1a4y). `[mut_chain]` refers to the chain containing the mutation. `[wildtype]`, `[mutant]`, and `[resid]` represent the wild-type amino acid, mutant amino acid, and mutation position, respectively. `[partnerA_partnerB]` describes the two interacting partners in the protein complex, such as A\_B.

A specific example is:

```bash
python run.py c 0 1a4y A E A 401 A_B
```

To perform batch predictions for multiple samples, first organize the relevant mutation data in `pred_data.csv`. The input format should match that used for individual samples. Once `pred_data.csv` is complete, execute the following command in the terminal to obtain the prediction results. The results will be saved in `PredictedResults.txt`.

```bash
python run.py c 1
```
where `1` indicates that the program is set to perform predictions for multiple samples.`c` refers the input of a complex structure.
Additionally, if using monomer structures as input, you can obtain the prediction results by executing the following command in the terminal. The results will be saved in `predicted_results.txt`.
```bash
python run.py m [apo mutation chain] [apo partner chain] [mut_chain] [wildtype] [mutant] [resid]
python run.py m 1a4y_A 1a4y_B A E A 401
```
where `m` refers the input of monomer structures.`[apo mutation chain]` indicates the monomer structure of the mutated chain, while `[apo partner chain]` indicates the monomer structure of the partner chain.
