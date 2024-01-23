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

Executing the above command will automatically install the Anaconda virtual environment. Upon completion, a virtual environment named "piano" will be created. In addition, the model weights can be obtained through [Zenodo-PIANO](https://doi.org/10.5281/zenodo.10551028). After downloading, place it in piano/Data/model_params.

## Step 3\: Download required software

The download links for various software packages are provided below. After downloading, you can install it directly according to the official tutorial.

[PSI-BLAST](https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html)

[PSI-BLAST Database\:Uniref90](https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/)

[HHblits](https://github.com/soedinglab/hh-suite)

[HHblits Database\:Uniref30](https://gwdu111.gwdg.de/\~compbiol/uniclust/2023_02/)

[NACCESS](http://www.bioinf.manchester.ac.uk/naccess/)

[MSMS](https://ccsb.scripps.edu/msms/downloads/)

Please install these software in the piano/Software. In addition, [mkdssp](https://github.com/cmbi/hssp/releases) needs to be granted certain permissions, which can be operated by executing the following command：

```bash
chmod a+x mkdssp
```

## Step 4\: Running PIANO

Activate the installed piano virtual environment and ensure that the current working directory is PIANO.

```bash
conda activate piano
```

If you want to obtain the prediction results of a single sample, directly execute the following command on the command line to obtain the prediction results, and the results are saved in the PredictedResults.txt:

```bash
python run.py 0 [pdb name] [mut_chain] [wildtype] [mutant] [resid] [partnerA_partnerB]
```

where the digit 0 signifies that the program performs predictions for individual samples only. \[pdb name] is the name of the complex to be predicted, such as 1a4y. \[mut\_chain] is the name of the mutated chain. \[wildtype], \[mutant], and \[resid] are wild-type amino acid, mutant amino acid, and the mutation position, respectively. \[partnerA\_partnerB] describes the two interaction partners in the protein complex, such as A\_B.

A specific example is:

```bash
python run.py 0 1a4y A E A 401 A_B
```

If you want to perform batch prediction of multiple samples, please first organize the relevant mutation sample data in pred\_data.csv. The input format is consistent with the data format within an individual sample. After the completion of filling in pred\_data.csv, execute the following command in the command line to obtain the prediction results. The results are saved in the PredictedResults.txt:

```bash
python run.py 1
```

where the digit 1 indicates that the program conducts predictions for multiple samples.
