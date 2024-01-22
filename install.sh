if [ ! -f "piano/Data/model_params/piano_model.pkl" ];
then
	wget -P piano/Data/model_params/ https://zenodo.org/api/records/10512324/files-archive   # need to 
fi

conda create -n piano python==3.8 -y

source activate piano
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install scikit-learn
pip install biopython
pip install pandas
pip install networkx
conda install -c schrodinger pymol -y



