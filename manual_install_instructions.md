# Manual Installation w/o installing all of NNPDF

```
mamba create -n supernet -y && conda activate supernet
mamba install python=3.10 jax=0.4.13 ml_dtypes optax=0.1.7 flax chex=0.1.83
mamba install flit -c conda-forge -y
mamba install lhapdf prompt_toolkit seaborn h5py dask
pip install validobj pineappl "ruamel.yaml<0.18.0" ultranest
```

Now we need to do some manual stuff:
Download the nnpdf repository.
```
cd nnpdf/validphys2
pip install -e .
```
Do the same for reportengine, clone the repository and then
```
cd reportengine
pip install -e .
```
Then continue:
```
mkdir ~/miniforge3/envs/supernet/share/NNPDF
cp nnpdf/libnnpdf/nnprofile.yaml.in ~/miniforge3/envs/supernet/share/NNPDF/nnprofile.yaml
cd ~/miniforge3/envs/supernet/share/NNPDF
mkdir results
cp -r nnpdf/nnpdfcpp/data .
```
Modifying the paths at the beginning of nnprofile.yaml, e.g.
```
data_path: '@PROFILE_PREFIX@/data/' ->  data_path: '/Users/luca/opt/miniconda3/envs/nnpdf-dev/share/NNPDF/data/'`
```

Finally install supernet and the various models:
```
cd super_net
flit install --symlink
cd models/wmin
flit install --symlink
cd models/grid_pdf
flit install --symlink
```
