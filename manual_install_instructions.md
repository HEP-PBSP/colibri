# Manual Installation w/o installing all of NNPDF

```
mamba create -n supernet -y && conda activate supernet
mamba install python=3.10 jax=0.4.11 ml_dtypes=0.2.0 optax flax
mamba install flit -c conda-forge -y
mamba install lhapdf prompt_toolkit seaborn
pip install reportengine validobj pineappl "ruamel.yaml<0.18.0" ultranest
```

Now we need to do some manual stuff:
Download the nnpdf repository.
```
cd nnpdf/validphys2
pip install -e .
```
Create a NNPDF folder in the conda environment and copy the nnprofile.yaml.in, modifying the paths at the beginning, e.g.
```
data_path: '@PROFILE_PREFIX@/data/' ->  data_path: '/Users/luca/opt/miniconda3/envs/nnpdf-dev/share/NNPDF/data/'`
```
Then continue:
```
mkdir ~/miniforge3/envs/supernet/share/NNPDF
cp nnpdf/libnnpdf/nnprofile.yaml.in ~/miniforge3/envs/supernet/share/NNPDF/nnprofile.yaml
cd ~/miniforge3/envs/supernet/share/NNPDF
mkdir results
cp -r nnpdf/nnpdfcpp/data .
```

Finally install supernet and the various apps:
```
cd super_net
flit install --symlink
cd wmin
flit install --symlink
```