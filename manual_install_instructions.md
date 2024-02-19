# Manual Installation w/o installing all of NNPDF

First we need to create a conda environment with the following packages:
```
mamba create -n colibri -y && mamba activate colibri
mamba install python=3.11 flit mpich mpi4py lhapdf prompt_toolkit seaborn h5py dask rust eko pandoc -c conda-forge -y
pip install validobj pineappl "ruamel.yaml<0.18.0" ultranest jax optax flax dill
```
If running on HPC GPU, add the following
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Now we need to do some manual stuff:
Download the nnpdf repository. 
We need to checkout to a specific commit where validphys was still a separate Python package.
```
git clone https://github.com/NNPDF/nnpdf.git
cd nnpdf/validphys2
git checkout ce6c05c
pip install -e .
```
Do the same for reportengine, clone the repository and then
```
git clone https://github.com/NNPDF/reportengine.git
cd reportengine
pip install -e .
```
Then continue:
```
mkdir ~/miniconda3/envs/colibri/share/NNPDF
cp nnpdf/libnnpdf/nnprofile.yaml.in ~/miniconda3/envs/colibri/share/NNPDF/nnprofile.yaml
cd ~/miniconda3/envs/colibri/share/NNPDF
mkdir results
cp -r nnpdf/nnpdfcpp/data .
```

We need to modify the first two paths at the beginning of nnprofile.yaml:
```
data_path: '@PROFILE_PREFIX@/data/' ->  data_path: '/Users/YourUsername/miniconda3/envs/nnpdf-dev/share/NNPDF/data/'
results_path: '@PROFILE_PREFIX@/results/' -> results_path: '/Users/YourUsername/miniconda3/envs/colibri/share/NNPDF/results/'
```

Finally install colibri and the various models:
```
cd colibri
flit install --symlink
cd models/wmin
flit install --symlink
cd models/grid_pdf
flit install --symlink
```
