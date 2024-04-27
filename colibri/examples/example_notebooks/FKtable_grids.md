# Interpolation of FKtable on new x-grid


```python
from validphys.coredata import FKTableData
from validphys.api import API
from validphys.fkparser import load_fktable
import numpy as np
import pandas as pd
from scipy.interpolate import (interp1d,RegularGridInterpolator, 
                               LinearNDInterpolator, interp2d, griddata,
                              NearestNDInterpolator, interpn)
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import warnings
# interp2d is deprecated and scipy complains
warnings.filterwarnings("ignore")
```

    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
    Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.


### Simple Example of Interpolation using scipy.interpolate


```python
x = np.array([1,2,3])
y = x
# default `kind` for spline interpolator is linear
f = interp1d(x, y, fill_value="extrapolate")
x_new = np.linspace(0.1,1,10)

f(x_new) - x_new
```




    array([-2.77555756e-17, -5.55111512e-17,  0.00000000e+00,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00])




```python
inp = {
    "fit": "210713-n3fit-001",
    "dataset_inputs": {"from_": "fit"},
    "use_cuts": "internal",
    "theoryid": 200
}

inp_400 = {
    "fit": "210713-n3fit-001",
    "dataset_inputs": {"from_": "fit"},
    "use_cuts": "internal",
    "theoryid": 400
}
```


```python
data = API.data(**inp)
data_400 = API.data(**inp_400)
```


```python
datasets = data.datasets
datasets_400 = data_400.datasets

fktable_dis = load_fktable(datasets[0].fkspecs[0])
fktable_dis_400 = load_fktable(datasets_400[0].fkspecs[0])

fktable_had = load_fktable(datasets[22].fkspecs[0])
fktable_had_400 = load_fktable(datasets_400[22].fkspecs[0])

new_xgrid = fktable_dis_400.xgrid
```

## DIS FKTable interpolation


```python
def interp1d_(fktable, fktable_400):
    """
    """
    
    xgrid = fktable.xgrid
    xgrid_new = fktable_400.xgrid

    dfs=[]
    for d, grp in fktable.sigma.groupby('data'): 

        x_vals = xgrid[grp.index.get_level_values('x')]

        interpolators = {col: interp1d(x_vals, grp[col].values, kind='slinear', fill_value=0, bounds_error=False)
                        for col in grp.columns}
        
        tmp_index = pd.MultiIndex.from_product([[d],range(len(xgrid_new))], names=['data','x'])

        d = dict()
        
        for col in grp.columns:
            d[f"{col}"] = interpolators[col](xgrid_new)

        tmp_df = pd.DataFrame(d,index=tmp_index)

        dfs.append(tmp_df)
    
    return pd.concat(dfs, axis=0)


```

## Test of DIS interpolation


```python
import time

# probably need to ignore the broken datasets as in the imagepdf nb

for ds, ds_400 in zip(datasets, datasets_400):
    ds_name = str(ds)

    for fk, fk_400 in zip(ds.fkspecs, ds_400.fkspecs):
        tab=load_fktable(fk)
        tab_400=load_fktable(fk_400)        

        if not (tab.hadronic and tab_400.hadronic):
            
            print()
            print(f"Dataset = {ds_name}")
            t0=time.time()
            interp1d_(tab, tab_400)
            t1=time.time()
            print(f'time needed for griddata = {t1-t0}')
            print()
```

    
    Dataset = NMCPD_dw_ite
    time needed for griddata = 0.15586400032043457
    
    
    Dataset = NMCPD_dw_ite
    time needed for griddata = 0.15121102333068848
    
    
    Dataset = NMC
    time needed for griddata = 0.31128811836242676
    
    
    Dataset = SLACP_dwsh
    time needed for griddata = 0.11115813255310059
    
    
    Dataset = SLACD_dw_ite
    time needed for griddata = 0.10625815391540527
    
    
    Dataset = BCDMSP_dwsh
    time needed for griddata = 0.30986690521240234
    
    
    Dataset = BCDMSD_dw_ite
    time needed for griddata = 0.20801424980163574
    
    
    Dataset = CHORUSNUPb_dw_ite
    time needed for griddata = 0.6159191131591797
    
    
    Dataset = CHORUSNBPb_dw_ite
    time needed for griddata = 0.5928218364715576
    
    
    Dataset = NTVNUDMNFe_dw_ite
    time needed for griddata = 0.05926990509033203
    
    
    Dataset = NTVNBDMNFe_dw_ite
    time needed for griddata = 0.04531574249267578
    
    
    Dataset = HERACOMBNCEM
    time needed for griddata = 0.1915600299835205
    
    
    Dataset = HERACOMBNCEP460
    time needed for griddata = 0.2445387840270996
    
    
    Dataset = HERACOMBNCEP575
    time needed for griddata = 0.32440900802612305
    
    
    Dataset = HERACOMBNCEP820
    time needed for griddata = 0.08336305618286133
    
    
    Dataset = HERACOMBNCEP920
    time needed for griddata = 0.44548487663269043
    
    
    Dataset = HERACOMBCCEM
    time needed for griddata = 0.04938817024230957
    
    
    Dataset = HERACOMBCCEP
    time needed for griddata = 0.0921788215637207
    
    
    Dataset = HERACOMB_SIGMARED_C
    time needed for griddata = 0.04336881637573242
    
    
    Dataset = HERACOMB_SIGMARED_B
    time needed for griddata = 0.0192410945892334
    



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[7], line 9
          6 ds_name = str(ds)
          8 for fk, fk_400 in zip(ds.fkspecs, ds_400.fkspecs):
    ----> 9     tab=load_fktable(fk)
         10     tab_400=load_fktable(fk_400)        
         12     if not (tab.hadronic and tab_400.hadronic):


    File ~/miniconda3/envs/colibri/lib/python3.9/site-packages/validphys/fkparser.py:57, in load_fktable(spec)
         55 if spec.legacy:
         56     with open_fkpath(spec.fkpath) as handle:
    ---> 57         tabledata = parse_fktable(handle)
         58 else:
         59     tabledata = pineappl_reader(spec)


    File ~/miniconda3/envs/colibri/lib/python3.9/site-packages/validphys/fkparser.py:306, in parse_fktable(f)
        304 _check_required_sections(res, lineno)
        305 Q0 = res['TheoryInfo']['Q0']
    --> 306 sigma = _build_sigma(f, res)
        307 hadronic = res['GridInfo'].hadronic
        308 ndata = res['GridInfo'].ndata


    File ~/miniconda3/envs/colibri/lib/python3.9/site-packages/validphys/fkparser.py:217, in _build_sigma(f, res)
        214 gi = res["GridInfo"]
        215 fm = res["FlavourMap"]
        216 table = (
    --> 217     _parse_hadronic_fast_kernel(f) if gi.hadronic else _parse_dis_fast_kernel(f)
        218 )
        219 # Filter out empty flavour indices
        220 table = table.loc[:, fm.ravel()]


    File ~/miniconda3/envs/colibri/lib/python3.9/site-packages/validphys/fkparser.py:173, in _parse_hadronic_fast_kernel(f)
        168 """Parse the FastKernel secrion of an hadronic FKTable into a DataFrame.
        169 ``f`` should be a stream containing only the section"""
        170 # Note that we need the slower whitespace here because it turns out
        171 # that there are fktables where space and tab are used as separators
        172 # within the same table.
    --> 173 df = pd.read_csv(f, sep=r'\s+', header=None, index_col=(0,1,2))
        174 df.columns = list(range(14*14))
        175 df.index.names = ['data', 'x1', 'x2']


    File ~/miniconda3/envs/colibri/lib/python3.9/site-packages/pandas/util/_decorators.py:211, in deprecate_kwarg.<locals>._deprecate_kwarg.<locals>.wrapper(*args, **kwargs)
        209     else:
        210         kwargs[new_arg_name] = new_arg_value
    --> 211 return func(*args, **kwargs)


    File ~/miniconda3/envs/colibri/lib/python3.9/site-packages/pandas/util/_decorators.py:331, in deprecate_nonkeyword_arguments.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
        325 if len(args) > num_allow_args:
        326     warnings.warn(
        327         msg.format(arguments=_format_argument_list(allow_args)),
        328         FutureWarning,
        329         stacklevel=find_stack_level(),
        330     )
    --> 331 return func(*args, **kwargs)


    File ~/miniconda3/envs/colibri/lib/python3.9/site-packages/pandas/io/parsers/readers.py:950, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, error_bad_lines, warn_bad_lines, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options)
        935 kwds_defaults = _refine_defaults_read(
        936     dialect,
        937     delimiter,
       (...)
        946     defaults={"delimiter": ","},
        947 )
        948 kwds.update(kwds_defaults)
    --> 950 return _read(filepath_or_buffer, kwds)


    File ~/miniconda3/envs/colibri/lib/python3.9/site-packages/pandas/io/parsers/readers.py:611, in _read(filepath_or_buffer, kwds)
        608     return parser
        610 with parser:
    --> 611     return parser.read(nrows)


    File ~/miniconda3/envs/colibri/lib/python3.9/site-packages/pandas/io/parsers/readers.py:1778, in TextFileReader.read(self, nrows)
       1771 nrows = validate_integer("nrows", nrows)
       1772 try:
       1773     # error: "ParserBase" has no attribute "read"
       1774     (
       1775         index,
       1776         columns,
       1777         col_dict,
    -> 1778     ) = self._engine.read(  # type: ignore[attr-defined]
       1779         nrows
       1780     )
       1781 except Exception:
       1782     self.close()


    File ~/miniconda3/envs/colibri/lib/python3.9/site-packages/pandas/io/parsers/c_parser_wrapper.py:230, in CParserWrapper.read(self, nrows)
        228 try:
        229     if self.low_memory:
    --> 230         chunks = self._reader.read_low_memory(nrows)
        231         # destructive to chunks
        232         data = _concatenate_chunks(chunks)


    File ~/miniconda3/envs/colibri/lib/python3.9/site-packages/pandas/_libs/parsers.pyx:808, in pandas._libs.parsers.TextReader.read_low_memory()


    File ~/miniconda3/envs/colibri/lib/python3.9/site-packages/pandas/_libs/parsers.pyx:866, in pandas._libs.parsers.TextReader._read_rows()


    File ~/miniconda3/envs/colibri/lib/python3.9/site-packages/pandas/_libs/parsers.pyx:852, in pandas._libs.parsers.TextReader._tokenize_rows()


    File ~/miniconda3/envs/colibri/lib/python3.9/site-packages/pandas/_libs/parsers.pyx:1965, in pandas._libs.parsers.raise_parser_error()


    File ~/miniconda3/envs/colibri/lib/python3.9/_compression.py:68, in DecompressReader.readinto(self, b)
         66 def readinto(self, b):
         67     with memoryview(b) as view, view.cast("B") as byte_view:
    ---> 68         data = self.read(len(byte_view))
         69         byte_view[:len(data)] = data
         70     return len(data)


    File ~/miniconda3/envs/colibri/lib/python3.9/gzip.py:495, in _GzipReader.read(self, size)
        492 # Read a chunk of data from the file
        493 buf = self._fp.read(io.DEFAULT_BUFFER_SIZE)
    --> 495 uncompress = self._decompressor.decompress(buf, size)
        496 if self._decompressor.unconsumed_tail != b"":
        497     self._fp.prepend(self._decompressor.unconsumed_tail)


    KeyboardInterrupt: 



```python
print(datasets[0])
dis_200 = load_fktable(datasets[0].fkspecs[1])
sigma_dis_200 = dis_200.sigma
# get only first data point
sigma_dis_200 = sigma_dis_200[sigma_dis_200.index.get_level_values('data')==31]
xgrid_dis_200 = dis_200.xgrid[sigma_dis_200.index.get_level_values('x')]

dis_400 = load_fktable(datasets_400[0].fkspecs[0])
sigma_dis_400 = dis_400.sigma
# get only first data point
sigma_dis_400 = sigma_dis_400[sigma_dis_400.index.get_level_values('data')==31]
xgrid_dis_400 = dis_400.xgrid[sigma_dis_400.index.get_level_values('x')]

# sigma_dis_interpolated = interp1d_(load_fktable(datasets[0].fkspecs[0]), load_fktable(datasets_400[0].fkspecs[0]))
sigma_dis_interpolated = interp1d_(load_fktable(datasets[0].fkspecs[0]), load_fktable(datasets_400[0].fkspecs[0]))
sigma_dis_interpolated = sigma_dis_interpolated[sigma_dis_interpolated.index.get_level_values('data')==31]
```

    NMCPD_dw_ite



```python
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=xgrid_dis_200, y=sigma_dis_200[1],
                    mode='lines+markers',
                    name='Theory 200 grid'))

fig.add_trace(go.Scatter(x=xgrid_dis_400,
                         y=sigma_dis_interpolated['1'],
                    mode='lines+markers',
                    name='Linear Spline Interpolation Th 200'))



fig.add_trace(go.Scatter(x=xgrid_dis_400, y=sigma_dis_400[1],
                    mode='lines+markers',
                    name='Theory 400 grid'))



# fig.add_trace(go.Scatter(x=full_df.x_new, y=full_df['1_new_quad'],
#                     mode='lines+markers',
#                     name='Quadratic Spline Interpolation'))

# fig.add_trace(go.Scatter(x=full_df.x_new, y=full_df['1_new_cube'],
#                     mode='lines+markers',
#                     name='Cubic Spline Interpolation'))

fig.show()
```


<div>                            <div id="7da3e447-949c-4fb5-8eea-117bcfdb8522" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("7da3e447-949c-4fb5-8eea-117bcfdb8522")) {                    Plotly.newPlot(                        "7da3e447-949c-4fb5-8eea-117bcfdb8522",                        [{"mode":"lines+markers","name":"Theory 200 grid","x":[0.006462086188072037,0.008211413818346237,0.010406430101627047,0.013145242377006214,0.016539677064696088,0.020713544990341646,0.025799184275975186,0.031932166531904796,0.03924441821929895,0.04785645013746829,0.05786975813970817,0.06936058914559116,0.08237606093198396,0.09693312844280857,0.11302028259806587,0.13060136558252977,0.1496206304489491,0.17000817878382546,0.19168509767881273,0.21456787498065333,0.23857191133923947,0.26361412378371596,0.28961474037494994,0.31649843172242276,0.34419493216439917,0.3726392886892116,0.40177185182881514,0.4315380974447697,0.4618883454802988,0.492777422893488,0.5241643033000918,0.5560117448787213,0.5882859401636978,0.6209561857723189,0.6539945762940527,0.6873757240323636,0.721076504650414,0.7550758277669313,0.7893544309706461,0.8238946954308234,0.8586804811765834,0.8936969801311311,0.9289305850712899,0.9643687728045999],"y":[0.013610492460429668,0.32843849062919617,-0.3177516758441925,0.4095399677753449,-0.19767372310161593,0.05769628286361694,-0.0072703263722360125,0.003850334323942661,0.002066008746623993,0.001697177649475634,0.0013284384040161967,0.0010505188256502151,0.0008483259589411318,0.0007001857738941908,0.0005920151597820222,0.0005129195051267743,0.00045524336746893823,0.00041097291978076095,0.00037707842420786614,0.0003517012228257954,0.0003309185558464378,0.00031332951039075846,0.000298800237942487,0.00028669816674664617,0.0002758092596195638,0.0002657259174156934,0.00025578521308489144,0.0002490580955054611,0.00024161722103599456,0.00023465292179025712,0.00022810461814515293,0.0002220404276158661,0.00021637015743181107,0.0002108952758135274,0.000205698175705038,0.0002006861614063382,0.0001958767679752782,0.00019109112326987088,0.00018677502521313727,0.00018273277964908632,0.00017888678121380508,0.00017505075084045527,0.0001714187819743529,0.00016785717161837965],"type":"scatter"},{"mode":"lines+markers","name":"Linear Spline Interpolation Th 200","x":[1.9999999999999954e-07,3.034304765867952e-07,4.6035014748963906e-07,6.984208530700364e-07,1.0596094959101024e-06,1.607585498470808e-06,2.438943292891682e-06,3.7002272069854957e-06,5.613757716930151e-06,8.516806677573355e-06,1.292101569074731e-05,1.9602505002391748e-05,2.97384953722449e-05,4.511438394964044e-05,6.843744918967896e-05,0.00010381172986576898,0.00015745605600841445,0.00023878782918561914,0.00036205449638139736,0.0005487795323670796,0.0008314068836488144,0.0012586797144272762,0.0019034634022867384,0.0028738675812817515,0.004328500638820811,0.006496206194633799,0.009699159574043398,0.014375068581090129,0.02108918668378717,0.030521584007828916,0.04341491741702269,0.060480028754447364,0.08228122126204893,0.10914375746330703,0.14112080644440345,0.17802566042569432,0.2195041265003886,0.2651137041582823,0.31438740076927585,0.3668753186482242,0.4221667753589648,0.4798989029610255,0.5397572337880445,0.601472197967335,0.6648139482473823,0.7295868442414312,0.7956242522922756,0.8627839323906108,0.9309440808717544,1.0],"y":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.01975109938702404,-0.1095385234215183,0.18954241966307928,0.052897639912334725,0.001292588519437729,0.0018873970311346971,0.0012653058674487352,0.000849799275087993,0.0006180810568574901,0.0004810191807191719,0.0003984366134296341,0.00034742742195628825,0.0003124915375010116,0.0002876484769760488,0.00026776920787738604,0.00025117599731811467,0.0002375565328451333,0.00022513550989966292,0.000214160403598677,0.0002040737004965117,0.000194678877609479,0.00018604126831171084,0.00017843725130754575,0.0001712164215278027,0.0],"type":"scatter"},{"mode":"lines+markers","name":"Theory 400 grid","x":[1.9999999999999954e-07,3.034304765867952e-07,4.6035014748963906e-07,6.984208530700364e-07,1.0596094959101024e-06,1.607585498470808e-06,2.438943292891682e-06,3.7002272069854957e-06,5.613757716930151e-06,8.516806677573355e-06,1.292101569074731e-05,1.9602505002391748e-05,2.97384953722449e-05,4.511438394964044e-05,6.843744918967896e-05,0.00010381172986576898,0.00015745605600841445,0.00023878782918561914,0.00036205449638139736,0.0005487795323670796,0.0008314068836488144,0.0012586797144272762,0.0019034634022867384,0.0028738675812817515,0.004328500638820811,0.006496206194633799,0.009699159574043398,0.014375068581090129,0.02108918668378717,0.030521584007828916,0.04341491741702269,0.060480028754447364,0.08228122126204893,0.10914375746330703,0.14112080644440345,0.17802566042569432,0.2195041265003886,0.2651137041582823,0.31438740076927585,0.3668753186482242,0.4221667753589648,0.4798989029610255,0.5397572337880445,0.601472197967335,0.6648139482473823,0.7295868442414312,0.7956242522922756,0.8627839323906108,0.9309440808717544,1.0],"y":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.3575792031968011e-12,1.6859392479550802e-06,-0.014312599882404,0.13885470213407738,0.10990634548471011,0.06587401687144452,-0.017809172458870225,0.008050032920748832,0.0031825827492339745,0.0021003522932367662,0.001390251653186891,0.0009831713184126616,0.0007463249322032248,0.0006072365052776278,0.0005224533117269683,0.0004675062470826981,0.00042920088578586,0.00040049902470882604,0.0003776210948303441,0.00035831193038073664,0.0003412402033203867,0.00032577022896865254,0.0003117734096932652,0.0002967212078762319,0.0003017843753274685,0.0002333905539475964,0.0003355598374216454,8.768209535503794e-05],"type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('7da3e447-949c-4fb5-8eea-117bcfdb8522');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


## Hadronic FKTable Interpolation 

### Example1: Using deprecated interp2d

### Simple Example of 2D interpolation:

Given 3 data points and for each of them a function evaluated on a 2x2 xgrid. The function we want to fit is: f(x1,x2,data)=x1+x2 + data



```python
ex_xgrid = np.array([0.1,0.2])

ex_dict = {"data":[0,0,0,0,1,1,1,1,2,2,2,2],"x1":[0,0,1,1,0,0,1,1,0,0,1,1],"x2":[0,1,0,1,0,1,0,1,0,1,0,1],
          "col1":[0.2,0.3,0.3,0.4,1.2,1.3,1.3,1.4,2.2,2.3,2.3,2.4]}

ex_df = pd.DataFrame.from_dict(ex_dict)
ex_df.set_index(['data', 'x1', 'x2'], inplace=True)


ex_dfs = []

for d, grp in ex_df.groupby('data'):
    x1_vals = ex_xgrid[grp.index.get_level_values('x1')]
    x2_vals = ex_xgrid[grp.index.get_level_values('x2')]

    interpolator = interp2d(x1_vals,x2_vals, grp['col1'])

    # test the interpolator
    new_ex_xgrid = np.array([0.1,0.15,0.2])
    len_grid = len(new_ex_xgrid)
    
    interpolated_grid = interpolator(new_ex_xgrid,new_ex_xgrid)
    
    tmp_index = pd.MultiIndex.from_product([range(3),range(3)], names=['x1','x2'])
    tmp_df = pd.DataFrame({'col1':interpolated_grid.flatten()},index=tmp_index)
    
    ex_dfs.append(tmp_df)

        
pd.concat(ex_dfs, axis=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>col1</th>
    </tr>
    <tr>
      <th>x1</th>
      <th>x2</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">0</th>
      <th>0</th>
      <td>0.20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.30</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">1</th>
      <th>0</th>
      <td>0.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.35</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2</th>
      <th>0</th>
      <td>0.30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.40</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">0</th>
      <th>0</th>
      <td>1.20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.30</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">1</th>
      <th>0</th>
      <td>1.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.35</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2</th>
      <th>0</th>
      <td>1.30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.40</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">0</th>
      <th>0</th>
      <td>2.20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.30</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">1</th>
      <th>0</th>
      <td>2.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.35</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2</th>
      <th>0</th>
      <td>2.30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.40</td>
    </tr>
  </tbody>
</table>
</div>




```python


def interp2d_(fktable, fktable_400):
    """
    """
    
    xgrid = fktable.xgrid
    xgrid_new = fktable_400.xgrid

    dfs=[]
    for d, grp in fktable.sigma.groupby('data'): 

        x1_vals = xgrid[grp.index.get_level_values('x1')]
        x2_vals = xgrid[grp.index.get_level_values('x2')]

        interpolators = [interp2d(x1_vals,x2_vals, grp[col].values)
                        for col in grp.columns]

        tmp_index = pd.MultiIndex.from_product([[d],range(50),range(50)], names=['data','x1','x2'])

        d = dict()

        for col, interp in zip(grp.columns, interpolators):
            d[f"{col}"] = interp(xgrid_new, xgrid_new).flatten()

        tmp_df = pd.DataFrame(d,index=tmp_index)

        dfs.append(tmp_df)
    
    return pd.concat(dfs, axis=0)

```

### Example2 Using: griddata

#### Simple example


```python
ex_xgrid = np.array([0.1,0.2])

ex_dict = {"data":[0,0,0,0,1,1,1,1,2,2,2,2],"x1":[0,0,1,1,0,0,1,1,0,0,1,1],"x2":[0,1,0,1,0,1,0,1,0,1,0,1],
          "col1":[0.2,0.3,0.3,0.4,1.2,1.3,1.3,1.4,2.2,2.3,2.3,2.4]}

ex_df = pd.DataFrame.from_dict(ex_dict)
ex_df.set_index(['data', 'x1', 'x2'], inplace=True)


ex_dfs = []

for d, grp in ex_df.groupby('data'):
    x1_vals = ex_xgrid[grp.index.get_level_values('x1')]
    x2_vals = ex_xgrid[grp.index.get_level_values('x2')]
    

    # test the interpolator
    new_ex_xgrid = np.array([0.05,0.15,0.25])
    len_grid = len(new_ex_xgrid)
    
    new_ex_xgrid_mesh = np.meshgrid(new_ex_xgrid, new_ex_xgrid)

    # Flatten the grid into two separate 1D arrays
    new_ex_xgrid_flat = np.ravel(new_ex_xgrid_mesh[0])
    new_ex_ygrid_flat = np.ravel(new_ex_xgrid_mesh[1])

    # Evaluate griddata at all combinations of new_ex_xgrid values
    interpolated_grid = griddata(points=(x1_vals, x2_vals), 
                            values=grp['col1'], 
                            xi=(new_ex_xgrid_flat, new_ex_ygrid_flat), method='linear', fill_value=0)


    
    tmp_index = pd.MultiIndex.from_product([range(len(new_ex_xgrid)),range(len(new_ex_xgrid))], names=['x1','x2'])
    tmp_df = pd.DataFrame({'col1':interpolated_grid},index=tmp_index)
    
    ex_dfs.append(tmp_df)

        
pd.concat(ex_dfs, axis=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>col1</th>
    </tr>
    <tr>
      <th>x1</th>
      <th>x2</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">0</th>
      <th>0</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">1</th>
      <th>0</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2</th>
      <th>0</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">0</th>
      <th>0</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">1</th>
      <th>0</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2</th>
      <th>0</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">0</th>
      <th>0</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">1</th>
      <th>0</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2</th>
      <th>0</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# # possibility, but quite slow
# grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
# grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')

# fill_value = 123  # Whatever you like
# grid_z0[np.isnan(grid_z1)] = fill_value
```


```python
def griddata_(fktable, fktable_400):
    """
    """
    
    xgrid = fktable.xgrid
    xgrid_new = fktable_400.xgrid

    dfs=[]
    for d, grp in fktable.sigma.groupby('data'): 

        x1_vals = xgrid[grp.index.get_level_values('x1')]
        x2_vals = xgrid[grp.index.get_level_values('x2')]

        new_xgrid_mesh = np.meshgrid(xgrid_new, xgrid_new)

        # Flatten the grid into two separate 1D arrays
        new_x1grid_flat = np.ravel(new_xgrid_mesh[0])
        new_x2grid_flat = np.ravel(new_xgrid_mesh[1])

#         interpolated_grids = {col: griddata(points=(x1_vals,x2_vals), values=grp[col].values,
#                                       xi=(new_x1grid_flat,new_x2grid_flat), method='nearest')
#                         for col in grp.columns}
        
        interpolated_grids = {col: griddata(points=(x1_vals,x2_vals), values=grp[col].values,
                              xi=(new_x1grid_flat,new_x2grid_flat), method='linear', fill_value=0)
                for col in grp.columns}

        tmp_index = pd.MultiIndex.from_product([[d],range(len(xgrid_new)),range(len(xgrid_new))], names=['data','x1','x2'])

        col_dict = dict()

        # set to zero in extrapolation regions
        idx_x1_m = np.where(new_x1grid_flat<np.min(x1_vals))[0]
        idx_x1_p = np.where(new_x1grid_flat>np.max(x1_vals))[0]
        idx_x1 = np.unique(np.concatenate((idx_x1_m, idx_x1_p)))
        
        idx_x2_m = np.where(new_x2grid_flat<np.min(x2_vals))[0]
        idx_x2_p = np.where(new_x2grid_flat>np.max(x2_vals))[0]
        idx_x2 = np.unique(np.concatenate((idx_x2_m, idx_x2_p)))
        
        extrapolation_region = np.unique(np.concatenate((idx_x1,idx_x2)))    

        for col in grp.columns:            
#             interpolated_grids[col][extrapolation_region]=0
            col_dict[f"{col}"] = interpolated_grids[col]
        
        tmp_df = pd.DataFrame(col_dict,index=tmp_index)

        dfs.append(tmp_df)

    return pd.concat(dfs, axis=0)
```

## Test of Hadronic Implementation


```python
ds=20
dp=4

had_200 = load_fktable(datasets[ds].fkspecs[0])
sigma_had_200 = had_200.sigma
# get only 1 datapoint
sigma_had_200 = sigma_had_200[sigma_had_200.index.get_level_values('data')==dp]
# fixed x1, sigma as func of x2
sigma_had_200 = sigma_had_200[sigma_had_200.index.get_level_values('x1')==sigma_had_200.index.get_level_values('x1')[33]]

x2grid_had_200 = had_200.xgrid[sigma_had_200.index.get_level_values('x2')]
x1val_200 = np.unique(had_200.xgrid[sigma_had_200.index.get_level_values('x1')])

sigma_had_200

had_400 = load_fktable(datasets_400[ds].fkspecs[0])
sigma_had_400 = had_400.sigma
# get only 1 datapoint
sigma_had_400 = sigma_had_400[sigma_had_400.index.get_level_values('data')==dp]
# fixed x1, sigma as func of x2
x1_idx = np.where(np.abs(had_400.xgrid - x1val_200) == np.min(np.abs(had_400.xgrid - x1val_200)) )[0][0]
sigma_had_400 = sigma_had_400[sigma_had_400.index.get_level_values('x1')==x1_idx]

x2grid_had_400 = had_400.xgrid[sigma_had_400.index.get_level_values('x2')]

sigma_had_interpolated = griddata_(load_fktable(datasets[ds].fkspecs[0]), load_fktable(datasets_400[ds].fkspecs[0]))
# get only 1 datapoint
sigma_had_interpolated = sigma_had_interpolated[sigma_had_interpolated.index.get_level_values('data')==dp]
# fixed x1, sigma as func of x2
sigma_had_interpolated = sigma_had_interpolated[sigma_had_interpolated.index.get_level_values('x1')==x1_idx]

```


```python
fig = go.Figure()
col=15
# Add traces
fig.add_trace(go.Scatter(x=x2grid_had_200, y=sigma_had_200[col],
                    mode='lines+markers',
                    name='Theory 200 grid'))


fig.add_trace(go.Scatter(x=x2grid_had_400, y=sigma_had_400[col],
                    mode='lines+markers',
                    name='Theory 400 grid'))

fig.add_trace(go.Scatter(x=x2grid_had_400,
                         y=sigma_had_interpolated[f'{col}'],
                    mode='lines+markers',
                    name='2D Interpolation Th 200'))



fig.show()
```


<div>                            <div id="e11d77f2-22ca-4560-b38a-33bde05e6f9b" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("e11d77f2-22ca-4560-b38a-33bde05e6f9b")) {                    Plotly.newPlot(                        "e11d77f2-22ca-4560-b38a-33bde05e6f9b",                        [{"mode":"lines+markers","name":"Theory 200 grid","x":[0.1513691864136627,0.1718072020185354,0.19351950774110482,0.21642316669124803,0.24043441342442204,0.26547114785744835,0.2914546456721418,0.318310629525828,0.34596985003778685,0.37436831014382704,0.40344724285484723,0.43315292784227316,0.46343641021473797,0.494253166710632,0.5255627504338757,0.5573284347365492,0.5895168692559658,0.6220977557689522,0.6550435478676653,0.6883291760310861,0.7219317980960877,0.7558305741699504,0.7900064644737235,0.8244420483287564,0.8591213624003013,0.8940297563271963],"y":[0.18225168647646905,1.0490174342536924,0.26929575316071513,-0.08426702954292298,0.046448550272136936,0.048851375808119775,0.018510924029499293,0.01574986524615437,0.014746610250659287,0.012295299571305514,0.010689684822186829,0.009558908556997776,0.008626838553082198,0.0079042628101632,0.007251609638929366,0.006721795297656209,0.0062897831968031824,0.00590937329946086,0.005573545897677541,0.005279701333791017,0.005019848321266474,0.0047878971067070964,0.004581053611356765,0.004475081469900906,0.004205947879739106,0.00651823904676363],"type":"scatter"},{"mode":"lines+markers","name":"Theory 400 grid","x":[1.9999999999999954e-07,3.034304765867952e-07,4.6035014748963906e-07,6.984208530700364e-07,1.0596094959101024e-06,1.607585498470808e-06,2.438943292891682e-06,3.7002272069854957e-06,5.613757716930151e-06,8.516806677573355e-06,1.292101569074731e-05,1.9602505002391748e-05,2.97384953722449e-05,4.511438394964044e-05,6.843744918967896e-05,0.00010381172986576898,0.00015745605600841445,0.00023878782918561914,0.00036205449638139736,0.0005487795323670796,0.0008314068836488144,0.0012586797144272762,0.0019034634022867384,0.0028738675812817515,0.004328500638820811,0.006496206194633799,0.009699159574043398,0.014375068581090129,0.02108918668378717,0.030521584007828916,0.04341491741702269,0.060480028754447364,0.08228122126204893,0.10914375746330703,0.14112080644440345,0.17802566042569432,0.2195041265003886,0.2651137041582823,0.31438740076927585,0.3668753186482242,0.4221667753589648,0.4798989029610255,0.5397572337880445,0.601472197967335,0.6648139482473823,0.7295868442414312,0.7956242522922756,0.8627839323906108,0.9309440808717544,1.0],"y":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.9371902569835221,0.3927603321442107,-0.021585278752808815,0.08563394258120446,0.05310500169542,0.04071666905438808,0.03442615839986706,0.03057235385061151,0.02475622323826615,0.023301616986257354,0.022004748686230494,0.02016696062248097,0.016476026495557166,0.018753082690227522,0.4056271597499533],"type":"scatter"},{"mode":"lines+markers","name":"2D Interpolation Th 200","x":[1.9999999999999954e-07,3.034304765867952e-07,4.6035014748963906e-07,6.984208530700364e-07,1.0596094959101024e-06,1.607585498470808e-06,2.438943292891682e-06,3.7002272069854957e-06,5.613757716930151e-06,8.516806677573355e-06,1.292101569074731e-05,1.9602505002391748e-05,2.97384953722449e-05,4.511438394964044e-05,6.843744918967896e-05,0.00010381172986576898,0.00015745605600841445,0.00023878782918561914,0.00036205449638139736,0.0005487795323670796,0.0008314068836488144,0.0012586797144272762,0.0019034634022867384,0.0028738675812817515,0.004328500638820811,0.006496206194633799,0.009699159574043398,0.014375068581090129,0.02108918668378717,0.030521584007828916,0.04341491741702269,0.060480028754447364,0.08228122126204893,0.10914375746330703,0.14112080644440345,0.17802566042569432,0.2195041265003886,0.2651137041582823,0.31438740076927585,0.3668753186482242,0.4221667753589648,0.4798989029610255,0.5397572337880445,0.601472197967335,0.6648139482473823,0.7295868442414312,0.7956242522922756,0.8627839323906108,0.9309440808717544,1.0],"y":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6023889782464709,-0.04885784146320546,0.039283006599300854,0.013518200988580639,0.010905970820639889,0.008230670108261298,0.006968275619640984,0.005942800648539437,0.0051470330585847305,0.004599584018059476,0.004168210044838479,0.00384374206366952,0.003644640554089193,0.0,0.0],"type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('e11d77f2-22ca-4560-b38a-33bde05e6f9b');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
sigma_had_200[sigma_had_200.index.get_level_values('x1')==3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>19</th>
      <th>20</th>
      <th>24</th>
      <th>25</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>...</th>
      <th>146</th>
      <th>150</th>
      <th>151</th>
      <th>155</th>
      <th>156</th>
      <th>157</th>
      <th>159</th>
      <th>160</th>
      <th>164</th>
      <th>165</th>
    </tr>
    <tr>
      <th>data</th>
      <th>x1</th>
      <th>x2</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="35" valign="top">5</th>
      <th rowspan="35" valign="top">3</th>
      <th>5</th>
      <td>0.000031</td>
      <td>-3.571636e-08</td>
      <td>-2.347446e-22</td>
      <td>-2.464362e-22</td>
      <td>0.000000e+00</td>
      <td>6.229416e-06</td>
      <td>-6.229416e-06</td>
      <td>-6.060400e-08</td>
      <td>6.522245e-11</td>
      <td>-1.626196e-24</td>
      <td>...</td>
      <td>-5.856424e-24</td>
      <td>1.245883e-05</td>
      <td>2.076471e-06</td>
      <td>-6.229456e-06</td>
      <td>7.143258e-09</td>
      <td>0.000000e+00</td>
      <td>-5.856424e-24</td>
      <td>0.000000e+00</td>
      <td>2.076471e-06</td>
      <td>1.453529e-05</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.000471</td>
      <td>-9.049938e-07</td>
      <td>-7.402461e-21</td>
      <td>-3.793357e-21</td>
      <td>-4.042137e-26</td>
      <td>9.410777e-05</td>
      <td>-9.410776e-05</td>
      <td>-9.666424e-07</td>
      <td>1.342987e-09</td>
      <td>9.005689e-24</td>
      <td>...</td>
      <td>1.002761e-21</td>
      <td>1.882156e-04</td>
      <td>3.136927e-05</td>
      <td>-9.410807e-05</td>
      <td>1.809979e-07</td>
      <td>0.000000e+00</td>
      <td>1.002761e-21</td>
      <td>7.885926e-21</td>
      <td>3.136927e-05</td>
      <td>2.195849e-04</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.000361</td>
      <td>-1.823013e-06</td>
      <td>3.694787e-22</td>
      <td>-4.686687e-21</td>
      <td>0.000000e+00</td>
      <td>7.223379e-05</td>
      <td>-7.223378e-05</td>
      <td>-8.003624e-07</td>
      <td>1.619324e-09</td>
      <td>2.744588e-23</td>
      <td>...</td>
      <td>-6.774060e-22</td>
      <td>1.444678e-04</td>
      <td>2.407797e-05</td>
      <td>-7.223533e-05</td>
      <td>3.645988e-07</td>
      <td>6.345037e-23</td>
      <td>-6.774060e-22</td>
      <td>-9.348281e-22</td>
      <td>2.407797e-05</td>
      <td>1.685458e-04</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.000001</td>
      <td>-3.412575e-07</td>
      <td>6.821376e-21</td>
      <td>1.659741e-21</td>
      <td>4.718722e-22</td>
      <td>2.444969e-07</td>
      <td>-2.444943e-07</td>
      <td>-3.747381e-08</td>
      <td>-2.829476e-09</td>
      <td>5.131658e-24</td>
      <td>...</td>
      <td>1.719491e-22</td>
      <td>4.891188e-07</td>
      <td>8.152071e-08</td>
      <td>-2.449965e-07</td>
      <td>6.824501e-08</td>
      <td>0.000000e+00</td>
      <td>1.719491e-22</td>
      <td>4.573919e-21</td>
      <td>8.152071e-08</td>
      <td>5.706414e-07</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.000004</td>
      <td>1.404160e-06</td>
      <td>-2.003831e-21</td>
      <td>-2.200502e-22</td>
      <td>4.503577e-22</td>
      <td>-8.558997e-07</td>
      <td>8.558993e-07</td>
      <td>-2.657237e-09</td>
      <td>-7.044804e-09</td>
      <td>2.121233e-24</td>
      <td>...</td>
      <td>3.428177e-23</td>
      <td>-1.711756e-06</td>
      <td>-2.852925e-07</td>
      <td>8.563489e-07</td>
      <td>-2.808397e-07</td>
      <td>3.458650e-22</td>
      <td>3.428177e-23</td>
      <td>-3.310715e-23</td>
      <td>-2.852925e-07</td>
      <td>-1.997048e-06</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.000034</td>
      <td>1.860175e-06</td>
      <td>-2.269753e-21</td>
      <td>6.673897e-23</td>
      <td>1.487929e-22</td>
      <td>6.787464e-06</td>
      <td>-6.787463e-06</td>
      <td>-7.964934e-08</td>
      <td>-8.238936e-09</td>
      <td>-7.312612e-25</td>
      <td>...</td>
      <td>-3.974452e-23</td>
      <td>1.357497e-05</td>
      <td>2.262495e-06</td>
      <td>-6.786154e-06</td>
      <td>-3.720433e-07</td>
      <td>0.000000e+00</td>
      <td>-3.974452e-23</td>
      <td>-6.884459e-22</td>
      <td>2.262495e-06</td>
      <td>1.583746e-05</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.000019</td>
      <td>2.062199e-06</td>
      <td>-1.659004e-21</td>
      <td>-2.123182e-22</td>
      <td>5.162952e-22</td>
      <td>3.846456e-06</td>
      <td>-3.846458e-06</td>
      <td>-4.752754e-08</td>
      <td>-8.802672e-09</td>
      <td>8.887208e-25</td>
      <td>...</td>
      <td>-1.695651e-22</td>
      <td>7.692950e-06</td>
      <td>1.282158e-06</td>
      <td>-3.844616e-06</td>
      <td>-4.124483e-07</td>
      <td>2.180259e-23</td>
      <td>-1.695651e-22</td>
      <td>4.694446e-22</td>
      <td>1.282158e-06</td>
      <td>8.975107e-06</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.000011</td>
      <td>2.294315e-06</td>
      <td>1.312326e-22</td>
      <td>-2.564064e-22</td>
      <td>-2.434220e-22</td>
      <td>2.155623e-06</td>
      <td>-2.155623e-06</td>
      <td>-2.744046e-08</td>
      <td>-9.348312e-09</td>
      <td>4.098128e-25</td>
      <td>...</td>
      <td>-1.538835e-22</td>
      <td>4.311267e-06</td>
      <td>7.185443e-07</td>
      <td>-2.153467e-06</td>
      <td>-4.588719e-07</td>
      <td>-1.156787e-22</td>
      <td>-1.538835e-22</td>
      <td>0.000000e+00</td>
      <td>7.185443e-07</td>
      <td>5.029814e-06</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.000010</td>
      <td>2.411945e-06</td>
      <td>9.953018e-22</td>
      <td>-3.502632e-22</td>
      <td>1.070315e-22</td>
      <td>1.953358e-06</td>
      <td>-1.953359e-06</td>
      <td>-2.381107e-08</td>
      <td>-9.576574e-09</td>
      <td>-3.960128e-24</td>
      <td>...</td>
      <td>-1.302840e-23</td>
      <td>3.906732e-06</td>
      <td>6.511219e-07</td>
      <td>-1.951078e-06</td>
      <td>-4.823978e-07</td>
      <td>6.751691e-23</td>
      <td>-1.302840e-23</td>
      <td>-1.156423e-21</td>
      <td>6.511219e-07</td>
      <td>4.557855e-06</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.000008</td>
      <td>2.457784e-06</td>
      <td>1.640672e-21</td>
      <td>1.246212e-22</td>
      <td>1.446957e-22</td>
      <td>1.681613e-06</td>
      <td>-1.681611e-06</td>
      <td>-2.000669e-08</td>
      <td>-9.594939e-09</td>
      <td>1.586525e-24</td>
      <td>...</td>
      <td>5.506638e-24</td>
      <td>3.363235e-06</td>
      <td>5.605392e-07</td>
      <td>-1.679488e-06</td>
      <td>-4.915652e-07</td>
      <td>2.200854e-24</td>
      <td>5.506638e-24</td>
      <td>-7.571696e-23</td>
      <td>5.605392e-07</td>
      <td>3.923774e-06</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.000007</td>
      <td>2.470180e-06</td>
      <td>1.374566e-21</td>
      <td>-1.690613e-23</td>
      <td>-2.714812e-22</td>
      <td>1.401205e-06</td>
      <td>-1.401207e-06</td>
      <td>-1.634923e-08</td>
      <td>-9.515576e-09</td>
      <td>6.106692e-25</td>
      <td>...</td>
      <td>-1.060071e-22</td>
      <td>2.802420e-06</td>
      <td>4.670699e-07</td>
      <td>-1.399405e-06</td>
      <td>-4.940444e-07</td>
      <td>-5.616951e-23</td>
      <td>-1.060071e-22</td>
      <td>6.644517e-22</td>
      <td>4.670699e-07</td>
      <td>3.269489e-06</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.000006</td>
      <td>2.461037e-06</td>
      <td>1.027197e-21</td>
      <td>-1.211539e-22</td>
      <td>-1.473655e-23</td>
      <td>1.221484e-06</td>
      <td>-1.221484e-06</td>
      <td>-1.395098e-08</td>
      <td>-9.372130e-09</td>
      <td>-1.411491e-25</td>
      <td>...</td>
      <td>4.950137e-23</td>
      <td>2.442973e-06</td>
      <td>4.071620e-07</td>
      <td>-1.220210e-06</td>
      <td>-4.922155e-07</td>
      <td>4.035335e-23</td>
      <td>4.950137e-23</td>
      <td>3.988075e-22</td>
      <td>4.071620e-07</td>
      <td>2.850135e-06</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.000005</td>
      <td>2.433742e-06</td>
      <td>5.390216e-22</td>
      <td>2.049448e-22</td>
      <td>-3.450799e-23</td>
      <td>1.067757e-06</td>
      <td>-1.067757e-06</td>
      <td>-1.194810e-08</td>
      <td>-9.182985e-09</td>
      <td>-2.959539e-25</td>
      <td>...</td>
      <td>-1.002241e-22</td>
      <td>2.135517e-06</td>
      <td>3.559195e-07</td>
      <td>-1.067163e-06</td>
      <td>-4.867562e-07</td>
      <td>6.515796e-24</td>
      <td>-1.002241e-22</td>
      <td>-2.292640e-23</td>
      <td>3.559195e-07</td>
      <td>2.491436e-06</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.000005</td>
      <td>2.391082e-06</td>
      <td>-3.341502e-22</td>
      <td>-5.696743e-24</td>
      <td>5.996929e-24</td>
      <td>9.351804e-07</td>
      <td>-9.351809e-07</td>
      <td>-1.025120e-08</td>
      <td>-8.964309e-09</td>
      <td>-5.907571e-25</td>
      <td>...</td>
      <td>2.615037e-23</td>
      <td>1.870363e-06</td>
      <td>3.117269e-07</td>
      <td>-9.353842e-07</td>
      <td>-4.782240e-07</td>
      <td>-2.433751e-23</td>
      <td>2.615037e-23</td>
      <td>-2.344561e-23</td>
      <td>3.117269e-07</td>
      <td>2.182090e-06</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.000004</td>
      <td>2.343491e-06</td>
      <td>3.771601e-22</td>
      <td>1.616341e-22</td>
      <td>-9.183716e-24</td>
      <td>8.393584e-07</td>
      <td>-8.393583e-07</td>
      <td>-9.029136e-09</td>
      <td>-8.733884e-09</td>
      <td>9.093220e-25</td>
      <td>...</td>
      <td>1.182106e-23</td>
      <td>1.678719e-06</td>
      <td>2.797865e-07</td>
      <td>-8.404699e-07</td>
      <td>-4.687056e-07</td>
      <td>4.095066e-23</td>
      <td>1.182106e-23</td>
      <td>5.272725e-22</td>
      <td>2.797865e-07</td>
      <td>1.958504e-06</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.000004</td>
      <td>2.294769e-06</td>
      <td>-1.875424e-22</td>
      <td>1.857198e-22</td>
      <td>1.169091e-23</td>
      <td>7.603407e-07</td>
      <td>-7.603407e-07</td>
      <td>-8.042636e-09</td>
      <td>-8.500330e-09</td>
      <td>6.425622e-25</td>
      <td>...</td>
      <td>5.193772e-23</td>
      <td>1.520682e-06</td>
      <td>2.534468e-07</td>
      <td>-7.624470e-07</td>
      <td>-4.589606e-07</td>
      <td>4.083924e-23</td>
      <td>5.193772e-23</td>
      <td>-6.973304e-24</td>
      <td>2.534468e-07</td>
      <td>1.774129e-06</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.000003</td>
      <td>2.242866e-06</td>
      <td>-3.543874e-22</td>
      <td>8.512086e-23</td>
      <td>-1.844024e-23</td>
      <td>6.885547e-07</td>
      <td>-6.885546e-07</td>
      <td>-7.166899e-09</td>
      <td>-8.265656e-09</td>
      <td>-6.265194e-25</td>
      <td>...</td>
      <td>1.165464e-23</td>
      <td>1.377108e-06</td>
      <td>2.295182e-07</td>
      <td>-6.917322e-07</td>
      <td>-4.485801e-07</td>
      <td>6.233600e-23</td>
      <td>1.165464e-23</td>
      <td>8.010907e-23</td>
      <td>2.295182e-07</td>
      <td>1.606626e-06</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.000003</td>
      <td>2.192014e-06</td>
      <td>2.456437e-22</td>
      <td>2.226656e-22</td>
      <td>3.207217e-22</td>
      <td>6.344365e-07</td>
      <td>-6.344367e-07</td>
      <td>-6.525372e-09</td>
      <td>-8.036822e-09</td>
      <td>3.647970e-25</td>
      <td>...</td>
      <td>2.052546e-23</td>
      <td>1.268872e-06</td>
      <td>2.114787e-07</td>
      <td>-6.387358e-07</td>
      <td>-4.384092e-07</td>
      <td>3.253272e-22</td>
      <td>2.052546e-23</td>
      <td>-7.883289e-23</td>
      <td>2.114787e-07</td>
      <td>1.480351e-06</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.000003</td>
      <td>2.140745e-06</td>
      <td>-4.973361e-22</td>
      <td>1.462274e-23</td>
      <td>-5.643607e-23</td>
      <td>5.815834e-07</td>
      <td>-5.815835e-07</td>
      <td>-5.908700e-09</td>
      <td>-7.813397e-09</td>
      <td>-1.840760e-24</td>
      <td>...</td>
      <td>-5.053512e-23</td>
      <td>1.163166e-06</td>
      <td>1.938610e-07</td>
      <td>-5.870463e-07</td>
      <td>-4.281555e-07</td>
      <td>5.909912e-24</td>
      <td>-5.053512e-23</td>
      <td>-5.711933e-23</td>
      <td>1.938610e-07</td>
      <td>1.357027e-06</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.000003</td>
      <td>2.088470e-06</td>
      <td>-2.708403e-23</td>
      <td>9.058139e-23</td>
      <td>1.445176e-23</td>
      <td>5.345953e-07</td>
      <td>-5.345950e-07</td>
      <td>-5.370381e-09</td>
      <td>-7.595929e-09</td>
      <td>-2.322879e-25</td>
      <td>...</td>
      <td>-1.835229e-23</td>
      <td>1.069189e-06</td>
      <td>1.781982e-07</td>
      <td>-5.412482e-07</td>
      <td>-4.177002e-07</td>
      <td>-4.663677e-25</td>
      <td>-1.835229e-23</td>
      <td>-3.013614e-24</td>
      <td>1.781982e-07</td>
      <td>1.247388e-06</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.000003</td>
      <td>2.038251e-06</td>
      <td>-2.406535e-22</td>
      <td>2.982936e-23</td>
      <td>1.802947e-23</td>
      <td>4.972722e-07</td>
      <td>-4.972726e-07</td>
      <td>-4.958607e-09</td>
      <td>-7.387871e-09</td>
      <td>-5.484519e-26</td>
      <td>...</td>
      <td>3.834777e-23</td>
      <td>9.945429e-07</td>
      <td>1.657572e-07</td>
      <td>-5.051490e-07</td>
      <td>-4.076561e-07</td>
      <td>1.137629e-23</td>
      <td>3.834777e-23</td>
      <td>-6.196778e-25</td>
      <td>1.657572e-07</td>
      <td>1.160300e-06</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.000002</td>
      <td>1.990294e-06</td>
      <td>-1.007321e-22</td>
      <td>-1.423298e-22</td>
      <td>-3.406102e-24</td>
      <td>4.642348e-07</td>
      <td>-4.642348e-07</td>
      <td>-4.605900e-09</td>
      <td>-7.189148e-09</td>
      <td>5.624355e-25</td>
      <td>...</td>
      <td>1.184346e-23</td>
      <td>9.284690e-07</td>
      <td>1.547447e-07</td>
      <td>-4.735001e-07</td>
      <td>-3.980648e-07</td>
      <td>6.396647e-24</td>
      <td>1.184346e-23</td>
      <td>1.007270e-24</td>
      <td>1.547447e-07</td>
      <td>1.083212e-06</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.000002</td>
      <td>1.943447e-06</td>
      <td>1.988325e-22</td>
      <td>1.958416e-23</td>
      <td>1.820095e-24</td>
      <td>4.340682e-07</td>
      <td>-4.340684e-07</td>
      <td>-4.289044e-09</td>
      <td>-6.998420e-09</td>
      <td>1.114477e-25</td>
      <td>...</td>
      <td>5.563375e-24</td>
      <td>8.681341e-07</td>
      <td>1.446891e-07</td>
      <td>-4.445111e-07</td>
      <td>-3.886951e-07</td>
      <td>-1.295253e-24</td>
      <td>5.563375e-24</td>
      <td>-7.667315e-24</td>
      <td>1.446891e-07</td>
      <td>1.012823e-06</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.000002</td>
      <td>1.898113e-06</td>
      <td>-4.046807e-22</td>
      <td>7.766506e-23</td>
      <td>3.051080e-24</td>
      <td>4.071860e-07</td>
      <td>-4.071859e-07</td>
      <td>-4.013612e-09</td>
      <td>-6.816482e-09</td>
      <td>-5.875273e-25</td>
      <td>...</td>
      <td>-1.717921e-23</td>
      <td>8.143699e-07</td>
      <td>1.357283e-07</td>
      <td>-4.187518e-07</td>
      <td>-3.796277e-07</td>
      <td>1.567269e-23</td>
      <td>-1.717921e-23</td>
      <td>-1.326672e-23</td>
      <td>1.357283e-07</td>
      <td>9.500984e-07</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.000002</td>
      <td>1.858890e-06</td>
      <td>-3.159037e-23</td>
      <td>7.838814e-23</td>
      <td>1.117154e-23</td>
      <td>3.831456e-07</td>
      <td>-3.831455e-07</td>
      <td>-3.779775e-09</td>
      <td>-6.652167e-09</td>
      <td>3.027751e-25</td>
      <td>...</td>
      <td>-1.864709e-23</td>
      <td>7.662888e-07</td>
      <td>1.277148e-07</td>
      <td>-3.963612e-07</td>
      <td>-3.717831e-07</td>
      <td>-4.650662e-24</td>
      <td>-1.864709e-23</td>
      <td>-5.243123e-24</td>
      <td>1.277148e-07</td>
      <td>8.940043e-07</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.000002</td>
      <td>1.817126e-06</td>
      <td>4.861619e-22</td>
      <td>2.884338e-23</td>
      <td>-7.344062e-24</td>
      <td>3.614600e-07</td>
      <td>-3.614599e-07</td>
      <td>-3.570122e-09</td>
      <td>-6.487940e-09</td>
      <td>-6.754201e-26</td>
      <td>...</td>
      <td>-1.060474e-23</td>
      <td>7.229179e-07</td>
      <td>1.204864e-07</td>
      <td>-3.757778e-07</td>
      <td>-3.634302e-07</td>
      <td>-7.939083e-24</td>
      <td>-1.060474e-23</td>
      <td>-1.137205e-22</td>
      <td>1.204864e-07</td>
      <td>8.434041e-07</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.000002</td>
      <td>1.773098e-06</td>
      <td>-3.028709e-22</td>
      <td>6.827333e-23</td>
      <td>6.849051e-23</td>
      <td>3.407246e-07</td>
      <td>-3.407247e-07</td>
      <td>-3.367048e-09</td>
      <td>-6.323628e-09</td>
      <td>-2.730875e-25</td>
      <td>...</td>
      <td>-1.925134e-23</td>
      <td>6.814478e-07</td>
      <td>1.135747e-07</td>
      <td>-3.558616e-07</td>
      <td>-3.546247e-07</td>
      <td>-1.288002e-24</td>
      <td>-1.925134e-23</td>
      <td>1.931524e-22</td>
      <td>1.135747e-07</td>
      <td>7.950218e-07</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.000002</td>
      <td>1.733839e-06</td>
      <td>6.362701e-23</td>
      <td>9.819492e-23</td>
      <td>3.895716e-23</td>
      <td>3.241920e-07</td>
      <td>-3.241919e-07</td>
      <td>-3.222019e-09</td>
      <td>-6.172312e-09</td>
      <td>2.995048e-25</td>
      <td>...</td>
      <td>-2.653414e-23</td>
      <td>6.483817e-07</td>
      <td>1.080636e-07</td>
      <td>-3.404357e-07</td>
      <td>-3.467726e-07</td>
      <td>-1.515058e-23</td>
      <td>-2.653414e-23</td>
      <td>5.557933e-23</td>
      <td>1.080636e-07</td>
      <td>7.564460e-07</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.000002</td>
      <td>1.697790e-06</td>
      <td>2.457424e-22</td>
      <td>9.058339e-23</td>
      <td>2.210077e-22</td>
      <td>3.091386e-07</td>
      <td>-3.091387e-07</td>
      <td>-3.091573e-09</td>
      <td>-6.030883e-09</td>
      <td>5.077337e-25</td>
      <td>...</td>
      <td>6.220481e-24</td>
      <td>6.182744e-07</td>
      <td>1.030457e-07</td>
      <td>-3.262478e-07</td>
      <td>-3.395626e-07</td>
      <td>1.312136e-22</td>
      <td>6.220481e-24</td>
      <td>-3.337722e-23</td>
      <td>1.030457e-07</td>
      <td>7.213201e-07</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.000002</td>
      <td>1.660911e-06</td>
      <td>-7.596831e-23</td>
      <td>4.576457e-23</td>
      <td>9.985430e-24</td>
      <td>2.906731e-07</td>
      <td>-2.906730e-07</td>
      <td>-2.914996e-09</td>
      <td>-5.893791e-09</td>
      <td>2.936095e-25</td>
      <td>...</td>
      <td>2.529342e-23</td>
      <td>5.813443e-07</td>
      <td>9.689056e-08</td>
      <td>-3.090106e-07</td>
      <td>-3.321865e-07</td>
      <td>4.249254e-24</td>
      <td>2.529342e-23</td>
      <td>-1.390438e-23</td>
      <td>9.689056e-08</td>
      <td>6.782341e-07</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.000001</td>
      <td>1.626865e-06</td>
      <td>5.732778e-23</td>
      <td>3.175686e-23</td>
      <td>3.575086e-25</td>
      <td>2.797486e-07</td>
      <td>-2.797486e-07</td>
      <td>-2.841240e-09</td>
      <td>-5.764250e-09</td>
      <td>2.019642e-25</td>
      <td>...</td>
      <td>-3.163365e-24</td>
      <td>5.594946e-07</td>
      <td>9.324907e-08</td>
      <td>-2.992742e-07</td>
      <td>-3.253779e-07</td>
      <td>-8.987062e-23</td>
      <td>-3.163365e-24</td>
      <td>6.061356e-23</td>
      <td>9.324907e-08</td>
      <td>6.527436e-07</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.000001</td>
      <td>1.593929e-06</td>
      <td>-1.461566e-22</td>
      <td>2.109933e-23</td>
      <td>2.805942e-24</td>
      <td>2.654515e-07</td>
      <td>-2.654514e-07</td>
      <td>-2.714197e-09</td>
      <td>-5.640455e-09</td>
      <td>8.379422e-26</td>
      <td>...</td>
      <td>-1.218651e-23</td>
      <td>5.309007e-07</td>
      <td>8.848340e-08</td>
      <td>-2.860000e-07</td>
      <td>-3.187905e-07</td>
      <td>-2.988342e-24</td>
      <td>-1.218651e-23</td>
      <td>2.135931e-23</td>
      <td>8.848340e-08</td>
      <td>6.193837e-07</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.000001</td>
      <td>1.592943e-06</td>
      <td>4.467603e-23</td>
      <td>-5.208351e-23</td>
      <td>-1.071359e-23</td>
      <td>2.581726e-07</td>
      <td>-2.581727e-07</td>
      <td>-2.664648e-09</td>
      <td>-5.638260e-09</td>
      <td>2.794696e-26</td>
      <td>...</td>
      <td>-6.031248e-24</td>
      <td>5.163432e-07</td>
      <td>8.605715e-08</td>
      <td>-2.801229e-07</td>
      <td>-3.185930e-07</td>
      <td>-1.875668e-24</td>
      <td>-6.031248e-24</td>
      <td>3.809133e-24</td>
      <td>8.605715e-08</td>
      <td>6.024002e-07</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.000001</td>
      <td>1.531824e-06</td>
      <td>3.511682e-22</td>
      <td>1.625476e-23</td>
      <td>9.194051e-24</td>
      <td>2.425858e-07</td>
      <td>-2.425859e-07</td>
      <td>-2.536921e-09</td>
      <td>-5.417002e-09</td>
      <td>-2.165092e-25</td>
      <td>...</td>
      <td>1.590749e-23</td>
      <td>4.851694e-07</td>
      <td>8.086155e-08</td>
      <td>-2.652689e-07</td>
      <td>-3.063692e-07</td>
      <td>-1.991929e-24</td>
      <td>1.590749e-23</td>
      <td>1.037937e-23</td>
      <td>8.086155e-08</td>
      <td>5.660311e-07</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.000002</td>
      <td>2.446829e-06</td>
      <td>-1.056329e-22</td>
      <td>-4.170457e-23</td>
      <td>6.260293e-23</td>
      <td>3.707257e-07</td>
      <td>-3.707257e-07</td>
      <td>-3.926404e-09</td>
      <td>-8.688645e-09</td>
      <td>-2.843522e-26</td>
      <td>...</td>
      <td>-2.372616e-23</td>
      <td>7.414483e-07</td>
      <td>1.235746e-07</td>
      <td>-4.098901e-07</td>
      <td>-4.893724e-07</td>
      <td>4.430170e-24</td>
      <td>-2.372616e-23</td>
      <td>2.155068e-25</td>
      <td>1.235746e-07</td>
      <td>8.650230e-07</td>
    </tr>
  </tbody>
</table>
<p>35 rows × 56 columns</p>
</div>



## Test griddata versus interp2d


```python
sigma_1 = interp2d_(fktable_had, fktable_had_400)
sigma_2 = griddata_(fktable_had, fktable_had_400)
```


```python
import time

# probably need to ignore the broken datasets as in the imagepdf nb

for ds, ds_400 in zip(datasets, datasets_400):
    ds_name = str(ds)

    for fk, fk_400 in zip(ds.fkspecs, ds_400.fkspecs):
        tab=load_fktable(fk)
        tab_400=load_fktable(fk_400)        

        if tab.hadronic and tab_400.hadronic:
            
            print()
            print(f"Dataset = {ds_name}")
            t0=time.time()
            griddata_(tab, tab_400)
            t1=time.time()
            print(f'time needed for griddata = {t1-t0}')
            print()
#             t0=time.time()
#             interp2d_(tab, tab_400)
#             t1=time.time()
#             print(f'time needed for interp2d = {t1-t0}')
            
            
```

    
    Dataset = DYE886R_dw_ite
    time needed for griddata = 2.8714170455932617
    
    
    Dataset = DYE886R_dw_ite
    time needed for griddata = 3.4969239234924316
    
    
    Dataset = DYE886P
    time needed for griddata = 34.607521057128906
    
    
    Dataset = DYE605_dw_ite
    time needed for griddata = 41.15037703514099
    
    
    Dataset = DYE906R_dw_ite
    time needed for griddata = 1.9671008586883545
    
    
    Dataset = DYE906R_dw_ite
    time needed for griddata = 1.9105918407440186
    
    
    Dataset = DYE906R_dw_ite
    time needed for griddata = 1.7240381240844727
    
    
    Dataset = DYE906R_dw_ite



    ---------------------------------------------------------------------------

    QhullError                                Traceback (most recent call last)

    Cell In[64], line 17
         15 print(f"Dataset = {ds_name}")
         16 t0=time.time()
    ---> 17 griddata_(tab, tab_400)
         18 t1=time.time()
         19 print(f'time needed for griddata = {t1-t0}')


    Cell In[62], line 20, in griddata_(fktable, fktable_400)
         17 new_x1grid_flat = np.ravel(new_xgrid_mesh[0])
         18 new_x2grid_flat = np.ravel(new_xgrid_mesh[1])
    ---> 20 interpolated_grids = [griddata(points=(x1_vals,x2_vals), values=grp[col].values,
         21                               xi=(new_x1grid_flat,new_x2grid_flat), method='linear', fill_value=0)
         22                 for col in grp.columns]
         24 tmp_index = pd.MultiIndex.from_product([[d],range(50),range(50)], names=['data','x1','x2'])
         26 d = dict()


    Cell In[62], line 20, in <listcomp>(.0)
         17 new_x1grid_flat = np.ravel(new_xgrid_mesh[0])
         18 new_x2grid_flat = np.ravel(new_xgrid_mesh[1])
    ---> 20 interpolated_grids = [griddata(points=(x1_vals,x2_vals), values=grp[col].values,
         21                               xi=(new_x1grid_flat,new_x2grid_flat), method='linear', fill_value=0)
         22                 for col in grp.columns]
         24 tmp_index = pd.MultiIndex.from_product([[d],range(50),range(50)], names=['data','x1','x2'])
         26 d = dict()


    File ~/miniconda3/envs/colibri/lib/python3.9/site-packages/scipy/interpolate/_ndgriddata.py:264, in griddata(points, values, xi, method, fill_value, rescale)
        262     return ip(xi)
        263 elif method == 'linear':
    --> 264     ip = LinearNDInterpolator(points, values, fill_value=fill_value,
        265                               rescale=rescale)
        266     return ip(xi)
        267 elif method == 'cubic' and ndim == 2:


    File interpnd.pyx:281, in scipy.interpolate.interpnd.LinearNDInterpolator.__init__()


    File _qhull.pyx:1841, in scipy.spatial._qhull.Delaunay.__init__()


    File _qhull.pyx:353, in scipy.spatial._qhull._Qhull.__init__()


    QhullError: QH6154 Qhull precision error: Initial simplex is flat (facet 1 is coplanar with the interior point)
    
    While executing:  | qhull d Qz Q12 Qbb Qt Qc
    Options selected for Qhull 2019.1.r 2019/06/21:
      run-id 1095309642  delaunay  Qz-infinity-point  Q12-allow-wide  Qbbound-last
      Qtriangulate  Qcoplanar-keep  _pre-merge  _zero-centrum  Qinterior-keep
      Pgood  _max-width 0.46  Error-roundoff 1.3e-15  _one-merge 9e-15
      Visible-distance 2.6e-15  U-max-coplanar 2.6e-15  Width-outside 5.2e-15
      _wide-facet 1.5e-14  _maxoutside 1e-14
    
    The input to qhull appears to be less than 3 dimensional, or a
    computation has overflowed.
    
    Qhull could not construct a clearly convex simplex from points:
    - p1(v4):  0.49  0.93 0.022
    - p22(v3):   0.7  0.93  0.93
    - p21(v2):  0.93  0.93  0.73
    - p0(v1):  0.47  0.93     0
    
    The center point is coplanar with a facet, or a vertex is coplanar
    with a neighboring facet.  The maximum round off error for
    computing distances is 1.3e-15.  The center point, facets and distances
    to the center point are as follows:
    
    center point   0.6488   0.9306   0.4214
    
    facet p22 p21 p0 distance=    0
    facet p1 p21 p0 distance=    0
    facet p1 p22 p0 distance=    0
    facet p1 p22 p21 distance=    0
    
    These points either have a maximum or minimum x-coordinate, or
    they maximize the determinant for k coordinates.  Trial points
    are first selected from points that maximize a coordinate.
    
    The min and max coordinates for each dimension are:
      0:    0.4736    0.9306  difference= 0.457
      1:    0.9306    0.9306  difference=    0
      2:         0    0.9306  difference= 0.9306
    
    If the input should be full dimensional, you have several options that
    may determine an initial simplex:
      - use 'QJ'  to joggle the input and make it full dimensional
      - use 'QbB' to scale the points to the unit cube
      - use 'QR0' to randomly rotate the input for different maximum points
      - use 'Qs'  to search all points for the initial simplex
      - use 'En'  to specify a maximum roundoff error less than 1.3e-15.
      - trace execution with 'T3' to see the determinant for each point.
    
    If the input is lower dimensional:
      - use 'QJ' to joggle the input and make it full dimensional
      - use 'Qbk:0Bk:0' to delete coordinate k from the input.  You should
        pick the coordinate with the least range.  The hull will have the
        correct topology.
      - determine the flat containing the points, rotate the points
        into a coordinate plane, and delete the other coordinates.
      - add one or more points to make the input full dimensional.




```python
np.max(sigma_1.to_numpy()- sigma_2.to_numpy())
```




    0.00019656477386740742




```python

```
