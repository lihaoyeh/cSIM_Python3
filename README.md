# cSIM with speckle in Python3

This is a Python3 implementation of coherent structured illumination microscopy algorithm with speckle <br/>
```cSIM_speckle_simulation.ipynb```: Jupyter notebook for creating simulation data for cSIM processing <br/>
```Preprocess.ipynb```: Jupyter notebook for generating appropriate exp dataset for optimization <br/>
```cSIM_main.ipynb```: Jupyter notebook for main optimization algorithm to process dataset <br/>
```cSIM_func.py```: Processing functions with numpy implementation <br/>
```cSIM_func_af.py```: Processing functions with arrayfire implementation <br/>
```dftregistration.py```: DFT registration python code translated by [1] to python from original MATLAB code in [2] <br/>
```zernfun.py```: Zernike polynomial generation function translated from original MATLAB code in [3] <br/>


[1] https://github.com/keflavich/image_registration <br/>
[2] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient subpixel image registration algorithms," Opt. Lett. 33, 156-158 (2008). <br/>
[3] https://www.mathworks.com/matlabcentral/fileexchange/7687-zernike-polynomials

## Environment requirement
Python 3.6, ArrayFire <br/>

1. Follow http://arrayfire.org/docs/index.htm for ArrayFire installation
2. Follow https://github.com/arrayfire/arrayfire-python to install ArrayFire Python and set the path to the libraries

## Data download
You can find sample experiment data from here: TBD
