# Automatic Differentiation of Variable and Fixed Speed Heat Pumps With Smart Meter Data

Author: Tobias Brudermüller (Brudermueller), Bits to Energy Lab, ETH Zurich: tbrudermuell@ethz.ch

This repository contains the Python code and trained models for the [following paper](https://ieeexplore.ieee.org/abstract/document/9961055): 

*T. Brudermueller, F. Wirth, A. Weigert and T. Staake, "Automatic Differentiation of Variable and Fixed Speed Heat Pumps With Smart Meter Data," 2022 IEEE International Conference on Communications, Control, and Computing Technologies for Smart Grids (SmartGridComm), Singapore, Singapore, 2022, pp. 412-418, doi: 10.1109/SmartGridComm52983.2022.9961055.*

For detailed explanations about underlying assumptions, and implementations please refer to this source. 
Further, if you make use of this repository of paper, please **use the following citation**: 
```
@INPROCEEDINGS{9961055,
  author={Brudermueller, Tobias and Wirth, Florian and Weigert, Andreas and Staake, Thorsten},
  booktitle={2022 IEEE International Conference on Communications, Control, and Computing Technologies for Smart Grids (SmartGridComm)}, 
  title={Automatic Differentiation of Variable and Fixed Speed Heat Pumps With Smart Meter Data}, 
  year={2022},
  volume={},
  number={},
  pages={412-418},
  doi={10.1109/SmartGridComm52983.2022.9961055}}
```

### Abstract 

With the increasing prevalence of heat pumps in private households, the need for optimization is growing. At the same time, the growing number of active smart electricity meters generates data that can be used for remote monitoring. In this paper, we focus on the automatic differentiation between fixed speed and variable speed heat pumps using smart meter data. This distinction is relevant because it is necessary for evaluating the state or cyclic behavior of a heat pump. In addition, identifying fixed speed heat pumps is important because they are known to be the less efficient systems and therefore may be preferred targets in energy efficiency or replacement campaigns. Our methods are applied to electricity data from 171 Swiss households with a resolution of 15 minutes. In this setting, a K-Nearest Neighbor model achieves a mean AUC of 0.976 compared to 0.5 of a biased random guess model.

### Installation 

If you want to use your Python 3.7 interpreter directly, please check that you have the following packages pip-installed: ```numpy```, ```pandas```, ```scikit-learn```, and ```scipy```. Otherwise, if you want to create an anaconda environment named ```hp_type```, you can use the following commands, which makes use of the file ```installation/requirements.yml```: 

1. Navigating into installation folder: ```cd [....]/installation```
2. Installing environment: ```conda env create -n hp_type -f requirements.yml``` 
3. Opening environment for a session: ```conda activate hp_type```
4. Close environment after a session: ```conda deactivate```

**NOTE:**: The code also uses parallel processing to compute features and therefore uses the ```functools``` and ```multiprocess``` packages included in the standard Python library, same as ```pickle```.

### Usage 

Our methods assume smart meter data in the form of energy measurements (in kWh) at 15-minute resolution and are designed for residential use only. Our algorithms expect a data frame for each household that maps a consumption value to the corresponding timestamp. An example of the expected data format can be found in ```data/smd_example.csv```. 

The source code is located in the ```src``` folder. The file ```src/feature_generation.py``` defines and computes the features from a pandas data frame containing the smart meter data (in a parallel fashion).
It can be considered a subroutine for the ```src/classification_routine.py``` file, which is the main script you can use as a starting point to test the algorithms with the sample data or to customize them for use with your own data. It also loads the trained models, which are located in the ```models``` folder. In summary, the ```src/classification_routine.py``` file should be fairly self-explanatory after having read the paper. You can run it as follows:

1. Navigating into installation folder: ```cd [....]/src```
2. Run script: ```python classification_routine.py``` 


### Additional Note

In the paper, the algorithms are evaluated without a majority vote at the end, which is different in this repository. Consider a time horizon of one week used for the models. Then in the paper, if a household has forecasts for several weeks, we treat each week as an individual observation to compute the performance values. However, in a real-world application, one would need a single prediction for each individual household.  Therefore, we consider the class that is predicted most frequently as the class that represents the entire household. In addition, this approach allows us to compute a confidence level in the prediction with respect to the dominance of the majority class.
