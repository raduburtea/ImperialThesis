# ImperialThesis

This repository contains the code used to produce the results in the "Constrained Reinforcement Learning for Process Optimization" project.
The OMLT folder contains a slightly modified OMLT package, models_paper contains the implementation of the 
OMLT-DDPG, SAFE, RCPO and DDPG algorithms and the rllab folder the code for the TRPO and CPO experiments. The code found in the production_run.py file implements the SAFE algorithm, 
as well as the framework for running the production runs for the other algorithms.
Some sample trained models are included in the following folders: 
- OMLT-DPPG: models_paper/models-OMLT 
- RCPO: models_paper/models-RCPO 
- CPO, TRPO: models_paper/models-PO 

**Disclaimer**: the OMLT package seems to not be usable on the M1 Mac architecture. The matrix multiplcation 
operation used in the neural networks is read as 'Mul' instead of 'MatMul', which is not supported by the OMLT
package and will throw an error.

## Running the Experiments

In order to run any of the experiments you would first have to install the required packages. In order to to this 
cd into the ImperialThesis (```cd ImperialThesis```) directory and run the following commad:
```
pip install -r requirements.txt
```
You also need to install the Lasagne package separately as the version required to run with Theano is not available on pip anymore. For this please run the following: ``` pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip ```

Ipopt also has to be installed with homebrew, by running the following command: ```brew install ipopt```

To run each of the experiments included in this repository please cd in the ImperialThesis directory and run the followig commands:
- OMLT-DDPG: ```python models_paper/omltddpg.py```
- RCPO: ```python models_paper/rcpo.py```
- CPO: ```python rllab/sandbox/cpo/experiments/CPO_inventory_management.py```
- TRPO: ```python rllab/examples/trpo_network_management.py```
- SAFE: ```python models_paper/production_run.py``` - for this file you have to make sure that one of the following algorithm names is used: 
RCPO- , CPO-, TRPO-, SAFE- or OMLT-, followed by any other metadata you might want to use. Make sure that for the SAFE algorithm a valid 
model from the models_OMLT folder is selected
