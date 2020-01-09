# Bidirectional Molecule Generation with Recurrent Neural Networks

This is the supporting code for: Grisoni F., Moret M., Lingwood R., Schneider G., "Bidirectional Molecule Generation with Recurrent Neural Networks". *Journal of Chemical Information and Modeling* (2020). Available [here](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00943).

You can use this repository for the generation of SMILES with bidirectional
recurrent neural networks (RNNs). In addition to the methods' code, several pre-trained models for each approach are included.

The following methods are implemented:
* **Bidirectional Molecule Design by Alternate Learning** (BIMODAL), designed for SMILES generation – see [Grisoni *et al.* 2020](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00943).
* **Synchronous Forward Backward RNN** (FB-RNN), based on [Mou *et al.* 2016](https://arxiv.org/pdf/1512.06612.pdf).
* **Neural Autoregressive Distribution Estimator** (NADE), re-adapted for SMILES generation from [Berglund *et al.* 2015](http://papers.nips.cc/paper/5651-bidirectional-recurrent-neural-networks-as-generative-models.pdf).
* **Forward RNN**, *i.e.*, unidirectional RNN for SMILES generation. 


## Table of Contents
1. [Prerequisites](#Prerequisites)
2. [Using the Code](#Using_the_code)
    1. [Sampling from a pre-trained model](#Sample)
    2. [Training a model on your data](#Training) 
    3. [Fine-tuning a model on your data](#Finetuning) 
4. [Authors](#Authors)
5. [License](#License)
6. [How to cite](#cite) 

## Prerequisites<a name="Prerequisites"></a>

This repository can be cloned with the following command:

```
git clone https://github.com/ETHmodlab/BIMODAL
```

To install the necessary packages to run the code, we recommend using [conda](https://www.anaconda.com/download/). 
Once conda is installed, you can install the virtual environment:

```
cd path/to/repository/
conda env create -f brnn.yml
```

To activate the dedicated environment:
```
conda activate brnn
```

Your code should now be ready to use!

# Using the code <a name="Using_the_code"></a>
## Sampling from a pre-trained model <a name="Sample"></a>

In this repository, we provide you with 22 pre-trained models you can use for sampling (stored in [evaluation/](evaluation/)).
These models were trained on a set of 271,914 bioactive molecules from ChEMBL22 (K<sub>d/I</sub>/IC<sub>50</sub>/EC<sub>50</sub> <1μM), for 10 epochs.    

To sample SMILES, you can create a new file in [model/](model/) and use the *Sampler class*. 
For example, to sample from the pre-trained BIMODAL model with 512 units:

```
from sample import Sampler
experiment_name = 'BIMODAL_fixed_512'
s = Sampler(experiment_name)
s.sample(N=100, stor_dir='../evaluation', T=0.7, fold=[1], epoch=[9], valid=True, novel=True, unique=True, write_csv=True)
```

Parameters:
* *experiment_name* (str): name of the experiment with pre-trained model you want to sample from (you can find pre-trained models in [evaluation/](evaluation/))
* *stor_dir* (str): directory where the models are stored. The sampled SMILES will also be saved there (if write_csv=True)
* *N* (int): number of SMILES to sample
* *T* (float): sampling temperature
* *fold* (list of int): number of folds to use for sampling
* *epoch* (list of int): epoch(s) to use for sampling
* *valid* (bool): if set to *True*, only generate valid SMILES are accepted (increases the sampling time)
* *novel* (bool): if set to *True*, only generate novel SMILES (increases the sampling time)
* *unique* (bool): if set to *True*, only generate unique SMILES are provided (increases the sampling time)
* *write_csv* (bool): if set to *True*, the .csv file of the generated smiles will be exported in the specified directory.

*Notes*: 
- For the provided pre-trained models, only *fold=[1]* and *epoch=[9]* are provided.
- The list of available models and their description are provided in [evaluation/model_names.md](evaluation/model_names.md)

## Training a New Model
Alternatively, if you want to pre-train a model on your own data, you will need to execute three steps: (i) data processing (ii) training and (iii) evaluation.
Please be aware that you will need the access to a GPU to pre-train your own model as this is a computationally intensive step.

### Preprocessing
Data can be processed by using [preprocessing/main_preprocessor.py](preprocessing/main_preprocessor.py):
```
from main_preprocessor import preprocess_data
preprocess_data(filename_in='../data/chembl_smiles', model_type='BIMODAL', starting_point='fixed', augmentation=1)
```
Parameters:
* *filename_in* (str): name of the file containing the SMILES strings (.csv or .tar.xz)
* *model_type* (str): name of the chosen generative method
* *starting_point* (str): starting point type ('fixed' or 'random')
* *augmentation*(int): augmentation folds [Default = 1]

*Notes*:
* In [preprocessing/main_preprocessor.py](preprocessing/main_preprocessor.py) you will find info regarding advanced options for pre-processing (e.g., stereochemistry, canonicalization, etc.)
* Please note that the pre-treated data will have to be stored in [data/](data/).

### Training

Training requires a parameter file (.ini) with a given set of parameters. You can find examples for all models in [experiments/](experiments/), and further details about the parameters below:


|Section		|Parameter     	| Description			|Comments|
| --- | --- | --- | --- |	
|Model		|model         	| Type				| ForwardRNN, FBRNN, BIMODAL, NADE  |
| 		|hidden_units	| Number of hidden units	|	Suggested value: 256 for ForwardRNN, FBRNN and NADE;  128 for BIMODAL|
|		|generation	| To be defined only for NADE (other models defined through preprocessing) 			| fixed, random |
|Data		|data		| Name of data file		| Has to be located in data/ |
| 		|encoding_size  | Number of different SMILES tokens		| 55 |
|		|molecular_size	| Length of string with padding	| See preprocessing |
|		|missing_token	| To add in the parameter file only for NADE			| M |
|Training	|epochs		| Number of epochs		|  Suggested value: 10 |
|		|learning_rate	| Learning rate			|  Suggested value: 0.001|
|		|n_folds	| Folds in cross-validation	| See below: More than 1 for cross_validation, 1 to use only one fold of the data for validation |
|		|batch_size	| Batch size			|  Suggested value: 128  |
|Evaluation	| samples	| Number of generated SMILES after each epoch |  |
|		| temp		| Sampling temperature		| Suggested value: 0.7 |
|		| starting_token	| Starting token for sampling	| G for all models except NADE, which requires a sequence consisting of missing values (see publication)	|
	
Note:
- Be aware that value such as the number of tokens or the missing token for NADE have to be defined as in the example above. We kept those as parameters such that you can easily change them if you wish to use this code for other applications.


Options for training:

- Cross-validation: 
```
from trainer import Trainer

t = Trainer(experiment_name = 'BIMODAL_fixed_512')
t.cross_validation(stor_dir = '../evaluation/', restart = False)
```

- Single run: 1/*n_folds* of data used for validation
```
from trainer import Trainer

t = Trainer(experiment_name = 'BIMODAL_fixed_512')
t.single_run(stor_dir = '../evaluation/', restart = False)
```

Parameters:   
* *experiment_name* :  Name of parameter file (.ini)
* *stor_dir*: Directory where outputs can be found
* *restart*: If true, automatic restart from saved models (e.g. to be used if your training was interrupted before completion)

### Evaluation

You can do the evaluation of the outputs of your experiment with the [evaluation/main_evaluator.py](evaluation/main_evaluator.py) with the following possibilities:   

```
from evaluation import Evaluator

stor_dir = '../evaluation/'
e = Evaluator(experiment_name = 'BIMODAL_fixed_512')
# Plot training and validation loss within one figure
e.eval_training_validation(stor_dir=stor_dir)
# Plot percentage of novel, valid and unique SMILES
e.eval_molecule(stor_dir=stor_dir)
```

Parameters:
* *experiment_name*:  Name parameter file (.ini)
* *stor_dir*: Directory where outputs can be found

Note:
- the losses plot can be found, in that case, in '{experiment_name}/statistic/all_statistic.png'
- the novel, valid and unique SMILES plot can be found, in that case, in '../evaluation/{experiment_name}/molecules/novel_valid_unique_molecules.png'    

## Fine-tuning a model<a name="Finetuning"></a>

Fine-tuning requires a pre-trained model and a parameter file (.ini).
Examples of the parameter files (BIMODAL and ForwardRNN) are provided in [experiments/](experiments/).

You can start the sampling procedure with [model/main_fine_tuner.py](model/main_fine_tuner.py)


|Section		|Parameter     	| Description			|Comments |
| --- | --- | --- | --- |	
|Model		|model         	| Type				| ForwardRNN, FBRNN, BIMODAL, NADE  |
| 		|hidden_units	| Number of hidden units	|	Suggested value: 256 for ForwardRNN, FBRNN and NADE;  128 for BIMODAL|
|		|generation	| Only NADE (other models defined through preprocessing) 			| fixed, random |
|Data		|data		| Name of data file		| Has to be located in data/ |
| 		|encoding_size  | Number of different SMILES tokens		| 55 |
|		|molecular_size	| Length of string with padding	| See preprocessing |
|		|missing_token	| To add in the parameter file only for NADE			| M |
|Training	|epochs		| Number of epochs		|  Suggested value: 10 |
|		|learning_rate	| Learning rate			|  Suggested value: 0.001|
|		|batch_size	| Batch size			|  Suggested value: 128  |
|Evaluation	| samples	| Number of generated SMILES after each epoch |  |
|		| temp		| Sampling temperature		| Suggested value: 0.7 |
|		| starting_token	| Starting token for sampling	| G for all models except NADE, which requires a sequence consisting of missing values (see publication)	|
|Fine-Tuning		|start_model         	| Name of pre-trained model to be used for fine-tuning				|   |

To fine-tune a model, you can run:

```
t = FineTuner(experiment_name = 'BIMODAL_random_512_FineTuning_template')
t.fine_tuning(stor_dir='../evaluation/', restart=False)
```

Parameters:
* *experiment_name*:  Name parameter file (.ini)
* *stor_dir*: Directory where outputs can be found
* *restart*: If True, automatic restart from saved models (e.g. to be used if your training was interrupted before completion)
   
Note:
-  The batch size should not exceed the number of SMILES that you have in your fine-tuning file (taking into account the data augmentation).


## Authors<a name="Authors"></a>

* Robin Lingwood (https://github.com/robinlingwood)
* Francesca Grisoni (https://github.com/grisoniFr)
* Michael Moret (https://github.com/michael1788)

See also the list of [contributors](https://github.com/ETHmodlab/Bidirectional_RNNs/contributors) who participated in this project.


## License<a name="License"></a>

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

## How to Cite <a name="cite"></a>

If you use this code (or parts thereof), please cite it as:

```
@article{grisoni2020,
  title={Bidirectional Molecule Generation with Recurrent Neural Networks},
  author={Grisoni, Francesca and Moret, Michael and Lingwood, Robin and Schneider, Gisbert},
  journal={Journal of Chemical Information and Modeling},
  volume={Article ASAP},
  number={},
  pages={},
  year={2020},
  doi = {10.1021/acs.jcim.9b00943},
  url = {https://pubs.acs.org/doi/10.1021/acs.jcim.9b00943},
 publisher={ACS Publications}
}
```
