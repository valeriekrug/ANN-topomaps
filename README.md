# ANN-topomaps

This repository complements the paper
```
Valerie Krug, Raihan Kabir Ratul, Christopher Olson and Sebastian Stober.
Visualizing Deep Neural Networks with Topographic Activation Maps. 2023.
In HHAI 2023: Augmenting Human Intellect (pp. 138-152). IOS Press.
```
available at [https://doi.org/10.3233/FAIA230080](https://doi.org/10.3233/FAIA230080)

## Usage
This section describes how to compute topographic maps of existing Deep Neural Network models.  
Topographic map layouts and visualizations are computed with the `run_config.py` according to a configuration specified in a `config.json`.

`python3 run_config.py -c </path/to/config.json> -o </output/path/>`

In the publication, robustness analyses are performed for multiple runs of the same configuration. These experiments are conducted with the `run_experiment.py` script, specifying the number of repetitions with the `-r` argument.

`python3 run_experiment.py -c </path/to/config.json> -o </output/path/> -r 100`

### interactive notebook
For a more interactive use, we provide an ipython notebook in a shared [Google Drive folder](https://drive.google.com/drive/folders/1EXcOStfZklJ0IeaA9A1SnXbY8HOCNmGZ?usp=sharing).     
The directory contains a copy of this git repository's source code as well as the pre-trained models used in the publication.  
Instead of using configuration files in json format, you can interactively choose and run the configuration with a user interface.

## reproducibility
The shared Google Drive folder includes the models used in the publication.  
Each experiment can be reproduced using the respective configuration files in `configs/paper`.

For the racial bias experiment that uses the FairFace data set, you need to first preprocess the data set.  
To this end, download the data set from https://github.com/dchen236/FairFace ([Data subsection](https://github.com/dchen236/FairFace#data)).  
Then, use the provided preprocessing script to generate a TF data set:  
`python3 preprocess_fairface.py -i <path/to/fairface/> -o </output/path/>`

## Customization
By default, we only allow models that we used in the publication for ease of reproducibility.  
adding a pre-trained model:
- add a new model to the `models/` directory (saved with `tensorflow.keras.Model.save()`)  
  Note: the code expects the naming format `<traindata>-<modelname>` (e.g. `MNIST-MLP`)
- add the new data-model combination to `allowed_model_values` in `src.config.check_general_params()`
adding a data set
- provide a new data loader in `src.models.load_data()`
- add the new data set name to `allowed_dataset_values` in `src.config.check_general_params()`

## Visualizing Training Processes

`python3 create_checkpoints.py -c "</path/to/config.json>"`

`sh run_topomapcreator_from_checkpoint.sh`
