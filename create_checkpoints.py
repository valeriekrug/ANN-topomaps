import sys
import os
import ipywidgets as widgets
import re

sys.path.insert(0, os.path.join('/scratch/python_envs/annalyzer_env/python/lib/python3.8/site-packages/'))
os.environ['HTTP_PROXY'] = 'http://fp.cs.ovgu.de:3210/'
os.environ['HTTPS_PROXY'] = 'http://fp.cs.ovgu.de:3210/'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["TF_GPU_ALLOCATOR"]= "cuda_malloc_async"

import argparse

from src.neuron_activations import get_NAPs, get_neuron_activations, get_neuron_activations_of_checkpoint, get_NAPs_of_checkpoint
from src.topomap_class import TopomapVisualizer, load_experiment
from src.config import load_config

config_path = "configs/examples/mnist_mlp.json"
force_recompute = True
create_topomap_video = True
output_dir = os.path.join("output_base_path", config_path.split('/')[-1].split('.')[0])
checkpoint_path = 'output_base_path/checkpoints_for_video/'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
    print("Created File " + str(output_dir))
    
config = load_config(config_path)
model_name = config["model"]
data_name = config["data"]
layers_of_interest = config["layers"]
distance_metric = config["distance_metric"]
nap_params = config['nap_params']
topomap_params = config['topomap_params']
plot_params = config['plot_params']

NAPs, inputs = get_NAPs(model_name, data_name, layers_of_interest, nap_params["n_random_examples"], error_mode=nap_params["error_mode"], create_video_frames=True, checkpoint_path=checkpoint_path)

