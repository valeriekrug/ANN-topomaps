import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

from tkinter import *
from tkinter import ttk

from src.neuron_activations import get_NAPs, get_neuron_activations
from src.topomap_class import TopomapVisualizer, load_experiment
from src.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='path to config.json', required=True)
parser.add_argument('-o', '--output', help='output directory', required=True)
parser.add_argument('-f', '--force-recompute', help='force recomputing of existing runs',
                    required=False, action='store_true')
parser.add_argument('-v', '--video', help='video player for topomaps over epoch', required=False, action='store_true')

args = parser.parse_args()
config_path = args.config
force_recompute = args.force_recompute
create_topomap_video = args.video
output_dir = os.path.join(args.output, config_path.split('/')[-1].split('.')[0])
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

config = load_config(config_path)
model_name = config["model"]
data_name = config["data"]
layers_of_interest = config["layers"]
distance_metric = config["distance_metric"]
nap_params = config['nap_params']
topomap_params = config['topomap_params']
plot_params = config['plot_params']

precomputed_path = os.path.join(output_dir, "saved_experiment.pkl")
if os.path.isfile(precomputed_path) and not force_recompute:
    ANNScan = load_experiment(precomputed_path)

    # optional commands to restore the variables from the main script.
    # NAPs = {"values": ANNScan.naps,
    #         "error_mode": ANNScan.error_mode}
    # inputs = ANNScan.inputs
    # neuron_activations = ANNScan.neuron_activations

    print("loaded experiment '" + output_dir + "'. To recompute the experiment, use -f/--force-recompute flag.")
else:
    NAPs, inputs = get_NAPs(model_name,
                            data_name,
                            layers_of_interest,
                            nap_params["n_random_examples"],
                            error_mode=nap_params["error_mode"])

    neuron_activations = None
    if not topomap_params["neuron_activations"]["use_naps"]:
        neuron_activations = get_neuron_activations(model_name,
                                                    data_name,
                                                    layers_of_interest,
                                                    topomap_params["neuron_activations"])

    ANNScan = TopomapVisualizer(NAPs,
                                inputs=inputs,
                                neuron_activations=neuron_activations,
                                distance_metric=distance_metric)

    ANNScan.set_plot_params(plot_params)

    for layer in layers_of_interest:
        for method in topomap_params['methods']:
            ANNScan.compute_topomap(method, layer)

    ANNScan.align_topomaps_layerwise(layers_of_interest[0])

    ANNScan.save(precomputed_path)

# using the computed or loaded model
for layer in layers_of_interest:
    for method in topomap_params['methods']:
        # ANNScan.plot_dendrogram(layer, orientation='top', output_dir=output_dir)
        ANNScan.plot_topomap(method, layer, interpolated=True, output_dir=output_dir)
        # ANNScan.plot_topomap(method, layer, interpolated=False, output_dir=output_dir)
