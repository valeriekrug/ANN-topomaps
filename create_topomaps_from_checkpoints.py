import sys
import os
import ipywidgets as widgets
import re
import argparse
import pickle

os.environ['HTTP_PROXY'] = 'http://fp.cs.ovgu.de:3210/'
os.environ['HTTPS_PROXY'] = 'http://fp.cs.ovgu.de:3210/'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0,'/scratch/python_envs/annalyzer_env/python/lib/python3.8/site-packages/')
#os.environ["TF_GPU_ALLOCATOR"]= "cuda_malloc_async"

import argparse

from src.neuron_activations import get_NAPs, get_neuron_activations, get_neuron_activations_of_checkpoint, get_NAPs_of_checkpoint
from src.topomap_class import TopomapVisualizer, load_experiment
from src.config import load_config
from src.topomap_quality_utils import compute_topomap_image_quality

config_path = "configs/examples/mnist_mlp.json"
output_dir = os.path.join("output_base_path", config_path.split('/')[-1].split('.')[0])
checkpoint_path = 'output_base_path/checkpoints_for_video/'

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', help='Current checkpoint index', required=True)

args = parser.parse_args()
index = (int(args.index))

config = load_config(config_path)
model_name = config["model"]
data_name = config["data"]
layers_of_interest = config["layers"]
distance_metric = config["distance_metric"]
nap_params = config['nap_params']
topomap_params = config['topomap_params']
plot_params = config['plot_params']

checkpoints = os.listdir(checkpoint_path)

checkpoints.sort()
checkpoints = sorted(checkpoints, key=lambda s: int(re.search(r'(?<=epoch_)\d+', s).group()))

coordinates = None
group_order = None
neuron_activations = None
ANNScan = None
distance_mat = None
group_dist = None
links = None
input_representatives = None

if not index==0:
    with open('objs.pkl', 'rb') as f: 
        coordinates, group_order, distance_mat, group_dist, links, input_representatives, neuron_activations = pickle.load(f)
    with open('topomap_qualities.pkl', 'rb') as f:
        qualities = pickle.load(f)
print("Beginning with checkpoint: " + str(index))
i = len(checkpoints) - index - 1
print("Calculating NAPs..")
NAPs, inputs = get_NAPs_of_checkpoint(model_name,
                data_name,
                layers_of_interest,
                nap_params["n_random_examples"],
                error_mode=nap_params["error_mode"], 
                checkpoint_path=os.path.join(checkpoint_path, checkpoints[i]))
if(index==0):
    print("Calculating Neuron Activations..")
    neuron_activations = get_neuron_activations_of_checkpoint(model_name,
                                                              data_name,
                                                              layers_of_interest,
                                                              topomap_params["neuron_activations"],
                                                              os.path.join(checkpoint_path, checkpoints[i]))
    print("Creating TopomapVisualizer..")  
    ANNScan = TopomapVisualizer(NAPs,
                        inputs=inputs,
                        neuron_activations=neuron_activations,
                        distance_metric=distance_metric)
else:
    print("Creating TopomapVisualizer..")  
    ANNScan = TopomapVisualizer(NAPs,
                            inputs=inputs,
                            neuron_activations=neuron_activations,
                            distance_metric=distance_metric,
                            distance_mat=distance_mat,
                            group_dist=group_dist,
                            links=links,
                            group_ord=group_order,
                            do_distance_calculations=False
                               )

print("Setting PlotParams for TopomapVisualizer..")  
ANNScan.set_plot_params(plot_params)

# At the newest checkpoints calculate the coordinates, otherwise take the coordinates of the latest checkpoint
if(index==0):
    print("Calculating coordinates and determining group order..") 
    for layer in layers_of_interest:
        for method in topomap_params['methods']:
            # Here the coordinates are set.
            ANNScan.compute_topomap(method, layer)
    coordinates = ANNScan.topomaps
    group_order = ANNScan.group_order
    distance_mat = ANNScan.distance_matrices
    group_dist = ANNScan.group_distances
    links = ANNScan.linkages
else:
    print("Setting coordinates and group order..") 
    ANNScan.topomaps = coordinates
    print("Setting old input representatives..") 
    ANNScan.plotted_inputs = input_representatives

#ANNScan.align_topomaps_layerwise(layers_of_interest[0])

print("Computing Topomaps and saving them..")
quality_per_layer = []
for layer in layers_of_interest:
    for method in topomap_params['methods']:
        # ANNScan.plot_dendrogram(layer, orientation='top', output_dir=output_dir)
        if input_representatives is None and (index==0):
            input_representatives = ANNScan.plotted_inputs
        quality_result, _ = ANNScan.plot_topomap(method, layer, interpolated=True, output_dir=output_dir, as_img=True, epoch=checkpoints[i], use_same_input_as_representative=True, return_quality=True)
        quality_per_layer.append(quality_result)
        # ANNScan.plot_topomap(method, layer, interpolated=False, output_dir=output_dir)
        
if(index==0):        
    with open('objs.pkl', 'wb') as f:
            pickle.dump([coordinates, group_order, distance_mat, group_dist, links, input_representatives, neuron_activations], f)
    qualities = [quality_per_layer]
    #qualities.append(quality_per_layer)
    with open('topomap_qualities.pkl', 'wb') as f:
            pickle.dump(qualities, f)
else:
    # Append at beginning, as we are iterating backwards over the topomaps
    qualities = [quality_per_layer] + qualities
    with open('topomap_qualities.pkl', 'wb') as f:
            pickle.dump(qualities, f)
ANNScan = None
NAPs = None
inputs = None
    