import os
import argparse

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from src.neuron_activations import get_NAPs
from src.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='path to config.json', required=True)

args = parser.parse_args()

config_path = args.config
force_recompute = True
create_topomap_video = True
output_dir = os.path.join("output_base_path", config_path.split('/')[-1].split('.')[0])
checkpoint_path = 'checkpoints_for_tp/'
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

