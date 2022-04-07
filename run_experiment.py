"""
Script with additional parameters to conduct robustness experiments by performing repeated computations.
Parameters:
    '-c',  '--config'            path to config.json
    '-o',  '--output'            output directory
    '-f',  '--force-recompute'   force recomputing of existing runs (optional), default=False
    '-r',  '--repeat'            number of repetitions (optional), default=1
    '-rt', '--repeat-topomap'    repeat only topomap layout computation (optional), default=False
    '-t',  '--time'              store runtime (optional), default=False

Output:
    - saved evaluation metrics:   <output_dir>/evaluation.npy
    - plotted evaluation metrics: <output_dir>/eval_<metric>.pdf
"""

import shutil
import os
import argparse
import time
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from src.neuron_activations import get_NAPs, get_neuron_activations
from src.topomap_class import TopomapVisualizer
from src.config import load_config, assert_element_of


def visualize_method_error(eval_array_path, method_names, metric_ranges, has_auc=False, plot_runtimes=False,
                           output_dir=None):
    """
    function for plotting evaluation results
    :param eval_array_path: path to evaluation.npy
    :param method_names: list of names of layouting methods
    :param metric_ranges: index ranges for each metric
    :param has_auc: whether the range includes an AUC score
    :param plot_runtimes: whether to plot a runtime plot
    :param output_dir: output directory for the plots
    :return: None
    """
    colors = np.array(['#E377C2', '#3C75AF', '#EF8536', '#529E3E', '#C43932', '#85584E', '#8D69B8'])
    method_names = np.array(method_names)
    eval_array = np.load(eval_array_path)

    for metric in metric_ranges.keys():
        from_idx, to_idx = metric_ranges[metric]
        if has_auc:
            mse_errs = eval_array[0, :, from_idx:to_idx - 1, :]
            aucs = eval_array[0, :, to_idx - 1, :]
        else:
            mse_errs = eval_array[0, :, from_idx:to_idx, :]
            aucs = None
        runtimes = eval_array[0, :, -1, :]

        method_order = np.argsort(np.mean(aucs, axis=1))[::-1]
        method_names_ordered = method_names[method_order]
        colors_ordered = colors[method_order]
        mse_errs = mse_errs[method_order, :]
        aucs = aucs[method_order, :]
        runtimes = runtimes[method_order, :]

        fig, axes = plt.subplots(2, 1, figsize=(3.5, 5))

        ax = axes[0]
        idx = 0
        mean_errors = np.mean(mse_errs, axis=2)
        for err, color in zip(mean_errors[::-1], colors_ordered[::-1]):
            ax.plot(err, label=method_names_ordered[::-1][idx], color=color)
            idx += 1
            ax.set_ylim([0, 78])
        ax.legend()
        if metric == 'resize_mse':
            ax.set_xlabel('shrink size (px)')
            ax.set_xticks(np.arange(0.0, 10.0, 1.0), labels=np.arange(10, 56, 5)[::-1])
        elif metric == 'blur_mse':
            ax.set_xlabel('Gaussian blur radius (px)')
            ax.set_xticks(np.arange(0.0, 10.0, 1.0), labels=np.arange(2, 21, 2))
        ax.set_ylabel('MSE')

        ax = axes[1]

        violin_parts = ax.violinplot(np.transpose(aucs), vert=False, showmeans=True)

        for key in violin_parts.keys():
            if key != 'bodies':
                violin_parts[key].set_color('black')
        for pc, color in zip(violin_parts['bodies'], colors_ordered):
            pc.set_color(color)

        ax.set_yticks(np.arange(1, len(method_names) + 1))
        ax.set_yticklabels(method_names_ordered)

        ax.set_xlabel('MSE AUC')

        fig.tight_layout()

        if output_dir is None:
            plt.show()
        else:
            fig.savefig(os.path.join(output_dir, 'eval_' + metric + '.pdf'),
                        format='pdf',
                        # bbox_inches='tight',
                        pad_inches=0.02)

            plt.close(fig)

        if plot_runtimes:
            fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.5))

            violin_parts = plt.violinplot(np.transpose(runtimes), vert=False, showmedians=True)

            for key in violin_parts.keys():
                if key != 'bodies':
                    violin_parts[key].set_color('black')
            for pc, color in zip(violin_parts['bodies'], colors_ordered):
                pc.set_color(color)

            ax.set_yticks([])
            ax.set_yticklabels([])
            handles = [mlines.Line2D([], [], color=c, marker='o', linestyle='None',markersize=8, label=m)
                       for c, m in zip(colors_ordered[::-1], method_names_ordered[::-1])]
            ax.legend(handles=handles, handlelength=0.5, loc='lower right')

            ax.set_xlabel('runtime (s)')

            if output_dir is None:
                plt.show()
            else:
                fig.savefig(os.path.join(output_dir, 'eval_runtimes.pdf'),
                            format='pdf',
                            bbox_inches='tight')

                plt.close(fig)


# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='path to config.json', required=True)
parser.add_argument('-o', '--output', help='output directory', required=True)
parser.add_argument('-f', '--force-recompute', help='force recomputing of existing runs',
                    required=False, action='store_true')
parser.add_argument('-r', '--repeat', help='number of repetitions', required=False)
parser.add_argument('-rt', '--repeat-topomap', help='repeat only topomap layout computation',
                    required=False, action='store_true')
parser.add_argument('-t', '--time', help='store runtime', required=False, action='store_true')
args = parser.parse_args()
config_path = args.config
force_recompute = args.force_recompute
output_dir = os.path.join(args.output, config_path.split('/')[-1].split('.')[0])
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

n_repetitions = args.repeat
n_repetitions = 1 if n_repetitions is None else int(n_repetitions)
repeat_topomap_only = args.repeat_topomap
track_time = args.time

config = load_config(config_path)
model_name = config["model"]
data_name = config["data"]
if "data_path" in config.keys():
    data_path = config["data_path"]
else:
    data_path = None
layers_of_interest = config["layers"]
distance_metric = config["distance_metric"]
nap_params = config['nap_params']
topomap_params = config['topomap_params']
plot_params = config['plot_params']

# initialize numpy array for all evaluation values
n_metrics = 0
eval_metrics = []
eval_params = []
metric_ranges = dict()
# shrink/grow adds 10 single MSEs and 1 AU(MSE)C value
eval_metrics.append("resize_mse")
eval_params.append(np.arange(10))
metric_ranges["resize_mse"] = [n_metrics, n_metrics + 11]
n_metrics += 11

eval_metrics.append("blur_mse")
eval_params.append(np.arange(10))
metric_ranges["blur_mse"] = [n_metrics, n_metrics + 11]
n_metrics += 11

if track_time:
    # increase number of metrics to store by one (last position is time)
    n_metrics += 1

assert_element_of(eval_metrics, ['resize_mse', 'blur_mse'], 'eval_metrics', each_element=True)

# running repeated analyses according to the config or loading the evaluation.npy if it is pre-computed
if force_recompute or not os.path.isfile(os.path.join(output_dir, "evaluation.npy")):
    eval_results = np.zeros([len(layers_of_interest),
                             len(topomap_params['methods']),
                             n_metrics,
                             n_repetitions])

    for rep in range(n_repetitions):
        if rep == 0 or not repeat_topomap_only:
            NAPs, inputs = get_NAPs(model_name,
                                    data_name,
                                    layers_of_interest,
                                    nap_params["n_random_examples"],
                                    error_mode=nap_params["error_mode"],
                                    data_path=data_path)

            neuron_activations = None
            if not topomap_params["neuron_activations"]["use_naps"]:
                neuron_activations = get_neuron_activations(model_name,
                                                            data_name,
                                                            layers_of_interest,
                                                            topomap_params["neuron_activations"],
                                                            data_path=data_path)

            ANNScan = TopomapVisualizer(NAPs,
                                        inputs=inputs,
                                        neuron_activations=neuron_activations,
                                        distance_metric=distance_metric)

            ANNScan.set_plot_params(plot_params)

        ANNScan.clear_topomaps()  # make sure to also recompute the PSO initialization
        for l_id, layer in enumerate(layers_of_interest):
            for m_id, method in enumerate(topomap_params['methods']):
                if track_time:
                    start = time.time()

                ANNScan.compute_topomap(method, layer)
                if rep == 0:
                    ANNScan.plot_topomap(method, layer, interpolated=True, output_dir=output_dir)

                if track_time:
                    end = time.time()
                    time_elapsed = end - start
                    eval_results[l_id, m_id, -1, rep] = time_elapsed

                eval_example_output_dir = os.path.join(output_dir, "layer"+str(layer)+"_"+method)
                if not os.path.isdir(eval_example_output_dir):
                    os.makedirs(eval_example_output_dir)
                for metric, params in zip(eval_metrics, eval_params):

                    evaluation_measure, has_auc = ANNScan.evaluate_topomap_quality(layer, method, metric, params,
                                                                                   output_dir=eval_example_output_dir)
                    from_idx, to_idx = metric_ranges[metric]

                    # for 'rep' repetitions, store the quality
                    # for each layer, layouting method, evaluation metric and parameter
                    eval_results[l_id, m_id, from_idx:to_idx, rep] = evaluation_measure
                shutil.rmtree(eval_example_output_dir)

    np.save(os.path.join(output_dir, "evaluation.npy"), eval_results)
else:
    print("using evaluation.npy. To force recomputation, use the -f flag.")

# plot the computed or loaded evaluation measures
visualize_method_error(eval_array_path=os.path.join(output_dir, "evaluation.npy"),
                       method_names=topomap_params['methods'],
                       metric_ranges=metric_ranges,
                       has_auc=True,  # has_auc,
                       plot_runtimes=track_time,
                       output_dir=output_dir)
