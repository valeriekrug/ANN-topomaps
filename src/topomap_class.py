import os.path
import copy
import pickle
import shutil
import re

import numpy as np
from scipy import interpolate

from src.topomap_quality_utils import compute_topomap_image_quality
from src.topomaps import *
from src.utils import *
from src.visualization import *
from matplotlib import cm
from matplotlib.colors import Normalize

def load_experiment(pkl_file_path):
    with open(pkl_file_path, 'rb') as object_file:
        saved_experiment = pickle.load(object_file)

    return saved_experiment


def reshape_to_2d(ndarray):
    values_shape = ndarray.shape
    if len(values_shape) < 2:
        ndarray = np.expand_dims(ndarray, 0)
        values_shape = ndarray.shape

    if len(values_shape) <= 2:
        ndarray = np.transpose(ndarray)
    else:
        ndarray = np.transpose(np.reshape(ndarray,
                                          [np.prod(values_shape[:-1]),
                                           values_shape[-1]]
                                          )
                               )
    return ndarray


class TopomapVisualizer:
    def __init__(self, nap_values, inputs=None, group_names=None,
                 neuron_activations=None, distance_metric='cosine', plot_layout=None, distance_mat=None, group_dist=None, links=None, group_ord=None, do_distance_calculations=True
                ):

        self.error_mode = nap_values['error_mode']
        nap_values = nap_values['values']
        self.n_groups = nap_values[next(iter(nap_values.keys()))].shape[0]

        self.inputs = inputs
        if inputs is not None:
            group_names = [*self.inputs.keys()]

        if self.error_mode in ['binary_contrast', 'confusion_matrix']:
            n_lbls = len(np.unique([s.split('_')[0] for s in group_names]))
            if self.error_mode == 'binary_contrast':
                plot_layout = (2, n_lbls)
            elif self.error_mode == 'confusion_matrix':
                plot_layout = (n_lbls, n_lbls)
        else:
            if plot_layout is not None:
                assert self.n_groups <= np.prod(plot_layout), "too many groups for given plot layout"
                assert len(plot_layout) == 2, "plot layout must be 2-dimensional"
            else:
                nrow = 1
                ncol = self.n_groups  # //2
                plot_layout = (nrow, ncol)
        self.plot_layout = plot_layout
        
        self.plotted_inputs = None

        if group_names is None:
            group_names = np.arange(self.n_groups)

        assert self.n_groups == len(group_names), "number of group names does not match first dim. of values"

        self.group_names = np.array(group_names, 'str')

        self.naps = nap_values
        if neuron_activations is None:
            self.neuron_activations = dict()
        else:
            self.neuron_activations = neuron_activations

        self.naps_per_channel = dict()
        self.channel_color_per_group = dict()
        self.clim = dict()

        self.topomaps = dict()
        if distance_mat is not None:
            self.distance_matrices = distance_mat
        else:
            self.distance_matrices = dict()
            
        if distance_mat is not None:
            self.distance_matrices = distance_mat
        else:
            self.distance_matrices = dict()
          
        if group_dist is not None:
            self.group_distances = group_dist
        else:
            self.group_distances = dict()
        
        if links is not None:
            self.linkages = links
        else:
            self.linkages = dict()
        
        if group_ord is not None:
            self.group_order = group_ord
        else:
            self.group_order = dict()

        if not do_distance_calculations:
            self.used_metric = distance_metric
        else:
            self.used_metric = None

        self.layers = [*self.naps.keys()]

        for layer in self.layers:
            self.naps_per_channel[layer] = reshape_to_2d(self.naps[layer])

            nap_values_shape = self.naps[layer].shape
            if len(nap_values_shape) <= 2:
                self.channel_color_per_group[layer] = dict(zip(group_names, self.naps[layer]))
            else:
                avg_dims = tuple(np.arange(1, len(nap_values_shape) - 1))
                aggregated_naps = np.mean(self.naps[layer], avg_dims)
                self.channel_color_per_group[layer] = dict(zip(group_names, aggregated_naps))

            self.clim[layer] = symmetric_clim(np.concatenate([*self.channel_color_per_group[layer].values()]))
            if neuron_activations is None:
                self.neuron_activations[layer] = np.transpose(np.stack([*self.channel_color_per_group[layer].values()]))
            else:
                self.neuron_activations[layer] = reshape_to_2d(self.neuron_activations[layer])

            # essentially each topomap is only a coordinate assigned to each neuron/feature map
            self.topomaps[layer] = dict()

            # to avoid recomputing distance matrices, keep them in a dict
            if do_distance_calculations:
                self.distance_matrices[layer] = dict()
                self.group_distances[layer] = dict()
                self.linkages[layer] = dict()
                self.group_order[layer] = dict()

        if do_distance_calculations:
            self.change_distance_metric(distance_metric)

        self.plot_params = {'cmap': 'bwr',
                            'scale': 2,
                            'ordered': True,
                            'avg_inputs': False,
                            'use_title': True}

    def save(self, output_path):
        with open(output_path, 'wb') as output_file:
            pickle.dump(self, output_file)

    def set_plot_params(self, plot_params):
        self.plot_params = plot_params
        self.plot_params['cmap'] = 'bwr'

    def get_group_ordering(self, layer):
        return self.group_order[layer][self.used_metric]

    def get_group_distance(self, layer):
        return self.group_distances[layer][self.used_metric]

    def change_distance_metric(self, metric):
        for layer in self.layers:
            self.change_neuron_distance_metric(metric, layer)
            self.change_group_distance_metric(metric, layer)
        self.used_metric = metric

    def change_neuron_distance_metric(self, metric, layer):
        if metric not in self.distance_matrices[layer].keys():
            dist_mat = compute_distance_metric(self.neuron_activations[layer], metric)
            self.distance_matrices[layer][metric] = dist_mat

    def change_group_distance_metric(self, metric, layer):
        if metric not in self.group_distances[layer].keys():

            correct_group_ids = np.arange(len(self.group_names))
            if self.error_mode in ['binary_contrast', 'confusion_matrix']:
                lbl_pred_matrix = np.array([s.split('_') for s in self.group_names])
                if self.error_mode == 'binary_contrast':
                    correct_group_ids = np.argwhere(lbl_pred_matrix[:, 1] == 'correct')[:, 0]
                elif self.error_mode == 'confusion_matrix':
                    correct_group_ids = np.argwhere(lbl_pred_matrix[:, 0] == lbl_pred_matrix[:, 1])[:, 0]

            dist_mat = compute_distance_metric(np.transpose(self.naps_per_channel[layer][:, correct_group_ids]), metric)
            self.group_distances[layer][metric] = dist_mat

            linkage = compute_linkage(dist_mat)
            self.linkages[layer][metric] = linkage

            group_order = get_dendrogram_order(linkage)
            self.group_order[layer][metric] = self.group_names[correct_group_ids][group_order]

    def clear_topomaps(self):
        self.topomaps = dict()
        for layer in self.layers:
            self.topomaps[layer] = dict()

    def compute_topomap(self, method, layer):
        # compute topomap - method dependent
        # convert to coordinates - method dependent
        # store coordinates in self.computed_topomaps - method independent

        dist_mat = self.distance_matrices[layer][self.used_metric]
        computed_methods = self.topomaps[layer].keys()

        if method == 'random':
            coordinates = random_layout(squareform(dist_mat).shape[0])
        elif method == 'random_PSO':
            if 'random' not in computed_methods:
                self.compute_topomap('random', layer)
            coordinates = train_PSO(dist_mat,
                                    pos_init=self.topomaps[layer]['random'])
        elif method == 'PSO':
            coordinates = train_PSO(dist_mat)
        elif method == 'graph':
            graph = make_threshold_graph(dist_mat,
                                         np.percentile(dist_mat, 7.5))
            coordinates = layout_graph(graph)
        elif method == 'graph_PSO':
            if 'graph' not in computed_methods:
                self.compute_topomap('graph', layer)
            coordinates = train_PSO(dist_mat,
                                    pos_init=self.topomaps[layer]['graph'])
        elif method == 'PCA':
            coordinates = compute_PCA(self.neuron_activations[layer])
        elif method == 'PCA_PSO':
            if 'PCA' not in computed_methods:
                self.compute_topomap('PCA', layer)
            coordinates = train_PSO(dist_mat,
                                    pos_init=self.topomaps[layer]['PCA'])
        elif method == 'TSNE':
            coordinates = compute_TSNE(self.neuron_activations[layer])
        elif method == 'TSNE_PSO':
            if 'TSNE' not in computed_methods:
                self.compute_topomap('TSNE', layer)
            coordinates = train_PSO(dist_mat,
                                    pos_init=self.topomaps[layer]['TSNE'])
        elif method == 'UMAP':
            coordinates = compute_UMAP(self.neuron_activations[layer])
        elif method == 'UMAP_PSO':
            if 'UMAP' not in computed_methods:
                self.compute_topomap('UMAP', layer)
            coordinates = train_PSO(dist_mat,
                                    pos_init=self.topomaps[layer]['UMAP'])
        elif method == 'SOM':
            coordinates = train_SOM(self.neuron_activations[layer])
        elif method == 'SOM_PSO':
            if 'SOM' not in computed_methods:
                self.compute_topomap('SOM', layer)
            coordinates = train_PSO(dist_mat,
                                    pos_init=self.topomaps[layer]['SOM'])
        else:
            coordinates = None

        # scale coordinates to be normalized between 0,1 in both dimensions
        coordinates = coordinates - np.min(coordinates, 0)
        coordinates = coordinates / np.max(coordinates, 0)
        self.topomaps[layer][method] = coordinates

    def evaluate_topomap_quality(self, layer, method, metric, params, compute_auc=True, output_dir=None):
        was_plot_individually = self.plot_params['plot_individually']
        self.plot_params['plot_individually'] = True
        temp_dir = os.path.join(output_dir, 'topo_tmp')
        output_path = self.plot_topomap(method, layer, output_dir=temp_dir)

        quality_array = compute_topomap_image_quality(output_path, metric, params, return_auc=compute_auc, output_dir=None)
        self.plot_params['plot_individually'] = was_plot_individually
        shutil.rmtree(temp_dir)
        return (quality_array, compute_auc)


    def align_topomap_to(self, layer, reference_layer, methods=None):
        if methods is None:
            methods = [*self.topomaps[reference_layer].keys()]
        for method in methods:
            topomap_target = self.topomaps[layer][method]
            topomap_reference = self.topomaps[reference_layer][method]

            # get colors across all groups for both layers
            # and normalize them to represent plotting colors
            colors_target = stack_ndarray_dict(copy.copy(self.channel_color_per_group[layer]))
            colors_target = np.transpose(colors_target)
            colors_target /= get_absmax(colors_target)
            colors_reference = stack_ndarray_dict(copy.copy(self.channel_color_per_group[reference_layer]))
            colors_reference = np.transpose(colors_reference)
            colors_reference /= get_absmax(colors_reference)
            color_dists = euclidean_distances(colors_reference, colors_target)

            rot_coords = get_all_rotations(topomap_target)
            dist_mats = get_rotation_dist_mats(topomap_reference, rot_coords)

            similarity_mats = np.sqrt(2) - dist_mats
            similarity_mats[similarity_mats < np.median(similarity_mats)] = 0
            color_similarity = np.max(color_dists) - color_dists

            weighted_similarity_mats = similarity_mats * color_similarity

            similarity_score = np.mean(weighted_similarity_mats, axis=(1, 2))
            opt_rotation_idx = np.argmax(similarity_score)
            self.topomaps[layer][method] = rot_coords[opt_rotation_idx]

    def align_topomaps_layerwise(self, reference_layer=None):
        if reference_layer is None:
            reference_layer = self.layers[0]

        for layer in self.layers:
            if layer != reference_layer:
                self.align_topomap_to(layer, reference_layer)

    def plot_input_on_axis(self, ax, group_idx):
        if group_idx in self.group_names:
            group_inputs = self.inputs[group_idx]

            if self.plot_params['avg_inputs']:
                group_inputs = np.mean(group_inputs, 0)
            else:
                random_example_id = np.random.permutation(np.arange(len(group_inputs)))[0]
                group_inputs = group_inputs[random_example_id]
            if group_inputs.shape[-1] == 1:
                ax.imshow(group_inputs[:, :, 0],
                          clim=[0, 1],
                          cmap='Greys')
            else:
                ax.imshow(group_inputs)
        else:
            ax.set_axis_off()

        ax.set_xticks([])
        ax.set_yticks([])
        if self.plot_params['use_title']:
            ax.set_title(str(group_idx))
            
    def _plot_consistant_input_on_axis(self, ax, group_idx):
        if group_idx in self.group_names:
            group_inputs = self.inputs[group_idx]
            group_idx = re.search(r'\d+', group_idx).group()

            if self.plot_params['avg_inputs']:
                group_inputs = np.mean(group_inputs, 0)
            else:
                if self.plotted_inputs is None:
                    group_names = [*self.inputs.keys()]
                    
                    self.plotted_inputs = [None] * len(np.unique([s.split('_')[0] for s in group_names]))
                    random_example_id = np.random.permutation(np.arange(len(group_inputs)))[0]
                    group_inputs = group_inputs[random_example_id]
                    self.plotted_inputs[int(group_idx)] = group_inputs
                elif self.plotted_inputs[int(group_idx)] is None:
                    #print(group_inputs)
                    random_example_id = np.random.permutation(np.arange(len(group_inputs)))[0]
                    group_inputs = group_inputs[random_example_id]
                    self.plotted_inputs[int(group_idx)] = group_inputs
                else:
                    group_inputs = self.plotted_inputs[int(group_idx)]
            if group_inputs.shape[-1] == 1:
                ax.imshow(group_inputs[:, :, 0],
                          clim=[0, 1],
                          cmap='Greys')
            else:
                ax.imshow(group_inputs)
        else:
            ax.set_axis_off()

        ax.set_xticks([])
        ax.set_yticks([])
        if self.plot_params['use_title']:
            ax.set_title(str(group_idx))

    def plot_topomap_on_axis(self, ax, coordinate_information, layer, group_idx, interpolated, method=None,
                             output_dir=None, return_quality=False):
        axes = [ax]
        plot_individually = self.plot_params['plot_individually']
        if plot_individually:
            scale = self.plot_params['scale']
            fig, ax_s = plt.subplots(1, 1, figsize=(scale, 0.95 * scale))
            axes.append(ax_s)

        if group_idx in self.group_names:
            colors = self.channel_color_per_group[layer][group_idx]
            if interpolated:
                positions, x, y, xx, yy = coordinate_information
                new_xy = interpolate.griddata(positions,
                                              colors,
                                              (xx, yy),
                                              method='linear')
                if return_quality:
                    interpolated_colors = np.transpose(new_xy[:, ::-1])
                    c_min, c_max = self.clim[layer]
                    norm = Normalize(c_min, c_max)
                    image_array = norm(interpolated_colors)
                    image_array = cm.bwr(image_array)[:, :, :3]
                    quality_results = compute_topomap_image_quality([image_array], 'components', params=None)
                for ax in axes:
                    ax.imshow(np.transpose(new_xy[:, ::-1]),
                              clim=self.clim[layer],
                              cmap=self.plot_params['cmap'])
            else:
                positions, x, y = coordinate_information
                for ax in axes:
                    ax.scatter(x, y,
                               c=colors,
                               vmin=self.clim[layer][0],
                               vmax=self.clim[layer][1],
                               cmap=self.plot_params['cmap'])
        else:
            ax.imshow(np.zeros([100,100]),
                      clim=[0,1],
                      cmap='Greys')

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        if self.plot_params['use_title']:
            ax.set_title(str(group_idx))
        if plot_individually:
            axes[1].set_axis_off()
            fig.savefig(os.path.join(output_dir, str(group_idx) + '.pdf'),
                        format='pdf', bbox_inches='tight', pad_inches=0.02)

            plt.close(fig)
        if return_quality:
            return quality_results[0]

    def plot_topomap(self, method, layer, interpolated=True, resolution=100, output_dir=None, as_img=False, epoch=None, use_same_input_as_representative = False, return_quality = False):
        positions = self.topomaps[layer][method]
        x, y = np.transpose(positions)

        ordered = self.plot_params['ordered']
        scale = self.plot_params['scale']
        plot_individually = self.plot_params['plot_individually']

        nrow, ncol = self.plot_layout

        plot_inputs = self.plot_params['plot_inputs']
        if plot_inputs:
            if self.inputs is None:
                print('To create topomaps with inputs, you need to provide them when creating the object.')
                plot_inputs = False
                self.plot_params['plot_inputs'] = False
            else:
                if self.error_mode == 'binary_contrast':
                    if not use_same_input_as_representative:
                        nrow += 2
                    else:
                        nrow += 1
                elif self.error_mode == 'confusion_matrix':
                    nrow += 1
                    ncol += 1
                else:
                    nrow *= 2

        fig, axes = plt.subplots(nrow, ncol, figsize=(scale * ncol, scale * 0.95 * nrow))

        if nrow == 1:
            axes = np.expand_dims(axes, 0)
        elif ncol == 1:
            axes = np.expand_dims(axes, 1)

        if interpolated:
            path_term_type = 'interpolated'
            xx_min, yy_min = np.min(positions, axis=0)
            xx_max, yy_max = np.max(positions, axis=0)

            xx, yy = np.mgrid[xx_min:xx_max:complex(resolution), yy_min:yy_max:complex(resolution)]

            coordinate_information = [positions, x, y, xx, yy]
        else:
            path_term_type = 'scatter'
            coordinate_information = [positions, x, y]

        output_file_stem = None
        if output_dir is not None:
            output_file_stem = os.path.join(output_dir,
                                            'topomap_layer' + str(layer) + '_' + method + '_' + path_term_type)
        if plot_individually:
            if not os.path.isdir(output_file_stem):
                os.makedirs(output_file_stem)

        if self.error_mode in ['binary_contrast', 'confusion_matrix']:
            if ordered:
                order = [g.split('_')[0] for g in self.get_group_ordering(layer)]
            else:
                lbls = [g.split('_')[0] for g in self.group_names]
                order = np.arange(len(np.unique(lbls)))

            if self.error_mode == 'binary_contrast':
                quality_results = []
                for col, lbl in enumerate(order):
                    group_idx = str(lbl) + '_correct'

                    row = 0
                    if plot_inputs:
                        ax = axes[row, col]
                        if not use_same_input_as_representative:
                            self.plot_input_on_axis(ax, group_idx)
                        else:
                            self._plot_consistant_input_on_axis(ax, group_idx)
                        row += 1
                    ax = axes[row, col]
                    quality_results.append(self.plot_topomap_on_axis(ax, coordinate_information, layer, group_idx, interpolated, method,
                                          output_dir=output_file_stem, return_quality=True))
                    #self.plot_topomap_on_axis(ax, coordinate_information, layer, group_idx, interpolated,
                    #                          output_dir=output_file_stem)
                    row += 1

                    group_idx = str(lbl) + '_wrong'
                    ax = axes[row, col]
                    self.plot_topomap_on_axis(ax, coordinate_information, layer, group_idx, interpolated,
                                              output_dir=output_file_stem)
                    if plot_inputs and not use_same_input_as_representative:
                        row += 1
                        ax = axes[3, col]
                        self.plot_input_on_axis(ax, group_idx)

            elif self.error_mode == 'confusion_matrix':
                for row, lbl in enumerate(order):
                    if plot_inputs:
                        self.plot_input_on_axis(axes[0, 0], '')
                        group_idx = str(lbl) + '_' + str(lbl)
                        ax = axes[row + 1, 0]
                        self.plot_input_on_axis(ax, group_idx)
                        ax = axes[0, row + 1]
                        self.plot_input_on_axis(ax, group_idx)
                    for col, pred in enumerate(order):
                        if plot_inputs:
                            ax = axes[row + 1, col + 1]
                        else:
                            ax = axes[row, col]
                        group_idx = str(lbl) + '_' + str(pred)
                        self.plot_topomap_on_axis(ax, coordinate_information, layer, group_idx, interpolated,
                                                  output_dir=output_file_stem)

        else:
            if ordered:
                order = self.get_group_ordering(layer)
            else:
                order = self.group_names
            quality_results = []
            for plot_idx, group_idx in enumerate(order):
                row = plot_idx // ncol
                col = plot_idx % ncol

                ax = axes[row, col]
                if plot_inputs:
                    if not use_same_input_as_representative:
                        self.plot_input_on_axis(ax, group_idx)
                    else:
                        self._plot_consistant_input_on_axis(ax, group_idx)
                    ax = axes[row + (nrow // 2), col]

                quality_results.append(self.plot_topomap_on_axis(ax, coordinate_information, layer, group_idx, interpolated, method,
                                          output_dir=output_file_stem, return_quality=True))

        if output_dir is None:
            plt.show()
        else:
            fig.tight_layout()
            if not as_img:
                plt.savefig(
                    output_file_stem + '.pdf',
                    format='pdf', bbox_inches='tight', pad_inches=0.02)
            else:
                current_epoch = (re.search(r'(?<=epoch_)\d+', epoch).group())
                current_batch = (re.search(r'(?<=batch_)\d+', epoch).group())
                plt.savefig(
                    output_file_stem + '_epoch-' + current_epoch + '_batch-' + current_batch + '.png',
                    format='png', bbox_inches='tight', pad_inches=0.02)
            plt.close(fig)
        if return_quality:
            return quality_results, output_file_stem
        else:
            return output_file_stem

    def plot_inputs(self, layer, output_dir=None):
        nrow, ncol = self.plot_layout
        ordered = self.plot_params['ordered']
        scale = self.plot_params['scale']

        fig, axes = plt.subplots(nrow, ncol, figsize=(scale * ncol, scale * 0.95 * nrow))
        if nrow == 1:
            axes = np.expand_dims(axes, 0)
        elif ncol == 1:
            axes = np.expand_dims(axes, 1)

        group_names = [*self.inputs.keys()]

        if self.error_mode in ['binary_contrast', 'confusion_matrix']:
            if ordered:
                order = [g.split('_')[0] for g in self.get_group_ordering(layer)]
            else:
                lbls = [g.split('_')[0] for g in group_names]
                order = np.arange(len(np.unique(lbls)))

            if self.error_mode == 'binary_contrast':
                for col, lbl in enumerate(order):
                    for row, suffix in enumerate(['_correct', '_wrong']):
                        ax = axes[row, col]
                        group_idx = str(lbl) + suffix
                        self.plot_input_on_axis(ax, group_idx)
            elif self.error_mode == 'confusion_matrix':
                for row, lbl in enumerate(order):
                    for col, pred in enumerate(order):
                        ax = axes[row, col]
                        group_idx = str(lbl) + '_' + str(pred)
                        self.plot_input_on_axis(ax, group_idx)

        else:
            if ordered:
                order = self.get_group_ordering(layer)
            else:
                order = self.group_names
            for plot_idx, group_idx in enumerate(order):
                row = plot_idx // ncol
                col = plot_idx % ncol
                ax = axes[row, col]
                self.plot_input_on_axis(ax, group_idx)

        if output_dir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(output_dir, 'inputs_layer' + str(layer) + '(order).pdf'),
                        format='pdf')
            plt.close(fig)

    def plot_dendrogram(self, layer, orientation='bottom', output_dir=None):
        plot_dendrogram(self.linkages[layer][self.used_metric], layer, self.plot_params['scale'], self.plot_layout[1],
                        orientation=orientation, output_dir=output_dir)
