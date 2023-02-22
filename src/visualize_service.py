import logging
import os

import configparser
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve
from minisom import MiniSom
from sklearn.metrics.pairwise import pairwise_distances
import seaborn as sns
from scipy.cluster import hierarchy
from scipy import interpolate
from PIL import Image, ImageDraw
import tensorflow as tf
import json
import base64, io

from Utility.topomap_utils import *
from constants.visualize_constants import FEATURE_VISUALIZATION_TYPES
from services.file_service import FileService
from services.model_service import ModelService
from services.profile_service import ProfileService
from Utility.visualize_utils import get_transparent_image, get_base64_image, generate_non_responsive_neurons

logger = logging.getLogger('visualize_service')


def get_ax_position(ax):
    position = ax.get_position()
    position_array = np.array([position.x0,
                               position.y0,
                               position.width,
                               position.height])
    return position_array


def get_absmax_clim(nd_array):
    a = np.max(np.abs(nd_array))
    return (np.array([-a, a]))


class VisualizeService(object):
    def __init__(self, nap_settings, nap_path, profile_name):
        self.nap_settings = nap_settings
        self.nap_path = nap_path
        self.profile_name = profile_name

        self.config = configparser.ConfigParser()
        self.config.read('../configuration.val')

        self.file_service = FileService()
        # self.profile_name = profile_name
        # self.nap_service = NapService()
        # self.profile_service = ProfileService(model_path, dataset_path)

    def create_plot_paths(self, subdirectory_name, layer_names):
        path = self.config['analyze']['path']
        # TODO following line is only for development phase
        path = path.replace('cache', 'project')
        plot_path = os.path.join(path, self.profile_name, "visualize", subdirectory_name)

        for layer_name in layer_names:
            layer_plot_path = os.path.join(plot_path, layer_name)
            if not os.path.isdir(layer_plot_path):
                os.makedirs(layer_plot_path)
        return plot_path

    def get_layer_NAP_results(self, nap_service, layer_name):
        nap_result = np.load(nap_service[layer_name], allow_pickle=True)
        # nap_values = self.nap_service.get(self.model_path, layer_name)
        groups = [c_n for c_n in nap_result]
        n_groups = len(nap_result)

        nap_values = []
        # for group in np.arange(n_groups):
        for group in groups:
            group_nap = {
                "g_n": group,
                "g_nap": nap_result[str(group)]
            }
            nap_values.append(group_nap)
        return groups, n_groups, nap_values

    def plot_nap(self,
                 layer_names=None,
                 grid=None,
                 use_rgb=False,
                 diverging_cmap=None,
                 symmetric_cmap=None,
                 transpose=False,
                 invert_y=False):
        nap = self.file_service.get_by_layer_names(layer_names, self.nap_path)

        plot_path = self.create_plot_paths("NAP", layer_names)

        for layer_name in layer_names:
            layer_plot_path = os.path.join(plot_path, layer_name)

            groups, n_groups, nap_values = self.get_layer_NAP_results(nap, layer_name)

            if len(nap_values[0]['g_nap'].shape) == 1:
                print(f"skipping nap for {layer_name}")
                continue

            if grid is not None:
                # derive a grid size automatically or validate the given one
                if grid is None:
                    n_cols = 8
                    n_rows = np.ceil(n_groups / n_cols).astype('int')
                else:
                    n_rows, n_cols = grid
                    if n_cols * n_rows < n_groups:
                        raise ValueError(
                            'grid has {} positions, but there are {} groups'.format(n_cols * n_rows, n_groups))

            # validate usage of RGB
            if use_rgb:
                n_channels = nap_values[0]["g_nap"].shape[-1]
                if n_channels != 3:
                    warnings.warn(
                        'use_RGB is ignored. NAP values have {} channels, but RGB requires exactly 3.'.format(
                            n_channels))
                    use_rgb = False

            # derive the global min, max and abs_max over all groups
            glob_min, glob_max, glob_abs_max = [np.inf, -np.inf, -np.inf]
            for nap_value in nap_values:
                group_nap = nap_value["g_nap"]
                if not use_rgb and len(group_nap.shape) > 2:
                    group_nap = np.mean(group_nap, -1)
                group_min = np.min(group_nap)
                group_max = np.max(group_nap)
                group_abs_max = np.max(np.abs(group_nap))

                if group_min < glob_min:
                    glob_min = group_min
                if group_max > glob_max:
                    glob_max = group_max
                if group_abs_max > glob_abs_max:
                    glob_abs_max = group_abs_max

            if use_rgb:
                if glob_min < 0 or glob_max > 1:
                    if self.nap_settings['normalized']:
                        warn_text = 'use_RGB is ignored.' \
                                    'The NAPs are normalized, and therefore have no RGB value range [0..1].'
                    else:
                        warn_text = "use_RGB is ignored." \
                                    "NAP values are in [{},{}], which is no proper RGB value range [0..1]." \
                            .format(glob_min, glob_max)
                    warnings.warn(warn_text)
                    use_rgb = False

            # set the colormap
            if use_rgb:
                cmap = None
            elif diverging_cmap or (diverging_cmap == None and self.nap_settings['is_normalized']):
                cmap = 'bwr'
            else:
                cmap = 'Greys_r'

            if symmetric_cmap or (symmetric_cmap == None and self.nap_settings['is_normalized']):
                clim = (-glob_abs_max, glob_abs_max)
            else:
                clim = (glob_min, glob_max)

            # initialize a figure of the given grid size
            # (TODO include proper figure height for non-square NAPs)
            if grid is not None:
                grid_fig, grid_axes = plt.subplots(n_rows, n_cols)
                grid_fig.set_figwidth(20)
                grid_fig.set_figheight(n_rows * (20 / n_cols))
                flat_axes = grid_axes.flatten()
                for ax in flat_axes:
                    ax.axis('off')

            # plot each group on an individual axis
            for i, nap_value in enumerate(nap_values):
                group_nap = np.array(nap_value["g_nap"])
                if use_rgb or len(group_nap.shape) == 2:
                    plot_mat = group_nap
                else:
                    plot_mat = np.mean(group_nap, -1)
                if transpose:
                    plot_mat = np.transpose(plot_mat)
                if invert_y:
                    plot_mat = plot_mat[::-1, :]

                invididual_fig, ax = plt.subplots()
                invididual_fig.set_figheight(5)
                invididual_fig.set_figwidth(5)
                axes = [ax]

                if grid is not None:
                    axes.append(flat_axes[i])

                for ax in axes:
                    ax.imshow(plot_mat, cmap=cmap, clim=clim)
                    ax.axis('on')
                    ax.set_xticks([])
                    ax.set_yticks([])

                invididual_fig.savefig(layer_plot_path + "/" + nap_value["g_n"] + ".png", bbox_inches='tight')
                plt.close(invididual_fig)

            if grid is not None:
                grid_fig.savefig(layer_plot_path + "/all_groups.png", bbox_inches='tight')
                plt.close(grid_fig)

            # add a second plot with the colorbar
            # # TODO create a single plot for both parts
            # if not use_rgb:
            #     fig, axes = plt.subplots()
            #     fig.set_figwidth(20)
            #     fig.set_figheight(0.3)
            #     fig.colorbar(im, cax=axes, orientation='horizontal')
            #     plt.show()

        result = {
            "basic_path": plot_path,
            "layer_names": layer_names,
            "groups": groups
        }

        return result

    def visualize_clustering(self,
                             layer_names=None,
                             cluster_metric='euclidean',
                             linkage_method='complete',
                             score_percentile=80,
                             font_scale=1):

        nap = self.file_service.get_by_layer_names(layer_names, self.nap_path)

        plot_path = self.create_plot_paths("clustermap", layer_names)

        ordering_dict = dict()
        for layer_name in layer_names:
            layer_plot_path = os.path.join(plot_path, layer_name)

            groups, n_groups, nap_values = self.get_layer_NAP_results(nap, layer_name)

            group_names = [nap_value["g_n"] for nap_value in nap_values]
            naps_to_show = []
            profile_values = np.array([np.array(nap_value["g_nap"]).flatten() for nap_value in nap_values])

            distance_mat = pairwise_distances(profile_values, metric=cluster_metric)
            linkage = hierarchy.linkage(profile_values, metric=cluster_metric, method=linkage_method,
                                        optimal_ordering=True)

            dendrogram = hierarchy.dendrogram(linkage, no_plot=True)
            dendrogram_order = np.asarray(dendrogram['leaves'], dtype='int32')

            ordering_dict[layer_name] = [int(idx) for idx in dendrogram_order]

            distance_threshold = np.percentile(linkage[:, 2], float(score_percentile))
            clustering = hierarchy.fcluster(linkage,
                                            distance_threshold,
                                            'distance')
            ordered_clustering = clustering[dendrogram_order]
            existing_clusters = np.unique(clustering)

            network_palette = sns.husl_palette(len(existing_clusters), l=0.8)
            network_lut = dict(zip(existing_clusters, network_palette))
            colors = np.array([network_lut[i] for i in clustering])
            sns.set(font_scale=font_scale)
            clustermap = sns.clustermap(distance_mat,
                                        row_linkage=linkage,
                                        col_linkage=linkage,
                                        row_colors=colors,
                                        col_colors=colors,
                                        cmap='Reds',
                                        xticklabels=group_names,
                                        yticklabels=group_names)

            clustermap.cax.set_visible(False)

            heatmap_ax = clustermap.ax_heatmap
            heatmap_ax.invert_xaxis()
            clustermap.ax_col_dendrogram.invert_xaxis()
            clustermap.ax_col_colors.invert_xaxis()

            cluster_id = ordered_clustering[0]
            for row_col_idx, next_cluster_id in enumerate(ordered_clustering):
                if next_cluster_id != cluster_id:
                    heatmap_ax.axhline(row_col_idx, c="w", linewidth=2, alpha=1)
                    heatmap_ax.axvline(row_col_idx, c="w", linewidth=2, alpha=1)
                    cluster_id = next_cluster_id

            heatmap_ax.yaxis.set_ticks_position("left")
            heatmap_ax.yaxis.set_ticklabels(clustermap.ax_heatmap.yaxis.get_ticklabels(), rotation=0)
            heatmap_ax.xaxis.set_ticks_position("top")
            heatmap_ax.xaxis.set_ticklabels(clustermap.ax_heatmap.xaxis.get_ticklabels(), rotation=90)

            dendrogram_axis_offset = 0.05
            dendrogram_shrink_factor = 0.5

            position_array = get_ax_position(clustermap.ax_col_dendrogram)
            position_array[1] += dendrogram_axis_offset * position_array[3]
            position_array[3] *= dendrogram_shrink_factor
            clustermap.ax_col_dendrogram.set_position(position_array)

            position_array = get_ax_position(clustermap.ax_row_dendrogram)
            position_array[2] *= (1 - dendrogram_axis_offset)
            position_array[0] += position_array[2] * dendrogram_shrink_factor
            position_array[2] *= dendrogram_shrink_factor
            clustermap.ax_row_dendrogram.set_position(position_array)

            color_axis_resize_factor = 1.5

            position_array = get_ax_position(clustermap.ax_col_colors)
            position_array[3] *= color_axis_resize_factor
            clustermap.ax_col_colors.set_position(position_array)

            position_array = get_ax_position(clustermap.ax_row_colors)
            position_array[0] -= (color_axis_resize_factor - 1) * position_array[2]
            position_array[2] = position_array[2] * color_axis_resize_factor + 0.1
            clustermap.ax_row_colors.set_position(position_array)

            plt.savefig(layer_plot_path + "/plot.png", bbox_inches='tight')
            plt.close()
            # plt.show()

        # writing ordering of all layers into a single json file
        with open(plot_path + "/ordering.json", "w") as outfile:
            ordering_dict_json = json.dumps(ordering_dict)
            outfile.write(ordering_dict_json)

        result = {
            "basic_path": plot_path,
            "layer_names": layer_names,
            "ordering": ordering_dict_json
        }

        return result

    def plot_topomap(self,
                     layer_names=None,
                     grid=None,
                     method=None,
                     method_params=None):

        if method in ["graph", "PSO", "graph_PSO",
                      "SOM", "SOM_PSO", "PCA", "PCA_PSO",
                      "TSNE", "TSNE_PSO", "UMAP", "UMAP_PSO"]:
            plot_path = self.create_plot_paths("topomap", layer_names)

            n_epochs = method_params['n_epochs']
            nap = self.file_service.get_by_layer_names(layer_names, self.nap_path)

            for layer_name in layer_names:
                layer_plot_path = os.path.join(plot_path, layer_name)

                groups, n_groups, nap_values = self.get_layer_NAP_results(nap, layer_name)

                # TODO check why dense is excluded
                if len(nap_values[0]['g_nap'].shape) == 1:
                    print(f"skipping nap for {layer_name}")
                    continue

                group_names = [nap_value["g_n"] for nap_value in nap_values]
                n_groups = len(group_names)
                profile_values = np.array([np.array(nap_value["g_nap"]) for nap_value in nap_values])
                # add dimension for dense layers to match convolution layer dimensionality
                profile_shape = profile_values.shape

                # compute topomaps only if channel dimension is larger than one
                if profile_shape[-1] > 1:
                    # expand dims of NAPs if it is a 1D convolutional layer
                    if len(profile_shape) == 3:
                        profile_values = np.expand_dims(profile_values, 1)
                        profile_shape = profile_values.shape

                    # flattened profile per channel - each column contains nap values of one output channel, all groups stacked
                    flat_channel_profile = np.transpose(np.reshape(profile_values,
                                                                   [np.prod(profile_shape[:-1]), profile_shape[-1]]
                                                                   ))
                    # remove channel positions which are (almost) zero everywhere
                    flat_channel_profile = flat_channel_profile[:, np.max(np.abs(flat_channel_profile), 0) > 0.01]

                    dist_mat = compute_distance_metric(flat_channel_profile,
                                                       'euclidean')

                    # compute topomap coordinates depending on method
                    # TODO potential for SOLID principle
                    init_method = method.split("_")[0]
                    if init_method == 'PSO':
                        init_coordinates = train_PSO(dist_mat)
                    elif init_method == 'graph':
                        graph = make_threshold_graph(dist_mat,
                                                     np.percentile(dist_mat, 7.5))
                        init_coordinates = layout_graph(graph)
                    elif init_method == 'PCA':
                        init_coordinates = compute_PCA(flat_channel_profile)
                    elif init_method == 'TSNE':
                        init_coordinates = compute_TSNE(flat_channel_profile)
                    elif init_method == 'UMAP':
                        init_coordinates = compute_UMAP(flat_channel_profile)
                    elif init_method == 'SOM':
                        init_coordinates = train_SOM(flat_channel_profile)
                    else:
                        init_coordinates = None  # not reachable, only for completeness

                    if method[-4:] == '_PSO':
                        coordinates = train_PSO(dist_mat,
                                                pos_init=init_coordinates)
                    else:
                        coordinates = init_coordinates

                    # scale coordinates to be normalized between 0,1 in both dimensions
                    coordinates = coordinates - np.min(coordinates, 0)
                    coordinates = coordinates / np.max(coordinates, 0)

                    # x, y = np.transpose(coordinates)

                    xx_min, yy_min = np.min(coordinates, axis=0)
                    xx_max, yy_max = np.max(coordinates, axis=0)
                    resolution = 100

                    xx, yy = np.mgrid[xx_min:xx_max:complex(resolution), yy_min:yy_max:complex(resolution)]

                    # TODO allow other options than global average pooling for color aggregation
                    colors = profile_values
                    while len(colors.shape) > 2:
                        colors = np.mean(colors, 1)

                    # np.save(layer_plot_path + "/topomap_colors.npy", colors)
                    coordinates = coordinates[:, ::-1]  # to account for the transpose in the interpolation imshow
                    coordinates[:, 0] = 1 - coordinates[:, 0]  # to account for the invert_y in the interpolation imshow
                    np.save(layer_plot_path + "/topomap_coordinates.npy",
                            coordinates)
                    interpolated_colors = np.zeros([n_groups, resolution, resolution])

                    clim = get_absmax_clim(colors)

                    # individual topomap plots are always generated
                    # if grid is given the plots are additionally saved as a merged single plot
                    if grid is not None:
                        grid_fig, grid_axes = plt.subplots(grid[0], grid[1], figsize=(2 * grid[1], 2 * grid[0]))
                        flat_axes = grid_axes.flatten()
                        for ax in flat_axes:
                            ax.axis('off')
                    for g_idx in range(n_groups):
                        g_name = group_names[g_idx]
                        individual_fig, individual_ax = plt.subplots(1, 1, figsize=(5, 5))
                        axes = [individual_ax]
                        if grid is not None:
                            row = g_idx // grid[1]
                            col = g_idx % grid[1]
                            grid_ax = grid_axes[row, col]
                            grid_ax.set_title(group_names[g_idx])
                            axes.append(grid_ax)

                        new_xy = interpolate.griddata(coordinates,
                                                      colors[g_idx],
                                                      (xx, yy),
                                                      method='linear')
                        new_xy[np.isnan(new_xy)] = 0
                        interpolated_colors[g_idx] = np.transpose(new_xy[:, ::-1])

                        # create the same topomap plot on individual and grid plot
                        for ax in axes:
                            ax.imshow(interpolated_colors[g_idx],
                                      clim=clim,
                                      cmap='bwr')
                            # ax.set_title(g_name)

                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_facecolor("white")

                        individual_fig.savefig(layer_plot_path + "/" + g_name + ".png", bbox_inches='tight')
                        plt.close(individual_fig)
                    if grid is not None:
                        grid_fig.savefig(layer_plot_path + "/" + "topomaps" + ".png", bbox_inches='tight')
                        plt.close(grid_fig)

                    np.save(layer_plot_path + "/topomap_interpolated_colors.npy",
                            interpolated_colors)

                else:
                    print(f"skipping topomap for {layer_name}, because topomaps need multiple channels")
        else:
            print(f"{method} is not available")

        result = {
            "basic_path": plot_path,
            "layer_names": layer_names,
            "groups": groups
        }

        return result

    def get_responsive_topomap_region_neurons(self,
                                              layer,
                                              group_idx,
                                              size=25,
                                              shape='circular',
                                              absolute=True):
        plot_path = self.create_plot_paths("topomap", [layer])

        layer_plot_path = os.path.join(plot_path, layer)

        topomap_coordinates = np.load(layer_plot_path + "/topomap_coordinates.npy")
        # topomap_colors = np.load(layer_plot_path + "/topomap_colors.npy")
        topomap_interpolated_colors = np.load(layer_plot_path + "/topomap_interpolated_colors.npy")

        topomap = topomap_interpolated_colors[group_idx]

        if size == 1:
            avg_pooled_topomap = topomap
        else:
            if shape == 'rectangular':
                kernel = np.ones([size, size])
                kernel = kernel / (size ** 2)
            elif shape == 'circular':

                center = 2*[int(size / 2)]
                radius = size//2

                y, x = np.ogrid[:size, :size]  # x y order does not matter for circular
                dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

                kernel = (dist_from_center <= radius).astype('int')
                kernel = kernel / np.sum(kernel)
            else:
                print(f"{shape} is not available")

            # convolve pads more at the end, if size is an odd number
            avg_pooled_topomap = convolve(topomap,
                                          kernel,
                                          mode='constant',
                                          cval=float(np.mean(topomap)))

        if absolute:
            avg_pooled_topomap = np.abs(avg_pooled_topomap)

        responsive_region_idx = np.argwhere(avg_pooled_topomap == np.max(avg_pooled_topomap))[0]

        slice_offset_low = (size - 1) // 2
        slice_offset_high = (size // 2) + 1
        lbounds = responsive_region_idx - slice_offset_low
        xfrom, yfrom = lbounds
        ubounds = responsive_region_idx + slice_offset_high
        xto, yto = ubounds
        # responsive_region = (slice(xfrom, xto),
        #                      slice(yfrom, yto))

        if shape == 'rectangular':
            coord_above_lbound = topomap_coordinates >= lbounds / 100
            coord_below_ubound = topomap_coordinates <= ubounds / 100
            coord_within_bounds = np.logical_and(coord_above_lbound, coord_below_ubound)
            both_coords_within_bounds = np.logical_and(coord_within_bounds[:, 0], coord_within_bounds[:, 1])

            responsive_neurons = np.argwhere(both_coords_within_bounds)[:, 0]
            # responsive_coordinates = topomap_coordinates[responsive_neurons]
        elif shape == 'circular':
            dist_from_argmax = np.sqrt(np.sum((topomap_coordinates - (responsive_region_idx/100)) ** 2, 1))
            responsive_neurons = np.argwhere(dist_from_argmax <= (radius/100))[:, 0]
        else:
            print(f"{shape} is not available")

        # nap = self.file_service.get_by_layer_names([layer], self.nap_path)
        # groups, n_groups, _ = self.get_layer_NAP_results(nap, layer)
        # topomap_image_path = layer_plot_path + "/" + groups[group_idx] + ".png"
        # topomap_image = np.asarray(Image.open(topomap_image_path))[:, :, :-1]
        topomap_shape = topomap.shape
        selection_image = np.zeros(topomap_shape)
        selection_image.fill(255)

        if shape == 'rectangular':
            selection_image[xfrom:xto, yfrom:yto] = 0
        elif shape == 'circular':
            selection_image[xfrom:xto, yfrom:yto] = 1 - kernel
            selection_image[selection_image == 1] = 255
        else:
            print(f"{shape} is not available")

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(selection_image,
                  clim=[0,255],
                  cmap='Greys_r')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")

        selection_image_path = layer_plot_path + "/selection.png"
        fig.savefig(selection_image_path, bbox_inches='tight')
        plt.close(fig)

        selection_image = np.asarray(Image.open(selection_image_path))[:, :, :-1]
        selection_image = Image.fromarray(selection_image)
        selection_image = get_transparent_image(selection_image)

        responsive_neurons = list(map(int, responsive_neurons))

        return {
            'responsive_neurons': responsive_neurons,
            'active_regions': get_base64_image(selection_image)
        }

    def get_manually_selected_neurons(self,
                                      layer_name,
                                      group_idx,
                                      selection_image):
        logging.info("get_manually_selected_neurons method has started")

        plot_path = self.create_plot_paths("topomap", [layer_name])
        nap = self.file_service.get_by_layer_names([layer_name], self.nap_path)
        layer_plot_path = os.path.join(plot_path, layer_name)

        groups, n_groups, _ = self.get_layer_NAP_results(nap, layer_name)
        topomap_image_path = layer_plot_path + "/" + groups[group_idx] + ".png"

        topomap_coordinates = np.load(layer_plot_path + "/topomap_coordinates.npy")

        topomap_image = np.asarray(Image.open(topomap_image_path))[:, :, :-1]
        selection_image = np.asarray(Image.open(io.BytesIO(base64.b64decode(selection_image))))[:, :, :-1]

        # convert selection to binary mask
        selection_image = np.mean(selection_image, 2)
        selection_image[selection_image > 245] = -1
        selection_image[selection_image != -1] = 1
        selection_image[selection_image == -1] = 0

        # filling: if necessary, fill the selection area
        content = np.copy(np.uint8(selection_image))
        mask = np.copy(content)
        mask = np.pad(mask, 1)
        content *= 255

        # filling: extracting background by filling from a non-selected point as seed
        # seed = tuple(np.argwhere(content == 0)[0])
        # filled = cv2.floodFill(image=content,
        #                       mask=mask,
        #                       seedPoint=seed,
        #                       newVal=255)

        # filling: invert background image to get selection + filled holes of selection
        # selection_image = selection_image + 1 - (filled[1]) / 255.

        # remove outer whitespace in topomap from both inputs
        orig_shape = topomap_image.shape
        is_white_pixel = np.mean(topomap_image, 2) == 255

        is_white_row = np.sum(is_white_pixel, 1) == orig_shape[1]
        is_white_col = np.sum(is_white_pixel, 0) == orig_shape[0]

        # topomap_image = topomap_image[np.logical_not(is_white_row)]
        # topomap_image = topomap_image[:, np.logical_not(is_white_col)]
        selection_image = selection_image[np.logical_not(is_white_row)]
        selection_image = selection_image[:, np.logical_not(is_white_col)]

        # divide into grid based on topomap dim
        selection_image_shape = np.array(selection_image.shape)

        topomap_coordinates_scaled = (topomap_coordinates * (selection_image_shape - 1)).astype('int')
        is_in_selection = selection_image[topomap_coordinates_scaled[:, 0], topomap_coordinates_scaled[:, 1]]

        responsive_neurons = np.argwhere(is_in_selection > 0.98)[:, 0]   # == 1 with allowing small error
        responsive_neurons = list(map(int, responsive_neurons))

        return responsive_neurons

    def feature_visualization(self, dataset_path, model_path, visualize_settings):
        logging.info("feature_visualization method has started")

        model_service = ModelService(model_path)
        outputs = model_service.get_model_outputs()
        model = model_service.get_differentiable_model(outputs)
        layers = model_service.get_layers()

        profile_service = ProfileService(model_path, dataset_path)
        profile = profile_service.get_method_by_name(self.profile_name)
        model_info = profile["model"]
        channels_first = model_info["chf"]
        activation_result = profile["mthds"][0]["activation"]
        layer_shapes = self.file_service.get_shape_by_layer_names(layers, activation_result)

        input_layer_shape = layer_shapes[
            model.input.name] if model.input.name in layer_shapes else model_service.get_shape()

        feature_visualization_results = []

        if channels_first:
            input_layer_shape = [input_layer_shape[-1]] + input_layer_shape[0:-1]

        # Batch Optimization is found to be slower than optimizing individual images
        for visualize_setting in visualize_settings:
            layer_name = visualize_setting["layer_name"]
            layer_idx = layers.index(layer_name)
            layer_shape = layer_shapes[layer_name]

            group_idx = visualize_setting["group_idx"]
            nap = self.file_service.get_by_layer_names([layer_name], self.nap_path)
            groups, n_groups, nap_values = self.get_layer_NAP_results(nap, layer_name)
            group_name = groups[group_idx]
            profile_values = np.array([np.array(nap_value["g_nap"]) for nap_value in nap_values])
            group_activation = profile_values[group_idx]

            responsive_neurons = visualize_setting["responsive_neurons"]
            is_responsive = visualize_setting["is_responsive"]
            feature_visualization_type = visualize_setting["feature_visualization_type"]

            image_shape = (1, *tuple(input_layer_shape))
            img = tf.Variable(tf.random.uniform(image_shape))

            if is_responsive:
                neurons = responsive_neurons
            else:
                neurons = generate_non_responsive_neurons(responsive_neurons, no_neurons=layer_shape[-1])

            if len(neurons) == 0:
                logging.info(f"skipping feature visualization for layer {layer_name}, group {group_name}")
                continue

            for step in range(50):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(img)
                    img.assign_add(tf.random.uniform(image_shape, 0, 0.01))
                    activations = model(img, training=False)
                    activations_layer = activations[layer_idx]
                    if channels_first:
                        activations_layer = tf.transpose(activations_layer,
                                                         [0] + list(np.arange(2, len(activations_layer.shape))) + [1])

                    activations_neurons = tf.gather(activations_layer[0], neurons, axis=-1)

                    if feature_visualization_type == FEATURE_VISUALIZATION_TYPES.activation_maximization.value:
                        loss = tf.reduce_mean(activations_neurons)

                    elif feature_visualization_type == FEATURE_VISUALIZATION_TYPES.optimization_over_group_averages.value:
                        group_activation_neurons = tf.cast(tf.gather(group_activation, neurons, axis=-1), tf.float32)
                        loss = tf.reduce_mean(tf.math.square(activations_neurons - group_activation_neurons))

                grads = tape.gradient(loss, img)
                grads = tf.sign(grads)

                if feature_visualization_type == FEATURE_VISUALIZATION_TYPES.activation_maximization.value:
                    epsilon = 0.1
                    img.assign_add(epsilon * grads)  # Gradient Ascent

                elif feature_visualization_type == FEATURE_VISUALIZATION_TYPES.optimization_over_group_averages.value:
                    epsilon = 0.1
                    img.assign_sub(epsilon * grads)  # Gradient Descent

                img.assign(tf.clip_by_value(img, 0., 1.))
                img.assign(gaussian_filter(img.numpy(), sigma=0.3))

            plt.axis('off')
            plt.imshow(img[0])
            bytesIO = io.BytesIO()
            plt.savefig(bytesIO, format='png', pad_inches=0, bbox_inches='tight')
            bytesIO.seek(0)

            optimized_image = base64.b64encode(bytesIO.read()).decode('utf-8')

            feature_visualization_result = {
                "feature_visualization_type": feature_visualization_type,
                "is_responsive": is_responsive,
                "group_name": group_name,
                "layer_name": layer_name,
                "image": optimized_image
            }
            feature_visualization_results.append(feature_visualization_result)

        return feature_visualization_results
