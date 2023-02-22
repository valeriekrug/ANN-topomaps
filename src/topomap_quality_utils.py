import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import auc
#from Utility.topomap_utils import *
import cv2


def compute_topomap_layout(channel_profile, topomap_method):
    dist_mat = compute_distance_metric(channel_profile, 'euclidean')

    init_method = topomap_method.split("_")[0]
    if init_method == 'PSO':
        init_coordinates = train_PSO(dist_mat)
    elif init_method == 'graph':
        graph = make_threshold_graph(dist_mat,
                                     np.percentile(dist_mat, 7.5))
        init_coordinates = layout_graph(graph)
    elif init_method == 'PCA':
        init_coordinates = compute_PCA(channel_profile)
    elif init_method == 'TSNE':
        init_coordinates = compute_TSNE(channel_profile)
    elif init_method == 'UMAP':
        init_coordinates = compute_UMAP(channel_profile)
    elif init_method == 'SOM':
        init_coordinates = train_SOM(channel_profile)
    elif init_method == 'random':
        init_coordinates = random_layout(channel_profile.shape[0])
    else:
        init_coordinates = None  # not reachable, only for completeness

    if topomap_method[-4:] == '_PSO':
        coordinates = train_PSO(dist_mat,
                                pos_init=init_coordinates)
    else:
        coordinates = init_coordinates

    # scale coordinates to be normalized between 0,1 in both dimensions
    coordinates = coordinates - np.min(coordinates, 0)
    coordinates = coordinates / np.max(coordinates, 0)

    return coordinates

def coordinates_to_interpolate_grid(coordinates, resolution=100):
    coordinates = coordinates[:, ::-1]  # to account for the transpose in the interpolation imshow
    coordinates[:, 0] = 1 - coordinates[:, 0]  # to account for the invert_y in the interpolation imshow
    # x, y = np.transpose(coordinates)

    xx_min, yy_min = np.min(coordinates, axis=0)
    xx_max, yy_max = np.max(coordinates, axis=0)

    xx, yy = np.mgrid[xx_min:xx_max:complex(resolution), yy_min:yy_max:complex(resolution)]

    return xx, yy

def topomap_rescale(img, shrink_size):
    shrink_size = 55 - (shrink_size * 5)
    res_img = img.resize((shrink_size, shrink_size))
    res_img_big = res_img.resize(img.size)
    arr_img_rescaled = np.asarray(res_img_big)
    return arr_img_rescaled


def topomap_blur(img, radius):
    radius = (radius * 0.5) + 0.5
    blur_img = img.filter(ImageFilter.GaussianBlur(radius))
    arr_img_blurred = np.asarray(blur_img)
    return arr_img_blurred


def topomap_mse(arr_img, metric, params):

    img_manipulation_fn = None
    if metric == 'resize_mse':
        img_manipulation_fn = topomap_rescale
    elif metric == 'blur_mse':
        img_manipulation_fn = topomap_blur

    errs = np.zeros(len(params))
    img, arr_img_converted  = ndarray_to_image(arr_img, return_array=True)
    for i, param in enumerate(params):
        arr_img_changed = img_manipulation_fn(img, param)

        err = np.abs(arr_img_converted[:,:,:-1] - arr_img_changed[:,:,:-1]) / 255.0
        err = np.sum(err) / np.sum(arr_img_converted[:,:,:-1] / 255.0)
        errs[i] = err

    img.close()
    return errs


def compute_topomap_image_quality(image_arrays, metric, params=None, return_auc=False):
    if 'mse' in metric:
        group_errors = []
        for group_image in image_arrays:
            group_errors.append(topomap_mse(group_image,
                                            metric,
                                            params))
        group_errors = np.array(group_errors)
        quality_values = list(np.mean(group_errors, axis=0))

        if return_auc:
            mse_auc = auc(np.arange(len(quality_values)), quality_values)
            quality_values.append(mse_auc)

        return np.array(quality_values)
    elif metric == 'components':
        n_components_per_group = list()
        avg_component_area_per_group = list()
        for arr_img in image_arrays:
            _, arr_img = ndarray_to_image(arr_img, return_array=True)
            arr_img = arr_img[:,:,:-1]
            # cv2.imwrite('/project/ankrug/PhD_Thesis/experiments/01_input.png', arr_img)

            n_components = 0
            total_component_size = 0
            for channel_id, channel_name in zip([0,2],['red','blue']):
                # img_grey = cv2.cvtColor(arr_img, cv2.COLOR_BGR2GRAY)
                img_grey_channel = arr_img[:,:,channel_id]
                # set a thresh
                thresh = 230
                # get threshold image
                ret, thresh_img = cv2.threshold(img_grey_channel, thresh, 255, cv2.THRESH_BINARY)
                thresh_img = 255 - thresh_img
                retval, labels = cv2.connectedComponents(thresh_img)
                component_ids, component_sizes = np.unique(labels,return_counts=True)
                is_component = component_sizes >= 10
                bg_label = int(np.argwhere(component_ids == labels[0,0]))
                bg_size = component_sizes[bg_label]
                relevant_component_size = np.sum(component_sizes[is_component]) - bg_size
                n_channel_components = np.sum(is_component) -1 #-1 to account for background component

                total_component_size = total_component_size + relevant_component_size
                n_components = n_components + n_channel_components

                # Map component labels to hue val
                # label_hue = np.uint8(179 * labels / np.max(labels))
                # blank_ch = 255 * np.ones_like(label_hue)
                # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
                # # cvt to BGR for display
                # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
                # # set bg label to black
                # labeled_img[label_hue == 0] = 0

                # cv2.imwrite('/project/ankrug/PhD_Thesis/experiments/02_' + channel_name + '.png', img_grey_channel)
                # cv2.imwrite('/project/ankrug/PhD_Thesis/experiments/03_binary_' + channel_name + '.png', thresh_img)
                # cv2.imwrite('/project/ankrug/PhD_Thesis/experiments/04_components_'+channel_name+'.png', labeled_img)
            n_components_per_group.append(n_components)
            if n_components>0:
                avg_component_area_per_group.append(total_component_size / n_components)
            else:
                avg_component_area_per_group.append(0)


            # # find contours
            # contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # # create an empty image for contours
            # img_contours = np.zeros(arr_img.shape)
            # # draw the contours on the empty image
            # cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
            # # save image
            # cv2.imwrite('/project/ankrug/PhD_Thesis/experiments/04_contours.png',img_contours)
        avg_n_components = np.mean(n_components_per_group)
        avg_component_area = np.mean(avg_component_area_per_group)
        return np.array([[avg_n_components, avg_component_area]])
    elif metric == 'group_dist':
        # group distance is not a good measure
        # e.g. if many overall inactive FMs are collapsed in one position,
        #      it'll create higher group distances
        #      but this is not a good layout
        flat_image_arrays = np.reshape(image_arrays, [image_arrays.shape[0], np.prod(image_arrays.shape[1:])])
        dmat = compute_distance_metric(flat_image_arrays, 'cityblock')
        normalizer = np.sum(np.abs(flat_image_arrays), 1)
        normalizer = np.mean(normalizer)
        avg_group_dist = np.mean(dmat / normalizer)
        return np.array([avg_group_dist])
    else:
        raise ValueError("metric " + metric + " not available.")

def ndarray_to_image(arr_img, return_array=False):
    arr_img_converted = np.zeros((arr_img.shape[0], arr_img.shape[1], 4))
    arr_img_converted[:, :, :-1] = arr_img
    arr_img_converted[:, :, -1] = 1
    arr_img_converted = (arr_img_converted * 255).astype('uint8')
    img = Image.fromarray(arr_img_converted)
    if return_array:
        return img, arr_img_converted
    else:
        return img